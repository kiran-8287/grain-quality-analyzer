"""Train an isolated YOLO-seg student on classical auto-label pseudo-masks."""
import argparse
import csv
import os
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "ultralytics_config"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "torch_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "mpl_cache"))

import numpy as np
import torch
from ultralytics import YOLO

SEED = 0
DATA = ROOT / "dataset_1536_curated" / "data.yaml"
RUNS = ROOT / "runs"
WEIGHTS = ROOT / "models" / "weak_count.pt"
MAX_DET = 2000


def score_counts(model, split):
    image_dir = ROOT / "dataset_1536_curated" / "images" / split
    targets = {}
    with (ROOT / "pseudo_counts.csv").open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                targets[row["file"]] = (int(row["pseudo_rice"]), int(row["pseudo_foreign"]))
    rows = []
    for result in model.predict(source=str(image_dir), imgsz=1024, conf=0.25,
                                iou=0.6, max_det=MAX_DET, batch=1, device=0,
                                stream=True, verbose=False):
        assert MAX_DET >= 2000
        counts = [0, 0]
        if result.boxes is not None:
            for cls in result.boxes.cls.int().tolist():
                if cls in (0, 1):
                    counts[cls] += 1
        name = Path(result.path).name
        expected = targets[name]
        rows.append({"file": name, "pseudo_rice": expected[0], "pseudo_foreign": expected[1],
                     "pred_rice": counts[0], "pred_foreign": counts[1],
                     "abs_rice": abs(counts[0] - expected[0]),
                     "abs_foreign": abs(counts[1] - expected[1])})
    return rows


def summarize(rows, split):
    if not rows:
        print(f"{split}: no images")
        return
    mae_r = np.mean([r["abs_rice"] for r in rows])
    mae_f = np.mean([r["abs_foreign"] for r in rows])
    within3 = np.mean([(r["abs_rice"] <= 3 and r["abs_foreign"] <= 1) for r in rows])
    print(f"{split} count fit vs auto-label pseudo-counts: rice MAE={mae_r:.2f}, "
          f"foreign MAE={mae_f:.2f}, both within rice±3/foreign±1={within3:.1%}")


def write_count_report(model):
    train_rows = score_counts(model, "train")
    val_rows = score_counts(model, "val")
    all_rows = [{**r, "split": s} for s, rr in (("train", train_rows), ("val", val_rows)) for r in rr]
    report = ROOT / "count_fit_vs_pseudo_labels.csv"
    with report.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summarize(train_rows, "TRAIN")
    summarize(val_rows, "PSEUDO-VAL")
    print(f"Count-fit report (pseudo targets only): {report}")
    print("These are agreement scores against auto-labels, not ground-truth accuracy.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="weak_count_yolo")
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--resume", action="store_true",
                        help="resume this run from its saved last.pt checkpoint")
    parser.add_argument("--score-only", action="store_true",
                        help="score saved best weights against pseudo-label counts")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this RTX 3050 experiment")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    if not DATA.is_file():
        raise SystemExit(f"Missing {DATA}; run build_pseudo_dataset.py first")
    if args.score_only:
        if not WEIGHTS.is_file():
            raise SystemExit(f"Missing trained weights: {WEIGHTS}")
        write_count_report(YOLO(str(WEIGHTS)))
        return
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    checkpoint = RUNS / args.name / "weights" / "last.pt"
    if args.resume and not checkpoint.is_file():
        raise SystemExit(f"No resume checkpoint found: {checkpoint}")
    plans = [(1, 1024)] if args.resume else [(1, 1024), (1, 896), (1, 768)]
    for attempt, (batch, imgsz) in enumerate(plans):
        model = None
        try:
            model = YOLO(str(checkpoint) if args.resume else args.model)
            if args.resume:
                model.train(resume=True)
            else:
                model.train(data=str(DATA), imgsz=imgsz, epochs=args.epochs, patience=15,
                            batch=batch, workers=0, device=0, seed=SEED, deterministic=True,
                            cos_lr=True, amp=True, close_mosaic=10, mosaic=1.0,
                            degrees=90, flipud=0.5, fliplr=0.5, scale=0.35,
                            translate=0.1, hsv_h=0.01, hsv_s=0.4, hsv_v=0.4,
                            overlap_mask=True, mask_ratio=8, max_det=MAX_DET, project=str(RUNS),
                            name=args.name, exist_ok=True, plots=True)
            imgsz = model.trainer.args.imgsz
            val = model.val(data=str(DATA), imgsz=imgsz, split="val", max_det=MAX_DET,
                            device=0, plots=True)
            print(f"PSEUDO-VAL box mAP50={val.box.map50:.4f}")
            print(f"PSEUDO-VAL mask mAP50={val.seg.map50:.4f}")
            for cls, score in enumerate(val.seg.ap50):
                print(f"PSEUDO-VAL class {cls} mask mAP50={score:.4f}")
            best = Path(model.trainer.best)
            WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best, WEIGHTS)
            print(f"weights: {WEIGHTS}")
            write_count_report(model)
            return
        except Exception as exc:
            oom = (isinstance(exc, (torch.cuda.OutOfMemoryError, MemoryError))
                   or "out of memory" in str(exc).lower()
                   or "unable to allocate" in str(exc).lower())
            if args.resume or not oom or attempt == len(plans) - 1:
                raise
            print(f"CUDA OOM at batch={batch}, imgsz={imgsz}; retrying "
                  f"batch={plans[attempt + 1][0]}, imgsz={plans[attempt + 1][1]}")
            if model is not None:
                del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
