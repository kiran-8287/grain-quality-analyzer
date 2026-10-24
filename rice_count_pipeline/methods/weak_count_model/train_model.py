"""Train an isolated YOLO-seg student on classical auto-label pseudo-masks."""
import argparse
import csv
import json
import os
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
DATA_HOME = PROJECT.parent / "data" / "rice_count"
REPO_ROOT = PROJECT.parent
ARTIFACTS = REPO_ROOT / "models" / "experiments" / "weak_count_model"
RUNTIME_CACHE = DATA_HOME / "work" / "runtime_cache"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ARTIFACTS / "ultralytics_config"))
os.environ.setdefault("TORCH_HOME", str(RUNTIME_CACHE / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))

import numpy as np
import torch
from ultralytics import YOLO

SEED = 0
DATASET = DATA_HOME / "datasets" / "weak_count_model_all_raw"
DATA = DATASET / "data.yaml"
PSEUDO_COUNTS = DATA_HOME / "work" / "weak_count_model_all_raw" / "pseudo_counts.csv"
RUNS = ARTIFACTS / "runs"
WEIGHTS = ARTIFACTS / "weak_count.pt"
MAX_DET = 3000


def score_counts(model, imgsz):
    image_dir = DATASET / "images" / "all"
    targets = {}
    with PSEUDO_COUNTS.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            targets[row["file"]] = (int(row["pseudo_rice"]), int(row["pseudo_foreign"]))
    rows = []
    for result in model.predict(source=str(image_dir), imgsz=imgsz, conf=0.35,
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
    if len(rows) != len(targets):
        raise RuntimeError(f"Predicted {len(rows)} images but found {len(targets)} pseudo-count labels")
    return rows


def summarize(rows, label):
    if not rows:
        print(f"{label}: no images")
        return
    mae_r = np.mean([r["abs_rice"] for r in rows])
    mae_f = np.mean([r["abs_foreign"] for r in rows])
    within3 = np.mean([(r["abs_rice"] <= 3 and r["abs_foreign"] <= 1) for r in rows])
    print(f"{label} count fit vs autolabel pseudo-counts: rice MAE={mae_r:.2f}, "
          f"foreign MAE={mae_f:.2f}, both within rice±3/foreign±1={within3:.1%}")


def write_count_report(model, imgsz):
    rows = score_counts(model, imgsz)
    all_rows = [{**r, "split": "all_raw_training_images_in_sample"} for r in rows]
    report = ARTIFACTS / "all_raw_count_fit_vs_pseudo_labels.csv"
    with report.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summarize(rows, "ALL-RAW TRAIN FIT (in-sample)")
    print(f"Count-fit report (pseudo targets only): {report}")
    print("These are agreement scores against auto-labels, not ground-truth accuracy.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="weak_count_lab_v3_30ep_768")
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--resume", action="store_true",
                        help="resume this run from its saved last.pt checkpoint")
    parser.add_argument("--score-only", action="store_true",
                        help="score saved best weights against pseudo-label counts")
    parser.add_argument("--validate-only", action="store_true",
                        help="run in-sample pseudo-mask metrics on saved weights with zero workers")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this RTX 3050 experiment")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    if not DATA.is_file():
        raise SystemExit(f"Missing {DATA}; run build_all_raw_dataset.py first")
    if args.validate_only:
        if not WEIGHTS.is_file():
            raise SystemExit(f"Missing trained weights: {WEIGHTS}")
        metrics = YOLO(str(WEIGHTS)).val(data=str(DATA), imgsz=args.imgsz, split="val",
                                        batch=1, workers=0, max_det=MAX_DET,
                                        device=0, plots=False)
        report = {
            "scope": "in-sample agreement against proxy masks on all 860 training photos",
            "imgsz": args.imgsz,
            "box_precision": float(metrics.box.mp), "box_recall": float(metrics.box.mr),
            "box_map50": float(metrics.box.map50), "box_map50_95": float(metrics.box.map),
            "mask_precision": float(metrics.seg.mp), "mask_recall": float(metrics.seg.mr),
            "mask_map50": float(metrics.seg.map50), "mask_map50_95": float(metrics.seg.map),
            "rice_mask_ap50": float(metrics.seg.ap50[0]),
            "foreign_mask_ap50": float(metrics.seg.ap50[1]),
        }
        output = ARTIFACTS / "in_sample_proxy_mask_metrics.json"
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"IN-SAMPLE proxy-mask metrics: {report}")
        print(f"Metrics JSON: {output}")
        return
    if args.score_only:
        if not WEIGHTS.is_file():
            raise SystemExit(f"Missing trained weights: {WEIGHTS}")
        write_count_report(YOLO(str(WEIGHTS)), args.imgsz)
        return
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    checkpoint = RUNS / args.name / "weights" / "last.pt"
    if args.resume and not checkpoint.is_file():
        raise SystemExit(f"No resume checkpoint found: {checkpoint}")
    plans = [(2, args.imgsz), (1, args.imgsz), (1, 896), (1, 768)] if args.resume else [
        (1, args.imgsz), (1, min(args.imgsz, 896)),
        (1, min(args.imgsz, 768)), (1, min(args.imgsz, 640))]
    plans = list(dict.fromkeys(plans))
    trained_model = None
    trained_imgsz = None
    for attempt, (batch, imgsz) in enumerate(plans):
        model = None
        try:
            model = YOLO(str(checkpoint) if args.resume else args.model)
            if args.resume:
                model.train(resume=True)
            else:
                model.train(data=str(DATA), imgsz=imgsz, epochs=args.epochs, patience=25,
                            batch=batch, workers=0, device=0, seed=SEED, deterministic=True,
                            val=False,
                            cos_lr=True, amp=True, close_mosaic=0, mosaic=0.0,
                            degrees=90, flipud=0.5, fliplr=0.5, scale=0.35,
                            translate=0.1, hsv_h=0.01, hsv_s=0.4, hsv_v=0.4,
                            overlap_mask=True, mask_ratio=4, max_det=MAX_DET, project=str(RUNS),
                            name=args.name, exist_ok=True, plots=False)
            trained_imgsz = int(model.trainer.args.imgsz)
            best = Path(model.trainer.best)
            if not best.is_file():
                best = Path(model.trainer.last)
            WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best, WEIGHTS)
            trained_model = YOLO(str(WEIGHTS))
            print(f"Training completed at imgsz={trained_imgsz}, batch={batch}; weights: {WEIGHTS}")
            break
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
    if trained_model is None:
        raise RuntimeError("Training did not complete on any GPU memory plan")

    # This is a fit check against the proxy labels on those same 860 photos,
    # not independent validation accuracy.
    val_imgsz = trained_imgsz
    while True:
        try:
            val = trained_model.val(data=str(DATA), imgsz=val_imgsz, split="val", batch=1,
                                    workers=0, max_det=MAX_DET, device=0, plots=False)
            print(f"IN-SAMPLE TRAIN-SET mask precision={val.seg.mp:.4f}, "
                  f"recall={val.seg.mr:.4f}, mAP50-95={val.seg.map:.4f}")
            print(f"IN-SAMPLE TRAIN-SET mask mAP50 at imgsz={val_imgsz}="
                  f"{val.seg.map50:.4f}")
            for cls, score in enumerate(val.seg.ap50):
                print(f"IN-SAMPLE TRAIN-SET class {cls} mask mAP50={score:.4f}")
            break
        except Exception as exc:
            oom = (isinstance(exc, (torch.cuda.OutOfMemoryError, MemoryError))
                   or "out of memory" in str(exc).lower()
                   or "unable to allocate" in str(exc).lower())
            if not oom or val_imgsz <= 640:
                raise
            val_imgsz = max(640, val_imgsz - 128)
            torch.cuda.empty_cache()
            print(f"CUDA OOM during in-sample mAP; retrying at imgsz={val_imgsz}")
    score_imgsz = trained_imgsz
    while True:
        try:
            write_count_report(trained_model, score_imgsz)
            break
        except Exception as exc:
            oom = (isinstance(exc, (torch.cuda.OutOfMemoryError, MemoryError))
                   or "out of memory" in str(exc).lower()
                   or "unable to allocate" in str(exc).lower())
            if not oom or score_imgsz <= 640:
                raise
            score_imgsz = max(640, score_imgsz - 128)
            torch.cuda.empty_cache()
            print(f"CUDA OOM during proxy count fit; retrying at imgsz={score_imgsz}")


if __name__ == "__main__":
    main()
