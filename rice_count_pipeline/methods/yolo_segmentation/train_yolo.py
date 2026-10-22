import argparse
import shutil
from pathlib import Path

try:
    from .runtime_paths import ROOT
except ImportError:
    from runtime_paths import ROOT
import torch
from ultralytics import YOLO

DATA_HOME = ROOT.parent / "data" / "rice_count"
ARTIFACTS = ROOT.parent / "models" / "experiments" / "yolo_segmentation"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="yolo11s-seg.pt")
    args = ap.parse_args()
    data = DATA_HOME / "datasets" / "yolo_segmentation" / "data.yaml"
    if not data.is_file():
        raise SystemExit(f"Missing dataset config: {data}; run build_dataset.py --source reviewed first")
    plans = [(4, 1280), (2, 1280), (2, 1024)]
    for attempt, (batch, imgsz) in enumerate(plans):
        model = None
        try:
            model = YOLO(args.model)
            model.train(data=str(data), imgsz=imgsz, epochs=150, patience=30, batch=batch,
                        cos_lr=True, amp=True, workers=2, close_mosaic=15, mosaic=1.0,
                        degrees=90, flipud=0.5, fliplr=0.5, scale=0.4, translate=0.1,
                        hsv_h=0.01, hsv_s=0.4, hsv_v=0.4, copy_paste=0.2, mixup=0,
                        overlap_mask=True, max_det=1000, seed=0,
                        project=str(ARTIFACTS / "runs"),
                        name=args.name, exist_ok=True)
            metrics = model.val(data=str(data), imgsz=imgsz, split="val", max_det=2000)
            print(f"val mask mAP50={metrics.seg.map50:.4f}")
            for i, score in enumerate(metrics.seg.ap50):
                print(f"class {i} mask mAP50={score:.4f}")
            best = Path(model.trainer.best)
            dest = ARTIFACTS / f"{args.name}.pt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best, dest)
            print(f"weights: {dest}")
            return
        except Exception as exc:
            oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "cuda out of memory" in str(exc).lower()
            if not oom or attempt == len(plans) - 1:
                raise
            print(f"CUDA OOM at batch={batch}, imgsz={imgsz}; retrying batch={plans[attempt + 1][0]}, imgsz={plans[attempt + 1][1]}")
            if model is not None: del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
