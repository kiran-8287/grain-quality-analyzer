#!/usr/bin/env python3
"""
train_yolo.py -- Ultralytics YOLO for FOREIGN MATTER (husks, stones, ...) on the green sheet.

Rice grains are deliberately NOT labelled here: they are the "background" this detector must
learn to ignore, which is exactly what you want for impurity counting.

Dataset file (written for you by prepare_data.py as dataset/yolo/foreign.yaml):

    path: /abs/path/to/dataset/yolo
    train: images/train
    val: images/val
    names:
      0: foreign_matter          # or  0: husk / 1: stone  if annotated separately

Example
-------
python train_yolo.py --data dataset/yolo/foreign.yaml --model yolo11s.pt --imgsz 1280 --batch 8
python train_yolo.py --data dataset/yolo/foreign.yaml --p2          # extra stride-4 head for tiny objects

Small-object checklist
----------------------
1. Effective pixel size matters more than any hyper-parameter. YOLO letterboxes a 4000 px
   photo down to --imgsz. If foreign pieces end up smaller than ~16 px, raise --imgsz
   (1536-1920) or tile the photos into overlapping 1280 px crops before training
   (SAHI `slice_coco` does this) and tile again at inference.
2. --p2 adds a stride-4 detection head (yolov8-p2). Costs ~25-40% more compute.
3. Do not use rotation augmentation with axis-aligned boxes: it inflates the labels.
   Your physical shaking already provides rotation; we only flip.
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to foreign.yaml")
    ap.add_argument("--model", default="yolo11s.pt", help="yolo11n/s/m.pt or yolov8s.pt ...")
    ap.add_argument("--p2", action="store_true", help="use yolov8s-p2 (extra high-resolution head)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8, help="-1 = auto (single GPU only)")
    ap.add_argument("--device", default="0", help="'0', '0,1' or 'cpu'")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", default="runs/foreign")
    ap.add_argument("--name", default="exp")
    ap.add_argument("--patience", type=int, default=30, help="early-stop patience (epochs)")
    args = ap.parse_args()

    if args.p2:
        # build the P2 architecture and transfer every COCO weight whose shape still matches
        model = YOLO("yolov8s-p2.yaml").load("yolov8s.pt")
    else:
        model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        seed=0,
        deterministic=True,
        # ---- optimiser ---------------------------------------------------------------------
        optimizer="AdamW",      # robust on small datasets; lr0=0.002 is a sensible AdamW start for 1 class
        lr0=0.002,
        lrf=0.01,               # final lr = lr0 * lrf
        cos_lr=True,
        warmup_epochs=3,
        weight_decay=0.0005,
        # ---- augmentation (small-object friendly) -----------------------------------------------
        mosaic=1.0,             # 4 photos per sample: many more small objects per batch
        close_mosaic=15,        # last 15 epochs without mosaic -> matches real photos
        scale=0.3,              # default 0.5 shrinks already tiny objects too aggressively
        translate=0.1,
        degrees=0.0,            # no rotation: would inflate axis-aligned boxes
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.5,             # a top-down sheet has no 'up'
        hsv_h=0.01,             # keep hue nearly fixed: sheet colour is a stable cue
        hsv_s=0.5,
        hsv_v=0.3,
        mixup=0.0,
        copy_paste=0.0,
        # ---- misc ------------------------------------------------------------------------------
        max_det=300,            # ~50 impurities per photo expected; plenty of head-room
        amp=True,
        plots=True,
        val=True,
    )

    metrics = model.val(data=args.data, imgsz=args.imgsz, split="val", conf=0.001, iou=0.6)
    print(f"\nval  mAP50 {metrics.box.map50:.3f} | mAP50-95 {metrics.box.map:.3f} | "
          f"precision {metrics.box.mp:.3f} | recall {metrics.box.mr:.3f}")
    print("weights: ", model.trainer.best)
    print("Inference tip: model.predict(img, imgsz=%d, conf=0.25, iou=0.5, max_det=300)" % args.imgsz)


if __name__ == "__main__":
    main()
