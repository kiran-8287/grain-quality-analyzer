"""Run the experimental rice/foreign model over filename-counted test photos."""
import argparse
import csv
import gc
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
REPO_ROOT = PROJECT.parent
DATA_HOME = REPO_ROOT / "data" / "rice_count"
ARTIFACTS = REPO_ROOT / "models" / "experiments" / "weak_count_model"
RESULTS = ROOT / "results"
RUNTIME_CACHE = DATA_HOME / "work" / "runtime_cache"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ARTIFACTS / "ultralytics_config"))
os.environ.setdefault("TORCH_HOME", str(RUNTIME_CACHE / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))

import cv2
import numpy as np
import torch
from ultralytics import YOLO

WEIGHTS = ARTIFACTS / "weak_count.pt"
NAME_COUNTS = re.compile(r"(\d+)\s*whole\s*,?\s*(\d+)\s*broken\s*,?\s*(\d+)\s*stones", re.I)
MAX_DET = 3000
CONFIDENCE = 0.35


def predict_with_memory_fallback(model, path, requested_imgsz):
    import torch
    sizes = list(dict.fromkeys([requested_imgsz, min(requested_imgsz, 896),
                                min(requested_imgsz, 768), min(requested_imgsz, 640)]))
    for imgsz in sizes:
        try:
            result = model.predict(str(path), imgsz=imgsz, conf=CONFIDENCE, iou=0.6,
                                   max_det=MAX_DET, device=0, verbose=False)[0]
            return result, imgsz
        except Exception as exc:
            oom = (isinstance(exc, (torch.cuda.OutOfMemoryError, MemoryError))
                   or "out of memory" in str(exc).lower())
            if not oom or imgsz == sizes[-1]:
                raise
            torch.cuda.empty_cache()
            print(f"CUDA OOM on {path.name} at imgsz={imgsz}; retrying smaller")


def expected_counts(path):
    match = NAME_COUNTS.search(path.stem)
    if not match:
        return None
    whole, broken, foreign = map(int, match.groups())
    return whole, broken, whole + broken, foreign


def save_mask_overlay(path, result, target, rice_count, foreign_count,
                      rice_pct, foreign_pct, long_side=1536):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read {path}")
    h, w = image.shape[:2]
    scale = min(1.0, long_side / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (round(w * scale), round(h * scale)),
                           interpolation=cv2.INTER_AREA)
    overlay = image.copy()
    mask_data = result.masks.data if result.masks is not None else None
    classes = result.boxes.cls.int().tolist() if result.boxes is not None else []
    colors = ((255, 0, 255), (0, 165, 255))
    target_size = (image.shape[1], image.shape[0])
    color_layers = {cls: np.full_like(overlay, color) for cls, color in enumerate(colors)}
    if mask_data is not None:
        for i, cls in enumerate(classes):
            if cls not in (0, 1) or i >= len(mask_data):
                continue
            mask = mask_data[i].detach().to("cpu").numpy()
            mask = cv2.resize((mask > 0).astype(np.uint8), target_size,
                              interpolation=cv2.INTER_NEAREST)
            cv2.copyTo(color_layers[cls], mask, overlay)
    image = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)
    cv2.putText(image, f"rice={rice_count} ({rice_pct:.1f}%)  foreign={foreign_count} ({foreign_pct:.1f}%)",
                (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(target), image):
        raise OSError(f"Could not save mask overlay: {target}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DATA_HOME / "test")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--out", type=Path, default=RESULTS / "test_set")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this experiment")
    if not args.weights.is_file():
        raise SystemExit(f"Missing weights: {args.weights}")
    paths = sorted(p for p in args.source.iterdir()
                   if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        raise SystemExit(f"No images under {args.source}")
    out = args.out
    overlays = out / "overlays"
    mask_overlays = out / "mask_overlays"
    overlays.mkdir(parents=True, exist_ok=True)
    mask_overlays.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    rows = []
    for i, path in enumerate(paths, 1):
        assert MAX_DET >= 2000
        result, used_imgsz = predict_with_memory_fallback(model, path, args.imgsz)
        counts = [0, 0]
        if result.boxes is not None:
            for cls in result.boxes.cls.int().tolist():
                if cls in (0, 1):
                    counts[cls] += 1
        total = sum(counts)
        rice_pct = 100 * counts[0] / total if total else 0.0
        foreign_pct = 100 * counts[1] / total if total else 0.0
        truth = expected_counts(path)
        whole, broken, true_rice, true_foreign = truth or (None, None, None, None)
        rice_match = counts[0] == true_rice if truth else None
        foreign_match = counts[1] == true_foreign if truth else None
        rows.append({
            "file": path.name, "true_whole": whole, "true_broken": broken,
            "true_rice_total": true_rice, "true_foreign": true_foreign,
            "pred_rice": counts[0], "pred_foreign": counts[1],
            "inference_imgsz": used_imgsz, "confidence": CONFIDENCE,
            "pred_rice_pct": round(rice_pct, 2), "pred_foreign_pct": round(foreign_pct, 2),
            "true_rice_pct": round(100 * true_rice / (true_rice + true_foreign), 2) if truth else None,
            "true_foreign_pct": round(100 * true_foreign / (true_rice + true_foreign), 2) if truth else None,
            "rice_abs_error": abs(counts[0] - true_rice) if truth else None,
            "foreign_abs_error": abs(counts[1] - true_foreign) if truth else None,
            "rice_match": rice_match, "foreign_match": foreign_match,
            "both_match": rice_match and foreign_match if truth else None,
        })
        image = result.plot(masks=False)
        status = (f"expected rice={true_rice} foreign={true_foreign}" if truth
                  else "filename counts unavailable")
        cv2.putText(image, f"rice={counts[0]} ({rice_pct:.1f}%)  foreign={counts[1]} ({foreign_pct:.1f}%)",
                    (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(image, status, (24, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(overlays / path.name), image)
        save_mask_overlay(path, result, mask_overlays / path.name, counts[0], counts[1],
                          rice_pct, foreign_pct)
        print(f"{i}/{len(paths)} {path.name}: predicted {counts[0]} rice ({rice_pct:.1f}%), "
              f"{counts[1]} foreign ({foreign_pct:.1f}%); expected {true_rice}/{true_foreign}; "
              f"match rice={rice_match} foreign={foreign_match}; imgsz={used_imgsz}")
        del result, image
        torch.cuda.empty_cache()
        gc.collect()

    report = out / "test_counts.csv"
    with report.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    known = [r for r in rows if r["both_match"] is not None]
    if known:
        n = len(known)
        both = sum(r["both_match"] for r in known)
        rice_mae = sum(r["rice_abs_error"] for r in known) / n
        foreign_mae = sum(r["foreign_abs_error"] for r in known) / n
        print(f"SUMMARY {n} named images: exact both={both}/{n}; "
              f"rice MAE={rice_mae:.2f}; foreign MAE={foreign_mae:.2f}")
    print(f"CSV: {report}")
    print(f"Overlays: {overlays}")
    print(f"Mask overlays: {mask_overlays}")
    print("Counts are experimental; validate predictions against the filename counts.")


if __name__ == "__main__":
    main()
