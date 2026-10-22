"""Count rice and foreign instances with the isolated experimental model."""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
REPO_ROOT = PROJECT.parent
DATA_HOME = REPO_ROOT / "data" / "rice_count"
ARTIFACTS = REPO_ROOT / "models" / "experiments" / "weak_count_model"
RUNTIME_CACHE = DATA_HOME / "work" / "runtime_cache"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ARTIFACTS / "ultralytics_config"))
os.environ.setdefault("TORCH_HOME", str(RUNTIME_CACHE / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))

import cv2
from ultralytics import YOLO

WEIGHTS = ARTIFACTS / "weak_count.pt"
MAX_DET = 2000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not args.image.is_file() or not args.weights.is_file():
        raise SystemExit("Image and trained experiment weights must both exist")
    assert MAX_DET >= 2000
    model = YOLO(str(args.weights))
    result = model.predict(str(args.image), imgsz=1280, conf=0.25, iou=0.6,
                           max_det=MAX_DET, verbose=False)[0]
    counts = [0, 0]
    if result.boxes is not None:
        for cls in result.boxes.cls.int().tolist():
            if cls in (0, 1):
                counts[cls] += 1
    total = sum(counts)
    rice_pct = 100 * counts[0] / total if total else 0.0
    foreign_pct = 100 * counts[1] / total if total else 0.0
    print(f"EXPERIMENTAL counts: rice={counts[0]}, foreign={counts[1]}; "
          f"percentages={rice_pct:.1f}%/{foreign_pct:.1f}%")
    print("Predictions are not verified until compared with the image's exact counts.")
    vis = result.plot()
    cv2.putText(vis, f"rice={counts[0]} foreign={counts[1]}", (24, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
    target = args.out or ARTIFACTS / "predictions" / f"{args.image.stem}_count.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), vis):
        raise OSError(f"Could not write overlay: {target}")
    print(f"overlay: {target}")


if __name__ == "__main__":
    main()
