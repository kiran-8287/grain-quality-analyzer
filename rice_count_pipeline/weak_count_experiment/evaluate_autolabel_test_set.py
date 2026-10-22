"""Evaluate the unchanged backend HSV/KMeans auto-labeler on count-labeled photos."""
import argparse
import csv
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(WORKSPACE / "backend" / "models"))
from autolabel import label_image  # noqa: E402

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

NAME_COUNTS = re.compile(r"(\d+)\s*whole\s*,?\s*(\d+)\s*broken\s*,?\s*(\d+)\s*stones", re.I)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
LONG_SIDE = 1536


def expected_counts(path):
    match = NAME_COUNTS.search(path.stem)
    if not match:
        return None
    whole, broken, foreign = map(int, match.groups())
    return whole, broken, whole + broken, foreign


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=PROJECT / "data" / "test")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "predictions" / "test_set" / "autolabel_proxy")
    args = parser.parse_args()
    paths = sorted(p for p in args.source.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise SystemExit(f"No images under {args.source}")
    overlays = args.out / "overlays"
    overlays.mkdir(parents=True, exist_ok=True)
    rows = []

    for i, path in enumerate(paths, 1):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        h, w = image.shape[:2]
        scale = min(1.0, LONG_SIDE / max(h, w))
        vis = (cv2.resize(image, (round(w * scale), round(h * scale)),
                          interpolation=cv2.INTER_AREA) if scale < 1 else image.copy())
        counts = {"rice": 0, "foreign": 0}

        def draw_instance(mask, cls):
            counts[cls] += 1
            color = (255, 0, 255) if cls == "rice" else (0, 0, 255)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if scale < 1:
                contours = [np.round(c.astype(np.float32) * scale).astype(np.int32)
                            for c in contours]
            cv2.drawContours(vis, contours, -1, color, 2, cv2.LINE_AA)

        start = time.perf_counter()
        with threadpool_limits(limits=1):
            _, _, ref_area = label_image(path, instance_callback=draw_instance)
        rice, foreign = counts["rice"], counts["foreign"]
        total = rice + foreign
        rice_pct = 100 * rice / total if total else 0.0
        foreign_pct = 100 * foreign / total if total else 0.0
        truth = expected_counts(path)
        whole, broken, true_rice, true_foreign = truth or (None, None, None, None)
        target = overlays / path.name
        cv2.putText(vis, f"PROXY rice={rice} ({rice_pct:.1f}%) foreign={foreign} ({foreign_pct:.1f}%)",
                    (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        if truth:
            cv2.putText(vis, f"expected rice={true_rice} foreign={true_foreign}",
                        (16, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
                        cv2.LINE_AA)
        if not cv2.imwrite(str(target), vis):
            raise OSError(f"Could not save overlay: {target}")
        rows.append({
            "file": path.name, "true_whole": whole, "true_broken": broken,
            "true_rice_total": true_rice, "true_foreign": true_foreign,
            "proxy_rice_instances": rice, "proxy_foreign_blobs": foreign,
            "proxy_rice_pct": round(rice_pct, 2), "proxy_foreign_pct": round(foreign_pct, 2),
            "rice_signed_error": rice - true_rice if truth else None,
            "foreign_signed_error": foreign - true_foreign if truth else None,
            "rice_exact": rice == true_rice if truth else None,
            "foreign_exact": foreign == true_foreign if truth else None,
            "single_grain_reference_area_px": round(ref_area, 2),
            "seconds": round(time.perf_counter() - start, 2),
            "overlay": str(target),
        })
        print(f"{i}/{len(paths)} {path.name}: proxy rice={rice} ({rice_pct:.1f}%), "
              f"foreign={foreign} ({foreign_pct:.1f}%); expected {true_rice}/{true_foreign}; "
              f"errors {rice - true_rice:+}/{foreign - true_foreign:+}", flush=True)

    report = args.out / "autolabel_proxy_counts.csv"
    with report.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    known = [r for r in rows if r["rice_signed_error"] is not None]
    if known:
        n = len(known)
        rice_mae = sum(abs(r["rice_signed_error"]) for r in known) / n
        foreign_mae = sum(abs(r["foreign_signed_error"]) for r in known) / n
        rice_exact = sum(r["rice_exact"] for r in known)
        foreign_exact = sum(r["foreign_exact"] for r in known)
        print(f"SUMMARY {n} images: exact rice={rice_exact}/{n}; exact foreign={foreign_exact}/{n}; "
              f"rice MAE={rice_mae:.2f}; foreign MAE={foreign_mae:.2f}")
    print(f"CSV: {report}")
    print(f"Overlays: {overlays}")
    print("These are fixed-rule proxy counts, not trained-model predictions.")


if __name__ == "__main__":
    main()
