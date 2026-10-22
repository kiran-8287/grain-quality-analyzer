"""Run the existing experimental area counter on filename-counted test images."""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "models"))
from area_count import CALIBRATION, count_image  # noqa: E402

import cv2

NAME_COUNTS = re.compile(r"(\d+)\s*whole\s*,?\s*(\d+)\s*broken\s*,?\s*(\d+)\s*stones", re.I)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def expected_counts(path):
    match = NAME_COUNTS.search(path.stem)
    if not match:
        return None
    whole, broken, foreign = map(int, match.groups())
    return whole, broken, whole + broken, foreign


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=PROJECT / "data" / "test")
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    parser.add_argument("--out", type=Path, default=ROOT / "predictions" / "test_set" / "area")
    args = parser.parse_args()
    if not args.calibration.is_file():
        raise SystemExit(f"Missing area calibration: {args.calibration}")
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    paths = sorted(p for p in args.source.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise SystemExit(f"No images under {args.source}")
    overlays = args.out / "overlays"
    overlays.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, path in enumerate(paths, 1):
        result = count_image(path, calibration)
        rice, foreign = result["n_rice_estimate"], result["n_foreign_blobs"]
        total = rice + foreign
        rice_pct = 100 * rice / total if total else 0.0
        foreign_pct = 100 * foreign / total if total else 0.0
        truth = expected_counts(path)
        whole, broken, true_rice, true_foreign = truth or (None, None, None, None)
        rice_err = abs(rice - true_rice) if truth else None
        foreign_err = abs(foreign - true_foreign) if truth else None
        overlay = result.pop("overlay")
        cv2.putText(overlay, f"expected rice={true_rice} foreign={true_foreign}" if truth
                    else "expected counts unavailable", (16, 76), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2, cv2.LINE_AA)
        target = overlays / path.name
        if not cv2.imwrite(str(target), overlay):
            raise OSError(f"Could not save overlay: {target}")
        rows.append({
            "file": path.name, "true_whole": whole, "true_broken": broken,
            "true_rice_total": true_rice, "true_foreign": true_foreign,
            "area_rice_estimate": rice, "foreign_blob_count": foreign,
            "estimated_rice_pct": round(rice_pct, 2),
            "estimated_foreign_pct": round(foreign_pct, 2),
            "rice_abs_error": rice_err, "foreign_abs_error": foreign_err,
            "rice_exact": rice == true_rice if truth else None,
            "foreign_exact_blobs": foreign == true_foreign if truth else None,
            "grain_area_basis": result["grain_area_basis"],
            "rice_area_px": result["rice_area_px"],
            "grain_area_px": round(result["grain_area_px"], 2),
            "lone_grain_blobs": result["lone_grain_blobs"],
            "overlay": str(target),
        })
        print(f"{i}/{len(paths)} {path.name}: area rice~{rice} ({rice_pct:.1f}%), "
              f"foreign blobs={foreign} ({foreign_pct:.1f}%); expected "
              f"{true_rice}/{true_foreign}; abs error={rice_err}/{foreign_err}")

    report = args.out / "area_counts.csv"
    with report.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    known = [row for row in rows if row["rice_abs_error"] is not None]
    if known:
        n = len(known)
        rice_mae = sum(row["rice_abs_error"] for row in known) / n
        foreign_mae = sum(row["foreign_abs_error"] for row in known) / n
        rice_exact = sum(row["rice_exact"] for row in known)
        foreign_exact = sum(row["foreign_exact_blobs"] for row in known)
        print(f"SUMMARY {n} named images: exact rice={rice_exact}/{n}; "
              f"foreign blobs exact={foreign_exact}/{n}; rice MAE={rice_mae:.2f}; "
              f"foreign-blob MAE={foreign_mae:.2f}")
    print(f"CSV: {report}")
    print(f"Overlays: {overlays}")
    print("Area rice estimate and connected foreign-blob counts are experimental proxies.")


if __name__ == "__main__":
    main()
