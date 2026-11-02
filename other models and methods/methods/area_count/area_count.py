"""Experimental rice count from colored material area; not instance segmentation."""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2

METHOD = Path(__file__).resolve().parent
ROOT = METHOD.parent.parent
DATA_HOME = ROOT.parent / "data" / "rice_count"
sys.path.insert(0, str(ROOT / "tools"))
from audit_photos_v2 import blob_areas, load_small, lone_grain_area, masks  # noqa: E402

CALIBRATION = METHOD / "results" / "area_calibration.json"
OUT = METHOD / "results" / "predictions"


def calibrate(summary, out, expected_rice=100, expected_foreign=10, clump_grains=8):
    with summary.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    refs, grain_areas = [], []
    for row in rows:
        try:
            ratio = float(row["ratio"])
            foreign = int(float(row["stones"]))
            big = float(row["largest_blob_in_grains"])
            grain = float(row["grain_area"])
        except (KeyError, TypeError, ValueError):
            continue
        grain_areas.append(grain)
        if foreign == expected_foreign and big <= clump_grains:
            refs.append(ratio)
    if len(refs) < 20 or not grain_areas:
        raise ValueError(f"Insufficient calibration rows: {len(refs)} clean references")
    data = {
        "method": "experimental_area_proxy",
        "assumption": f"reference photos contain {expected_rice} rice grains each",
        "expected_foreign_for_reference": expected_foreign,
        "reference_photos": len(refs),
        "area_bias": statistics.median(refs) / expected_rice,
        "fallback_grain_area_px_at_1536": statistics.median(grain_areas),
        "source_summary": str(summary),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Experimental calibration saved: {out}")
    print(f"Assumption: {expected_rice} rice/reference photo; {len(refs)} reference photos")
    print(f"Area bias: {data['area_bias']:.4f}; fallback grain area: {data['fallback_grain_area_px_at_1536']:.0f} px")


def count_image(path, calibration):
    img = load_small(path, 1536)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    args = SimpleNamespace(a_thr=7, stone_a=126, stone_l=100,
                           min_rice_area=200, min_stone_area=700)
    rice, stone = masks(img, args)
    rice_area = int(cv2.countNonZero(rice))
    grain_area, n_lone = lone_grain_area(rice, args.min_rice_area)
    basis = "photo single grains" if grain_area else "training-set fallback"
    grain_area = grain_area or float(calibration["fallback_grain_area_px_at_1536"])
    bias = float(calibration["area_bias"])
    if grain_area <= 0 or bias <= 0:
        raise ValueError("Calibration has invalid grain area or bias")
    rice_est = round(rice_area / grain_area / bias)
    foreign_blobs = len(blob_areas(stone, args.min_stone_area))
    vis = img.copy()
    for mask, color in ((rice, (0, 255, 255)), (stone, (0, 0, 255))):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)
    cv2.putText(vis, f"EXPERIMENTAL rice~{rice_est} foreign blobs={foreign_blobs}",
                (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    return {"n_rice_estimate": rice_est, "n_foreign_blobs": foreign_blobs,
            "rice_area_px": rice_area, "grain_area_px": grain_area,
            "grain_area_basis": basis, "lone_grain_blobs": n_lone,
            "overlay": vis}


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("calibrate", help="derive an assumed-count calibration from audit CSV")
    fit.add_argument("--summary", type=Path, default=DATA_HOME / "work" / "audit" / "summary.csv")
    fit.add_argument("--out", type=Path, default=CALIBRATION)
    fit.add_argument("--expected-rice", type=int, default=100)
    fit.add_argument("--expected-foreign", type=int, default=10)
    predict = commands.add_parser("predict", help="estimate counts for one image")
    predict.add_argument("image", type=Path)
    predict.add_argument("--calibration", type=Path, default=CALIBRATION)
    predict.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "calibrate":
        calibrate(args.summary, args.out, args.expected_rice, args.expected_foreign)
        return
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    result = count_image(args.image, calibration)
    target = args.out or OUT / f"area_{args.image.stem}.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), result.pop("overlay")):
        raise OSError(f"Could not write overlay: {target}")
    print(f"EXPERIMENTAL estimate: rice={result['n_rice_estimate']} "
          f"foreign_blobs={result['n_foreign_blobs']} (not verified true counts)")
    print(f"Grain-area basis: {result['grain_area_basis']}; overlay: {target}")


if __name__ == "__main__":
    main()
