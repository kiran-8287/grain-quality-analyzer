"""Isolated experiment: apply array/morphology/watershed ideas to rice photos."""
import argparse
import csv
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audit_photos_v2 import masks  # noqa: E402


def segment(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    h, w = image.shape[:2]
    scale = 1536 / max(h, w)
    image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    rice, foreign = masks(image, argparse.Namespace(a_thr=7, stone_a=126, stone_l=100))

    # X-ray-derived cleanup: close pinholes, fill enclosed gaps, then label components.
    rice = cv2.morphologyEx(rice, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    rice = ndimage.binary_fill_holes(rice > 0).astype(np.uint8) * 255
    ncomp, comp, stats, _ = cv2.connectedComponentsWithStats(rice, connectivity=8)
    for label in range(1, ncomp):
        if stats[label, cv2.CC_STAT_AREA] < 200:
            rice[comp == label] = 0

    # Seed watershed with distance-transform peaks to split touching rice blobs.
    dist = cv2.distanceTransform(rice, cv2.DIST_L2, 5)
    maxima = cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))
    peak_mask = ((dist > 2.5) & (dist >= maxima - 1e-4)).astype(np.uint8)
    nseed, seeds = cv2.connectedComponents(peak_mask, connectivity=8)
    markers = np.zeros(rice.shape, np.int32)
    markers[rice == 0] = 1
    for seed in range(1, nseed):
        if np.count_nonzero(seeds == seed) >= 1:
            markers[seeds == seed] = seed + 1
    topography = np.clip(255 - dist * (255 / max(float(dist.max()), 1.0)), 0, 255).astype(np.uint8)
    labels = cv2.watershed(cv2.cvtColor(topography, cv2.COLOR_GRAY2BGR), markers)
    labels[rice == 0] = 0
    ids, areas = np.unique(labels[labels >= 2], return_counts=True)
    valid = ids[areas >= 200]
    instances = [(labels == i) for i in valid]
    _, foreign_labels, foreign_stats, _ = cv2.connectedComponentsWithStats(foreign, connectivity=8)
    foreign_n = int(np.count_nonzero(foreign_stats[1:, cv2.CC_STAT_AREA] >= 700))
    return image, instances, foreign_n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=ROOT / "data" / "work" / "xray_array_trial")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in a.images:
        image, instances, foreign_n = segment(path)
        vis = image.copy()
        for i, mask in enumerate(instances, 1):
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 255, 255), 1)
            ys, xs = np.where(mask)
            cv2.putText(vis, str(i), (int(xs.mean()), int(ys.mean())),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(vis, f"TRIAL rice={len(instances)} foreign_blobs={foreign_n}",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        target = a.out / f"{path.stem}_trial.jpg"
        cv2.imwrite(str(target), vis)
        rows.append({"image": str(path), "rice_instances": len(instances),
                     "foreign_blobs": foreign_n, "overlay": str(target)})
        print(f"{path.name}: rice instances={len(instances)}, foreign blobs={foreign_n}")
    with (a.out / "trial_counts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved trial outputs: {a.out.resolve()}")


if __name__ == "__main__":
    main()
