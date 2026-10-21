"""
audit_photos.py  --  STAGE 1: look at the data before labeling anything.

What it does
  1. Shrinks every photo to a fixed working size (so thresholds mean the same
     thing for 3072x4096 or any other resolution).
  2. Removes the green background (HSV "green key") -> everything else is a blob.
  3. Learns the typical single-grain area from ALL photos (histogram peak),
     then classifies blobs: speck / single / merged / huge.
  4. Estimates objects per photo and compares with the known total (100+10=110).
  5. Saves: contact sheet, overlays, summary.csv, histograms.

Usage
  python audit_photos.py "A:/grain/raw" --out audit_out
Tune if the overlays look wrong:  --h-lo --h-hi --s-min --min-area-frac
"""
import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png"}
COLORS = {0: (0, 0, 255), 1: (0, 255, 255), 2: (255, 0, 255), 3: (255, 255, 0)}
NAMES = {0: "speck", 1: "single", 2: "merged", 3: "huge"}


def load_small(path, long_side):
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def foreground_mask(img, a):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bg = cv2.inRange(hsv, (a.h_lo, a.s_min, 0), (a.h_hi, 255, 255))
    fg = cv2.bitwise_not(bg)
    k = max(3, round(0.003 * max(img.shape[:2])))  # kernel size relative to image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)   # kills tiny specks
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)  # fills pin-holes
    return fg


def blobs(img, a):
    """Return label image, areas of all blobs (index = label id)."""
    fg = foreground_mask(img, a)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    return fg, labels, stats[:, cv2.CC_STAT_AREA]


def find_grain_area(all_areas):
    """Typical single-grain area = peak of the log-area histogram."""
    all_areas = np.asarray(all_areas, dtype=np.float64)
    all_areas = all_areas[np.isfinite(all_areas) & (all_areas > 0)]
    if not all_areas.size:
        raise ValueError("No usable foreground blobs found; check the input folder and green-key thresholds.")
    la = np.log10(all_areas)
    hist, edges = np.histogram(la, bins=40)
    peak = (edges[hist.argmax()] + edges[hist.argmax() + 1]) / 2
    near = all_areas[np.abs(la - peak) < 0.3]  # within a factor of ~2
    return float(np.median(near))


def classify(area, g, huge_area):
    if area >= huge_area:
        return 3
    if area < 0.4 * g:
        return 0
    if area <= 1.8 * g:
        return 1
    return 2


def fit(img, w, h):
    canvas = np.zeros((h, w, 3), np.uint8)
    s = min(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    canvas[: r.shape[0], : r.shape[1]] = r
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--out", default="audit_out")
    p.add_argument("--long-side", type=int, default=1536)
    p.add_argument("--h-lo", type=int, default=35)    # OpenCV hue range is 0-179
    p.add_argument("--h-hi", type=int, default=90)
    p.add_argument("--s-min", type=int, default=40)
    p.add_argument("--min-area-frac", type=float, default=0.00003)  # of image area
    p.add_argument("--huge-frac", type=float, default=0.02)         # of image area
    p.add_argument("--expected-total", type=int, default=110)       # 100 rice + 10 stones
    p.add_argument("--n-sample", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(f for f in Path(a.folder).rglob("*") if f.is_file() and f.suffix.lower() in EXTS)
    if not paths:
        p.error(f"No photos found under {Path(a.folder).resolve()}")
    print(f"Found {len(paths)} photos")
    rng = random.Random(a.seed)

    # ---- contact sheet: raw thumbnails, to eyeball what the data really is
    thumbs = []
    for f in rng.sample(paths, min(24, len(paths))):
        img = load_small(f, 320)
        if img is not None:
            thumbs.append(fit(img, 240, 320))
    if not thumbs:
        raise ValueError("No readable photos found")
    cols = 6
    while len(thumbs) % cols:
        thumbs.append(np.zeros_like(thumbs[0]))
    rows = [np.hstack(thumbs[i:i + cols]) for i in range(0, len(thumbs), cols)]
    cv2.imwrite(str(out / "contact_sheet.jpg"), np.vstack(rows))

    # ---- pass 1: blob areas for every photo
    per_photo = []
    for i, f in enumerate(paths):
        img = load_small(f, a.long_side)
        if img is None:
            print("  unreadable:", f)
            continue
        img_area = img.shape[0] * img.shape[1]
        _, _, areas = blobs(img, a)
        areas = areas[1:]  # drop background label 0
        min_area = a.min_area_frac * img_area
        per_photo.append({
            "file": str(f), "img_area": img_area,
            "n_tiny": int((areas < min_area).sum()),
            "areas": areas[areas >= min_area].astype(float),
        })
        if (i + 1) % 50 == 0:
            print(f"  pass 1: {i + 1}/{len(paths)}")

    pooled_parts = [r["areas"][r["areas"] < a.huge_frac * r["img_area"]] for r in per_photo]
    pooled_parts = [part for part in pooled_parts if part.size]
    if not pooled_parts:
        raise ValueError("No non-huge foreground blobs found; check the green-key thresholds.")
    pooled = np.concatenate(pooled_parts)
    g = find_grain_area(pooled)
    print(f"\nTypical single-grain area (working size): {g:.0f} px  "
          f"(~{np.sqrt(g / 3):.0f} x {np.sqrt(g * 3):.0f} px if grain is 1:3)")

    # ---- pass 2: per-photo counts
    rows_csv = []
    for r in per_photo:
        huge_area = a.huge_frac * r["img_area"]
        cls = [classify(x, g, huge_area) for x in r["areas"]]
        cls = np.array(cls)
        singles = int((cls == 1).sum())
        merged_areas = r["areas"][cls == 2]
        est_total = singles + int(sum(max(2, round(x / g)) for x in merged_areas))
        rows_csv.append({
            "file": r["file"], "n_tiny_removed": r["n_tiny"], "specks": int((cls == 0).sum()),
            "singles": singles, "merged_blobs": int((cls == 2).sum()),
            "huge_blobs": int((cls == 3).sum()), "est_total": est_total,
            "blob_count": singles + int((cls == 2).sum()),
        })
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)

    est = np.array([r["est_total"] for r in rows_csv])
    blob = np.array([r["blob_count"] for r in rows_csv])
    # blob_count UNDER-counts when grains touch; est_total OVER-counts when stones are big.
    # So the true total (rice + stones) should usually sit between the two.
    between = (blob <= a.expected_total * 1.05) & (est >= a.expected_total * 0.95)
    print(f"\nObjects per photo (expected {a.expected_total}):")
    print(f"  blob_count (touching grains merged): min {blob.min()}  median {int(np.median(blob))}  max {blob.max()}")
    print(f"  est_total  (area-based)            : min {est.min()}  median {int(np.median(est))}  max {est.max()}")
    print(f"  expected total lies between the two: {between.sum()}/{len(est)} photos")
    print(f"  median tiny blobs removed per photo: {int(np.median([r['n_tiny_removed'] for r in rows_csv]))}")
    print(f"  photos with huge blobs (sheet edge / table / shadow): "
          f"{sum(r['huge_blobs'] > 0 for r in rows_csv)}")

    # ---- overlays for a few random photos
    for k, r in enumerate(rng.sample(per_photo, min(a.n_sample, len(per_photo)))):
        f = Path(r["file"])
        img = load_small(f, a.long_side)
        fg, labels, areas = blobs(img, a)
        img_area = img.shape[0] * img.shape[1]
        min_area = a.min_area_frac * img_area
        lut = np.full(len(areas), -1, np.int8)
        for lab in range(1, len(areas)):
            if areas[lab] >= min_area:
                lut[lab] = classify(areas[lab], g, a.huge_frac * img_area)
        cls_img = lut[labels]
        vis = img.copy()
        for c in range(4):
            m = (cls_img == c).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, COLORS[c], 2)
        for j, c in enumerate(range(4)):
            count = int(np.unique(labels[(cls_img == c) & (labels > 0)]).size)
            cv2.putText(vis, f"{NAMES[c]}: {count}",
                        (20, 40 + 40 * j), cv2.FONT_HERSHEY_SIMPLEX, 1.1, COLORS[c], 3)
        cv2.imwrite(str(out / f"overlay_{k:02d}.jpg"), vis)
        cv2.imwrite(str(out / f"mask_{k:02d}.png"), fg)

    # ---- histograms (optional)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].hist(np.log10(pooled), bins=60)
        ax[0].axvline(np.log10(g), color="r")
        ax[0].set_title("Blob areas (log10 px) - red = typical grain")
        ax[1].hist(est, bins=40)
        ax[1].axvline(a.expected_total, color="r")
        ax[1].set_title("Estimated objects per photo - red = expected")
        fig.tight_layout()
        fig.savefig(out / "histograms.png", dpi=120)
    except ImportError:
        print("(matplotlib not installed - skipped histograms.png)")

    print(f"\nSaved everything in: {out.resolve()}")


if __name__ == "__main__":
    main()
