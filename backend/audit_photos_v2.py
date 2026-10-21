"""
audit_photos_v2.py  --  STAGE 2: lighting-proof segmentation + count sanity check.

Why v1 failed: rice is PALE and low-saturation, the paper is strongly green and its
brightness changes a lot between photos. A fixed HSV range cannot cope with that.

Idea used here
  * Work in Lab colour. Channel a* = "how green vs neutral/red".
      paper  : a* ~ 85-90  (strongly green)       rice  : a* ~ 112-120 (neutral)
      stones : a* ~ 130+   (orange/brown) or very dark
  * Estimate the paper colour LOCALLY (big median filter) and subtract it,
    so bright / dark / shadowed photos all look alike.
  * Foreground = pixels much less green than the local paper.
  * Stones = foreground that is orange/brown (high a*) or very dark.  Rice = the rest.
  * Count check WITHOUT splitting touching grains:
        rice_est = rice_area / (area of ONE grain in THIS photo) / bias   ~ 100 when all grains are visible.
    One-grain area is measured per photo from lone grains (compact + elongated blobs),
    so photos taken from slightly different heights/zooms are handled fairly.
    'bias' (about 0.92) is calibrated from clean photos, using the fact that each has 100 grains.

v2.1 additions: per-photo grain size, per-folder summary, worst-photo list, --show.

Usage
  python audit_photos_v2.py "A:/grain/backend/data/raw" --out audit2_out
  python audit_photos_v2.py "A:/grain/backend/data/raw" --out audit2_out --show IMG20261020212111
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png"}
K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def load_small(path, long_side):
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def local_paper(img, frac=0.12):
    """Smooth estimate of the paper colour (grains cover too little area to fool a big median)."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    k = int(max(small.shape[:2]) * frac) | 1
    return cv2.resize(cv2.medianBlur(small, k), (w, h), interpolation=cv2.INTER_LINEAR)


def masks(img, a):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = cv2.cvtColor(local_paper(img), cv2.COLOR_BGR2LAB).astype(np.float32)
    a_s = cv2.GaussianBlur(lab[..., 1], (0, 0), 1.5)
    d_a = a_s - bg[..., 1]          # less green than the paper  -> positive
    d_l = lab[..., 0] - bg[..., 0]  # much darker than the paper -> very negative
    fg = ((d_a > a.a_thr) | (d_l < -45)).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, K5)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, K5)
    stone = ((fg > 0) & ((a_s > a.stone_a) | (lab[..., 0] < a.stone_l))).astype(np.uint8) * 255
    stone = cv2.dilate(cv2.morphologyEx(stone, cv2.MORPH_OPEN, K5), K5)  # include the stone's rim
    rice = cv2.bitwise_and(fg, cv2.bitwise_not(stone))
    rice = cv2.morphologyEx(rice, cv2.MORPH_OPEN, K5)
    return rice, stone


def blob_areas(mask, min_area):
    _, _, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    ar = st[1:, cv2.CC_STAT_AREA]
    return ar[ar >= min_area]


def lone_grain_area(mask, min_area):
    """Median area of blobs that look like ONE grain: compact (high solidity) and elongated."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areas = []
    for c in cnts:
        ar = cv2.contourArea(c) + 0.5 * cv2.arcLength(c, True) + 1  # ~ pixel count
        if ar < min_area:
            continue
        hull = cv2.contourArea(cv2.convexHull(c))
        if hull <= 0:
            continue
        (_, _), (w, h), _ = cv2.minAreaRect(c)
        aspect = max(w, h) / max(1.0, min(w, h))
        if ar / hull >= 0.93 and 2.4 <= aspect <= 5.5:
            areas.append(ar)
    return (float(np.median(areas)), len(areas)) if len(areas) >= 8 else (None, len(areas))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--out", default="audit2_out")
    p.add_argument("--long-side", type=int, default=1536)
    p.add_argument("--a-thr", type=float, default=7)       # how much less green than paper = foreground
    p.add_argument("--stone-a", type=float, default=126)   # a* above this = orange/brown stone
    p.add_argument("--stone-l", type=float, default=100)   # L* below this = very dark stone
    p.add_argument("--min-stone-area", type=int, default=700)
    p.add_argument("--min-rice-area", type=int, default=200)
    p.add_argument("--expected-rice", type=int, default=100)
    p.add_argument("--expected-stones", type=int, default=10)
    p.add_argument("--area-tol", type=float, default=0.08)  # allowed +-8% rice-count deviation
    p.add_argument("--stone-area-tol", type=float, default=0.20)  # total stone area vs usual
    p.add_argument("--clump-grains", type=float, default=8)  # a blob bigger than this many grains = clump
    p.add_argument("--n-sample", type=int, default=6)
    p.add_argument("--show", nargs="*", default=[], help="file-name parts to render overlays for")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    root = Path(a.folder)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(f for f in root.rglob("*") if f.is_file() and f.suffix.lower() in EXTS)
    if not paths:
        p.error(f"No photos found under {root.resolve()}")
    print(f"Found {len(paths)} photos")

    stats = []
    for i, f in enumerate(paths):
        img = load_small(f, a.long_side)
        if img is None:
            print("  unreadable:", f)
            continue
        rice, stone = masks(img, a)
        ra = blob_areas(rice, a.min_rice_area)
        sa = blob_areas(stone, a.min_stone_area)
        g, n_lone = lone_grain_area(rice, a.min_rice_area)
        stats.append({"file": str(f), "folder": str(f.parent.relative_to(root)),
                      "rice_area": int(ra.sum()), "rice_blobs": len(ra), "stones": len(sa),
                      "stone_area": int(sa.sum()),
                      "largest_rice_blob": int(ra.max()) if len(ra) else 0,
                      "grain_area": round(g) if g else None, "lone_grains": n_lone})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(paths)}")

    if not stats:
        raise ValueError("No readable photos found")
    # rice_area / lone-grain-area is scale-free but reads ~8% low (lone grains are a bit bigger than
    # the average grain). We KNOW each photo has 100 grains, so calibrate that bias from clean photos.
    for s in stats:
        g = s["grain_area"]
        s["ratio"] = s["rice_area"] / g if g else None
        s["largest_blob_in_grains"] = round(s["largest_rice_blob"] / g, 1) if g else None
    ref = [s["ratio"] for s in stats if s["ratio"] is not None and s["stones"] == a.expected_stones
           and s["largest_blob_in_grains"] <= a.clump_grains]
    ref = ref or [s["ratio"] for s in stats if s["ratio"] is not None]
    if not ref:
        raise ValueError("No photos had enough lone-grain blobs to calibrate the bias")
    bias = float(np.median(ref)) / a.expected_rice
    print(f"\nCalibration: lone-grain ratio bias = {bias:.3f} (from {len(ref)} reference photos)")
    exact = [s["stone_area"] for s in stats if s["stones"] == a.expected_stones]
    stone_ref = float(np.median(exact or [s["stone_area"] for s in stats])) or 1.0
    for s in stats:
        s["rice_est"] = round(s["ratio"] / bias, 1) if s["ratio"] is not None else None
        s["area_ok"] = s["rice_est"] is not None and abs(s["rice_est"] / a.expected_rice - 1) <= a.area_tol
        s["stones_ok"] = s["stones"] == a.expected_stones
        s["stone_area_ok"] = abs(s["stone_area"] / stone_ref - 1) <= a.stone_area_tol
        s["clumpy"] = s["ratio"] is not None and s["largest_blob_in_grains"] > a.clump_grains
        s["status"] = "PASS" if (s["area_ok"] and (s["stones_ok"] or s["stone_area_ok"]) and not s["clumpy"]) else "CHECK"
        s["ratio"] = round(s["ratio"], 1) if s["ratio"] is not None else None

    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(stats[0].keys()))
        w.writeheader()
        w.writerows(stats)

    ok = [s for s in stats if s["rice_est"] is not None]
    if not ok:
        raise ValueError("No photos had an estimated rice count")
    est = np.array([s["rice_est"] for s in ok])
    print(f"\nOVERALL  rice estimate (expected {a.expected_rice}): "
          f"p5 {np.percentile(est, 5):.0f} | median {np.median(est):.0f} | p95 {np.percentile(est, 95):.0f}")
    print(f"  area within +-{int(a.area_tol * 100)}% : {sum(s['area_ok'] for s in stats)}/{len(stats)}"
          f"   stones == {a.expected_stones}: {sum(s['stones_ok'] for s in stats)}/{len(stats)}"
          f"   clumpy: {sum(s['clumpy'] for s in stats)}   no grain size: {len(stats) - len(ok)}")
    sc = defaultdict(int)
    for x in stats:
        sc[x["stones"]] += 1
    print("  stones found per photo:", dict(sorted(sc.items())),
          f"| stone area ok (touching stones merged): {sum(x['stone_area_ok'] for x in stats)}/{len(stats)}")
    print(f"  PASS (usable for auto-labels): {sum(s['status'] == 'PASS' for s in stats)}/{len(stats)}")

    print("\nPER FOLDER")
    by = defaultdict(list)
    for s in stats:
        by[s["folder"]].append(s)
    print(f"  {'folder':32s} {'n':>4s} {'PASS':>5s} {'rice_est':>9s} {'blobs':>6s} {'bigblob':>8s} {'stones':>7s}")
    for fo, ss in sorted(by.items()):
        r = [x["rice_est"] for x in ss if x["rice_est"] is not None]
        bb = [x["largest_blob_in_grains"] for x in ss if x["largest_blob_in_grains"] is not None]
        print(f"  {fo[:32]:32s} {len(ss):4d} {sum(x['status'] == 'PASS' for x in ss):5d} "
              f"{(np.median(r) if r else float('nan')):9.0f} {np.median([x['rice_blobs'] for x in ss]):6.0f} "
              f"{(np.median(bb) if bb else float('nan')):8.1f} {np.median([x['stones'] for x in ss]):7.0f}")

    worst = sorted(stats, key=lambda s: -abs((s["rice_est"] or 0) - a.expected_rice))[:5]
    print("\nWORST 5 (look at these with --show):")
    for s in worst:
        print(f"  {Path(s['file']).name}  rice_est={s['rice_est']}  stones={s['stones']}  big blob={s['largest_blob_in_grains']} grains")

    # Overlays: yellow = rice, red = stones
    rng = random.Random(a.seed)
    valid_paths = [Path(s["file"]) for s in stats]
    chosen = {str(f): f"overlay_{k:02d}.jpg" for k, f in enumerate(rng.sample(valid_paths, min(a.n_sample, len(valid_paths))))}
    matched = set()
    for f in paths:
        if any(t in f.name for t in a.show):
            matched.update(t for t in a.show if t in f.name)
            chosen[str(f)] = f"overlay_{f.stem}.jpg"
    missing = [t for t in a.show if t not in matched]
    if missing:
        p.error(f"--show did not match photo name(s): {', '.join(missing)}")
    for fpath, name in chosen.items():
        img = load_small(fpath, a.long_side)
        rice, stone = masks(img, a)
        vis = img.copy()
        for m, col in ((rice, (0, 255, 255)), (stone, (0, 0, 255))):
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, col, 2)
        s = next(x for x in stats if x["file"] == fpath)
        cv2.putText(vis, f"rice~{s['rice_est']}  stones {s['stones']}  {s['status']}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.imwrite(str(out / name), vis)
    print(f"\nSaved everything in: {out.resolve()}")


if __name__ == "__main__":
    main()
