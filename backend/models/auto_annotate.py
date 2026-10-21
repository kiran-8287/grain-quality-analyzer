#!/usr/bin/env python3
"""
auto_annotate.py -- pseudo-label green-sheet photos and write a COCO json you can import into
CVAT / Roboflow and CORRECT instead of drawing every polygon by hand.

How it works (classical CV, no training needed)
-----------------------------------------------
1. Everything that is not saturated green is foreground (rice is white, stones are orange/brown/black).
2. Each connected blob is classified:
     * stone pixels are strongly saturated (rice core saturation ~40-60, stones >100) -> foreign_matter
     * a lone rice-coloured blob of normal size/shape -> whole_rice, or broken_rice when its length is
       < --broken-ratio x the typical whole-grain length (computed over ALL images, so it adapts to
       your rice variety and camera distance)
     * anything doubtful -> needs_review: touching/overlapping grains, grain+stone merged, blobs that
       touch the image border (cut-off grains would look "broken"), oversized blobs
3. Polygons are traced on a 4x up-sampled mask and mapped back to sub-pixel COCO coordinates, so the
   rasterised mask matches the thresholded mask (no systematic 0.5 px shrink/growth).

IMPORTANT
---------
* needs_review polygons are NOT valid training masks (they are 2 grains fused). In CVAT delete each one
  and draw/relabel the individual grains. If you leave grains unlabeled the model learns that touching
  grains are background -- the very case you most need it to handle. train_maskrcnn.py ignores the
  needs_review category automatically, so forgetting to delete them is safe, just wasteful.
* Boundaries follow the colour threshold. Check the overlay images, and validate mm accuracy against
  caliper measurements before trusting the lengths.

Usage
-----
python auto_annotate.py --images raw_images --out annotations_auto.json --overlay-dir overlays
(then open a few overlays: blue=whole, cyan=broken, red=foreign, orange=needs_review)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CATEGORIES = [
    {"id": 1, "name": "whole_rice"},
    {"id": 2, "name": "broken_rice"},
    {"id": 3, "name": "foreign_matter"},
    {"id": 4, "name": "needs_review"},
]
COLORS = {1: (255, 0, 0), 2: (255, 255, 0), 3: (0, 0, 255), 4: (0, 165, 255)}  # BGR


def segment(img, hue_range, sat_min):
    """Foreground mask = not saturated green. Returns (mask uint8 0/255, saturation channel)."""
    h, s, _ = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    green = (h >= hue_range[0]) & (h <= hue_range[1]) & (s >= sat_min)
    fg = (~green).astype(np.uint8) * 255
    k = np.ones((5, 5), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)   # remove specks of felt fibre
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)  # fill pin-holes inside grains
    return fg, s


def blob_features(img_path, args):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)  # EXIF orientation applied by OpenCV
    if img is None:
        raise FileNotFoundError(img_path)
    H, W = img.shape[:2]
    fg, sat = segment(img, args.hue_range, args.sat_min)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    min_area = args.min_area_frac * H * W
    blobs = []
    for i in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if area < min_area:
            continue
        roi = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        core = cv2.erode(roi, np.ones((3, 3), np.uint8), iterations=4)  # ignore blended edge pixels
        if core.sum() < 30:
            core = roi
        stone_frac = float((sat[y:y + h, x:x + w][core > 0] > args.stone_sat).mean())
        cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)
        hull = cv2.contourArea(cv2.convexHull(c))
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        blobs.append({
            "x": x, "y": y, "w": w, "h": h, "area": float(area), "roi": roi,
            "stone_frac": stone_frac, "solidity": float(area / max(hull, 1.0)),
            "length": float(max(rw, rh)),
            "border": x <= 0 or y <= 0 or x + w >= W or y + h >= H,
        })
    return (W, H), blobs


def polygon(roi, x, y, up=4, eps=0.75):
    """Trace the blob on an up-sampled mask -> flat COCO polygon in continuous pixel coordinates."""
    pad = cv2.copyMakeBorder(roi, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    big = cv2.resize(pad, None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    pts = (c + 0.5) / up + np.array([x - 2, y - 2], np.float32)  # pixel index -> continuous coords
    pts = cv2.approxPolyDP(pts.reshape(-1, 1, 2), eps, True).reshape(-1, 2)
    return pts


def classify(blobs_per_image, args):
    """Assign a COCO category id to every blob (two passes: reference size first, then labels)."""
    allb = [b for _, bl in blobs_per_image for b in bl]
    rice = [b for b in allb if b["stone_frac"] <= 0.1 and b["solidity"] >= 0.93 and not b["border"]]
    if len(rice) < 5:
        raise SystemExit("Too few clean rice blobs found -- check --hue-range / --sat-min on one photo.")
    ref_area = float(np.median([b["area"] for b in rice]))
    whole = [b for b in rice if 0.75 * ref_area <= b["area"] <= 1.3 * ref_area]
    ref_len = float(np.median([b["length"] for b in whole]))
    ref_area = float(np.median([b["area"] for b in whole]))
    for _, bl in blobs_per_image:
        for b in bl:
            sf = b["stone_frac"]
            if b["border"] or 0.1 < sf < 0.9:
                b["cat"] = 4                                       # cut by border / grain+stone merged
            elif sf >= 0.9:
                b["cat"] = 3 if b["area"] <= 4 * ref_area else 4   # stone (or a big fused lump)
            elif b["solidity"] < args.min_solidity or b["area"] > args.max_area_ratio * ref_area \
                    or b["area"] < 0.12 * ref_area:
                b["cat"] = 4                                       # touching grains / dust
            else:
                b["cat"] = 2 if b["length"] < args.broken_ratio * ref_len else 1
    return ref_area, ref_len


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="annotations_auto.json")
    ap.add_argument("--overlay-dir", default=None, help="write colour-coded QA overlays here")
    ap.add_argument("--hue-range", type=int, nargs=2, default=[60, 95], help="OpenCV hue (0-179) of the sheet")
    ap.add_argument("--sat-min", type=int, default=90, help="min saturation for a pixel to count as sheet")
    ap.add_argument("--stone-sat", type=int, default=100, help="core saturation above this = stone-coloured")
    ap.add_argument("--min-area-frac", type=float, default=1.2e-4, help="ignore blobs smaller than this fraction of the image")
    ap.add_argument("--broken-ratio", type=float, default=0.75, help="broken if length < ratio x whole length")
    ap.add_argument("--min-solidity", type=float, default=0.90)
    ap.add_argument("--max-area-ratio", type=float, default=1.5, help="blobs bigger than this x a whole grain = fused")
    ap.add_argument("--max-images", type=int, default=0, help="debug: only the first N images")
    args = ap.parse_args()

    paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.max_images:
        paths = paths[:args.max_images]
    if not paths:
        raise SystemExit(f"No images in {args.images}")

    per_image = []
    for p in paths:
        size, blobs = blob_features(p, args)
        per_image.append(((p, size), blobs))
    ref_area, ref_len = classify([(k, b) for k, b in per_image], args)
    print(f"reference whole grain: area {ref_area:.0f}px^2, length {ref_len:.0f}px "
          f"(broken < {args.broken_ratio * ref_len:.0f}px)")

    images, anns, aid = [], [], 1
    totals = {c["id"]: 0 for c in CATEGORIES}
    if args.overlay_dir:
        Path(args.overlay_dir).mkdir(parents=True, exist_ok=True)
    for img_id, ((p, (W, H)), blobs) in enumerate(per_image, start=1):
        images.append({"id": img_id, "file_name": p.name, "width": W, "height": H})
        counts = {c["id"]: 0 for c in CATEGORIES}
        overlay = cv2.imread(str(p)) if args.overlay_dir else None
        for b in blobs:
            pts = polygon(b["roi"], b["x"], b["y"])
            if len(pts) < 3:
                continue
            x1, y1, x2, y2 = pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()
            anns.append({"id": aid, "image_id": img_id, "category_id": b["cat"], "iscrowd": 0,
                         "segmentation": [pts.reshape(-1).round(2).tolist()],
                         "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                         "area": round(float(cv2.contourArea(pts.astype(np.float32))), 2)})
            aid += 1
            counts[b["cat"]] += 1
            if overlay is not None:
                cv2.polylines(overlay, [np.round(pts).astype(np.int32)], True, COLORS[b["cat"]], max(2, W // 600))
        for k, v in counts.items():
            totals[k] += v
        print(f"{p.name}: whole {counts[1]} | broken {counts[2]} | foreign {counts[3]} | needs_review {counts[4]}")
        if overlay is not None:
            cv2.imwrite(str(Path(args.overlay_dir) / (p.stem + "_overlay.jpg")),
                        cv2.resize(overlay, None, fx=0.5, fy=0.5), [cv2.IMWRITE_JPEG_QUALITY, 85])

    Path(args.out).write_text(json.dumps({"images": images, "annotations": anns, "categories": CATEGORIES}))
    n_rev = totals[4]
    print(f"\nwrote {args.out}: {len(images)} images, {len(anns)} polygons "
          f"(whole {totals[1]}, broken {totals[2]}, foreign {totals[3]}, needs_review {n_rev})")
    print("Next: import into CVAT (COCO 1.0), delete/redraw every needs_review polygon, fix any wrong labels, export.")


if __name__ == "__main__":
    main()
