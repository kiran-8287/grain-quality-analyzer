import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

MAX_DET = 2000
assert MAX_DET >= 2000


def _tiles(width, height, size, overlap):
    stride = max(1, int(size * (1 - overlap)))
    xs = list(range(0, max(1, width - size + 1), stride))
    ys = list(range(0, max(1, height - size + 1), stride))
    if not xs or xs[-1] + size < width: xs.append(max(0, width - size))
    if not ys or ys[-1] + size < height: ys.append(max(0, height - size))
    xs, ys = sorted(set(xs)), sorted(set(ys))
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            x2, y2 = min(width, x + size), min(height, y + size)
            left = x if xi == 0 else (xs[xi - 1] + size + x) / 2
            right = x2 if xi == len(xs) - 1 else (x2 + xs[xi + 1]) / 2
            top = y if yi == 0 else (ys[yi - 1] + size + y) / 2
            bottom = y2 if yi == len(ys) - 1 else (y2 + ys[yi + 1]) / 2
            yield x, y, x2, y2, left, top, right, bottom


def _instances(result, offset):
    if result.masks is None or result.boxes is None:
        return []
    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    found = []
    for mask, cls in zip(masks, classes):
        mask = (mask > 0).astype(np.uint8)
        if mask.shape != (result.orig_shape[0], result.orig_shape[1]):
            mask = cv2.resize(mask, (result.orig_shape[1], result.orig_shape[0]), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(mask)
        if moments["m00"] == 0: continue
        cx, cy = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
        x, y, ox, oy, right, bottom = offset
        gx, gy = cx + x, cy + y
        if not (ox <= gx < right and oy <= gy < bottom): continue
        rect = cv2.minAreaRect(contour)
        length, breadth = sorted(rect[1], reverse=True)
        if length <= 0 or breadth <= 0: continue
        poly = cv2.approxPolyDP(contour, 1.0, True).reshape(-1, 2)
        poly = [[int(px + x), int(py + y)] for px, py in poly]
        found.append({"class": int(cls), "polygon": poly, "length_px": float(length),
                      "breadth_px": float(breadth), "lb_ratio": float(length / breadth),
                      "broken": False})
    return found


def count_image(path, weights, conf=0.25, iou=0.6, tile=None, overlap=0.2, px_per_mm=None,
                broken_ratio=0.75):
    if tile is not None and tile < 1: raise ValueError("tile must be positive")
    if not 0 <= overlap < 1: raise ValueError("overlap must be in [0, 1)")
    if px_per_mm is not None and px_per_mm <= 0: raise ValueError("px_per_mm must be positive")
    image = cv2.imread(str(path))
    if image is None: raise FileNotFoundError(path)
    model = YOLO(str(weights))
    height, width = image.shape[:2]
    crops = [(0, 0, width, height, 0, 0, width, height)] if tile is None else list(_tiles(width, height, tile, overlap))
    instances = []
    for x, y, x2, y2, left, top, right, bottom in crops:
        result = model.predict(image[y:y2, x:x2], conf=conf, iou=iou, max_det=MAX_DET,
                               retina_masks=True, verbose=False)[0]
        assert model.predictor.args.max_det >= 2000, "YOLO max_det must be at least 2000"
        instances.extend(_instances(result, (x, y, left, top, right, bottom)))
    rice = [o for o in instances if o["class"] == 0]
    foreign = [o for o in instances if o["class"] == 1]
    long = sorted((o["length_px"] for o in rice), reverse=True)
    if long:
        reference = float(np.median(long[: max(1, (len(long) + 1) // 2)]))
        for o in rice: o["broken"] = o["length_px"] < broken_ratio * reference
    if px_per_mm:
        for o in instances:
            o["length_mm"] = o["length_px"] / px_per_mm
            o["breadth_mm"] = o["breadth_px"] / px_per_mm
    return {"n_rice": len(rice), "n_foreign": len(foreign), "rice": rice, "foreign": foreign}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--tile", type=int)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--px-per-mm", type=float)
    ap.add_argument("--broken-ratio", type=float, default=0.75)
    ap.add_argument("--overlay", type=Path)
    args = ap.parse_args()
    result = count_image(args.image, args.weights, args.conf, args.iou, args.tile,
                         args.overlap, args.px_per_mm, args.broken_ratio)
    print(f"rice={result['n_rice']} foreign={result['n_foreign']}")
    image = cv2.imread(str(args.image))
    for key, color in (("rice", (0, 220, 0)), ("foreign", (0, 0, 255))):
        for item in result[key]:
            cv2.polylines(image, [np.asarray(item["polygon"], np.int32)], True, color, 2)
    cv2.putText(image, f"Rice {result['n_rice']}  Foreign {result['n_foreign']}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 0), 3)
    overlay = args.overlay or args.image.with_name(args.image.stem + "_count.jpg")
    cv2.imwrite(str(overlay), image)
    print(f"overlay={overlay}")


if __name__ == "__main__":
    main()
