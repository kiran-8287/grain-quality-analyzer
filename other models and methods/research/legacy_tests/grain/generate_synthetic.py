"""Generate deterministic synthetic grain images and matching COCO polygons."""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


def generate(output_dir: Path, image_count: int = 8, seed: int = 42) -> Path:
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    images = []
    annotations = []
    ann_id = 1
    width, height = 1200, 1600
    categories = [
        {"id": 1, "name": "whole_rice"},
        {"id": 2, "name": "broken_rice"},
        {"id": 3, "name": "foreign_matter"},
        {"id": 4, "name": "needs_review"},
    ]

    for image_id in range(1, image_count + 1):
        noise = np.random.default_rng(seed + image_id).normal(0, 5, (height, width, 1))
        background = np.empty((height, width, 3), dtype=np.float32)
        background[:] = (45, 105, 38)  # BGR green cloth
        image = np.clip(background + noise, 0, 255).astype(np.uint8)
        placed = []

        for _ in range(45):
            is_broken = rng.random() < 0.28
            rx, ry = (rng.randint(27, 38), rng.randint(13, 18)) if is_broken else (rng.randint(48, 58), rng.randint(15, 20))
            for attempt in range(300):
                cx = rng.randint(rx + 18, width - rx - 18)
                cy = rng.randint(ry + 18, height - ry - 18)
                if all((cx - ox) ** 2 + (cy - oy) ** 2 > (rx + orx + 12) ** 2 for ox, oy, orx, _ in placed):
                    break
            else:
                continue
            angle = rng.randint(0, 179)
            axes = (rx, ry)
            cv2.ellipse(image, (cx, cy), axes, angle, 0, 360, (218, 225, 222), -1, cv2.LINE_AA)
            cv2.ellipse(image, (cx, cy), axes, angle, 0, 360, (238, 241, 237), 2, cv2.LINE_AA)
            contour = cv2.ellipse2Poly((cx, cy), axes, angle, 0, 360, 4).reshape(-1, 2)
            xs, ys = contour[:, 0], contour[:, 1]
            category_id = 2 if is_broken else 1
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [contour.astype(float).reshape(-1).tolist()],
                "area": float(cv2.contourArea(contour)),
                "bbox": [float(xs.min()), float(ys.min()), float(xs.max() - xs.min()), float(ys.max() - ys.min())],
                "iscrowd": 0,
            })
            ann_id += 1
            placed.append((cx, cy, rx, ry))

        for _ in range(5):
            radius = rng.randint(10, 18)
            cx = rng.randint(radius + 10, width - radius - 10)
            cy = rng.randint(radius + 10, height - radius - 10)
            axes = (radius + rng.randint(0, 8), radius)
            angle = rng.randint(0, 179)
            color = (rng.randint(20, 65), rng.randint(80, 145), rng.randint(130, 205))
            cv2.ellipse(image, (cx, cy), axes, angle, 0, 360, color, -1, cv2.LINE_AA)
            contour = cv2.ellipse2Poly((cx, cy), axes, angle, 0, 360, 4).reshape(-1, 2)
            xs, ys = contour[:, 0], contour[:, 1]
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 3,
                "segmentation": [contour.astype(float).reshape(-1).tolist()],
                "area": float(cv2.contourArea(contour)),
                "bbox": [float(xs.min()), float(ys.min()), float(xs.max() - xs.min()), float(ys.max() - ys.min())],
                "iscrowd": 0,
            })
            ann_id += 1

        file_name = f"grain_{image_id:02d}.png"
        if not cv2.imwrite(str(images_dir / file_name), image):
            raise OSError(f"Could not write {images_dir / file_name}")
        images.append({"id": image_id, "file_name": file_name, "width": width, "height": height})

    coco_path = output_dir / "synthetic.json"
    coco_path.write_text(json.dumps({
        "info": {"description": "Synthetic smoke-test data; not real-data validation"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }, indent=2), encoding="utf-8")
    return coco_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    options = parser.parse_args()
    print(generate(options.out, options.images, options.seed))
