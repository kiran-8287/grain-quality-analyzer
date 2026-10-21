#!/usr/bin/env python3
"""
prepare_data.py -- ONE annotation pass -> TWO training sets with an identical split.

Input
-----
A single COCO JSON (exported from CVAT / Roboflow / Label Studio) that contains ALL
three classes, e.g.  whole_rice, broken_rice, foreign_matter,  plus the image folder.

Output
------
out/maskrcnn/train.json, val.json     COCO files (all classes kept; train_maskrcnn.py
                                      picks the rice classes by name)
out/yolo/images/{train,val}/          images for the foreign-matter detector
out/yolo/labels/{train,val}/          YOLO txt labels  ->  class cx cy w h  (normalised)
out/yolo/foreign.yaml                 dataset file to pass to train_yolo.py

Why a single script?  Because both models must see exactly the same train/val images,
and because Roboflow/CVAT YOLO exports bring along every class (or polygons) you
do not want in the detector.

Usage
-----
python prepare_data.py --coco annotations.json --images raw_images/ --out dataset/ \
                       --val-frac 0.15 --foreign-classes foreign_matter
"""
import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, help="COCO json with all classes")
    ap.add_argument("--images", required=True, help="folder holding the images named in the json")
    ap.add_argument("--out", required=True, help="output root folder")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--foreign-classes", nargs="+", default=["foreign_matter"],
                    help="COCO category names that go to YOLO; list order = YOLO class ids. "
                         "Use e.g. 'husk stone' if you annotated them separately.")
    ap.add_argument("--symlink", action="store_true", help="symlink instead of copy images (Linux/macOS)")
    args = ap.parse_args()

    data = json.loads(Path(args.coco).read_text())
    cats = {c["id"]: c["name"] for c in data["categories"]}
    foreign_ids = {}  # coco category id -> yolo class id
    for yolo_id, name in enumerate(args.foreign_classes):
        match = [cid for cid, n in cats.items() if n == name]
        if not match:
            raise SystemExit(f"Category '{name}' not found. Categories in file: {sorted(cats.values())}")
        foreign_ids[match[0]] = yolo_id

    anns_by_img = defaultdict(list)
    for a in data["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    # ---- split by IMAGE (never by annotation) ----------------------------------------
    images = sorted(data["images"], key=lambda im: im["file_name"])
    random.Random(args.seed).shuffle(images)
    n_val = max(1, round(len(images) * args.val_frac)) if args.val_frac > 0 and len(images) > 1 else 0
    split = {"val": images[:n_val], "train": images[n_val:]}

    out = Path(args.out)
    for name in ("train", "val"):
        # ---- Mask R-CNN json -----------------------------------------------------------
        ids = {im["id"] for im in split[name]}
        sub = {
            "images": split[name],
            "annotations": [a for a in data["annotations"] if a["image_id"] in ids],
            "categories": data["categories"],
        }
        (out / "maskrcnn").mkdir(parents=True, exist_ok=True)
        (out / "maskrcnn" / f"{name}.json").write_text(json.dumps(sub))

        # ---- YOLO images + labels --------------------------------------------------------
        img_dir = out / "yolo" / "images" / name
        lbl_dir = out / "yolo" / "labels" / name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        counts = Counter()
        for im in split[name]:
            src = Path(args.images) / im["file_name"]
            dst = img_dir / Path(im["file_name"]).name
            if not src.exists():
                raise SystemExit(f"Missing image: {src}")
            if not dst.exists():
                if args.symlink:
                    dst.symlink_to(src.resolve())
                else:
                    shutil.copy2(src, dst)

            W, H = im["width"], im["height"]
            lines = []
            for a in anns_by_img[im["id"]]:
                if a["category_id"] not in foreign_ids or a.get("iscrowd", 0):
                    continue
                x, y, w, h = a["bbox"]
                x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)  # clip to image
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                lines.append(f"{foreign_ids[a['category_id']]} {cx:.6f} {cy:.6f} "
                             f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
                counts[args.foreign_classes[foreign_ids[a['category_id']]]] += 1
            # an empty .txt is valid: it marks the image as "no foreign matter" (background)
            (lbl_dir / (dst.stem + ".txt")).write_text("\n".join(lines))

        rice = Counter(cats[a["category_id"]] for a in sub["annotations"] if a["category_id"] not in foreign_ids)
        print(f"[{name}] {len(split[name])} images | YOLO boxes: {dict(counts)} | Mask R-CNN instances: {dict(rice)}")

    yaml_path = out / "yolo" / "foreign.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(args.foreign_classes))
    yaml_path.write_text(
        f"path: {(out / 'yolo').resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n"
    )
    print(f"\nWrote {yaml_path}\nMask R-CNN files: {out / 'maskrcnn'}/train.json, val.json")


if __name__ == "__main__":
    main()
