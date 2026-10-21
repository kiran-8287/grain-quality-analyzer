import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from autolabel import label_image
except ImportError as exc:
    raise SystemExit("Missing backend/models/autolabel.py; add the project label_image(path) module unchanged.") from exc

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
NAMES = {"rice": 0, "foreign": 1}


def foreground(path):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    h, s, _ = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    green = (h >= 55) & (h <= 100) & (s > 90)
    return cv2.resize((~green).astype(np.uint8), None, fx=1 / 8, fy=1 / 8,
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def make_groups(paths):
    masks = [foreground(p) for p in paths]
    parent = list(range(len(paths)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(paths)):
        for j in range(i):
            a, b = masks[i], masks[j]
            if a.shape != b.shape:
                continue
            denom = min(int(a.sum()), int(b.sum()))
            if denom and np.count_nonzero(a & b) / denom > 0.6:
                parent[find(i)] = find(j)
    roots = {}
    return [roots.setdefault(find(i), len(roots)) for i in range(len(paths))]


def polygons(mask):
    m = (np.asarray(mask).astype(np.uint8) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        if cv2.contourArea(c) < 400:
            continue
        p = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
        if len(p) >= 3:
            out.append(p)
    return out


def auto_labels(path, cache):
    key = str(path.resolve())
    if key not in cache:
        img, instances, ref_area = label_image(path)
        anns = []
        for mask, label in instances:
            if label not in NAMES:
                continue
            anns.extend((p, label) for p in polygons(mask))
        cache[key] = (img, anns, ref_area)
    return cache[key]


def coco_package(paths, cache, out):
    out.mkdir(parents=True, exist_ok=True)
    images, annotations, aid = [], [], 1
    for iid, path in enumerate(paths, 1):
        img, anns, _ = auto_labels(path, cache)
        h, w = img.shape[:2]
        shutil.copy2(path, out / path.name)
        images.append({"id": iid, "file_name": path.name, "width": w, "height": h})
        for poly, label in anns:
            x, y, bw, bh = cv2.boundingRect(poly.astype(np.int32))
            annotations.append({"id": aid, "image_id": iid, "category_id": NAMES[label] + 1,
                                "segmentation": [poly.astype(float).reshape(-1).tolist()],
                                "area": float(cv2.contourArea(poly.astype(np.float32))),
                                "bbox": [x, y, bw, bh], "iscrowd": 0})
            aid += 1
    data = {"info": {"description": "Rice and foreign matter review", "version": "1.0"},
            "licenses": [], "images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "rice"}, {"id": 2, "name": "foreign"}]}
    (out / "coco.json").write_text(json.dumps(data), encoding="utf-8")


def read_reviewed(root):
    results = {}
    for jf in root.rglob("*.json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8-sig"))
        except (ValueError, OSError):
            continue
        if not isinstance(data, dict) or not {"images", "annotations", "categories"} <= data.keys():
            continue
        cats = {c["id"]: c["name"].lower() for c in data["categories"]}
        ims = {i["id"]: i for i in data["images"]}
        by_image = {i: [] for i in ims}
        for a in data["annotations"]:
            by_image.setdefault(a["image_id"], []).append((a, cats.get(a["category_id"], "")))
        for iid, im in ims.items():
            name = Path(im["file_name"]).name
            image_path = next((p for p in [jf.parent / im["file_name"], *root.rglob(name)] if p.is_file()), None)
            if image_path:
                results[name] = (image_path, im, by_image.get(iid, []))
    return results


def reviewed_polygons(record, width, height):
    _, _, anns = record
    out = []
    for ann, category in anns:
        label = "rice" if "rice" in category and "foreign" not in category else "foreign" if "foreign" in category else None
        if label is None:
            continue
        seg = ann.get("segmentation", [])
        if isinstance(seg, list):
            for points in seg:
                p = np.asarray(points, dtype=np.float32).reshape(-1, 2)
                if len(p) >= 3:
                    mask = np.zeros((height, width), np.uint8)
                    cv2.fillPoly(mask, [np.round(p).astype(np.int32)], 1)
                    out.extend((q, label) for q in polygons(mask))
        elif isinstance(seg, dict) and isinstance(seg.get("counts"), list):
            from pycocotools import mask as mask_utils
            out.extend((q, label) for q in polygons(mask_utils.decode(seg)) if len(q) >= 3)
    return out


def write_yolo(path, labels, split):
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    dest = ROOT / "data" / "yolo" / "images" / split / path.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    lines = []
    for poly, label in labels:
        pts = np.asarray(poly).reshape(-1, 2)
        if len(pts) < 3 or cv2.contourArea(pts.astype(np.float32)) < 400:
            continue
        coords = " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in pts)
        lines.append(f"{NAMES[label]} {coords}")
    target = ROOT / "data" / "yolo" / "labels" / split / f"{path.stem}.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("classical", "reviewed"), required=True)
    ap.add_argument("--expected-rice", type=int, default=100)
    ap.add_argument("--expected-foreign", type=int, default=10)
    ap.add_argument("--tol", type=int, default=3)
    ap.add_argument("--overlays", type=int, default=0)
    args = ap.parse_args()
    random.seed(0)
    raw = sorted(p for p in (ROOT / "data" / "raw").rglob("*") if p.suffix.lower() in EXTS)
    if not raw:
        raise SystemExit(f"No photos found in {ROOT / 'data' / 'raw'}")
    gids = make_groups(raw)
    unique = sorted(set(gids))
    if len(unique) > 1:
        _, vi = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0).split(unique, groups=unique))
        val_groups = {unique[i] for i in vi}
    else:
        val_groups = set()
    report, cache, classical_counts = [], {}, {}
    for p in raw:
        _, anns, _ = auto_labels(p, cache)
        nr = sum(label == "rice" for _, label in anns)
        nf = sum(label == "foreign" for _, label in anns)
        classical_counts[p.name] = (nr, nf)
        report.append((p.name, gids[raw.index(p)], nr, nf,
                       nf != args.expected_foreign or abs(nr - args.expected_rice) > args.tol))
    work = ROOT / "data" / "work"
    work.mkdir(parents=True, exist_ok=True)
    with (work / "report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["photo", "group", "n_rice", "n_foreign", "needs_review"]); writer.writerows(report)
    with (work / "groups.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["photo", "group"]); writer.writerows((p.name, g) for p, g in zip(raw, gids))
    print(f"n photos={len(raw)} n groups={len(unique)} n flagged={sum(r[-1] for r in report)}")
    if args.source == "classical":
        if (work / "review").exists(): shutil.rmtree(work / "review")
        chosen = []
        for g in sorted(val_groups):
            chosen.extend(random.sample([p for p, pg in zip(raw, gids) if pg == g],
                                        min(2, sum(pg == g for pg in gids))))
        chosen += [path for name, g, _, _, flagged in report
                   for path in raw if path.name == name and flagged and g not in val_groups]
        coco_package(chosen, cache, work / "review")
        print(f"Review package: {len(chosen)} photos in {work / 'review'}")
    else:
        records = read_reviewed(work / "reviewed")
        group_by_name = {p.name: g for p, g in zip(raw, gids)}
        reviewed = {name: rec for name, rec in records.items() if name in group_by_name}
        if not reviewed:
            raise SystemExit("No reviewed COCO photos matched data/raw filenames in data/work/reviewed")
        base = ROOT / "data" / "yolo"
        if base.exists(): shutil.rmtree(base)
        raw_by_name = {p.name: p for p in raw}
        for name, record in reviewed.items():
            im = cv2.imread(str(record[0])); h, w = im.shape[:2]
            split = "val" if group_by_name[name] in val_groups else "train"
            write_yolo(raw_by_name[name], reviewed_polygons(record, w, h), split)
        for p, g in zip(raw, gids):
            if g not in val_groups and p.name not in reviewed:
                write_yolo(p, auto_labels(p, cache)[1], "train")
        for split in ("train", "val"):
            (base / f"{split}.txt").write_text("\n".join(str(p.resolve()) for p in (base / "images" / split).glob("*")), encoding="utf-8")
        (base / "data.yaml").write_text(yaml.safe_dump({"path": str(base.resolve()), "train": "train.txt", "val": "val.txt",
                                                              "names": {0: "rice", 1: "foreign"}}, sort_keys=False), encoding="utf-8")
        print(f"Wrote YOLO dataset: train={sum(1 for _ in (base / 'images/train').glob('*'))}, val={sum(1 for _ in (base / 'images/val').glob('*'))}")
    if args.overlays > 0:
        odir = work / "overlays"; odir.mkdir(parents=True, exist_ok=True)
        sample = random.sample(raw, min(args.overlays * 4, len(raw)))
        tiles = []
        for p in sample:
            img = cv2.imread(str(p))
            for poly, label in auto_labels(p, cache)[1]:
                cv2.polylines(img, [np.asarray(poly, np.int32)], True, (0, 220, 0) if label == "rice" else (0, 0, 255), 2)
            thumb = cv2.resize(img, (640, 640 * img.shape[0] // img.shape[1]))
            cv2.putText(thumb, p.name[:60], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 1)
            tiles.append(thumb)
        for si in range(args.overlays):
            group = tiles[si * 4:(si + 1) * 4]
            if not group: break
            target_h = max(im.shape[0] for im in group)
            padded = [cv2.copyMakeBorder(im, 0, target_h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT) for im in group]
            while len(padded) < 4:
                padded.append(np.zeros_like(padded[0]))
            rows = [cv2.hconcat(padded[i:i + 2]) for i in range(0, len(padded), 2)]
            cv2.imwrite(str(odir / f"contact_{si + 1:02d}.jpg"), cv2.vconcat(rows))


if __name__ == "__main__":
    main()
