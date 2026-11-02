"""Prepare one clean representative per provisional layout group for annotation review."""
import argparse
import csv
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
DATA_HOME = ROOT.parent / "data" / "rice_count"
SEED = 0
MIN_AREA = 400


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def source_path(name):
    path = Path(name)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def polygon_from_mask(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA:
            continue
        approx = cv2.approxPolyDP(contour, 1.0, True).reshape(-1, 2)
        if len(approx) >= 3:
            out.append(approx.astype(float).reshape(-1).tolist())
    return out


def label_polygons(path):
    """Run heavy full-resolution mask work in a disposable Windows worker."""
    sys.path.insert(0, str(ROOT))
    from methods.autolabel_proxy.autolabel import label_image
    source = Path(path)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        _, instances, _ = label_image(str(source))
    objects = []
    for mask, label in instances:
        polygons = polygon_from_mask(mask)
        objects.append((label, polygons))
    return objects


def run_label_worker(path, cache_file):
    command = [sys.executable, str(Path(__file__).resolve()), "--label-worker",
               str(path), str(cache_file)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr[-1000:] or f"worker exit {result.returncode}")
    return json.loads(cache_file.read_text(encoding="utf-8"))


def get_selection(expected_rice, expected_foreign, tol):
    audit = DATA_HOME / "work" / "audit" / "summary.csv"
    groups = DATA_HOME / "work" / "layout_groups" / "groups.csv"
    if not audit.is_file() or not groups.is_file():
        raise SystemExit("Run audit_photos_v2.py and find_layout_groups.py first.")
    audits = {Path(r["file"]).name: r for r in read_csv(audit)}
    group_rows = {Path(r["file"]).name: r for r in read_csv(groups)}
    paths = sorted(p for p in (DATA_HOME / "raw").rglob("*")
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    missing = [p.name for p in paths if p.name not in audits or p.name not in group_rows]
    if missing or len(audits) != len(paths) or len(group_rows) != len(paths):
        raise SystemExit(f"Audit/group CSVs do not match raw images (missing examples: {missing[:5]}). Rerun both.")

    by_group = {}
    for path in paths:
        row = audits[path.name]
        gid = group_rows[path.name]["group_id"]
        by_group.setdefault(gid, []).append((path, row))
    rng = random.Random(SEED)
    group_ids = sorted(by_group, key=lambda v: (int(v) if v.isdigit() else v))
    shuffled = group_ids[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(0.15 * len(shuffled))) if len(shuffled) > 1 else 0
    val_ids = set(shuffled[:n_val])

    selected = {}
    for gid, members in by_group.items():
        candidates = [(p, r) for p, r in members if r["status"].upper() == "PASS"]
        if not candidates:
            continue
        candidates.sort(key=lambda pr: (pr[1]["stones"] != "10",
                                       abs(float(pr[1]["rice_est"]) - 100),
                                       -int(pr[1]["lone_grains"]), pr[0].name))
        selected[gid] = candidates[0]

    work = DATA_HOME / "work"
    out = work / "round1_selection.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        fields = ["file", "group_id", "group_size", "split", "selected", "status",
                  "rice_est", "stones", "lone_grains", "largest_blob_in_grains"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for gid, members in sorted(by_group.items()):
            for path, row in members:
                chosen = selected.get(gid, (None,))[0] == path
                writer.writerow({"file": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                                 "group_id": gid, "group_size": len(members),
                                 "split": "val" if gid in val_ids else "train",
                                 "selected": int(chosen), "status": row["status"],
                                 "rice_est": row["rice_est"], "stones": row["stones"],
                                 "lone_grains": row["lone_grains"],
                                 "largest_blob_in_grains": row["largest_blob_in_grains"]})
    report_path = work / "report.csv"
    flagged = 0
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["photo", "n_rice", "n_foreign", "estimate_type", "needs_review"])
        writer.writeheader()
        for path in paths:
            row = audits[path.name]
            nr, nf = round(float(row["rice_est"])), int(row["stones"])
            needs_review = nf != expected_foreign or abs(nr - expected_rice) > tol
            flagged += needs_review
            writer.writerow({"photo": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                             "n_rice": nr, "n_foreign": nf,
                             "estimate_type": "area_proxy_not_instance_count",
                             "needs_review": int(needs_review)})
    print(f"audit proxy report: {report_path} flagged={flagged}")
    return selected, val_ids, len(paths), len(by_group)


def write_classical(selected, val_ids, overlays):
    review = DATA_HOME / "work" / "review"
    images_dir = review / "images"
    if review.exists() and (review / "coco.json").exists():
        raise SystemExit(f"Refusing to overwrite existing review package: {review / 'coco.json'}")
    images_dir.mkdir(parents=True, exist_ok=True)
    coco = {"info": {"description": "Rice and foreign matter instance proposals; review all masks."},
            "licenses": [], "images": [], "annotations": [],
            "categories": [{"id": 1, "name": "rice"}, {"id": 2, "name": "foreign"}]}
    rng = random.Random(SEED)
    gid_order = sorted(selected, key=lambda g: (int(g) if g.isdigit() else g))
    overlay_gids = set(rng.sample(gid_order, min(overlays * 4, len(gid_order)))) if overlays else set()
    overlay_dir = DATA_HOME / "work" / "overlays"
    tile_dir = overlay_dir / "_tiles"
    if overlays:
        tile_dir.mkdir(parents=True, exist_ok=True)
    ann_id = 1
    label_cache = DATA_HOME / "work" / "label_cache"
    label_cache.mkdir(parents=True, exist_ok=True)
    label_errors = []
    for image_id, gid in enumerate(gid_order, 1):
            path, _ = selected[gid]
            cached = label_cache / f"{path.stem}.json"
            try:
                labelled_objects = json.loads(cached.read_text(encoding="utf-8")) if cached.exists() else run_label_worker(path, cached)
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                label_errors.append(path.name)
                print(f"proposal failed for {path.name}; included for manual labeling: {exc}")
                labelled_objects = []
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"Could not read {path}")
            filename = path.name
            target = images_dir / filename
            if not target.exists():
                os.link(path, target)
            h, w = image.shape[:2]
            coco["images"].append({"id": image_id, "file_name": f"images/{filename}",
                                   "width": w, "height": h})
            for label, polygons in labelled_objects:
                for segmentation in polygons:
                    xs, ys = segmentation[0::2], segmentation[1::2]
                    x0, y0 = min(xs), min(ys)
                    bw, bh = max(xs) - x0, max(ys) - y0
                    area = float(cv2.contourArea(np.asarray(segmentation, np.float32).reshape(-1, 1, 2)))
                    if area < MIN_AREA or bw <= 0 or bh <= 0:
                        continue
                    coco["annotations"].append({"id": ann_id, "image_id": image_id,
                        "category_id": 1 if label == "rice" else 2, "segmentation": [segmentation],
                        "area": area, "bbox": [x0, y0, bw, bh], "iscrowd": 0})
                    ann_id += 1
            if gid in overlay_gids:
                vis = image.copy()
                for label, polygons in labelled_objects:
                    color = (0, 0, 255) if label == "foreign" else (255, 0, 255)
                    for polygon in polygons:
                        cv2.polylines(vis, [np.asarray(polygon, np.int32).reshape(-1, 1, 2)],
                                      True, color, 3)
                scale = min(1.0, 1200 / max(vis.shape[:2]))
                vis = cv2.resize(vis, None, fx=scale, fy=scale)
                cv2.putText(vis, path.name, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (255, 255, 0), 2)
                cv2.imwrite(str(tile_dir / f"{gid}_{path.stem}.jpg"), vis)
    if overlays:
        tiles = sorted(tile_dir.glob("*.jpg"))
        rng.shuffle(tiles)
        thumb_w, thumb_h = 480, 640
        for sheet_index in range(overlays):
            canvas = np.full((thumb_h * 2, thumb_w * 2, 3), 245, np.uint8)
            for slot, tile in enumerate(tiles[sheet_index * 4:(sheet_index + 1) * 4]):
                image = cv2.imread(str(tile))
                if image is None:
                    continue
                image = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
                y, x = divmod(slot, 2)
                canvas[y * thumb_h:(y + 1) * thumb_h, x * thumb_w:(x + 1) * thumb_w] = image
            cv2.imwrite(str(overlay_dir / f"contact_{sheet_index + 1:02d}.jpg"), canvas)
        for tile in tiles:
            tile.unlink()
        tile_dir.rmdir()
    coco["info"]["images_without_proposals"] = label_errors
    (review / "coco.json").write_text(json.dumps(coco), encoding="utf-8")
    print(f"Review proposals: {len(selected)} representative images, {len(coco['annotations'])} polygons")
    print(f"COCO review package: {review}")



def reviewed_exports(folder):
    files = sorted(folder.rglob("*.json"))
    if not files:
        raise SystemExit(f"No reviewed COCO JSON found under {folder}")
    by_name = {}
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        categories = {int(c["id"]): c["name"].lower() for c in data.get("categories", [])}
        images = {int(i["id"]): i for i in data.get("images", [])}
        anns = {}
        for ann in data.get("annotations", []):
            anns.setdefault(int(ann["image_id"]), []).append(ann)
        for iid, image in images.items():
            name = Path(image["file_name"]).name
            if name in by_name:
                raise SystemExit(f"Duplicate reviewed image basename: {name}")
            by_name[name] = (image, anns.get(iid, []), categories, path.parent)
    return by_name


def write_yolo(selected, val_ids):
    review = DATA_HOME / "work" / "reviewed"
    exports = reviewed_exports(review)
    selected_by_name = {path.name: (gid, path) for gid, (path, _) in selected.items()}
    missing = sorted(set(selected_by_name) - set(exports))
    if missing:
        raise SystemExit(f"Reviewed COCO export is missing {len(missing)} selected images, e.g. {missing[:5]}")
    output = DATA_HOME / "datasets" / "yolo_segmentation"
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to replace existing YOLO dataset directory: {output}")
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    used = {"train": 0, "val": 0}
    for name, (gid, source) in selected_by_name.items():
        image_info, anns, categories, json_dir = exports[name]
        split = "val" if gid in val_ids else "train"
        image_path = json_dir / image_info["file_name"]
        if not image_path.is_file():
            image_path = source
        image = cv2.imread(str(image_path))
        if image is None:
            raise SystemExit(f"Could not load reviewed image {image_path}")
        h, w = image.shape[:2]
        source_image = cv2.imread(str(source))
        if source_image is None or source_image.shape[:2] != (h, w):
            raise SystemExit(f"Reviewed image dimensions differ from source photo: {name}")
        target = output / "images" / split / name
        if not target.exists():
            os.link(source, target)
        lines = []
        for ann in anns:
            label = categories.get(int(ann["category_id"]), "")
            cls = 0 if label in {"rice", "grain"} else 1 if label in {"foreign", "stone", "non-rice"} else None
            if cls is None:
                raise SystemExit(f"Unknown annotation category {label!r} in {name}")
            segs = ann.get("segmentation")
            if not isinstance(segs, list) or (segs and not isinstance(segs[0], list)):
                raise SystemExit(f"Only polygon COCO exports are supported: {name}")
            for seg in segs:
                if len(seg) < 6:
                    continue
                pts = np.asarray(seg, dtype=float).reshape(-1, 2)
                if cv2.contourArea(pts.astype(np.float32)) < MIN_AREA:
                    continue
                pts[:, 0] = np.clip(pts[:, 0] / w, 0, 1)
                pts[:, 1] = np.clip(pts[:, 1] / h, 0, 1)
                lines.append(str(cls) + " " + " ".join(f"{v:.6f}" for v in pts.ravel()))
        (output / "labels" / split / f"{Path(name).stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        used[split] += 1
    if not used["train"] or not used["val"]:
        raise SystemExit(f"Reviewed dataset needs both train and val groups; got {used}")
    yaml = (f"path: {output.as_posix()}\ntrain: images/train\nval: images/val\n"
            "names:\n  0: rice\n  1: foreign\n")
    (output / "data.yaml").write_text(yaml, encoding="utf-8")
    print(f"YOLO dataset: train={used['train']} val={used['val']} images ({output})")


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--label-worker":
        payload = label_polygons(sys.argv[2])
        Path(sys.argv[3]).write_text(json.dumps(payload), encoding="utf-8")
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("classical", "reviewed"), required=True)
    parser.add_argument("--expected-rice", type=int, default=100)
    parser.add_argument("--expected-foreign", type=int, default=10)
    parser.add_argument("--tol", type=int, default=3)
    parser.add_argument("--overlays", type=int, default=0)
    args = parser.parse_args()
    if args.overlays < 0 or args.tol < 0 or args.expected_rice < 0 or args.expected_foreign < 0:
        parser.error("counts, tolerance, and overlays must be nonnegative")
    selected, val_ids, n_photos, n_groups = get_selection(args.expected_rice, args.expected_foreign, args.tol)
    print(f"photos={n_photos} groups={n_groups} selected clean representatives={len(selected)}")
    print(f"groups without an audit-PASS representative={n_groups - len(selected)}")
    if args.source == "classical":
        write_classical(selected, val_ids, args.overlays)
    else:
        write_yolo(selected, val_ids)


if __name__ == "__main__":
    main()
