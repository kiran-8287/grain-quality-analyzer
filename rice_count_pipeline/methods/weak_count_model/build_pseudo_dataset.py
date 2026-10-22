"""Build an isolated YOLO-seg dataset from the existing classical pseudo-labeler."""
import argparse
import csv
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
DATA_HOME = PROJECT.parent / "data" / "rice_count"
RAW = DATA_HOME / "raw"
GROUPS = DATA_HOME / "work" / "layout_groups" / "groups.csv"
DATASET = DATA_HOME / "datasets" / "weak_count_model"
CACHE = DATA_HOME / "work" / "weak_count_model" / "label_cache"
LABEL_INPUTS = DATA_HOME / "work" / "weak_count_model" / "label_inputs"
REPORTS = DATA_HOME / "work" / "weak_count_model"
SEED = 0
LONG_SIDE = 1536
MIN_AREA = 56
# The known setup is approximately 100 rice grains and 10 stones. Large
# pseudo-count outliers indicate labeler failures (not useful training truth).
RICE_COUNT_RANGE = (80, 130)
FOREIGN_COUNT_RANGE = (6, 14)


def natural_key(path):
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.name.lower())


def instances_to_polygons(path):
    sys.path.insert(0, str(PROJECT / "methods" / "autolabel_proxy"))
    from autolabel import label_image
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        _, instances, _ = label_image(str(path), min_area=MIN_AREA)
    result = []
    for mask, name in instances:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < MIN_AREA:
                continue
            points = cv2.approxPolyDP(contour, 1.0, True).reshape(-1, 2)
            if len(points) >= 3:
                result.append([0 if name == "rice" else 1,
                               points.astype(float).reshape(-1).tolist()])
    return result


def worker(image, output):
    objects = instances_to_polygons(Path(image))
    Path(output).write_text(json.dumps(objects), encoding="utf-8")


def get_labels(path):
    target = CACHE / f"{Path(path).stem}.json"
    if target.is_file():
        return json.loads(target.read_text(encoding="utf-8"))
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
               str(path), str(target)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=240)
    if result.returncode:
        raise RuntimeError(result.stderr[-1200:] or f"worker exit {result.returncode}")
    return json.loads(target.read_text(encoding="utf-8"))


def prepare_one(path, group_id):
    split = "val" if group_id in VAL_GROUPS else "train"
    image_dst = DATASET / "images" / split / path.name
    label_dst = DATASET / "labels" / split / f"{path.stem}.txt"
    if image_dst.is_file() and label_dst.is_file():
        counts = [0, 0]
        for line in label_dst.read_text(encoding="utf-8").splitlines():
            cls = int(line.split(maxsplit=1)[0])
            counts[cls] += 1
        return {"file": path.name, "group_id": group_id, "split": split,
                "pseudo_rice": counts[0], "pseudo_foreign": counts[1]}
    original = cv2.imread(str(path))
    if original is None:
        raise ValueError("image could not be decoded")
    h0, w0 = original.shape[:2]
    scale = min(1.0, LONG_SIDE / max(h0, w0))
    image = (cv2.resize(original, (round(w0 * scale), round(h0 * scale)),
                        interpolation=cv2.INTER_AREA) if scale < 1 else original)
    label_input = LABEL_INPUTS / f"{path.stem}.png"
    label_input.parent.mkdir(parents=True, exist_ok=True)
    objects = get_labels(label_input) if (CACHE / f"{path.stem}.json").is_file() else None
    if objects is None:
        if not cv2.imwrite(str(label_input), image):
            raise OSError(f"Could not write scaled label input: {label_input}")
        try:
            objects = get_labels(label_input)
        finally:
            label_input.unlink(missing_ok=True)
    if not objects:
        raise ValueError("pseudo-labeler returned no polygons")
    h, w = image.shape[:2]
    lines, counts = [], [0, 0]
    for cls, flat in objects:
        pts = np.asarray(flat, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 3 or cv2.contourArea(pts) < MIN_AREA:
            continue
        pts[:, 0] = np.clip(pts[:, 0] / w, 0, 1)
        pts[:, 1] = np.clip(pts[:, 1] / h, 0, 1)
        lines.append(str(cls) + " " + " ".join(f"{v:.6f}" for v in pts.ravel()))
        counts[cls] += 1
    if not lines:
        raise ValueError("no valid polygons after conversion")
    if not image_dst.exists():
        # Keep one photo payload; YOLO resizes it, and polygon coordinates are normalized.
        os.link(path, image_dst)
    label_dst.write_text("\n".join(lines), encoding="utf-8")
    return {"file": path.name, "group_id": group_id, "split": split,
            "pseudo_rice": counts[0], "pseudo_foreign": counts[1]}


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3])
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()
    if not 0.05 <= args.val_fraction <= 0.4:
        parser.error("--val-fraction must be between 0.05 and 0.4")
    if not GROUPS.is_file():
        raise SystemExit(f"Missing provisional layout groups: {GROUPS}")
    with GROUPS.open(newline="", encoding="utf-8-sig") as f:
        group_rows = {Path(row["file"]).name: row["group_id"] for row in csv.DictReader(f)}
    paths = sorted((p for p in RAW.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
                   key=natural_key)
    if not paths:
        raise SystemExit(f"No images under {RAW}")
    missing = [p.name for p in paths if p.name not in group_rows]
    if missing:
        raise SystemExit(f"Groups file misses {len(missing)} images; regenerate it first")

    group_ids = sorted(set(group_rows.values()), key=lambda x: int(x) if x.isdigit() else x)
    random.Random(SEED).shuffle(group_ids)
    n_val = max(1, round(len(group_ids) * args.val_fraction))
    global VAL_GROUPS
    VAL_GROUPS = set(group_ids[:n_val])
    for split in ("train", "val"):
        (DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

    rows, failed = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(prepare_one, path, group_rows[path.name]): path for path in paths}
        for i, future in enumerate(as_completed(futures), 1):
            path = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                failed.append({"file": path.name, "group_id": group_rows[path.name],
                               "split": "val" if group_rows[path.name] in VAL_GROUPS else "train",
                               "reason": str(exc)})
            if i % 25 == 0 or i == len(paths):
                print(f"processed {i}/{len(paths)}; usable={len(rows)}; skipped={len(failed)}",
                      flush=True)

    if not rows:
        raise SystemExit("No pseudo-labeled images were produced")
    for row in rows:
        valid = (RICE_COUNT_RANGE[0] <= row["pseudo_rice"] <= RICE_COUNT_RANGE[1]
                 and FOREIGN_COUNT_RANGE[0] <= row["pseudo_foreign"] <= FOREIGN_COUNT_RANGE[1])
        row["use_for_training"] = valid
        if not valid:
            for path in (DATASET / "images" / row["split"] / row["file"],
                         DATASET / "labels" / row["split"] / f"{Path(row['file']).stem}.txt"):
                path.unlink(missing_ok=True)
    usable = [r for r in rows if r["use_for_training"]]
    train_n = sum(r["split"] == "train" for r in usable)
    val_n = sum(r["split"] == "val" for r in usable)
    if not train_n or not val_n:
        raise SystemExit("Insufficient pseudo-labeled images for both train and validation")
    REPORTS.mkdir(parents=True, exist_ok=True)
    for name, data in (("pseudo_counts.csv", rows), ("skipped.csv", failed)):
        fields = list(data[0]) if data else ["file", "group_id", "split", "reason"]
        with (REPORTS / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    yaml = (f"path: {DATASET.as_posix()}\ntrain: images/train\nval: images/val\n"
            "names:\n  0: rice\n  1: foreign\n")
    (DATASET / "data.yaml").write_text(yaml, encoding="utf-8")
    print(f"weak-label images: {len(rows)} usable by labeler; {len(failed)} label failures")
    print(f"count-plausibility gate: rice {RICE_COUNT_RANGE[0]}-{RICE_COUNT_RANGE[1]}, "
          f"foreign {FOREIGN_COUNT_RANGE[0]}-{FOREIGN_COUNT_RANGE[1]}; "
          f"excluded {len(rows)-len(usable)} pseudo-count outliers")
    print(f"training dataset: {len(usable)} images; train={train_n} val={val_n}")
    print(f"pseudo-label report: {REPORTS / 'pseudo_counts.csv'}")


if __name__ == "__main__":
    main()
