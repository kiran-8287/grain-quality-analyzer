"""Build a YOLO-seg dataset from every raw image using the autolabel proxy."""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

METHOD = Path(__file__).resolve().parent
PIPELINE = METHOD.parent.parent
DATA_HOME = PIPELINE.parent / "data" / "rice_count"
RAW = DATA_HOME / "raw"
DATASET = DATA_HOME / "datasets" / "weak_count_model_all_raw"
WORK = DATA_HOME / "work" / "weak_count_model_all_raw"
CACHE = DATA_HOME / "work" / "weak_count_model" / "label_cache_lab_v3"
LABEL_INPUTS = WORK / "label_inputs"
COUNTS = WORK / "pseudo_counts.csv"
LONG_SIDE = 1536
MIN_AREA = 200
MIN_FOREIGN_AREA = 700
MAX_SPLIT_CLUSTERS = 50
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_IDS = {"rice": 0, "foreign": 1}

sys.path.insert(0, str(METHOD.parent / "autolabel_proxy"))
from autolabel import label_image  # noqa: E402


def natural_key(path):
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.name.lower())


def scaled_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not decode raw image: {path}")
    height, width = image.shape[:2]
    scale = min(1.0, LONG_SIDE / max(height, width))
    if scale < 1:
        image = cv2.resize(image, (round(width * scale), round(height * scale)),
                           interpolation=cv2.INTER_AREA)
    return image


def generate_cached_polygons(path):
    image = scaled_image(path)
    LABEL_INPUTS.mkdir(parents=True, exist_ok=True)
    temporary = LABEL_INPUTS / f"{path.stem}.jpg"
    if not cv2.imwrite(str(temporary), image):
        raise OSError(f"Could not save temporary label input: {temporary}")
    objects = []

    def collect(mask, class_name):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        # A KMeans partition can be disconnected. Keep one object per proxy
        # instance instead of converting every fragment to a false grain.
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < MIN_AREA:
            return
        polygon = cv2.approxPolyDP(contour, 1.0, True).reshape(-1, 2)
        if len(polygon) >= 3:
            objects.append([CLASS_IDS[class_name], polygon.astype(float).reshape(-1).tolist()])

    try:
        # KMeans runs for every detected rice clump. Keep BLAS/OpenMP from
        # creating a large thread team for each small clustering operation.
        with threadpool_limits(limits=1):
            label_image(temporary, min_area=MIN_AREA, min_foreign_area=MIN_FOREIGN_AREA,
                        instance_callback=collect, max_split_clusters=MAX_SPLIT_CLUSTERS)
    finally:
        temporary.unlink(missing_ok=True)
    cache_path = CACHE / f"{path.stem}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(objects), encoding="utf-8")
    return objects


def get_polygons(path):
    cache_path = CACHE / f"{path.stem}.json"
    if cache_path.is_file():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    return generate_cached_polygons(path)


def to_yolo_label(path, polygons, image=None):
    image = scaled_image(path) if image is None else image
    height, width = image.shape[:2]
    lines, counts = [], [0, 0]
    for class_id, flat in polygons:
        points = np.asarray(flat, dtype=np.float32).reshape(-1, 2)
        if class_id not in (0, 1) or len(points) < 3:
            continue
        if cv2.contourArea(points) < MIN_AREA:
            continue
        points[:, 0] = np.clip(points[:, 0] / width, 0, 1)
        points[:, 1] = np.clip(points[:, 1] / height, 0, 1)
        lines.append(str(class_id) + " " + " ".join(f"{value:.6f}" for value in points.ravel()))
        counts[class_id] += 1
    return "\n".join(lines), counts


def build():
    paths = sorted((p for p in RAW.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS), key=natural_key)
    if len(paths) != 860:
        raise SystemExit(f"Expected all 860 raw images, found {len(paths)} under {RAW}")
    stems = [p.stem for p in paths]
    if len(stems) != len(set(stems)):
        raise SystemExit("Raw image stems must be unique to create YOLO labels")

    image_dir = DATASET / "images" / "all"
    label_dir = DATASET / "labels" / "all"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    rows = []
    cache_hits = cache_misses = 0
    for index, path in enumerate(paths, 1):
        cache_path = CACHE / f"{path.stem}.json"
        if cache_path.is_file():
            cache_hits += 1
        else:
            cache_misses += 1
        polygons = get_polygons(path)
        training_image = scaled_image(path)
        label_text, counts = to_yolo_label(path, polygons, training_image)
        image_target = image_dir / path.name
        # Use the same 1536 px scale as mask generation. This preserves the
        # effective model input detail while reducing Windows RAM pressure.
        if not cv2.imwrite(str(image_target), training_image,
                           [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"Could not save training copy: {image_target}")
        (label_dir / f"{path.stem}.txt").write_text(label_text, encoding="utf-8")
        rows.append({
            "file": path.name,
            "split": "all_train_images",
            "pseudo_rice": counts[0],
            "pseudo_foreign": counts[1],
            "label_source": "autolabel_proxy_lab_largest_contour_per_instance",
            "label_area_threshold_px": MIN_AREA,
            "foreign_area_threshold_px": MIN_FOREIGN_AREA,
            "max_clusters_per_component": MAX_SPLIT_CLUSTERS,
        })
        if index % 25 == 0 or index == len(paths):
            print(f"labeled {index}/{len(paths)} raw images", flush=True)

    with COUNTS.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    data_yaml = (
        f"path: {DATASET.as_posix()}\n"
        "train: images/all\n"
        "val: images/all\n"
        "names:\n  0: rice\n  1: foreign\n"
    )
    (DATASET / "data.yaml").write_text(data_yaml, encoding="utf-8")
    for cache_file in DATASET.rglob("*.cache"):
        cache_file.unlink(missing_ok=True)
    print(f"Images in training split: {len(list(image_dir.iterdir()))}/{len(paths)}")
    print(f"Locally normalized Lab proxy masks (v3 cache): {cache_misses} generated, {cache_hits} reused")
    rice_counts = np.array([row["pseudo_rice"] for row in rows])
    foreign_counts = np.array([row["pseudo_foreign"] for row in rows])
    print(f"Mask instance counts rice min/median/max={rice_counts.min()}/"
          f"{np.median(rice_counts):.0f}/{rice_counts.max()}; "
          f"foreign={foreign_counts.min()}/{np.median(foreign_counts):.0f}/{foreign_counts.max()}")
    print(f"Pseudo-count CSV: {COUNTS}")
    print("All raw photos are included; known composition counts were not forced into masks.")
    print("The val path intentionally points to the same images as train; mAP is in-sample.")


if __name__ == "__main__":
    build()
