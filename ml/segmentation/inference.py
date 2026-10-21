"""
Clean public inference API for rice grain instance segmentation.

Exposes analyze_image() which returns a structured dict with rice grains,
foreign matter, unresolved clusters, and processing metadata.

Model loading priority:
  1. YOLOv8-seg (models/yolo_seg/run1_baseline/weights/best.pt)
  2. Mask R-CNN   (models/maskrcnn/best/model.pth)
  3. Classical CV fallback via ml.segmentation.segment_grains
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ml.config import (
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD,
    CONFIDENCE_THRESHOLDS,
    get_project_root,
    get_model_info,
    get_threshold,
)
from ml.segmentation.mask_ops import translate_points
from ml.segmentation.postprocessing import PostProcessor
from ml.segmentation.segmentation import segment_grains, SegmentationResult, GrainInstance

PROJECT_ROOT = get_project_root()

logger = logging.getLogger(__name__)

_MODEL_CACHE: Dict[str, Tuple[Any, str, str]] = {}


def _load_model() -> Tuple[Any, str, str]:
    """
    Try to load the best available segmentation model.

    Returns (model_obj, method_str, version_str).

    Priority:
      1. YOLO (models/yolo_seg/run1_baseline/weights/best.pt) -> "yolov8l-seg"
      2. MaskRCNN (models/maskrcnn/best/model.pth) -> "maskrcnn"
      3. Classical CV fallback -> "classical_cv"
    """
    cache_key = "segmentation_model"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    yolo_path = PROJECT_ROOT / "models" / "yolo_seg" / "run1_baseline" / "weights" / "best.pt"
    if yolo_path.exists():
        try:
            from ultralytics import YOLO

            model = YOLO(str(yolo_path))
            yolo_info = get_model_info("segmentation")
            version = yolo_info.get("version", "1.0.0")
            _MODEL_CACHE[cache_key] = (model, "yolov8l-seg", version)
            logger.info(f"Loaded YOLO segmentation model from {yolo_path}")
            return model, "yolov8l-seg", version
        except Exception as exc:
            logger.warning(f"Failed to load YOLO model: {exc}. Trying MaskRCNN.")

    maskrcnn_path = PROJECT_ROOT / "models" / "maskrcnn" / "best" / "model.pth"
    if maskrcnn_path.exists():
        try:
            import torch
            from torchvision.models.detection import maskrcnn_resnet50_fpn

            model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)
            state = torch.load(maskrcnn_path, map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                model.load_state_dict(state)
            model.eval()
            seg_info = get_model_info("segmentation")
            version = seg_info.get("version", "1.0.0")
            _MODEL_CACHE[cache_key] = (model, "maskrcnn", version)
            logger.info(f"Loaded MaskRCNN segmentation model from {maskrcnn_path}")
            return model, "maskrcnn", version
        except Exception as exc:
            logger.warning(f"Failed to load MaskRCNN model: {exc}. Falling back to Classical CV.")

    _MODEL_CACHE[cache_key] = (None, "classical_cv", "1.0.0")
    logger.info("Using Classical CV segmentation fallback (no ML model weights found).")
    return None, "classical_cv", "1.0.0"


def _run_yolo_inference(model: Any, img_bgr: np.ndarray) -> List[Dict]:
    """
    Run YOLOv8-seg inference and return raw detections as a list of dicts.

    Detection dict keys: bbox (x,y,w,h), mask (HxW uint8), confidence,
    category_id, category_name (optional).
    """
    try:
        results = model.predict(
            source=img_bgr,
            conf=float(get_threshold("segmentation", "confidence_threshold", 0.25)),
            iou=float(get_threshold("segmentation", "nms_threshold", 0.3)),
            verbose=False,
        )
    except Exception as exc:
        logger.error(f"YOLO inference failed: {exc}")
        return []

    h, w = img_bgr.shape[:2]
    detections: List[Dict] = []
    if not results:
        return detections

    result = results[0]
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    names = getattr(result, "names", {})

    if boxes is None:
        return detections

    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].cpu().numpy() if hasattr(boxes, "xyxy") else None
        conf = float(boxes.conf[i].cpu().numpy()) if hasattr(boxes, "conf") else 0.0
        cls_id = int(boxes.cls[i].cpu().numpy()) if hasattr(boxes, "cls") else 0

        if xyxy is not None and len(xyxy) == 4:
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            bx = int(max(0, x1))
            by = int(max(0, y1))
            bw = int(max(0, min(w, x2) - bx))
            bh = int(max(0, min(h, y2) - by))
        else:
            continue

        mask_arr = np.zeros((h, w), dtype=np.uint8)
        if masks is not None and i < len(masks):
            try:
                m_data = masks.data[i].cpu().numpy() if hasattr(masks, "data") else None
                if m_data is not None:
                    m_bool = m_data.squeeze() > 0.5
                    if m_bool.shape[:2] != (h, w):
                        m_bool = cv2.resize(
                            m_bool.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                    mask_arr[m_bool] = 255
            except Exception:
                pass

        if bw <= 0 or bh <= 0:
            continue

        detections.append(
            {
                "bbox": [bx, by, bw, bh],
                "mask": mask_arr,
                "confidence": conf,
                "category_id": cls_id,
                "category_name": names.get(cls_id, f"class_{cls_id}"),
            }
        )

    return detections


def _run_maskrcnn_inference(model: Any, img_tensor: Any) -> List[Dict]:
    """
    Run Mask R-CNN inference on a pre-prepared image tensor.

    Returns a list of raw detection dicts with the same schema as YOLO output.
    """
    try:
        import torch

        with torch.no_grad():
            outputs = model([img_tensor])
    except Exception as exc:
        logger.error(f"MaskRCNN inference failed: {exc}")
        return []

    detections: List[Dict] = []
    if not outputs:
        return detections

    out = outputs[0]
    boxes = out.get("boxes", None)
    masks = out.get("masks", None)
    scores = out.get("scores", None)
    labels = out.get("labels", None)

    if boxes is None:
        return detections

    boxes_np = boxes.cpu().numpy()
    scores_np = scores.cpu().numpy() if scores is not None else np.ones(len(boxes_np))
    labels_np = labels.cpu().numpy() if labels is not None else np.ones(len(boxes_np), dtype=int)
    masks_np = masks.cpu().numpy() if masks is not None else None

    _, _, h, w = getattr(img_tensor, "shape", (1, 3, 0, 0))

    for i in range(len(boxes_np)):
        x1, y1, x2, y2 = [float(v) for v in boxes_np[i]]
        bx = int(max(0, x1))
        by = int(max(0, y1))
        bw = int(max(0, min(w, x2) - bx))
        bh = int(max(0, min(h, y2) - by))
        if bw <= 0 or bh <= 0:
            continue

        mask_arr = np.zeros((h, w), dtype=np.uint8)
        if masks_np is not None:
            m_bool = masks_np[i].squeeze() > 0.5
            if m_bool.shape[:2] != (h, w):
                m_bool = cv2.resize(
                    m_bool.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            mask_arr[m_bool] = 255

        detections.append(
            {
                "bbox": [bx, by, bw, bh],
                "mask": mask_arr,
                "confidence": float(scores_np[i]),
                "category_id": int(labels_np[i]),
            }
        )

    return detections


def _run_classical_inference(img_bgr: np.ndarray) -> List[Dict]:
    """
    Run Classical CV segmentation fallback via ml.segmentation.segment_grains.

    Returns raw detection dicts matching the YOLO/MaskRCNN schema, using the
    class mapping convention where rice grain == category_id 1.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    seg_result: SegmentationResult = segment_grains(img_rgb, method="classical_cv")

    h, w = img_rgb.shape[:2]
    detections: List[Dict] = []

    grain: GrainInstance
    for grain in seg_result.grains:
        mask = grain.mask
        if mask is None:
            continue
        # The mask is a tight crop of the image; the offset says where it sits.
        offset = tuple(int(v) for v in (getattr(grain, "mask_offset", (0, 0)) or (0, 0)))

        x, y, bw, bh = grain.bbox
        detections.append(
            {
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "mask": mask.astype(np.uint8),
                "mask_offset": offset,
                "confidence": float(grain.confidence),
                "category_id": 1,
                "category_name": "rice_grain",
            }
        )

    return detections


def _mask_to_polygon(mask: np.ndarray) -> List[List[float]]:
    """
    Convert a binary mask to a simplified polygon outline.

    Returns [[x1, y1], [x2, y2], ...] as a list of float coordinate pairs.
    Returns an empty list if the mask has no valid contour.
    """
    if mask is None or not isinstance(mask, np.ndarray) or mask.size == 0:
        return []

    m_bin = (mask > 0).astype(np.uint8)
    if int(m_bin.sum()) == 0:
        return []

    contours, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 3:
        return []

    perimeter = cv2.arcLength(cnt, True)
    epsilon = max(1.0, 0.005 * perimeter)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    if len(approx) < 3:
        approx = cnt

    polygon: List[List[float]] = []
    for pt in approx:
        px, py = pt[0]
        polygon.append([float(px), float(py)])
    return polygon


def _is_unresolved_cluster(
    confidence: float,
    mask_w_h: Tuple[float, float],
    num_components: int = 1,
) -> bool:
    """
    Decide whether a detection should be flagged as an unresolved cluster.

    A detection is unresolved when it has very low confidence or a suspiciously
    merged appearance (unusually wide bounding box or multiple connected
    components within the mask). Unresolved clusters go into the
    unresolved_clusters list and are NEVER misclassified as foreign matter.
    """
    if confidence < 0.30:
        return True
    if num_components is not None and num_components >= 3:
        return True
    w, h = mask_w_h
    max_side = max(w, h)
    min_side = min(w, h) + 1e-6
    if max_side / min_side > 5.0:
        return True
    return False


def analyze_image(image: np.ndarray) -> Dict[str, Any]:
    """
    Public segmentation inference entry point.

    Args:
        image: Input image as HxWxC numpy array (BGR or RGB accepted).

    Returns dict with keys:
      rice_detected: bool
      rice_count: int
      foreign_matter_count: int
      unresolved_cluster_count: int
      grains: list[dict] — per-grain outputs with id, confidence,
              confidence_label, bbox, mask_polygon, centroid, area_pixels,
              is_touching, segmentation_method
      foreign_matter: list[dict] — id, class, confidence, bbox
      unresolved_clusters: list[dict]
      processing: dict with inference_ms, total_ms, tiling_used, num_tiles
      method: "yolov8l-seg" | "maskrcnn" | "classical_cv"
      model_version: str
    """
    total_start = time.perf_counter()

    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return {
            "rice_detected": False,
            "rice_count": 0,
            "foreign_matter_count": 0,
            "unresolved_cluster_count": 0,
            "grains": [],
            "foreign_matter": [],
            "unresolved_clusters": [],
            "processing": {
                "inference_ms": 0,
                "total_ms": 0,
                "tiling_used": False,
                "num_tiles": 1,
            },
            "method": "classical_cv",
            "model_version": "1.0.0",
        }

    if len(image.shape) == 2:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        ch = image.shape[2] if len(image.shape) == 3 else 3
        if ch == 4:
            img_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif ch == 3:
            if image.dtype == np.uint8:
                img_bgr = image.copy()
            else:
                img_bgr = (image * 255).astype(np.uint8)
        else:
            img_bgr = image.copy()

    h, w = img_bgr.shape[:2]

    post = PostProcessor(
        conf_threshold=float(get_threshold("segmentation", "confidence_threshold", 0.25)),
        mask_iou_threshold=float(get_threshold("segmentation", "nms_threshold", 0.5)),
        min_grain_area=int(get_threshold("segmentation", "min_grain_area_pixels", 50)),
        tile_size=640,
        tile_overlap=128,
    )

    use_tiling = max(h, w) > post.tile_size
    num_tiles = 1
    all_raw: List[Dict] = []

    model, method_str, model_version = _load_model()

    inference_start = time.perf_counter()

    if method_str == "yolov8l-seg" and model is not None:
        if use_tiling:
            tiles = post.split_into_tiles(img_bgr, post.tile_size, post.tile_overlap)
            num_tiles = len(tiles)
            for tile_img, (gx1, gy1, tw, th) in tiles:
                tile_raw = _run_yolo_inference(model, tile_img)
                for d in tile_raw:
                    d_global = post.map_tile_to_global(d, (gx1, gy1))
                    d_global["_tile_offset"] = (gx1, gy1, tw, th)
                    all_raw.append(d_global)
        else:
            all_raw = _run_yolo_inference(model, img_bgr)

    elif method_str == "maskrcnn" and model is not None:
        try:
            import torch
            from torchvision import transforms

            transform = transforms.Compose([transforms.ToTensor()])
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor = transform(img_rgb)
            if use_tiling:
                tiles = post.split_into_tiles(img_rgb, post.tile_size, post.tile_overlap)
                num_tiles = len(tiles)
                for tile_img, (gx1, gy1, tw, th) in tiles:
                    t_tile = transform(tile_img)
                    tile_raw = _run_maskrcnn_inference(model, t_tile)
                    for d in tile_raw:
                        d_global = post.map_tile_to_global(d, (gx1, gy1))
                        d_global["_tile_offset"] = (gx1, gy1, tw, th)
                        all_raw.append(d_global)
            else:
                all_raw = _run_maskrcnn_inference(model, tensor)
        except Exception as exc:
            logger.warning(f"MaskRCNN path failed: {exc}. Using classical fallback.")
            all_raw = _run_classical_inference(img_bgr)
            method_str = "classical_cv"
    else:
        all_raw = _run_classical_inference(img_bgr)

    inference_ms = int(round((time.perf_counter() - inference_start) * 1000))

    processed = post.run_full_pipeline(all_raw, (h, w), use_tiling=use_tiling)

    grains_out: List[Dict[str, Any]] = []
    foreign_matter_out: List[Dict[str, Any]] = []
    unresolved_out: List[Dict[str, Any]] = []
    fm_id_counter = 0
    uc_id_counter = 0

    fm_class_mapping = {
        0: "background",
        1: "rice_grain",
        2: "foreign_matter",
    }

    for det in processed:
        cat_id = int(det.get("category_id", 1))
        confidence = float(det.get("confidence", 0.0))
        bbox = det.get("bbox", [0, 0, 0, 0])
        bx, by, bw, bh = (
            [float(v) for v in bbox]
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4
            else [0.0, 0.0, 0.0, 0.0]
        )

        mask = det.get("mask")
        mask_offset = det.get("mask_offset") or (0, 0)
        area_pixels = int(np.sum(mask > 0)) if isinstance(mask, np.ndarray) else 0

        num_components = 1
        if isinstance(mask, np.ndarray):
            m_bin = (mask > 0).astype(np.uint8)
            n_cc, _, _, _ = cv2.connectedComponentsWithStats(m_bin, connectivity=8)
            num_components = max(0, n_cc - 1)

        is_cluster = _is_unresolved_cluster(confidence, (bw, bh), num_components)
        is_rice_class = cat_id in (1,) or (
            det.get("category_name", "").lower() in ("rice_grain", "rice", "grain")
        )
        is_fm_class = cat_id >= 2 or (
            det.get("category_name", "").lower()
            in ("foreign_matter", "stone", "inorganic", "organic", "other_foreign_matter")
        )

        if is_cluster and is_rice_class:
            uc_id_counter += 1
            unresolved_out.append(
                {
                    "id": uc_id_counter,
                    "confidence": round(confidence, 4),
                    "bbox": [int(bx), int(by), int(bw), int(bh)],
                    "area_pixels": area_pixels,
                    "num_components": num_components,
                    "reason": "low_confidence_or_merged_appearance",
                }
            )
            continue

        if is_fm_class and not is_rice_class:
            fm_id_counter += 1
            cls_name = det.get("category_name") or fm_class_mapping.get(
                cat_id, "foreign_matter"
            )
            foreign_matter_out.append(
                {
                    "id": fm_id_counter,
                    "class": cls_name,
                    "confidence": round(confidence, 4),
                    "bbox": [int(bx), int(by), int(bw), int(bh)],
                }
            )
            continue

        if not is_rice_class and not is_fm_class:
            if confidence >= 0.50:
                fm_id_counter += 1
                foreign_matter_out.append(
                    {
                        "id": fm_id_counter,
                        "class": "foreign_matter",
                        "confidence": round(confidence, 4),
                        "bbox": [int(bx), int(by), int(bw), int(bh)],
                    }
                )
            else:
                uc_id_counter += 1
                unresolved_out.append(
                    {
                        "id": uc_id_counter,
                        "confidence": round(confidence, 4),
                        "bbox": [int(bx), int(by), int(bw), int(bh)],
                        "area_pixels": area_pixels,
                        "num_components": num_components,
                        "reason": "ambiguous_class_low_confidence",
                    }
                )
            continue

        cx = float(bx + bw / 2.0)
        cy = float(by + bh / 2.0)
        if isinstance(mask, np.ndarray) and area_pixels > 0:
            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                cx = float(np.mean(xs)) + float(mask_offset[0])
                cy = float(np.mean(ys)) + float(mask_offset[1])

        polygon = (
            translate_points(_mask_to_polygon(mask), mask_offset)
            if isinstance(mask, np.ndarray)
            else []
        )

        grain_entry = {
            "id": int(det.get("id", len(grains_out) + 1)),
            "confidence": round(confidence, 4),
            "confidence_label": det.get(
                "confidence_label",
                PostProcessor.compute_confidence_label(confidence),
            ),
            "bbox": [int(bx), int(by), int(bw), int(bh)],
            "mask_polygon": polygon,
            "centroid": [round(cx, 2), round(cy, 2)],
            "area_pixels": area_pixels,
            "is_touching": bool(det.get("is_touching", False)),
            "segmentation_method": method_str,
        }
        grains_out.append(grain_entry)

    rice_count = len(grains_out)
    fm_count = len(foreign_matter_out)
    uc_count = len(unresolved_out)
    rice_detected = rice_count > 0

    total_ms = int(round((time.perf_counter() - total_start) * 1000))

    return {
        "rice_detected": rice_detected,
        "rice_count": rice_count,
        "foreign_matter_count": fm_count,
        "unresolved_cluster_count": uc_count,
        "grains": grains_out,
        "foreign_matter": foreign_matter_out,
        "unresolved_clusters": unresolved_out,
        "processing": {
            "inference_ms": inference_ms,
            "total_ms": total_ms,
            "tiling_used": use_tiling,
            "num_tiles": num_tiles,
        },
        "method": method_str,
        "model_version": model_version,
    }
