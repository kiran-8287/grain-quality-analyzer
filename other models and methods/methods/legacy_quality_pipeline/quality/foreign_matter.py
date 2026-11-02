"""
Foreign matter detection module.

Uses YOLO (when trained) or heuristic fallback to detect non-rice objects.
Foreign matter is detected on the FULL image, not only on rice-grain crops.

Official percentage limit is weight-based. Image-derived quantity is labelled:
"Image-based count/area estimate" — not official laboratory weight percentage.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ml.config import get_model_info, get_project_root, get_threshold
from ml.segmentation.mask_ops import draw_mask

logger = logging.getLogger(__name__)


@dataclass
class ForeignObject:
    """A detected foreign matter object."""
    foreign_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    area_pixels: int = 0

    def to_dict(self) -> Dict:
        return {
            "foreign_id": self.foreign_id,
            "class": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
            "area_pixels": self.area_pixels,
        }


@dataclass
class ForeignMatterResult:
    """Result of foreign matter detection."""
    objects: List[ForeignObject]
    foreign_object_count: int
    foreign_matter_fraction_estimate: float
    total_image_area: int
    method: str
    processing_time_seconds: float
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "objects": [obj.to_dict() for obj in self.objects],
            "foreign_object_count": self.foreign_object_count,
            "foreign_matter_fraction_estimate": round(self.foreign_matter_fraction_estimate, 6),
            "total_image_area": self.total_image_area,
            "method": self.method,
            "basis": "Image-based count/area estimate — not official laboratory weight percentage",
            "processing_time_seconds": round(self.processing_time_seconds, 3),
            "warnings": self.warnings,
        }


def detect_foreign_matter(
    image_rgb: np.ndarray,
    grain_masks: Optional[List[np.ndarray]] = None,
    grain_mask_offsets: Optional[List[Tuple[int, int]]] = None,
) -> ForeignMatterResult:
    """
    Detect foreign matter in the full image.

    Args:
        image_rgb: Full RGB image
        grain_masks: Optional list of grain masks (to exclude rice regions).
            Masks may be tight bounding-box crops.
        grain_mask_offsets: Optional image-space (x, y) offset for each mask.
            When omitted, masks are assumed to be full-frame masks.
    """
    start_time = time.time()

    model_info = get_model_info("foreign_matter")
    model_path = get_project_root() / model_info.get("weights_path", "")

    if model_path.exists() and model_info.get("status") == "trained":
        result = _detect_foreign_yolo(image_rgb, model_path, grain_masks)
    else:
        result = _detect_foreign_heuristic(
            image_rgb, grain_masks, grain_mask_offsets=grain_mask_offsets
        )

    result.processing_time_seconds = time.time() - start_time
    return result


def classify_non_rice_object(mean_rgb) -> str:
    """
    Colour rule that assigns one of the project's foreign-matter class names
    (models/foreign_matter/class_mapping.json) to a NON-RICE object.

    Shared by the foreign-matter heuristic and the rice gate so both channels
    always speak the same class vocabulary. This function must never return a
    rice class name — foreign matter is not rice.
    """
    arr = np.asarray(mean_rgb, dtype=np.float32).reshape(-1)
    r, g, b = float(arr[0]), float(arr[1]), float(arr[2])

    if r < 80 and g < 80 and b < 80:
        return "stone"
    if g > r + 30:
        return "organic"
    return "other_foreign_matter"


def merge_gate_foreign_objects(
    fm_result: ForeignMatterResult,
    rice_gate: Optional[Dict],
    iou_threshold: float = 0.5,
) -> ForeignMatterResult:
    """
    Merge the rice gate's non-rice detections into the foreign-matter result.

    The foreign-matter channel excludes regions already segmented as grains, so an
    object that the segmenter accepted as a "grain" would otherwise be missing from
    the foreign-matter report. The rice gate classifies every detected object
    independently (rice class vs. the foreign-matter class vocabulary), so its
    non-rice detections are authoritative and are merged in — deduplicated against
    the objects that were already reported.

    Foreign matter is NOT rice and must stay separately visible.
    """
    if not rice_gate or not rice_gate.get("detections"):
        return fm_result

    def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter <= 0:
            return 0.0
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    objects = list(fm_result.objects)
    added = 0

    for detection in rice_gate["detections"]:
        if detection.get("is_rice"):
            continue
        bbox = detection.get("bbox") or []
        if len(bbox) != 4:
            continue
        bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        if any(_iou(bbox, obj.bbox) >= iou_threshold for obj in objects):
            continue
        added += 1
        objects.append(
            ForeignObject(
                foreign_id=len(objects) + 1,
                class_name=str(detection.get("class_name", "other_foreign_matter")),
                confidence=float(detection.get("confidence", 0.0)),
                bbox=bbox,
                area_pixels=int(detection.get("area_pixels", 0)),
            )
        )

    if added == 0:
        return fm_result

    total_area = fm_result.total_image_area
    total_foreign_area = sum(obj.area_pixels for obj in objects)

    return ForeignMatterResult(
        objects=objects,
        foreign_object_count=len(objects),
        foreign_matter_fraction_estimate=(
            total_foreign_area / total_area if total_area > 0 else 0.0
        ),
        total_image_area=total_area,
        method=f"{fm_result.method}+rice_gate",
        processing_time_seconds=fm_result.processing_time_seconds,
        warnings=list(fm_result.warnings)
        + [
            f"{added} non-rice object(s) identified by the rice gate were added to the "
            "foreign-matter report (they were segmented as grain instances)."
        ],
    )


def _detect_foreign_yolo(
    image_rgb: np.ndarray,
    model_path,
    grain_masks: Optional[List[np.ndarray]],
) -> ForeignMatterResult:
    """YOLO-based foreign matter detection."""
    try:
        from ultralytics import YOLO
        
        model = YOLO(str(model_path))
        results = model(image_rgb, verbose=False)
        
        h, w = image_rgb.shape[:2]
        total_area = h * w
        objects = []
        total_foreign_area = 0
        
        for r in results:
            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = r.names.get(cls_id, "other_foreign_matter")
                
                bw = int(x2 - x1)
                bh = int(y2 - y1)
                area = bw * bh
                total_foreign_area += area
                
                objects.append(ForeignObject(
                    foreign_id=i + 1,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=(int(x1), int(y1), bw, bh),
                    area_pixels=area,
                ))
        
        fraction = total_foreign_area / total_area if total_area > 0 else 0
        
        return ForeignMatterResult(
            objects=objects,
            foreign_object_count=len(objects),
            foreign_matter_fraction_estimate=fraction,
            total_image_area=total_area,
            method="yolo11n",
            processing_time_seconds=0.0,
        )
    except Exception as e:
        logger.warning(f"YOLO inference failed: {e}. Using heuristic fallback.")
        return _detect_foreign_heuristic(image_rgb, grain_masks)


def _detect_foreign_heuristic(
    image_rgb: np.ndarray,
    grain_masks: Optional[List[np.ndarray]],
    grain_mask_offsets: Optional[List[Tuple[int, int]]] = None,
) -> ForeignMatterResult:
    """
    Heuristic fallback for foreign matter detection.
    Uses colour and shape analysis to identify non-rice objects.
    Clearly labelled as heuristic.
    """
    h, w = image_rgb.shape[:2]
    total_area = h * w
    warnings = [
        "YOLO foreign matter model not available. Using heuristic fallback. "
        "Run: python training/train_foreign_matter.py"
    ]
    
    # Create a combined rice mask
    rice_mask = np.zeros((h, w), dtype=np.uint8)
    if grain_masks:
        for i, gm in enumerate(grain_masks):
            if gm is None:
                continue
            if grain_mask_offsets is not None and i < len(grain_mask_offsets):
                # Cropped mask: draw it at its image-space offset.
                draw_mask(rice_mask, gm, grain_mask_offsets[i])
            elif gm.shape == (h, w):
                rice_mask = cv2.bitwise_or(rice_mask, gm)

    padding = max(
        0,
        int(get_threshold("foreign_matter", "grain_mask_exclusion_padding_pixels", 3)),
    )
    if padding:
        kernel_size = padding * 2 + 1
        padding_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        rice_mask = cv2.dilate(rice_mask, padding_kernel, iterations=1)
    
    # Look for objects that are NOT rice using the same background-aware foreground mask
    from ml.segmentation.segmentation import extract_foreground_mask
    binary, _, _ = extract_foreground_mask(image_rgb)
    
    # Exclude each rice mask plus a small edge tolerance so segmentation
    # boundary differences are not reported as foreign-matter fragments.
    non_rice = cv2.bitwise_and(binary, cv2.bitwise_not(rice_mask))
    
    # Find non-rice contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(non_rice, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    objects = []
    total_foreign_area = 0
    fid = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:  # Skip tiny noise
            continue
        
        x, y, cw, ch = cv2.boundingRect(cnt)
        
        # Classify by colour
        roi = image_rgb[y:y+ch, x:x+cw]
        if roi.size == 0:
            continue
        
        mean_color = np.mean(roi.reshape(-1, 3), axis=0)
        
        # Simple heuristic classification (shared with the rice gate)
        cls_name = classify_non_rice_object(mean_color)
        
        fid += 1
        total_foreign_area += area
        
        objects.append(ForeignObject(
            foreign_id=fid,
            class_name=cls_name,
            confidence=0.3,  # Low confidence for heuristic
            bbox=(x, y, cw, ch),
            area_pixels=int(area),
        ))
    
    fraction = total_foreign_area / total_area if total_area > 0 else 0
    
    return ForeignMatterResult(
        objects=objects,
        foreign_object_count=len(objects),
        foreign_matter_fraction_estimate=fraction,
        total_image_area=total_area,
        method="heuristic_fallback",
        processing_time_seconds=0.0,
        warnings=warnings,
    )
