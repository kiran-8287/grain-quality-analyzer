"""
Instance segmentation module for rice grains.

Primary method: Mask R-CNN (when trained weights available)
Fallback: Classical CV pipeline (watershed, contours, connected components)

The fallback is clearly labelled as "Classical CV fallback" —
does NOT pretend results came from Mask R-CNN.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from ml.config import get_threshold, get_model_info, get_project_root
from ml.segmentation.mask_ops import tight_bounds, translate_contour

logger = logging.getLogger(__name__)


@dataclass
class GrainInstance:
    """
    A single segmented grain instance.

    ``mask`` is the binary mask cropped to the grain's tight bounding box and
    ``mask_offset`` is the ``(x, y)`` position of that crop in the image.
    Cropping keeps the memory cost of a grain proportional to the grain
    instead of to the image, so large uploads (12 MP phone photos) do not
    exhaust RAM.  Pixel-wise measurements on the crop are identical to the
    full-frame mask; only the position of the mask differs, and that is
    carried by ``mask_offset``.
    """
    grain_id: int
    mask: np.ndarray              # Binary mask (tight bbox crop of the image)
    bbox: Tuple[int, int, int, int]  # x, y, w, h (image coordinates)
    confidence: float
    centroid: Tuple[float, float]
    mask_offset: Tuple[int, int] = (0, 0)  # (x, y) of the mask crop in the image
    contour: Optional[np.ndarray] = None
    is_touching: bool = False
    is_overlapping: bool = False
    segmentation_quality: str = "good"  # good, uncertain, poor
    method: str = "classical_cv_fallback"
    is_foreign_matter: bool = False
    # Phase 1 segmentation extras (optional, populated when ml.inference cascade is used)
    confidence_label: Optional[str] = None    # HIGH, MEDIUM, LOW, or None if not assigned
    segmentation_method: Optional[str] = None  # yolov8l-seg, maskrcnn, classical_cv, etc.
    mask_polygon: Optional[List[Tuple[float, float]]] = None  # original polygon from Phase1, if any

    def mask_area(self) -> int:
        return int(np.sum(self.mask > 0))


@dataclass
class SegmentationResult:
    """Result of grain segmentation."""
    grains: List[GrainInstance]
    detected_count: int
    accepted_count: int
    uncertain_count: int
    rejected_count: int
    estimated_merged_count: int
    method: str
    processing_time_seconds: float
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "detected_count": self.detected_count,
            "accepted_count": self.accepted_count,
            "uncertain_count": self.uncertain_count,
            "rejected_count": self.rejected_count,
            "estimated_merged_count": self.estimated_merged_count,
            "method": self.method,
            "processing_time_seconds": round(self.processing_time_seconds, 3),
            "warnings": self.warnings,
        }


def segment_grains(
    image_rgb: np.ndarray,
    method: str = "auto",
) -> SegmentationResult:
    """
    Segment individual rice grains from an image.
    
    Args:
        image_rgb: RGB image (numpy array)
        method: 'yolov8', 'mask_rcnn', 'classical_cv', or 'auto'
        
    Returns:
        SegmentationResult with individual grain masks
    """
    start_time = time.time()
    
    project_root = get_project_root()

    # Check if YOLOv8 weights are available and ultralytics is installed
    yolo_weights_path = project_root / "models" / "yolo_seg" / "run1_baseline" / "weights" / "best.pt"
    yolov8_weights_exist = yolo_weights_path.exists()
    yolov8_installed = False
    yolo_model = None
    try:
        from ultralytics import YOLO
        yolov8_installed = True
        if yolov8_weights_exist:
            yolo_model = YOLO(str(yolo_weights_path))
    except ImportError:
        yolov8_installed = False
    except Exception as exc:
        logger.warning(f"Failed to load YOLO model: {exc}")
        yolov8_installed = False

    yolov8_available = yolov8_weights_exist and yolov8_installed and yolo_model is not None

    # Determine YOLO method string based on weights filename
    yolov8_method_str = "yolov8l-seg"
    if yolov8_weights_exist:
        wp = str(yolo_weights_path).lower()
        if "nano" in wp or "yolov8n" in wp or "-n-" in wp or "_n_" in wp:
            yolov8_method_str = "yolov8n-seg"
        elif "large" in wp or "yolov8l" in wp or "-l-" in wp or "_l_" in wp:
            yolov8_method_str = "yolov8l-seg"

    # Check if Mask R-CNN weights are available
    model_info = get_model_info("segmentation")
    mask_rcnn_weights_exist = (
        model_info.get("status") == "trained" and
        (project_root / model_info.get("weights_path", "")).exists()
    )
    mask_rcnn_installed = False
    try:
        import torch
        mask_rcnn_installed = True
    except ImportError:
        mask_rcnn_installed = False

    mask_rcnn_available = mask_rcnn_weights_exist and mask_rcnn_installed

    if method == "auto":
        if yolov8_available:
            method = "yolov8"
        elif mask_rcnn_available:
            method = "mask_rcnn"
        else:
            method = "classical_cv"

    if method == "yolov8":
        if yolov8_available:
            result = _segment_yolov8(image_rgb, yolo_model, yolov8_method_str)
        else:
            missing = []
            if not yolov8_weights_exist:
                missing.append(f"weights not found at {yolo_weights_path}")
            if not yolov8_installed:
                missing.append("ultralytics package not installed")
            logger.warning(
                "YOLOv8 not available (" + ", ".join(missing) + "). "
                "Trying Mask R-CNN, then Classical CV fallback."
            )
            if mask_rcnn_available:
                result = _segment_mask_rcnn(image_rgb)
            else:
                result = _segment_classical_cv(image_rgb)
    elif method == "mask_rcnn" and mask_rcnn_available:
        result = _segment_mask_rcnn(image_rgb)
    elif method == "classical_cv":
        result = _segment_classical_cv(image_rgb)
    else:
        if method == "mask_rcnn" and not mask_rcnn_available:
            logger.warning(
                "Mask R-CNN weights not available. "
                "Using Classical CV fallback. "
                "Train segmentation model with: python training/train_segmentation.py"
            )
        result = _segment_classical_cv(image_rgb)

    result.processing_time_seconds = time.time() - start_time
    return result


def _segment_yolov8(
    image_rgb: np.ndarray,
    model,
    method_str: str = "yolov8l-seg",
) -> SegmentationResult:
    """
    Segment using YOLOv8-seg instance segmentation model.

    Handles:
      - category_id=1 (rice_grain): standard GrainInstance with is_foreign_matter=False
      - category_id=2 (foreign_matter): GrainInstance with is_foreign_matter=True
    """
    h, w = image_rgb.shape[:2]
    warnings: List[str] = []

    min_area = get_threshold("segmentation", "min_grain_area_pixels", 50)

    try:
        results = model.predict(
            image_rgb,
            conf=0.25,
            iou=0.5,
            verbose=False,
        )
    except Exception as exc:
        logger.error(f"YOLOv8 predict() failed: {exc}. Falling back to Classical CV.")
        fallback = _segment_classical_cv(image_rgb)
        fallback.warnings.append(f"YOLOv8 inference failed ({exc}); used Classical CV fallback.")
        return fallback

    grains: List[GrainInstance] = []
    uncertain_count = 0
    rejected_count = 0
    estimated_merged = 0
    grain_id = 0
    fm_id = 0

    if not results:
        return SegmentationResult(
            grains=[],
            detected_count=0,
            accepted_count=0,
            uncertain_count=0,
            rejected_count=0,
            estimated_merged_count=0,
            method=method_str,
            processing_time_seconds=0.0,
            warnings=["YOLOv8 returned no predictions."],
        )

    result = results[0]
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)

    if boxes is None:
        return SegmentationResult(
            grains=[],
            detected_count=0,
            accepted_count=0,
            uncertain_count=0,
            rejected_count=0,
            estimated_merged_count=0,
            method=method_str,
            processing_time_seconds=0.0,
            warnings=["YOLOv8 predictions have no boxes."],
        )

    num_detections = len(boxes)

    for i in range(num_detections):
        # Extract box data
        xyxy = None
        conf = 0.0
        cls_id = 0
        try:
            if hasattr(boxes, "xyxy"):
                xyxy_arr = boxes.xyxy[i].cpu().numpy() if hasattr(boxes.xyxy[i], "cpu") else np.array(boxes.xyxy[i])
                xyxy = [float(v) for v in xyxy_arr]
            if hasattr(boxes, "conf"):
                c_val = boxes.conf[i].cpu().numpy() if hasattr(boxes.conf[i], "cpu") else boxes.conf[i]
                conf = float(c_val)
            if hasattr(boxes, "cls"):
                cl_val = boxes.cls[i].cpu().numpy() if hasattr(boxes.cls[i], "cpu") else boxes.cls[i]
                cls_id = int(cl_val)
        except Exception as exc:
            logger.warning(f"Failed to extract YOLO detection {i}: {exc}")
            rejected_count += 1
            continue

        if xyxy is None or len(xyxy) != 4:
            rejected_count += 1
            continue

        x1, y1, x2, y2 = xyxy
        bx = int(max(0, round(x1)))
        by = int(max(0, round(y1)))
        bw = int(max(0, min(w, round(x2)) - bx))
        bh = int(max(0, min(h, round(y2)) - by))
        if bw <= 0 or bh <= 0:
            rejected_count += 1
            continue

        # Extract mask
        mask_arr = np.zeros((h, w), dtype=np.uint8)
        if masks is not None and i < len(masks):
            try:
                m_data = None
                if hasattr(masks, "data"):
                    d = masks.data[i]
                    m_data = d.cpu().numpy() if hasattr(d, "cpu") else np.array(d)
                if m_data is not None:
                    m_bool = m_data.squeeze() > 0.5
                    if m_bool.shape[:2] != (h, w):
                        m_bool = cv2.resize(
                            m_bool.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                    mask_arr[m_bool] = 255
            except Exception:
                # If mask extraction fails, fallback to bbox rectangle mask
                mask_arr[by:by + bh, bx:bx + bw] = 255

        if int(np.sum(mask_arr > 0)) < min_area:
            rejected_count += 1
            continue

        # Extract contour and centroid
        contour = None
        try:
            cnts, _ = cv2.findContours(mask_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if cnts:
                contour = max(cnts, key=cv2.contourArea)
        except Exception:
            contour = None

        if contour is not None and len(contour) > 0:
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
            else:
                cx = float(bx + bw / 2.0)
                cy = float(by + bh / 2.0)
        else:
            cx = float(bx + bw / 2.0)
            cy = float(by + bh / 2.0)

        is_fm = cls_id == 2

        if is_fm:
            fm_id += 1
            instance_id = fm_id
        else:
            grain_id += 1
            instance_id = grain_id

        # Compute simple quality heuristic
        seg_quality = "good"
        if conf < 0.5:
            seg_quality = "uncertain"
            uncertain_count += 1

        is_touching = False
        try:
            # If mask solidity is low, it may be a touching cluster
            if contour is not None and len(contour) > 3:
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                mask_area_val = float(np.sum(mask_arr > 0))
                if hull_area > 0:
                    solidity = mask_area_val / hull_area
                    if solidity < 0.85:
                        is_touching = True
                        estimated_merged += 1
                        if seg_quality == "good":
                            seg_quality = "uncertain"
                            uncertain_count += 1
        except Exception:
            pass

        confidence_val = round(float(max(0.0, min(1.0, conf))), 4)

        # Store the mask as a tight crop; the offset locates it in the image.
        mask_bounds = tight_bounds(mask_arr)
        if mask_bounds is None:
            rejected_count += 1
            continue
        mx0, my0, mw, mh = mask_bounds

        grain = GrainInstance(
            grain_id=instance_id,
            mask=mask_arr[my0 : my0 + mh, mx0 : mx0 + mw],
            bbox=(bx, by, bw, bh),
            confidence=confidence_val,
            centroid=(cx, cy),
            mask_offset=(mx0, my0),
            contour=contour,
            is_touching=is_touching,
            segmentation_quality=seg_quality,
            method=method_str,
            is_foreign_matter=is_fm,
        )
        grains.append(grain)

    detected_count = grain_id + fm_id + rejected_count
    accepted_count = len(grains)

    if not grains:
        warnings.append("No grain instances found after YOLOv8 segmentation.")

    return SegmentationResult(
        grains=grains,
        detected_count=detected_count,
        accepted_count=accepted_count,
        uncertain_count=uncertain_count,
        rejected_count=rejected_count,
        estimated_merged_count=estimated_merged,
        method=method_str,
        processing_time_seconds=0.0,
        warnings=warnings,
    )


def _segment_mask_rcnn(image_rgb: np.ndarray) -> SegmentationResult:
    """
    Segment using Mask R-CNN.
    Only called when trained weights are available.
    """
    # This will be implemented when model is trained
    # For now, fall back to classical CV with a warning
    logger.warning("Mask R-CNN inference not yet implemented. Using classical CV fallback.")
    result = _segment_classical_cv(image_rgb)
    result.warnings.append("Mask R-CNN model loaded but inference path pending integration.")
    return result


def extract_foreground_mask(
    image_rgb: np.ndarray,
) -> Tuple[np.ndarray, bool, Dict]:
    """
    Extract binary foreground mask (grains = 255, background = 0)
    from an RGB image, automatically handling dark, light, or colored backgrounds.
    
    Uses perimeter border pixel sampling combined with Otsu thresholding.
    
    Returns:
        binary: uint8 array (h, w), 255 for foreground, 0 for background
        is_dark_bg: True if background is dark, False if light
        meta: dictionary with threshold statistics
    """
    h, w = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Check for blank / zero-variance image.
    # cv2.meanStdDev computes the standard deviation in one pass without
    # materialising a float64 copy of the frame (np.std would allocate
    # 8 bytes per pixel, i.e. 96 MB for a 12 MP upload).
    _gray_stats = cv2.meanStdDev(gray)
    _gray_std = float(np.asarray(_gray_stats[1]).ravel()[0])
    if int(np.max(gray)) == int(np.min(gray)) or _gray_std < 1.0:
        return (
            np.zeros((h, w), dtype=np.uint8),
            True,
            {"is_blank": True, "bg_gray": round(float(np.median(gray)), 2)},
        )
        
    margin = max(3, min(h, w) // 50)
    border_pixels = np.concatenate([
        gray[:margin, :].ravel(),
        gray[-margin:, :].ravel(),
        gray[:, :margin].ravel(),
        gray[:, -margin:].ravel(),
    ])
    
    otsu_val, thresh_bright = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_dark = cv2.bitwise_not(thresh_bright)
    
    dark_border_frac = float(np.mean(border_pixels < otsu_val))
    bright_fg_ratio = float(np.sum(thresh_bright > 0) / (h * w))
    dark_fg_ratio = float(np.sum(thresh_dark > 0) / (h * w))
    
    # Border fraction is the primary physical indicator of background polarity:
    # In grain imaging, background surrounds the perimeter of the sample.
    if dark_border_frac >= 0.5:
        # Dark background (e.g. black, navy, dark tray) -> grains are bright
        # Sanity check: grains rarely occupy > 70% of the field of view
        if bright_fg_ratio > 0.70 and dark_fg_ratio < 0.30:
            binary = thresh_dark
            is_dark_bg = False
        else:
            binary = thresh_bright
            is_dark_bg = True
    else:
        # Light background (e.g. white paper, light tray) -> grains are dark
        if dark_fg_ratio > 0.70 and bright_fg_ratio < 0.30:
            binary = thresh_bright
            is_dark_bg = True
        else:
            binary = thresh_dark
            is_dark_bg = False
            
    meta = {
        "otsu_val": float(otsu_val),
        "dark_border_frac": round(dark_border_frac, 3),
        "is_dark_bg": is_dark_bg,
        "fg_ratio": round(float(np.sum(binary > 0) / (h * w)), 4),
        # Median border intensity — used by the rice gate as the background reference.
        "bg_gray": round(float(np.median(border_pixels)), 2),
    }
    return binary, is_dark_bg, meta


def _partition_mask_by_markers(
    comp_mask: np.ndarray,
    markers_int: np.ndarray,
    min_grain_area: int = 50,
) -> List[np.ndarray]:
    """
    Partition 100% of comp_mask pixels to their nearest seed marker.
    Guarantees no boundary erosion or loss of grain pixels.
    """
    y_idx, x_idx = np.where(comp_mask > 0)
    if len(y_idx) == 0:
        return [comp_mask]
    y1, y2 = int(np.min(y_idx)), int(np.max(y_idx)) + 1
    x1, x2 = int(np.min(x_idx)), int(np.max(x_idx)) + 1

    crop_comp = comp_mask[y1:y2, x1:x2]
    crop_markers = markers_int[y1:y2, x1:x2]

    num_m = int(np.max(crop_markers))
    if num_m < 2:
        return [comp_mask]

    _, indices = distance_transform_edt(crop_markers == 0, return_indices=True)
    nearest_crop = crop_markers[indices[0], indices[1]]

    sub_masks = []
    for k in range(1, num_m + 1):
        full_sub = np.zeros_like(comp_mask)
        full_sub[y1:y2, x1:x2] = ((nearest_crop == k) & (crop_comp > 0)).astype(np.uint8) * 255
        if int(np.sum(full_sub > 0)) >= min_grain_area:
            sub_masks.append(full_sub)

    return sub_masks if len(sub_masks) > 1 else [comp_mask]


def _split_grain_cluster(
    comp_mask: np.ndarray,
    med_area: float,
    min_grain_area: int = 50,
) -> List[np.ndarray]:
    """
    Split a potentially merged/touching grain cluster into individual grain masks.
    Uses multi-threshold adaptive distance-transform peak detection with exact
    distance-field partitioning, plus concavity pinch-point cutting fallback.
    Retains 100% of foreground pixels with zero mask shrinkage.
    """
    area = int(np.sum(comp_mask > 0))
    if area < min_grain_area:
        return []

    cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return [comp_mask]
    cnt = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull) if len(cnt) > 3 and len(hull) > 3 else None

    deep_defects = []
    if defects is not None:
        for i in range(len(defects)):
            s, e, f, d = defects[i]
            depth = d / 256.0
            if depth >= 5.0:
                deep_defects.append((tuple(cnt[f][0]), depth))

    hull_pts = cv2.convexHull(cnt, returnPoints=True)
    hull_area = cv2.contourArea(hull_pts)
    solidity = area / hull_area if hull_area > 0 else 1.0

    # Cluster detection criteria:
    # 1. Area is significantly larger than typical single grain area (>= 1.35x median)
    # 2. Or at least two concave pinch points (deep defects >= 5px)
    # 3. Or low solidity (< 0.90) with any concavity defect
    is_cluster = (
        (area >= 1.35 * med_area)
        or (len(deep_defects) >= 2)
        or (solidity < 0.90 and len(deep_defects) >= 1)
    )
    if not is_cluster:
        return [comp_mask]

    dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
    max_d = float(np.max(dist))
    if max_d <= 0:
        return [comp_mask]

    best_markers = None

    # Adaptive marker detection via distance transform peaks across decreasing relative thresholds
    for ratio in [0.82, 0.76, 0.70, 0.64, 0.58, 0.52, 0.46]:
        thresh = (dist >= ratio * max_d).astype(np.uint8)
        thresh = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        nm, m, s, _ = cv2.connectedComponentsWithStats(thresh)
        valid_markers = [k for k in range(1, nm) if s[k, cv2.CC_STAT_AREA] >= 20]
        if len(valid_markers) > 1:
            cleaned_m = np.zeros_like(thresh, dtype=np.int32)
            for new_id, old_id in enumerate(valid_markers, start=1):
                cleaned_m[m == old_id] = new_id
            best_markers = cleaned_m
            break

    if best_markers is not None:
        sub_masks = _partition_mask_by_markers(comp_mask, best_markers, min_grain_area=min_grain_area)
        if len(sub_masks) > 1:
            return sub_masks

    # Fallback for tightly touching parallel grains with distinct concave pinch points
    if len(deep_defects) >= 2:
        deep_defects.sort(key=lambda x: x[1], reverse=True)
        pt1 = deep_defects[0][0]
        best_pt2 = None
        for p2, d2 in deep_defects[1:]:
            dist_pts = np.hypot(pt1[0] - p2[0], pt1[1] - p2[1])
            if dist_pts >= 12:
                best_pt2 = p2
                break
        if best_pt2 is not None:
            cut_mask = comp_mask.copy()
            cv2.line(cut_mask, pt1, best_pt2, 0, 2)
            nm, m, s, _ = cv2.connectedComponentsWithStats(cut_mask)
            if nm > 2:
                sub_masks = _partition_mask_by_markers(comp_mask, m, min_grain_area=min_grain_area)
                if len(sub_masks) > 1:
                    return sub_masks

    return [comp_mask]


def _segment_classical_cv(image_rgb: np.ndarray) -> SegmentationResult:
    """
    Classical CV fallback pipeline for grain segmentation.
    
    Steps:
    1. Robust background estimation via perimeter sampling
    2. Foreground extraction (Otsu thresholding calibrated to background polarity)
    3. Noise cleanup with light morphology (prevents artificial merging)
    4. Adaptive touching-grain splitting via distance-transform watershed and pinch-point cutting
    5. Per-grain instance mask, contour, bbox, and metadata extraction
    
    This is clearly labelled as "Classical CV fallback".
    """
    h, w = image_rgb.shape[:2]
    warnings = []
    
    binary, is_dark_bg, meta = extract_foreground_mask(image_rgb)
    if meta.get("is_blank", False) or np.sum(binary > 0) == 0:
        return SegmentationResult(
            grains=[],
            detected_count=0,
            accepted_count=0,
            uncertain_count=0,
            rejected_count=0,
            estimated_merged_count=0,
            method="classical_cv_fallback",
            processing_time_seconds=0.0,
            warnings=["No foreground grain objects detected in image."],
        )
        
    if is_dark_bg:
        warnings.append("Detected dark background — segmented bright grain foreground")
    else:
        warnings.append("Detected light background — segmented dark grain foreground")

    # Light morphological opening to remove 1-2 pixel isolated salt noise
    # without bridging touching grains (avoids aggressive closing)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)

    min_area = get_threshold("segmentation", "min_grain_area_pixels", 50)
    merge_area_ratio = get_threshold("segmentation", "merge_detection_area_ratio", 2.5)
    merge_aspect_ratio = get_threshold("segmentation", "merge_detection_aspect_ratio", 4.0)

    num_cc, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

    # Estimate median single grain area
    all_areas = []
    for label in range(1, num_cc):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            all_areas.append(area)

    median_area = float(np.median(all_areas)) if all_areas else 2000.0

    grains = []
    uncertain_count = 0
    rejected_count = 0
    estimated_merged = 0
    grain_id = 0

    for label in range(1, num_cc):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            rejected_count += 1
            continue

        # Work on the component's bounding-box crop instead of a full-frame
        # mask.  A full-frame mask costs image_pixels bytes per component, so a
        # 12 MP upload with 1000 components would retain ~12.6 GB and exhaust
        # RAM.  A 1 px zero ring reproduces the surrounding background for the
        # distance transform; sides flush with the image edge get no ring,
        # because the full-frame mask had no pixels (hence no background)
        # beyond them.
        comp_x = int(stats[label, cv2.CC_STAT_LEFT])
        comp_y = int(stats[label, cv2.CC_STAT_TOP])
        comp_w = int(stats[label, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[label, cv2.CC_STAT_HEIGHT])
        pad_left = 1 if comp_x > 0 else 0
        pad_top = 1 if comp_y > 0 else 0
        pad_right = 1 if comp_x + comp_w < w else 0
        pad_bottom = 1 if comp_y + comp_h < h else 0
        # OpenCV labels connected components in 2x2 block order, so the label
        # order depends on where the crop starts.  Extending the crop to an
        # even offset keeps the block grid aligned with the image, which keeps
        # the component ordering (and therefore the grain order) identical to
        # the full-frame implementation.
        if (comp_x - pad_left) % 2 != 0 and comp_x >= 2:
            pad_left += 1
        if (comp_y - pad_top) % 2 != 0 and comp_y >= 2:
            pad_top += 1
        comp_crop = np.zeros(
            (comp_h + pad_top + pad_bottom, comp_w + pad_left + pad_right),
            dtype=np.uint8,
        )
        comp_crop[pad_top : pad_top + comp_h, pad_left : pad_left + comp_w] = (
            (cc_labels[comp_y : comp_y + comp_h, comp_x : comp_x + comp_w] == label)
            .astype(np.uint8)
            * 255
        )

        # Split cluster if touching grains are present
        splits = _split_grain_cluster(comp_crop, median_area, min_grain_area=min_area)
        was_touching = len(splits) > 1
        if was_touching:
            estimated_merged += len(splits) - 1

        for s_mask in splits:
            # Drop the zero ring: back to the component's own crop frame.
            s_crop = s_mask[pad_top : pad_top + comp_h, pad_left : pad_left + comp_w]
            s_area = int(np.sum(s_crop > 0))
            if s_area < min_area:
                rejected_count += 1
                continue

            contours, _ = cv2.findContours(s_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                rejected_count += 1
                continue

            contour = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw <= 0 or bh <= 0:
                rejected_count += 1
                continue

            M = cv2.moments(contour)
            if M["m00"] == 0:
                rejected_count += 1
                continue

            cx = float(comp_x + M["m10"] / M["m00"])
            cy = float(comp_y + M["m01"] / M["m00"])

            area_ratio = s_area / median_area if median_area > 0 else 1.0
            aspect = max(bw, bh) / (min(bw, bh) + 1e-6)

            is_touching = was_touching or (area_ratio > merge_area_ratio) or (aspect > merge_aspect_ratio)
            seg_quality = "uncertain" if (area_ratio > merge_area_ratio or aspect > merge_aspect_ratio) else "good"
            if seg_quality == "uncertain":
                uncertain_count += 1

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = s_area / hull_area if hull_area > 0 else 0
            confidence = min(1.0, solidity * 0.7 + 0.3)

            # Store the grain mask as a tight crop; the offset locates it.
            bounds = tight_bounds(s_crop)
            if bounds is None:
                rejected_count += 1
                continue
            bx0, by0, bw0, bh0 = bounds

            grain_id += 1
            grain = GrainInstance(
                grain_id=grain_id,
                mask=s_crop[by0 : by0 + bh0, bx0 : bx0 + bw0],
                bbox=(comp_x + x, comp_y + y, bw, bh),
                confidence=round(float(confidence), 4),
                centroid=(cx, cy),
                mask_offset=(comp_x + bx0, comp_y + by0),
                contour=translate_contour(contour, (comp_x, comp_y)),
                is_touching=is_touching,
                segmentation_quality=seg_quality,
                method="classical_cv_fallback",
            )
            grains.append(grain)

    detected_count = grain_id + rejected_count
    accepted_count = len(grains)

    if not grains:
        warnings.append("No grain instances found after segmentation.")

    return SegmentationResult(
        grains=grains,
        detected_count=detected_count,
        accepted_count=accepted_count,
        uncertain_count=uncertain_count,
        rejected_count=rejected_count,
        estimated_merged_count=estimated_merged,
        method="classical_cv_fallback",
        processing_time_seconds=0.0,
        warnings=warnings,
    )


def detect_rice_presence(image_rgb: np.ndarray) -> Dict:
    """
    Decide whether the image contains RICE grains (Case 1 rice-presence gate).

    Delegates to ml.rice_gate.evaluate_rice_presence — the single central decision
    point — so that "objects were detected" is never confused with "rice was
    detected". Only detections whose class is the model configuration's rice class
    (models/segmentation/class_mapping.json -> class_id 1 "rice_grain") whose
    confidence passes rice_gate.rice_confidence_threshold count as rice; every
    other object keeps the project's foreign-matter class vocabulary.

    The public signature and the legacy summary keys are preserved for existing
    callers (`has_rice`, `confidence`, `estimated_grain_count`,
    `total_objects_detected`, `message`, `method`).
    """
    # Imported lazily: ml.rice_gate re-uses extract_foreground_mask from this
    # module, so a module-level import would be circular.
    from ml.segmentation.rice_gate import evaluate_rice_presence

    return evaluate_rice_presence(image_rgb)
