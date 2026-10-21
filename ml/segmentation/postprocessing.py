"""
Post-processing for instance segmentation outputs.

Handles confidence filtering, mask-level NMS, global ID assignment,
geometry verification, touching detection, and tiling pipeline for
large or dense images.

Also exposes the shared `render_phase1_overlay()` used by both the
Phase 1 demo endpoint and the full dashboard pipeline so that grain
masks are always coloured consistently with HIGH/MEDIUM/LOW confidence
labels.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ml.config import (
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD,
    get_project_root,
    get_threshold,
)
from ml.segmentation.mask_ops import (
    dilated_overlap,
    extents_intersect,
    mask_iou,
)

PROJECT_ROOT = get_project_root()

logger = logging.getLogger(__name__)


# ============================================================================
# Confidence palettes — shared between routes.py (Phase 1 demo endpoint),
# RiceQualityPipeline (full dashboard annotated image), and frontend legends.
# ============================================================================

# Color scheme per confidence label (BGR convention? No -> always RGB here because
# all internal buffers in the project are RGB numpy arrays; OpenCV BGR conversion
# is done ONLY at the network I/O boundary in individual callers).
CONFIDENCE_COLORS_RGB: Dict[str, Tuple[int, int, int]] = {
    "HIGH":   (34, 197, 94),     # Bright green  -> clearly trustworthy
    "MEDIUM": (234, 179, 8),     # Amber/yellow -> usable, analyst should verify
    "LOW":    (249, 115, 22),    # Orange       -> poor, borderline invalid
    "UNKNOWN": (148, 163, 184),  # Slate        -> no confidence_label available
}

# Cycle palette used when confidence labels are not available but a stable
# per-grain distinguishable color is still desired (e.g. classical CV fallback
# where no model confidence exists, or comparison renders).
GRAIN_CYCLE_PALETTE_RGB: List[Tuple[int, int, int]] = [
    (100, 220, 120),
    (120, 180, 255),
    (255, 200, 100),
    (200, 130, 255),
    (100, 220, 220),
    (255, 160, 180),
    (220, 220, 100),
    (170, 220, 255),
]

# Foreign matter box + label color.
FOREIGN_MATTER_COLOR_RGB: Tuple[int, int, int] = (255, 60, 60)


def confidence_to_color(
    confidence: Optional[float],
    label: Optional[str] = None,
) -> Tuple[int, int, int]:
    """
    Map a (confidence float, optional label) pair to an RGB color triple.

    Priority: if ``label`` is provided (HIGH/MEDIUM/LOW/UNKNOWN) use that
    directly.  Otherwise derive the label from ``confidence`` using the
    shared thresholds exported from ``ml.inference``.
    """
    if label and label in CONFIDENCE_COLORS_RGB:
        return CONFIDENCE_COLORS_RGB[label]
    if confidence is None:
        return CONFIDENCE_COLORS_RGB["UNKNOWN"]
    if confidence >= CONFIDENCE_HIGH_THRESHOLD:
        return CONFIDENCE_COLORS_RGB["HIGH"]
    if confidence >= CONFIDENCE_MEDIUM_THRESHOLD:
        return CONFIDENCE_COLORS_RGB["MEDIUM"]
    return CONFIDENCE_COLORS_RGB["LOW"]


def render_phase1_overlay(
    img_rgb: np.ndarray,
    grains: List[Dict[str, Any]],
    foreign_matter: Optional[List[Dict[str, Any]]] = None,
    *,
    include_legend: bool = True,
    legend_method_label: Optional[str] = None,
    alpha: float = 0.38,
) -> np.ndarray:
    """
    Build a confidence-coloured segmentation overlay on top of ``img_rgb``.

    The exact same function is used by **both**:

    * ``POST /api/phase1/analyze`` — demo endpoint rendering, and
    * ``RiceQualityPipeline.analyze()`` — full dashboard annotated image.

    This guarantees the analyst sees identical mask colours whether they
    are inspecting the demo tab or the 14-parameter full-analysis dashboard.

    Args:
        img_rgb: HxWx3 uint8 RGB source image.
        grains: List of grain dicts in the ``Phase1AnalysisResponse`` format.
            Each grain must contain at least ``id``, ``bbox`` (x,y,w,h), and
            either ``mask_polygon`` (array of [x,y] points) OR a valid bbox
            which will be used as a fallback fill region.  Optional keys:
            ``confidence``, ``confidence_label``, ``centroid``.
        foreign_matter: Optional list of dicts, each with ``id``, ``bbox``,
            and ``class``.  Rendered as red boxes with FM#id labels.
        include_legend: When True, draw a top-right corner legend showing
            the HIGH/MEDIUM/LOW palette plus the method badge.
        legend_method_label: Short string, e.g. "YOLOv8l-seg / Run 1".  Only
            shown if ``include_legend`` is True.
        alpha: Blending weight for the semi-transparent mask fill.

    Returns:
        Modified HxWx3 uint8 RGB image.
    """
    overlay = img_rgb.copy()
    h, w = overlay.shape[:2]

    if not isinstance(grains, list):
        grains = []
    if not isinstance(foreign_matter, list):
        foreign_matter = [] or []

    # 1. Rice grain masks — colored by confidence_label (HIGH=green /
    #    MEDIUM=amber / LOW=orange / fallback=cycle palette).
    for idx, g in enumerate(grains):
        bbox = g.get("bbox", [0, 0, 0, 0])
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        bx, by, bw, bh = [int(v) for v in bbox]
        gid = int(g.get("id", idx + 1))

        # Decide per-grain color: prioritize whole vs broken status if available
        b_label = g.get("broken_label")
        if b_label == "broken":
            color = (239, 68, 68)   # Vivid red for broken grains
        elif b_label == "whole":
            color = (34, 197, 94)   # Vivid green for whole grains
        else:
            color = confidence_to_color(
                confidence=g.get("confidence"),
                label=g.get("confidence_label"),
            )
            # If color ended up UNKNOWN because neither label nor confidence was
            # provided, fall back to the per-grain cycle palette so adjacent
            # grains remain visually separable (useful for classical CV outputs).
            if color == CONFIDENCE_COLORS_RGB["UNKNOWN"] and "confidence" not in g:
                color = GRAIN_CYCLE_PALETTE_RGB[idx % len(GRAIN_CYCLE_PALETTE_RGB)]

        polygon = g.get("mask_polygon") or []
        mask_filled = None
        if isinstance(polygon, list) and len(polygon) >= 3:
            try:
                pts = np.array(
                    [[int(float(px)), int(float(py))] for [px, py] in polygon],
                    dtype=np.int32,
                ).reshape(-1, 1, 2)
                mask_canvas = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask_canvas, [pts], 255)
                mask_filled = mask_canvas
            except Exception:
                mask_filled = None

        if mask_filled is None and bw > 0 and bh > 0:
            # Fallback: fill bbox as mask
            mask_filled = np.zeros((h, w), dtype=np.uint8)
            x1 = max(0, bx)
            y1 = max(0, by)
            x2 = min(w, bx + bw)
            y2 = min(h, by + bh)
            mask_filled[y1:y2, x1:x2] = 255

        if mask_filled is not None:
            mask_y, mask_x = np.where(mask_filled > 0)
            if mask_x.size:
                bx = int(mask_x.min())
                by = int(mask_y.min())
                bw = int(mask_x.max() - bx + 1)
                bh = int(mask_y.max() - by + 1)

            sel = mask_filled > 0
            overlay[sel] = (
                (1.0 - alpha) * overlay[sel].astype(np.float32)
                + alpha * np.array(color, dtype=np.float32)
            ).astype(np.uint8)

            # Contour outline (darker/stronger version of fill color)
            try:
                cnts, _ = cv2.findContours(
                    mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if cnts:
                    outline_color = tuple(
                        min(255, int(c * 0.75)) for c in color
                    )
                    cv2.drawContours(overlay, cnts, -1, outline_color, 2)
            except Exception:
                pass

            if bw > 0 and bh > 0:
                x1 = max(0, min(w - 1, bx))
                y1 = max(0, min(h - 1, by))
                x2 = max(0, min(w - 1, bx + bw - 1))
                y2 = max(0, min(h - 1, by + bh - 1))
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)

        # Grain ID label — top-left of the bbox, on a dark pill so it stays
        # readable even on dark rice samples.
        label = f"#{gid}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        lx = max(0, bx)
        ly = max(th + baseline + 2, by - 4)
        if ly + th + 2 > h:
            ly = min(h - 2, by + bh // 2)
        cv2.rectangle(
            overlay,
            (lx, ly - th - baseline),
            (lx + tw + 4, ly + 2),
            (30, 30, 30),
            -1,
        )
        cv2.putText(
            overlay,
            label,
            (lx + 2, ly),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    # 2. Foreign matter — red boxes + label.
    fm_color = FOREIGN_MATTER_COLOR_RGB
    for f in foreign_matter:
        bbox = f.get("bbox", [0, 0, 0, 0])
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        bx, by, bw, bh = [int(v) for v in bbox]
        if bw <= 0 or bh <= 0:
            continue
        x1 = max(0, bx)
        y1 = max(0, by)
        x2 = min(w, bx + bw)
        y2 = min(h, by + bh)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), fm_color, 2)

        fid = int(f.get("id", 0))
        cls = f.get("class", "FM") or "FM"
        label = f"FM#{fid} {cls}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        ly = max(th + baseline + 2, y1 - 4)
        lx = max(0, x1)
        if ly + th + 2 > h:
            ly = y2 - 2
            if ly - th - baseline < 0:
                ly = y2
        cv2.rectangle(
            overlay,
            (lx, ly - th - baseline),
            (lx + tw + 4, ly + 2),
            fm_color,
            -1,
        )
        cv2.putText(
            overlay,
            label,
            (lx + 2, ly),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    # 3. Legend — top-right corner.
    if include_legend:
        legend_entries = [
            ("HIGH",   CONFIDENCE_COLORS_RGB["HIGH"],   f"≥ {CONFIDENCE_HIGH_THRESHOLD:.2f}"),
            ("MEDIUM", CONFIDENCE_COLORS_RGB["MEDIUM"], f"≥ {CONFIDENCE_MEDIUM_THRESHOLD:.2f}"),
            ("LOW",    CONFIDENCE_COLORS_RGB["LOW"],    "lower"),
        ]
        entry_h = 22
        pad = 8
        row_count = len(legend_entries) + (1 if legend_method_label else 0)
        box_w = 170
        box_h = pad * 2 + row_count * entry_h
        bx0 = w - box_w - pad
        by0 = pad
        # Ensure legend fits inside the image (safety for tiny thumbnails).
        if bx0 < 0 or by0 + box_h > h:
            bx0 = max(pad, w - box_w - pad)
            by0 = max(pad, h - box_h - pad)
        cv2.rectangle(
            overlay,
            (bx0, by0),
            (bx0 + box_w, by0 + box_h),
            (0, 0, 0),
            -1,
        )
        # Translucent dark background already applied via rect fill.
        row = 0
        if legend_method_label:
            cv2.putText(
                overlay,
                legend_method_label,
                (bx0 + pad, by0 + pad + 14 + row * entry_h),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (210, 220, 255),
                1,
                cv2.LINE_AA,
            )
            row += 1
        for label, color, threshold_text in legend_entries:
            cy = by0 + pad + 14 + row * entry_h
            cx_swatch = bx0 + pad
            sw = 14
            cv2.rectangle(
                overlay,
                (cx_swatch, cy - 11),
                (cx_swatch + sw, cy - 1),
                color,
                -1,
            )
            cv2.rectangle(
                overlay,
                (cx_swatch, cy - 11),
                (cx_swatch + sw, cy - 1),
                (80, 80, 80),
                1,
            )
            cv2.putText(
                overlay,
                f"{label}  {threshold_text}",
                (cx_swatch + sw + 6, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            row += 1

    return overlay


class PostProcessor:
    """
    Post-processes raw instance segmentation detections.

    Provides confidence filtering, mask IoU NMS, global ID assignment,
    geometry verification, confidence labelling, touching detection,
    and a tiling pipeline for large images.
    """

    def __init__(
        self,
        conf_threshold: float = 0.25,
        mask_iou_threshold: float = 0.5,
        min_grain_area: int = 50,
        tile_size: int = 640,
        tile_overlap: int = 128,
    ):
        self.conf_threshold = conf_threshold
        self.mask_iou_threshold = mask_iou_threshold
        self.min_grain_area = min_grain_area
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

    def filter_by_confidence(
        self, detections: List[Dict], threshold: float
    ) -> List[Dict]:
        """
        Remove detections whose confidence is below the threshold.

        Each detection is expected to have keys: bbox, mask, confidence,
        category_id (plus optional extras).
        """
        return [d for d in detections if float(d.get("confidence", 0.0)) >= threshold]

    @staticmethod
    def _compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        """Compute IoU between two binary masks (same shape or broadcastable)."""
        a = mask_a.astype(bool)
        b = mask_b.astype(bool)
        intersection = float(np.logical_and(a, b).sum())
        union = float(np.logical_or(a, b).sum())
        if union == 0:
            return 0.0
        return intersection / union

    def mask_iou_nms(
        self, detections: List[Dict], iou_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Secondary NMS at mask level.

        For pairs where box IoU passed the first stage but mask IoU > threshold,
        keep the detection with higher confidence and discard the other.

        Masks are stored as bounding-box crops with an image-space offset, so
        only pairs whose crops can possibly overlap are compared.  This keeps
        the stage usable on images with hundreds of objects.
        """
        if len(detections) <= 1:
            return list(detections)

        sorted_indices = sorted(
            range(len(detections)),
            key=lambda i: float(detections[i].get("confidence", 0.0)),
            reverse=True,
        )

        keep: List[int] = []
        suppressed: set = set()

        for idx_i, i in enumerate(sorted_indices):
            if i in suppressed:
                continue
            keep.append(i)
            det_i = detections[i]
            mask_i = det_i.get("mask")
            if mask_i is None:
                continue
            off_i = det_i.get("mask_offset") or (0, 0)
            for j in sorted_indices[idx_i + 1 :]:
                if j in suppressed:
                    continue
                det_j = detections[j]
                mask_j = det_j.get("mask")
                if mask_j is None:
                    continue
                off_j = det_j.get("mask_offset") or (0, 0)
                if not extents_intersect(off_i, mask_i.shape, off_j, mask_j.shape):
                    continue
                iou = mask_iou(mask_i, off_i, mask_j, off_j)
                if iou > mask_iou_threshold:
                    suppressed.add(j)

        return [detections[k] for k in keep]

    def assign_global_ids(self, detections: List[Dict]) -> List[Dict]:
        """
        Add a sequential 'id' field starting at 1.

        Ordering: by confidence descending, then by y (bbox top), then by x
        (bbox left), so IDs are stable and visually intuitive.
        """
        def sort_key(d: Dict):
            bbox = d.get("bbox", [0, 0, 0, 0])
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                x, y = float(bbox[0]), float(bbox[1])
            else:
                x, y = 0.0, 0.0
            return (-float(d.get("confidence", 0.0)), y, x)

        ordered = sorted(detections, key=sort_key)
        for seq, det in enumerate(ordered, start=1):
            det["id"] = seq
        return ordered

    def verify_geometry(
        self,
        detections: List[Dict],
        min_area: int,
        img_hw: Tuple[int, int],
    ) -> List[Dict]:
        """
        Remove detections whose mask area is below min_area or whose bbox
        lies fully outside the image.
        """
        img_h, img_w = img_hw
        kept: List[Dict] = []
        for det in detections:
            mask = det.get("mask")
            if mask is None:
                continue
            area = int(np.sum(mask > 0))
            if area < min_area:
                continue
            bbox = det.get("bbox", [0, 0, 0, 0])
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x, y, w, h = bbox
                if x + w <= 0 or y + h <= 0 or x >= img_w or y >= img_h:
                    continue
            kept.append(det)
        return kept

    @staticmethod
    def compute_confidence_label(confidence: float) -> str:
        """Return 'HIGH', 'MEDIUM', or 'LOW' label for a confidence score."""
        if confidence >= 0.80:
            return "HIGH"
        if confidence >= 0.50:
            return "MEDIUM"
        return "LOW"

    def detect_touching(self, detections: List[Dict]) -> List[Dict]:
        """
        For each detection, set is_touching=True when its mask dilated with a
        3x3 kernel (1 iteration) intersects any *other* detection's mask.

        Masks are bounding-box crops with image-space offsets; only pairs whose
        crops can overlap are compared.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        n = len(detections)
        for det in detections:
            det["is_touching"] = False

        masks = [
            (d.get("mask") if isinstance(d.get("mask"), np.ndarray) else None)
            for d in detections
        ]
        offsets = [
            (d.get("mask_offset") or (0, 0)) if isinstance(d.get("mask"), np.ndarray) else (0, 0)
            for d in detections
        ]

        for i in range(n):
            if masks[i] is None:
                continue
            for j in range(i + 1, n):
                if masks[j] is None:
                    continue
                # The dilated mask reaches one pixel beyond its crop.
                if not extents_intersect(
                    offsets[i], masks[i].shape, offsets[j], masks[j].shape, pad=1
                ):
                    continue
                overlap = dilated_overlap(
                    masks[i], offsets[i], masks[j], offsets[j], kernel=kernel
                )
                if overlap > 0:
                    detections[i]["is_touching"] = True
                    detections[j]["is_touching"] = True
        return detections

    # ------------------------------------------------------------------
    # Tiling pipeline
    # ------------------------------------------------------------------

    def split_into_tiles(
        self,
        image: np.ndarray,
        tile_size: int,
        overlap: int,
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Split an image into overlapping tiles.

        Returns list of (tile_image, (global_x1, global_y1, tile_w, tile_h)).
        """
        h, w = image.shape[:2]
        tiles: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []
        step = tile_size - overlap
        if step <= 0:
            step = tile_size

        y = 0
        while y < h:
            x = 0
            while x < w:
                x2 = min(x + tile_size, w)
                y2 = min(y + tile_size, h)
                tile = image[y:y2, x:x2].copy()
                tiles.append((tile, (x, y, x2 - x, y2 - y)))
                if x2 >= w:
                    break
                x += step
            if y2 >= h:
                break
            y += step
        return tiles

    @staticmethod
    def map_tile_to_global(
        detection: Dict, tile_offset: Tuple[int, int]
    ) -> Dict:
        """
        Offset bbox and mask coordinates from tile-local to global image space.

        tile_offset = (global_x1, global_y1) for the top-left of the tile.
        """
        out = dict(detection)
        ox, oy = tile_offset

        bbox = out.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            bx, by, bw, bh = bbox
            out["bbox"] = [int(bx + ox), int(by + oy), int(bw), int(bh)]

        mask = out.get("mask")
        if isinstance(mask, np.ndarray):
            mh, mw = mask.shape[:2]
            if ox > 0 or oy > 0:
                # Pad mask in-place for positive offsets; for negative offsets
                # the caller should not pass negative offsets (tiles always
                # start at non-negative global coordinates).
                pad_y = int(oy)
                pad_x = int(ox)
                new_mask = np.zeros(
                    (mh + pad_y, mw + pad_x), dtype=mask.dtype
                )
                new_mask[pad_y : pad_y + mh, pad_x : pad_x + mw] = mask
                out["mask"] = new_mask

        return out

    def merge_tile_detections(
        self,
        tile_results: List[Tuple[List[Dict], Tuple]],
        mask_iou_threshold: float = 0.5,
    ) -> List[Dict]:
        """
        Merge detections from multiple tiles into a single global list.

        tile_results: list of (detections_list, (gx1, gy1, tw, th))
        For any pair coming from different tiles whose mask IoU > threshold,
        keep the one with higher confidence.
        """
        all_dets: List[Dict] = []
        for dets, (gx1, gy1, _tw, _th) in tile_results:
            for d in dets:
                global_d = self.map_tile_to_global(d, (gx1, gy1))
                all_dets.append(global_d)

        if len(all_dets) <= 1:
            return all_dets

        sorted_indices = sorted(
            range(len(all_dets)),
            key=lambda i: float(all_dets[i].get("confidence", 0.0)),
            reverse=True,
        )

        keep: List[int] = []
        suppressed: set = set()

        for idx_i, i in enumerate(sorted_indices):
            if i in suppressed:
                continue
            keep.append(i)
            mask_i = all_dets[i].get("mask")
            if mask_i is None:
                continue
            for j in sorted_indices[idx_i + 1 :]:
                if j in suppressed:
                    continue
                mask_j = all_dets[j].get("mask")
                if mask_j is None:
                    continue
                if mask_i.shape != mask_j.shape:
                    h = max(mask_i.shape[0], mask_j.shape[0])
                    w = max(mask_i.shape[1], mask_j.shape[1])
                    mi = np.zeros((h, w), dtype=bool)
                    mj = np.zeros((h, w), dtype=bool)
                    mi[: mask_i.shape[0], : mask_i.shape[1]] = mask_i.astype(bool)
                    mj[: mask_j.shape[0], : mask_j.shape[1]] = mask_j.astype(bool)
                    iou = self._compute_mask_iou(mi, mj)
                else:
                    iou = self._compute_mask_iou(mask_i, mask_j)
                if iou > mask_iou_threshold:
                    suppressed.add(j)

        return [all_dets[k] for k in keep]

    # ------------------------------------------------------------------
    # Full pipeline orchestration
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        raw_detections: List[Dict],
        image_shape: Tuple[int, int],
        use_tiling: bool = False,
    ) -> List[Dict]:
        """
        Orchestrate the complete post-processing stack.

        Steps:
          1. confidence filter (threshold stored on this instance)
          2. (tile merge only if use_tiling is True and detections carry tile
             grouping info — here use_tiling skips tile merge when false)
          3. mask IoU NMS
          4. verify geometry (area + image bounds)
          5. assign global IDs
          6. detect touching
          7. attach confidence_label

        Returns the final detection list with enriched metadata.
        """
        detections = self.filter_by_confidence(raw_detections, self.conf_threshold)

        if use_tiling:
            grouped: List[Tuple[List[Dict], Tuple]] = []
            current_offset = (0, 0, 0, 0)
            bucket: List[Dict] = []
            for d in detections:
                offset = d.get("_tile_offset")
                if offset is None:
                    bucket.append(d)
                else:
                    grouped.append(([d], tuple(offset)))
            if bucket:
                grouped.append((bucket, current_offset))
            if grouped:
                detections = self.merge_tile_detections(
                    grouped, mask_iou_threshold=self.mask_iou_threshold
                )

        detections = self.mask_iou_nms(detections, iou_threshold=self.mask_iou_threshold)
        detections = self.verify_geometry(detections, self.min_grain_area, image_shape)
        detections = self.assign_global_ids(detections)
        detections = self.detect_touching(detections)

        for d in detections:
            conf = float(d.get("confidence", 0.0))
            d["confidence_label"] = self.compute_confidence_label(conf)

        return detections
