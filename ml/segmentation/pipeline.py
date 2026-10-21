"""
Rice Quality Analysis Pipeline.

Orchestrates the complete image-based quality analysis:
1. Validates and loads image
2. Detects rice presence (stops if no rice)
3. Segments individual grains — **Phase 1 cascade priority**:
   * YOLOv8l-seg trained weights (primary — green/yellow/orange HIGH/MEDIUM/LOW polygons)
   * Mask R-CNN baseline (comparison fallback)
   * Classical CV watershed (emergency fallback — never deleted per docs/MODEL.md cascade)
4. Detects calibration reference (ArUco or manual, else uncalibrated)
5. Computes per-grain geometry (Length, Breadth, L/B ratio, Area, Solidity, etc.)
6. Classifies 8 defect types per grain (multi-label)
7. Detects full-image foreign matter (YOLO or heuristic)
8. Computes a geometry outlier diagnostic; lower-class admixture is unsupported
9. Computes sample-level summary statistics
10. Evaluates image quality indicators & assigns tier
11. Compares observed image fractions with historical/reference rice limits
12. Generates annotated image with **Phase 1 confidence-coloured polygons**,
    Grain IDs, foreign-matter red FM#N boxes, and top-right legend badge.
"""

import base64
import io
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from ml.quality.admixture import detect_admixture
from ml.quality.calibration import detect_calibration
from ml.quality.classifiers import (
    batch_classify_damaged,
    batch_classify_sprouted_weevilled,
    classify_damaged,
    classify_immature_shrunken,
    classify_sprouted_weevilled,
)
from ml.quality.colour import (
    analyze_dehusked,
    analyze_discoloured,
    analyze_red,
    compute_reference_lab,
    extract_grain_lab_pixels,
)
from ml.config import (
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD,
    get_threshold,
    load_standards,
)
from ml.quality.foreign_matter import detect_foreign_matter, merge_gate_foreign_objects
from ml.quality.geometry import (
    classify_broken,
    compute_grain_geometry,
    compute_robust_whole_kernel_length,
    resolve_whole_kernel_reference,
)
from ml.quality.profiles import GrainProfile, get_default_grain_profile, load_grain_profile
from ml.segmentation.inference import analyze_image as phase1_analyze_image
from ml.segmentation.mask_ops import (
    draw_mask,
    tight_bounds,
    translate_contour,
)
from ml.segmentation.postprocessing import (
    PostProcessor,
    confidence_to_color,
    render_phase1_overlay,
)
from ml.segmentation.preprocessing import (
    ImageValidationError,
    load_image,
    resize_for_inference,
)
from ml.quality.quality import assess_image_quality
from ml.segmentation.rice_gate import (
    MESSAGE_NO_ANALYSABLE_RICE,
    STATUS_NO_ANALYSABLE_RICE,
    STATUS_NO_RICE_CLASS,
    STATUS_NOT_RICE,
)
from ml.segmentation.segmentation import (
    GrainInstance,
    detect_rice_presence,
    segment_grains,
)
from ml.standards.standards import compare_with_standards
from ml.quality.texture import (
    extract_chalky_features,
    heuristic_chalky_classification,
)
from backend.app.services.run_logger import structured_logger

logger = logging.getLogger(__name__)



class RiceQualityPipeline:
    """
    End-to-end analyzer for rice grain quality.
    """

    def __init__(self, small_sample_threshold: int = 30):
        self.small_sample_threshold = small_sample_threshold

    @staticmethod
    def _filter_grains_to_rice(
        grains: List[Any], rice_gate: Optional[Dict[str, Any]]
    ) -> List[Any]:
        """Keep only segmentation results whose boxes match actual rice detections."""
        if not rice_gate or not rice_gate.get("detections"):
            return grains

        rice_boxes = [
            tuple(d.get("bbox") or [])
            for d in rice_gate.get("detections", [])
            if d.get("is_rice") and len(d.get("bbox") or []) == 4
        ]
        if not rice_boxes:
            return []

        def _overlaps(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
            ax, ay, aw, ah = a
            bx, by, bw, bh = b
            x1 = max(ax, bx)
            y1 = max(ay, by)
            x2 = min(ax + aw, bx + bw)
            y2 = min(ay + ah, by + bh)
            return x2 > x1 and y2 > y1

        filtered = []
        for grain in grains:
            grain_box = tuple(grain.bbox)
            if any(_overlaps(grain_box, box) for box in rice_boxes):
                filtered.append(grain)
        return filtered

    # ------------------------------------------------------------------
    # Phase 1 → dashboard adapter.
    #
    # ml.inference.analyze_image() returns Phase1AnalysisResponse-style
    # dicts with polygon masks and HIGH/MEDIUM/LOW labels.  The rest of
    # this pipeline (geometry computation, 8 defect classifiers, summary
    # stats) eats GrainInstance dataclasses with uint8 .mask arrays plus
    # .contour, .bbox, .centroid.  This helper bridges the worlds so
    # downstream code runs on the new masking unchanged.
    # ------------------------------------------------------------------

    @staticmethod
    def _quality_from_label(label: Optional[str]) -> str:
        return {
            "HIGH": "excellent",
            "MEDIUM": "good",
            "LOW": "poor",
        }.get(label or "", "segmented")

    @staticmethod
    def _grain_image_crop(
        image_rgb: np.ndarray, grain: GrainInstance
    ) -> np.ndarray:
        """
        Return the image region covered by a grain.

        Grain masks are stored as a tight crop plus an image-space offset, so
        the matching image region is a cheap view instead of a full-frame
        copy.  Per-grain classifiers crop by the mask's own bounding box, so
        handing them this view yields exactly the same pixels as before.
        """
        ox, oy = grain.mask_offset
        mh, mw = grain.mask.shape[:2]
        return image_rgb[oy : oy + mh, ox : ox + mw]

    @staticmethod
    def _rasterize_polygon(
        h: int,
        w: int,
        polygon: Any,
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[Optional[np.ndarray], Tuple[int, int], Optional[np.ndarray]]:
        """
        Return (uint8_mask_crop, offset_xy, contour_Nx1x2_int32) from either a
        polygon list of [x,y] points OR a fallback bbox rectangle.

        The mask is rasterised into a crop-sized canvas instead of a
        full-frame one: a full-frame canvas per grain costs image_pixels bytes
        and a 12 MP upload with 1000 grains would need 12.6 GB.  ``offset_xy``
        is the position of the crop in image coordinates; the contour is
        returned in image coordinates.
        """
        bx, by, bw, bh = bbox
        mask_filled = None
        offset = (0, 0)
        if isinstance(polygon, (list, tuple)) and len(polygon) >= 3:
            try:
                pts = np.array(
                    [[int(float(px)), int(float(py))] for [px, py] in polygon],
                    dtype=np.int32,
                ).reshape(-1, 1, 2)
                # Crop the canvas to the polygon extent (clipped to the image)
                # and shift the polygon to the crop origin.
                x0 = max(0, int(pts[..., 0].min()))
                y0 = max(0, int(pts[..., 1].min()))
                x1 = min(w, int(pts[..., 0].max()) + 1)
                y1 = min(h, int(pts[..., 1].max()) + 1)
                if x1 > x0 and y1 > y0:
                    canvas = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
                    shifted = pts - np.array([x0, y0], dtype=np.int32)
                    cv2.fillPoly(canvas, [shifted], 255)
                    mask_filled = canvas
                    offset = (x0, y0)
            except Exception:
                mask_filled = None

        if mask_filled is None and bw > 0 and bh > 0:
            x1 = max(0, bx)
            y1 = max(0, by)
            x2 = min(w, bx + bw)
            y2 = min(h, by + bh)
            if x2 > x1 and y2 > y1:
                mask_filled = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                mask_filled[:, :] = 255
                offset = (x1, y1)

        contour = None
        if mask_filled is not None:
            try:
                cnts, _ = cv2.findContours(
                    mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if cnts:
                    # Largest contour only (polygon fill should give exactly one).
                    contour = max(cnts, key=cv2.contourArea)
                    contour = translate_contour(contour, offset)
            except Exception:
                contour = None

        return mask_filled, offset, contour

    def _phase1_to_dashboard_grains(
        self,
        image_rgb: np.ndarray,
        phase1_result: Dict[str, Any],
    ) -> Tuple[List[GrainInstance], List[Dict[str, Any]]]:
        """
        Convert the structured dict from ``ml.inference.analyze_image`` into
        (1) a list of ``GrainInstance`` dataclasses for the downstream
        geometry + defect classifier pipeline, and (2) a list of FM dicts
        tagged with ``provenance='phase1'`` so the FM merge step later can
        de-duplicate them against the gate's own FM detections.

        The returned grain list is sorted by ``grain.id`` to keep the
        dashboard viewer drop-down order identical to the rendered
        annotated image IDs.
        """
        h, w = image_rgb.shape[:2]
        method_used = phase1_result.get("method") or "classical_cv_fallback"

        raw_grains: List[Dict[str, Any]] = list(
            phase1_result.get("grains") or []
        )
        # Sort by ID for rendering consistency.
        def _gid(g: Dict[str, Any]) -> int:
            try:
                return int(g.get("id", 0))
            except Exception:
                return 0

        raw_grains.sort(key=_gid)

        dashboard_grains: List[GrainInstance] = []
        for g in raw_grains:
            bbox = g.get("bbox", [0, 0, 0, 0])
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            bx, by, bw, bh = [int(v) for v in bbox]
            gid = int(g.get("id", len(dashboard_grains) + 1))
            confidence = float(g.get("confidence") or 0.0)
            label = g.get("confidence_label")
            if label is None:
                label = PostProcessor.compute_confidence_label(confidence)
            centroid = g.get("centroid")
            if isinstance(centroid, (list, tuple)) and len(centroid) == 2:
                cx, cy = float(centroid[0]), float(centroid[1])
            else:
                cx = float(bx + bw / 2.0)
                cy = float(by + bh / 2.0)

            mask, mask_offset, contour = self._rasterize_polygon(
                h=h,
                w=w,
                polygon=g.get("mask_polygon"),
                bbox=(bx, by, bw, bh),
            )
            if mask is None:
                continue
            # Keep the mask as a tight crop; the offset locates it in the image.
            bounds = tight_bounds(mask)
            if bounds is None:
                continue
            mx0, my0, mw, mh = bounds
            mask = mask[my0 : my0 + mh, mx0 : mx0 + mw]
            mask_offset = (mask_offset[0] + mx0, mask_offset[1] + my0)
            area = int(np.sum(mask > 0))
            bx = mask_offset[0]
            by = mask_offset[1]
            bw = mw
            bh = mh
            if area <= 0 and bw * bh > 0:
                area = bw * bh

            instance = GrainInstance(
                grain_id=gid,
                mask=mask,
                bbox=(bx, by, bw, bh),
                confidence=confidence,
                centroid=(cx, cy),
                mask_offset=mask_offset,
                contour=contour,
                is_touching=bool(g.get("is_touching", False)),
                is_overlapping=False,
                segmentation_quality=self._quality_from_label(label),
                method=method_used,
                is_foreign_matter=False,
                confidence_label=label,
                segmentation_method=method_used,
                mask_polygon=g.get("mask_polygon"),
            )
            dashboard_grains.append(instance)

        # --- Foreign matter list with provenance tag -------------------
        fm_out: List[Dict[str, Any]] = []
        for f in list(phase1_result.get("foreign_matter") or []):
            bbox = f.get("bbox", [0, 0, 0, 0])
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            fm_out.append(
                {
                    "id": int(f.get("id", len(fm_out) + 1)),
                    "class_name": f.get("class") or "foreign_matter",
                    "bbox": tuple(int(v) for v in bbox),
                    "confidence": float(f.get("confidence") or 0.0),
                    "provenance": "phase1_segmentation",
                }
            )
        return dashboard_grains, fm_out


    def analyze(
        self,
        image_source: str | bytes,
        filename: str = "image.jpg",
        manual_scale: Optional[Dict] = None,
        grade: str = "grade_a",
        profile: Optional[Union[str, Dict[str, Any], GrainProfile]] = None,
        job_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute full analysis pipeline on an image.

        Args:
            image_source: File path or raw image bytes
            filename: Original filename (for extension check / logging)
            manual_scale: Optional {'reference_pixels': N, 'reference_mm': M}
            grade: 'grade_a' or 'common'
            profile: Optional GrainProfile instance, name string, or dict profile
            job_id: Optional unique job identifier for structured logging correlation

        Returns:
            Dict matching the complete project result schema.
        """
        t0 = time.monotonic()
        start_time = time.time()
        warnings: List[str] = []

        file_size = None
        if isinstance(image_source, bytes):
            file_size = len(image_source)
        elif isinstance(image_source, str):
            try:
                file_size = Path(image_source).stat().st_size
            except OSError:
                pass

        structured_logger.info(
            "analysis_started",
            job_id=job_id,
            request_id=request_id,
            filename=filename,
            file_size_bytes=file_size,
            input_source="upload" if isinstance(image_source, bytes) else "other",
        )

        # 1. Load and validate image
        try:
            image_rgb, image_info = load_image(image_source, filename=filename)
        except ImageValidationError as e:
            structured_logger.error(
                "analysis_failed",
                job_id=job_id,
                request_id=request_id,
                status="failed",
                failed_stage="image_decode",
                error_type="ValidationError",
                error_message=str(e),
                duration_ms=int((time.monotonic() - t0) * 1000),
            )
            return {
                "success": False,
                "error": str(e),
                "error_type": "ValidationError",
                "warnings": [str(e)],
            }
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            structured_logger.error(
                "analysis_failed",
                job_id=job_id,
                request_id=request_id,
                status="failed",
                failed_stage="image_decode",
                error_type="DecodeError",
                error_message=f"Image decoding error: {str(e)}",
                duration_ms=int((time.monotonic() - t0) * 1000),
            )
            return {
                "success": False,
                "error": f"Image decoding error: {str(e)}",
                "error_type": "DecodeError",
                "warnings": [f"Image decoding error: {str(e)}"],
            }

        h, w = image_rgb.shape[:2]
        megapixels = (h * w) / 1_000_000

        structured_logger.info(
            "image_decoded",
            job_id=job_id,
            request_id=request_id,
            width=w,
            height=h,
            format=getattr(image_info, "format_", getattr(image_info, "format", None)),
            file_size_bytes=file_size,
            source=getattr(image_info, "source", None),
        )

        # 2. Rice-presence gate (Case 1): "objects detected" != "rice detected".
        #    Only detections whose class is the model configuration's rice class
        #    (models/segmentation/class_mapping.json -> class_id 1 "rice_grain")
        #    with confidence >= rice_gate.rice_confidence_threshold count as rice.
        #    Foreign-matter detections are never treated as rice.
        structured_logger.info("rice_gate_started", job_id=job_id)
        _rice_gate_start = time.monotonic()
        rice_gate = detect_rice_presence(image_rgb)
        gate_status = rice_gate.get("status")
        gate_ms = int((time.monotonic() - _rice_gate_start) * 1000)
        structured_logger.info(
            "rice_gate_completed",
            job_id=job_id,
            request_id=request_id,
            duration_ms=gate_ms,
            status=gate_status,
            has_rice=rice_gate.get("has_rice", False),
            rice_detections=rice_gate.get("rice_detections", 0),
            total_detections=rice_gate.get("total_detections", 0),
        )
        if gate_status in (
            STATUS_NOT_RICE,
            STATUS_NO_ANALYSABLE_RICE,
            STATUS_NO_RICE_CLASS,
        ) or not rice_gate.get("has_rice", False):
            logger.info(
                "Rice gate FAILED — rice analysis not executed. %s",
                rice_gate.get("debug", ""),
            )
            no_rice_response = self._build_no_rice_response(
                image_info=image_info,
                image_rgb=image_rgb,
                rice_gate=rice_gate,
                warnings=warnings,
                start_time=start_time,
                job_id=job_id,
            )
            structured_logger.info(
                "analysis_completed",
                job_id=job_id,
                request_id=request_id,
                status="success",
                rice_detected=False,
                total_processing_ms=int((time.monotonic() - t0) * 1000),
            )
            return no_rice_response

        logger.info("Rice gate PASSED. %s", rice_gate.get("debug", ""))
        warnings.extend(rice_gate.get("warnings", []))

        if rice_gate["confidence"] < 0.5:
            warnings.append(
                "Rice may be present, but confidence is low. Results may be unreliable."
            )

        # 3. Detect calibration
        calibration_res = detect_calibration(image_rgb, manual_scale=manual_scale)
        pixels_per_mm = calibration_res.pixels_per_mm if calibration_res.validity == "valid" else None

        # 4. Grain Instance Segmentation — Phase 1 masking cascade
        #
        #    Priority (identical to docs/MODEL.md and ml.inference.analyze_image):
        #      1. YOLOv8l-seg trained weights -> mask polygons + confidence labels
        #      2. Mask R-CNN baseline         -> fallback (slower)
        #      3. Classical CV watershed      -> emergency fallback, never deleted
        #
        #    After Phase 1 returns polygons, we rasterize them to uint8 masks
        #    (see _phase1_to_dashboard_grains) so that Steps 5–11 (geometry, 8
        #    defect classifiers, foreign matter, summary stats, standards)
        #    continue to run completely unchanged on identical-shaped inputs.
        phase1_meta: Dict[str, Any] = {}
        phase1_fm_extra: List[Dict[str, Any]] = []
        seg_detected = 0
        seg_uncertain = 0
        seg_rejected = 0

        structured_logger.info("segmentation_started", job_id=job_id)
        _seg_start = time.monotonic()
        try:
            phase1_result = phase1_analyze_image(image_rgb)
            if not isinstance(phase1_result, dict):
                raise RuntimeError(
                    f"phase1_analyze_image returned non-dict: {type(phase1_result)}"
                )
            phase1_method = phase1_result.get("method") or "unknown"
            phase1_version = phase1_result.get("model_version") or ""
            processing = phase1_result.get("processing") or {}
            phase1_meta = {
                "segmentation_method_used": phase1_method,
                "model_version": phase1_version,
                "processing_inference_ms": int(processing.get("inference_ms") or 0),
                "processing_total_ms": int(processing.get("total_ms") or 0),
                "tiling_used": bool(processing.get("tiling_used", False)),
                "num_tiles": int(processing.get("num_tiles") or 0),
                "source": "phase1_cascade",
            }
            grains, phase1_fm_extra = self._phase1_to_dashboard_grains(
                image_rgb=image_rgb,
                phase1_result=phase1_result,
            )
            seg_detected = len(grains)
            seg_uncertain = sum(
                1
                for g in grains
                if (g.confidence_label or "").upper() in {"LOW", "MEDIUM"}
            )
            seg_rejected = 0
            warnings.append(
                f"Segmentation cascade used: {phase1_method} "
                f"(v{phase1_version or '?'}). "
                f"Rasterized {seg_detected} polygons → uint8 masks "
                f"for downstream geometry + defect pipeline."
            )
        except Exception as exc:
            # Cascade safety: if anything above failed, fall back to the
            # classical CV watershed segment_grains() we used before.  This
            # emergency fallback is NEVER deleted per docs/MODEL.md cascade.
            logger.warning(
                "Phase 1 segmentation cascade failed. "
                "Falling back to classical CV watershed. Error: %s",
                exc,
            )
            warnings.append(
                "Phase 1 masking cascade unavailable — "
                "falling back to classical CV watershed. "
                f"Reason: {exc!r}"
            )
            phase1_meta = {"source": "classical_cv_fallback", "fallback_reason": repr(exc)}
            seg_result = segment_grains(image_rgb, method="auto")
            warnings.extend(seg_result.warnings)
            grains = list(seg_result.grains)
            seg_detected = seg_result.detected_count
            seg_uncertain = seg_result.uncertain_count
            seg_rejected = seg_result.rejected_count

        # IMPORTANT: only rice-gate detections are allowed to survive into the
        # per-grain rice analysis. Non-rice detections remain in the foreign-matter
        # channel and must never become GrainInstance objects.
        pre_filter_count = len(grains)
        grains = self._filter_grains_to_rice(grains, rice_gate)
        if len(grains) != pre_filter_count:
            warnings.append(
                "Filtered non-rice segmented objects before per-grain rice analysis. "
                "Only rice-gate detections remain in the rice-grain pipeline."
            )
        n_grains = len(grains)
        seg_method = phase1_meta.get("segmentation_method_used") or phase1_meta.get("source") or "unknown"
        seg_ms = int((time.monotonic() - _seg_start) * 1000)
        structured_logger.info(
            "segmentation_completed",
            job_id=job_id,
            request_id=request_id,
            duration_ms=seg_ms,
            method=seg_method,
            grain_count=n_grains,
            detected=seg_detected,
            uncertain=seg_uncertain,
            rejected=seg_rejected,
        )

        if n_grains == 0:
            # Rice-class detections existed, but no analysable grain instance was
            # produced: the rice-analysis pipeline must not run on zero grains.
            logger.info(
                "No analysable rice grain instances (%s rice detections) — stopping.",
                rice_gate.get("rice_detections", 0),
            )
            stopped_gate = dict(rice_gate)
            stopped_gate["status"] = STATUS_NO_ANALYSABLE_RICE
            stopped_gate["analysis_stopped"] = True
            stopped_gate["message"] = MESSAGE_NO_ANALYSABLE_RICE
            zero_grain_response = self._build_no_rice_response(
                image_info=image_info,
                image_rgb=image_rgb,
                rice_gate=stopped_gate,
                warnings=warnings + ["Zero grain instances accepted."],
                start_time=start_time,
                message=MESSAGE_NO_ANALYSABLE_RICE,
                sample_overrides={
                    "total_detected": seg_detected,
                    "uncertain": seg_uncertain,
                    "rejected": seg_rejected,
                },
                phase1_meta=phase1_meta,
                job_id=job_id,
            )
            structured_logger.info(
                "analysis_completed",
                job_id=job_id,
                request_id=request_id,
                status="success",
                rice_detected=False,
                grain_count=0,
                total_processing_ms=int((time.monotonic() - t0) * 1000),
            )
            return zero_grain_response

        # 5. Per-grain Geometry
        structured_logger.info("geometry_started", job_id=job_id)
        _geom_start = time.monotonic()
        geometries = []
        grain_areas = []
        grain_confidences = []
        lengths = []

        for grain in grains:
            geom = compute_grain_geometry(
                grain_id=grain.grain_id,
                mask=grain.mask,
                pixels_per_mm=pixels_per_mm,
                offset=grain.mask_offset,
            )
            geometries.append(geom)
            grain_areas.append(geom.area_pixels)
            grain_confidences.append(grain.confidence)
            effective_length = (
                geom.effective_length_mm
                if geom.effective_length_mm is not None
                else (geom.length_mm if geom.length_mm is not None else geom.effective_length_pixels)
            )
            lengths.append(effective_length)

        geom_ms = int((time.monotonic() - _geom_start) * 1000)
        structured_logger.info(
            "geometry_completed",
            job_id=job_id,
            request_id=request_id,
            duration_ms=geom_ms,
            grain_count=len(geometries),
        )
        # 6. Broken Grain Reference Calculation (3-tier hierarchy: profile -> sample_derived -> undetermined)
        #
        # NOTE: We do NOT automatically load get_default_grain_profile() here.
        # Pixel-unit profiles are zoom-level dependent: the same physical rice kernel
        # measures ~148 px in a 50-grain image but ~374 px in a 1-grain image.
        # Silently injecting a pixel profile from a different imaging setup causes
        # incorrect broken classifications on any image not shot at that exact distance.
        #
        # The caller must explicitly supply a profile (via the `profile` parameter)
        # if they have a calibrated reference for their imaging setup.
        # Without a profile, the system falls back to sample-derived reference (Tier 2)
        # or marks the classification as undetermined (Tier 3).
        measurement_unit = "mm" if calibration_res.validity == "valid" else "pixels"
        active_profile = profile  # None is valid — triggers Tier 2 / Tier 3

        calibration_info = calibration_res.to_dict() if calibration_res else {}

        (
            whole_kernel_len,
            whole_len_source,
            whole_len_status,
            whole_len_meta,
        ) = resolve_whole_kernel_reference(
            lengths=lengths,
            profile=active_profile,
            geometries=geometries,
            measurement_unit=measurement_unit,
            calibration=calibration_info,
        )

        broken_threshold_fraction = float(get_threshold("broken", "whole_kernel_fraction", 0.75))
        small_broken_fraction = float(get_threshold("broken", "small_broken_max_fraction", 0.25))

        if whole_kernel_len is None:
            warnings.append(
                f"Whole-kernel length reference could not be reliably estimated: {whole_len_meta.get('reason', 'insufficient reference evidence')}."
            )
        elif whole_len_source == "profile":
            warnings.append(
                f"Whole-kernel reference loaded from profile '{whole_len_meta.get('profile_name', 'configured')}': {whole_kernel_len:.1f} {measurement_unit}."
            )
        else:
            warnings.append(
                f"Whole-kernel reference derived from candidate intact grains in sample: {whole_kernel_len:.1f} {measurement_unit} ({whole_len_meta.get('candidate_count', 0)} candidate grains)."
            )

        # Build LAB reference for discolouration
        all_labs = []
        for grain in grains:
            g_lab = extract_grain_lab_pixels(
                self._grain_image_crop(image_rgb, grain), grain.mask
            )
            if g_lab is not None and len(g_lab) > 0:
                all_labs.append(g_lab)
            ref_lab = compute_reference_lab(all_labs)

        # Population stats for immature/shrunken
        pop_breadths = [
            g.breadth_pixels for g in geometries if g.breadth_pixels > 0
        ]
        population_stats = {
            "median_breadth": (
                float(np.median(pop_breadths)) if pop_breadths else 1.0
            ),
            "median_area": (
                float(np.median(grain_areas)) if grain_areas else 1.0
            ),
        }

        # 7. Multi-Label Defect Classification per grain
        structured_logger.info("broken_classification_started", job_id=job_id)
        _bc_start = time.monotonic()
        classified_grains = []
        broken_labels = []
        damaged_count = 0
        discoloured_count = 0
        chalky_count = 0
        chalky_analyzed_count = 0
        red_count = 0
        dehusked_count = 0
        immature_count = 0
        sprouted_count = 0
        broken_count = 0

        # Run batch ML classifiers once across all grains (avoids N separate model forward passes).
        # For dense large samples, use a bounded fast path so the pipeline remains responsive
        # while preserving truthful heuristic fallbacks and clear method labels.
        grain_masks_list = [g.mask for g in grains]
        grain_crops = [self._grain_image_crop(image_rgb, g) for g in grains]
        grain_offsets = [g.mask_offset for g in grains]
        bulk_mode = len(grain_masks_list) >= 30
        if bulk_mode:
            warnings.append(
                "Large-sample fast path active: detailed per-grain defect classification is approximated "
                "to preserve runtime while keeping outputs labelled as heuristic."
            )
            damaged_results = [
                {
                    "damaged_label": "normal",
                    "damaged_probability": 0.0,
                    "confidence": 0.0,
                    "method": "fast_bulk_fallback_large_sample",
                    "_source": "engineering_heuristic",
                    "limitation": "Large-sample performance mode uses a bounded approximate assessment.",
                }
                for _ in grain_masks_list
            ]
            sprouted_results = [
                {
                    "sprouted_weevilled_label": "normal",
                    "probability": 0.0,
                    "confidence": 0.0,
                    "method": "fast_bulk_fallback_large_sample",
                    "_source": "engineering_heuristic",
                    "limitation": "Large-sample performance mode uses a bounded approximate assessment.",
                }
                for _ in grain_masks_list
            ]
        else:
            damaged_results = batch_classify_damaged(
                image_rgb, grain_masks_list, grain_crops=grain_crops
            )
            sprouted_results = batch_classify_sprouted_weevilled(
                image_rgb, grain_masks_list, grain_crops=grain_crops
            )

        for i, (grain, geom) in enumerate(zip(grains, geometries)):
            grain_crop = grain_crops[i]
            # Broken
            g_len = (
                geom.effective_length_mm
                if geom.effective_length_mm is not None
                else (geom.length_mm if geom.length_mm is not None else geom.effective_length_pixels)
            )
            broken_res = classify_broken(
                length=g_len,
                whole_kernel_length=whole_kernel_len,
                threshold_fraction=broken_threshold_fraction,
                small_broken_fraction=small_broken_fraction,
                reference_source=whole_len_source,
                reference_status=whole_len_status,
                measurement_quality=geom.measurement_quality,
            )
            broken_labels.append(broken_res["broken_label"])
            if broken_res["broken_label"] == "broken":
                broken_count += 1

            # Chalky
            chalky_feats = extract_chalky_features(grain_crop, grain.mask)
            chalky_res = heuristic_chalky_classification(chalky_feats)
            if chalky_res["chalky_label"] == "chalky":
                chalky_count += 1
            if chalky_res["chalky_label"] in ("chalky", "not_chalky"):
                chalky_analyzed_count += 1

            # Damaged (from pre-computed batch)
            damaged_res = damaged_results[i]
            if damaged_res.get("damaged_label") in ("damaged", "slightly_damaged"):
                damaged_count += 1

            # Discoloured
            discoloured_res = analyze_discoloured(
                grain_crop, grain.mask, reference_lab=ref_lab
            )
            if discoloured_res.get("discoloured_label") == "discoloured":
                discoloured_count += 1

            # Red
            red_res = analyze_red(grain_crop, grain.mask)
            if red_res.get("red_label") == "red":
                red_count += 1

            # Dehusked
            dehusked_res = analyze_dehusked(grain_crop, grain.mask)
            if dehusked_res.get("dehusked_label") == "dehusked":
                dehusked_count += 1

            # Immature / Shrunken
            immature_res = classify_immature_shrunken(
                geom.to_dict(), population_stats=population_stats
            )
            if immature_res.get("immature_shrunken_status") == "immature_shrunken":
                immature_count += 1

            # Sprouted / Weevilled (from pre-computed batch)
            sprouted_res = sprouted_results[i]
            if sprouted_res.get("sprouted_weevilled_label") == "sprouted_weevilled":
                sprouted_count += 1

            # RLE or polygon contour for grain mask representation
            contour_pts = []
            if grain.contour is not None:
                # Subsample contour for clean JSON export
                step = max(1, len(grain.contour) // 30)
                contour_pts = [
                    [int(pt[0][0]), int(pt[0][1])]
                    for pt in grain.contour[::step]
                ]

            grain_dict = {
                "id": grain.grain_id,
                "bbox": list(grain.bbox),
                "centroid": [round(c, 2) for c in grain.centroid],
                "confidence": grain.confidence,
                "confidence_label": grain.confidence_label,
                "segmentation_quality": grain.segmentation_quality,
                "is_touching": grain.is_touching,
                "contour": contour_pts,
                "mask_polygon": (
                    [[float(x), float(y)] for [x, y] in grain.mask_polygon]
                    if isinstance(grain.mask_polygon, (list, tuple)) and len(grain.mask_polygon) >= 3
                    else None
                ),
                "geometry": geom.to_dict(),
                "defects": {
                    "broken": broken_res,
                    "damaged": damaged_res,
                    "discoloured": discoloured_res,
                    "chalky": chalky_res,
                    "red": red_res,
                    "dehusked": dehusked_res,
                    "immature_shrunken": immature_res,
                    "sprouted_weevilled": sprouted_res,
                },
                "sample_level_parameters": {
                    "foreign_matter": "Sample-level parameter",
                    "admixture": "Sample-level parameter",
                    "total_count": "Sample-level parameter",
                },
            }
            classified_grains.append(grain_dict)

        # 8. Foreign Matter Detection (Full Image)
        #    The gate's non-rice detections are merged in so foreign matter is always
        #    reported separately from rice (CASE B: rice + foreign matter).
        #    Additionally, if Phase 1 cascade produced its own FM detections, we
        #    include them as extra inputs (tagged provenance='phase1_segmentation')
        #    and the merger de-duplicates by box-IoU against the gate FM list.
        grain_masks_list = [g.mask for g in grains]
        base_fm = detect_foreign_matter(
            image_rgb,
            grain_masks=grain_masks_list,
            grain_mask_offsets=grain_offsets,
        )
        extra_fm_dicts: List[Dict[str, Any]] = []
        for f in phase1_fm_extra:
            # Convert adapter tuple/dict to a ForeignObject-shaped object the
            # merge helper can consume.  We produce dicts with the same shape
            # as detect_foreign_matter output (bbox + class_name + confidence).
            extra_fm_dicts.append(
                {
                    "class_name": f.get("class_name") or "foreign_matter",
                    "bbox": tuple(f["bbox"]) if isinstance(f.get("bbox"), (list, tuple)) else (0, 0, 0, 0),
                    "confidence": float(f.get("confidence") or 0.0),
                    "provenance": f.get("provenance") or "phase1_segmentation",
                }
            )
        # Append extras to the FM pipeline's list before merge-gate step.
        merged_fm_list: List[Any] = list(getattr(base_fm, "objects", []) or [])
        if extra_fm_dicts:
            # Build a lightweight object matching the ForeignObject protocol
            # that merge_gate_foreign_objects expects (attribute-based access).
            class _AdHocFM:
                def __init__(self, d: Dict[str, Any]):
                    for k, v in d.items():
                        setattr(self, k, v)
            merged_fm_list.extend([_AdHocFM(d) for d in extra_fm_dicts])
            # Monkey-patch the base FM result's objects before merging.
            try:
                base_fm.objects = merged_fm_list
            except Exception:
                pass
        fm_result = merge_gate_foreign_objects(base_fm, rice_gate)
        warnings.extend(fm_result.warnings)

        # 9. Admixture Detection (Sample-Level)
        seg_qualities = [g.segmentation_quality for g in grains]
        geom_dicts = [g.to_dict() for g in geometries]
        admixture_res = detect_admixture(
            geom_dicts,
            broken_labels=broken_labels,
            segmentation_qualities=seg_qualities,
        )

        # 10. Sample Statistics & Edge Cases
        if n_grains < self.small_sample_threshold:
            if n_grains == 1:
                warnings.append(
                    "Sample size: 1 grain. Individual-grain analysis is possible, "
                    "but sample-level quality percentages are not representative of a larger rice lot."
                )
            else:
                warnings.append(
                    f"Small sample ({n_grains} grains): grain-level classifications are available, "
                    "but sample-level percentages may not represent the composition of a larger rice lot."
                )

        # Metric or pixel lengths
        len_vals = [
            g.length_mm if g.length_mm is not None else g.length_pixels
            for g in geometries
        ]
        brd_vals = [
            g.breadth_mm if g.breadth_mm is not None else g.breadth_pixels
            for g in geometries
        ]
        lb_vals = [g.lb_ratio for g in geometries if g.lb_ratio is not None]

        unit = "mm" if calibration_res.calibrated else "pixels"

        # Safe percentage helper
        def to_pct(count: int, total: int) -> float:
            return round((count / total) * 100.0, 2) if total > 0 else 0.0

        def analyzed_labels(defect_key: str, label_key: str, known_labels: Tuple[str, ...]) -> int:
            return sum(
                grain["defects"][defect_key].get(label_key) in known_labels
                for grain in classified_grains
            )

        broken_analyzed_count = sum(label in ("broken", "whole") for label in broken_labels)
        damaged_analyzed_count = analyzed_labels(
            "damaged", "damaged_label", ("damaged", "slightly_damaged", "normal")
        )
        discoloured_analyzed_count = analyzed_labels(
            "discoloured", "discoloured_label", ("discoloured", "normal")
        )
        red_analyzed_count = analyzed_labels("red", "red_label", ("red", "not_red"))
        dehusked_analyzed_count = analyzed_labels(
            "dehusked", "dehusked_label", ("dehusked", "not_dehusked")
        )
        immature_analyzed_count = analyzed_labels(
            "immature_shrunken", "immature_shrunken_status", ("immature_shrunken", "normal")
        )
        sprouted_analyzed_count = analyzed_labels(
            "sprouted_weevilled", "sprouted_weevilled_label", ("sprouted_weevilled", "normal")
        )
        admixture_analyzed_count = (
            int(admixture_res.get("valid_grains_used", 0))
            if admixture_res.get("admixture_status") in ("detected", "not_detected")
            else 0
        )

        whole_count = sum(1 for label in broken_labels if label == "whole")
        undetermined_count = sum(1 for label in broken_labels if label == "undetermined")

        bc_ms = int((time.monotonic() - _bc_start) * 1000)
        structured_logger.info(
            "broken_classification_completed",
            job_id=job_id,
            request_id=request_id,
            duration_ms=bc_ms,
            grain_count=len(classified_grains),
            whole_count=whole_count,
            broken_count=broken_count,
            undetermined_count=undetermined_count,
            reference_source=whole_len_source,
        )

        sample_summary = {
            "total_rice_grains": n_grains,
            "total_count": n_grains,
            "whole_count": whole_count,
            "broken_count": broken_count,
            "undetermined_count": undetermined_count,
            "uncertain_grains": seg_uncertain,
            "rejected_grains": seg_rejected,
            "foreign_matter_count": fm_result.foreign_object_count,
            "admixture_percentage": (
                admixture_res.get("admixture_percentage")
                if admixture_analyzed_count
                else None
            ),
            "admixture_percent": (
                admixture_res.get("admixture_percentage")
                if admixture_analyzed_count
                else None
            ),
            "admixture_count": admixture_res.get("admixture_count", 0),
            "admixture_analyzed_count": admixture_analyzed_count,
            "admixture_status": admixture_res.get("admixture_status", "unsupported"),
            "geometry_outlier_count": admixture_res.get("geometry_outlier_count"),
            "geometry_outlier_fraction": admixture_res.get("geometry_outlier_fraction"),
            "broken_percent": (
                round((broken_count / n_grains) * 100.0, 2)
                if (n_grains > 0 and broken_analyzed_count > 0)
                else None
            ),
            "whole_percent": (
                round((whole_count / n_grains) * 100.0, 2)
                if (n_grains > 0 and broken_analyzed_count > 0)
                else None
            ),
            "broken_analyzed_count": broken_analyzed_count,
            "whole_reference_length": round(whole_kernel_len, 2) if whole_kernel_len is not None else None,
            "reference_source": whole_len_source,
            "reference_status": whole_len_status,
            "reference_profile_name": whole_len_meta.get("profile_name"),
            "reference_data_status": whole_len_meta.get("reference_data_status"),
            "reference_unit": whole_len_meta.get("reference_unit", measurement_unit),
            "reference_production_eligible": whole_len_meta.get("production_eligible", False),
            "reference_calibration_required": whole_len_meta.get("calibration_required", False),
            "reference_calibration_status": whole_len_meta.get("calibration_status"),
            "reference_count": whole_len_meta.get("reference_count") or whole_len_meta.get("candidate_count"),
            "reference_explanation": whole_len_meta.get("reason"),
            "damaged_count": damaged_count,
            "damaged_percent": to_pct(damaged_count, damaged_analyzed_count) if damaged_analyzed_count else None,
            "damaged_analyzed_count": damaged_analyzed_count,
            "discoloured_count": discoloured_count,
            "discoloured_percent": to_pct(discoloured_count, discoloured_analyzed_count) if discoloured_analyzed_count else None,
            "discoloured_analyzed_count": discoloured_analyzed_count,
            "chalky_count": chalky_count,
            "chalky_percent": (
                to_pct(chalky_count, chalky_analyzed_count)
                if chalky_analyzed_count
                else None
            ),
            "chalky_analyzed_count": chalky_analyzed_count,
            "red_count": red_count,
            "red_percent": to_pct(red_count, red_analyzed_count) if red_analyzed_count else None,
            "red_analyzed_count": red_analyzed_count,
            "dehusked_count": dehusked_count,
            "dehusked_percent": to_pct(dehusked_count, dehusked_analyzed_count) if dehusked_analyzed_count else None,
            "dehusked_analyzed_count": dehusked_analyzed_count,
            "immature_count": immature_count,
            "immature_percent": to_pct(immature_count, immature_analyzed_count) if immature_analyzed_count else None,
            "immature_analyzed_count": immature_analyzed_count,
            "sprouted_count": sprouted_count,
            "sprouted_percent": to_pct(sprouted_count, sprouted_analyzed_count) if sprouted_analyzed_count else None,
            "sprouted_analyzed_count": sprouted_analyzed_count,
            "foreign_matter_percent": None,
            "foreign_matter_analyzed_count": 0,
            "average_length": round(float(np.mean(len_vals)), 2) if len_vals else 0.0,
            "median_length": round(float(np.median(len_vals)), 2) if len_vals else 0.0,
            "average_breadth": round(float(np.mean(brd_vals)), 2) if brd_vals else 0.0,
            "median_breadth": round(float(np.median(brd_vals)), 2) if brd_vals else 0.0,
            "average_lb_ratio": round(float(np.mean(lb_vals)), 2) if lb_vals else 0.0,
            "median_lb_ratio": round(float(np.median(lb_vals)), 2) if lb_vals else 0.0,
            "measurement_unit": unit,
            "calibration_mode": calibration_res.mode,
            "pixels_per_mm": calibration_res.pixels_per_mm,
            "is_small_sample": n_grains < self.small_sample_threshold,
            "label_note": (
                "Observed sample fraction"
                if n_grains < self.small_sample_threshold
                else "Sample percentage"
            ),
        }

        # 11. Image Quality Assessment
        grain_quality_mask = np.zeros((h, w), dtype=np.uint8)
        for grain_mask, offset in zip(grain_masks_list, grain_offsets):
            draw_mask(grain_quality_mask, grain_mask, offset)

        quality_res = assess_image_quality(
            image_rgb=image_rgb,
            grain_areas=grain_areas,
            grain_confidences=grain_confidences,
            uncertain_count=seg_uncertain,
            total_count=n_grains,
            segmentation_qualities=seg_qualities,
            grain_mask=grain_quality_mask,
        )

        # 12. Historical/reference screening with sample- and quality-based suppression.
        standards_res = compare_with_standards(
            sample_stats=sample_summary,
            total_count=n_grains,
            calibration_mode=calibration_res.mode,
            grade=grade,
            quality_tier=quality_res["quality_level"],
            minimum_sample_size=max(30, self.small_sample_threshold),
        )
        warnings.extend(standards_res.get("warnings", []))

        # 13. Generate Annotated Image
        structured_logger.info("annotation_started", job_id=job_id)
        _ann_start = time.monotonic()
        phase1_grains_for_render: List[Dict[str, Any]] = []
        for i, g in enumerate(grains):
            mask_polygon = None
            if isinstance(g.mask_polygon, (list, tuple)) and len(g.mask_polygon) >= 3:
                mask_polygon = [[float(x), float(y)] for [x, y] in g.mask_polygon]
            b_lbl = broken_labels[i] if i < len(broken_labels) else "undetermined"
            phase1_grains_for_render.append(
                {
                    "id": int(g.grain_id),
                    "bbox": [int(g.bbox[0]), int(g.bbox[1]), int(g.bbox[2]), int(g.bbox[3])],
                    "confidence": float(g.confidence),
                    "confidence_label": g.confidence_label,
                    "mask_polygon": mask_polygon,
                    "centroid": [float(g.centroid[0]), float(g.centroid[1])],
                    "broken_label": b_lbl,
                }
            )
        fm_for_render: List[Dict[str, Any]] = []
        for idx, fo in enumerate(list(fm_result.objects or [])):
            try:
                bbox_attr = getattr(fo, "bbox", None)
                if bbox_attr is None and isinstance(fo, dict):
                    bbox_attr = fo.get("bbox")
                if not isinstance(bbox_attr, (list, tuple)) or len(bbox_attr) != 4:
                    continue
                class_name = getattr(fo, "class_name", None) or (
                    fo.get("class_name") if isinstance(fo, dict) else "FM"
                ) or "FM"
                conf = float(getattr(fo, "confidence", 0.0) or 0.0)
                fm_for_render.append(
                    {
                        "id": int(idx + 1),
                        "class": str(class_name),
                        "confidence": conf,
                        "bbox": [int(v) for v in bbox_attr],
                    }
                )
            except Exception:
                continue
        legend_label = None
        if phase1_meta:
            m = phase1_meta.get("segmentation_method_used") or phase1_meta.get("source")
            v = phase1_meta.get("model_version") or ""
            if m:
                legend_label = f"{m}  v{v}" if v else str(m)
        overlay_rgb = render_phase1_overlay(
            img_rgb=image_rgb,
            grains=phase1_grains_for_render,
            foreign_matter=fm_for_render,
            include_legend=False,
            legend_method_label=legend_label,
        )
        annotated_b64 = self._encode_rgb_to_jpeg_b64(image_rgb=overlay_rgb)

        ann_ms = int((time.monotonic() - _ann_start) * 1000)
        structured_logger.info(
            "annotation_completed",
            job_id=job_id,
            request_id=request_id,
            duration_ms=ann_ms,
        )

        total_elapsed = time.time() - start_time

        # Build clean deduplicated warnings list
        unique_warnings = list(dict.fromkeys(warnings))

        # Touching/merged estimate: grain count minus unique grain centroids
        # within 1 px is a rough approximation; classical CV watershed had an
        # explicit estimator. Phase 1 cascade marks per-grain `is_touching`
        # directly so we can report that instead.
        estimated_merged = sum(1 for g in grains if g.is_touching)

        result_dict = {
            "success": True,
            "rice_detected": True,
            "image": image_info.to_dict(),
            "sample": {
                "total_detected": seg_detected,
                "analysed": n_grains,
                "uncertain": seg_uncertain,
                "rejected": seg_rejected,
                "estimated_merged": estimated_merged,
            },
            "calibration": calibration_res.to_dict(),
            "quality": quality_res,
            "grains": classified_grains,
            "foreign_matter": fm_result.to_dict()["objects"],
            "admixture": admixture_res,
            "summary": sample_summary,
            "standards": standards_res,
            "rice_gate": rice_gate,
            "annotated_image_base64": annotated_b64,
            "original_image_base64": self._encode_rgb_to_jpeg_b64(image_rgb=image_rgb),
            "warnings": unique_warnings,
            "processing_time_seconds": round(total_elapsed, 3),
        }

        # Optional, schema-permissive metadata block — preserved exactly as
        # produced by the cascade so frontend/dashboard panels can surface a
        # "Segmentation Method: YOLOv8l-seg" badge later.  No existing frontend
        # key depends on this; it is silently ignored by older clients.
        if phase1_meta:
            result_dict["segmentation_info"] = dict(phase1_meta)

        structured_logger.info(
            "analysis_completed",
            job_id=job_id,
            request_id=request_id,
            status="success",
            rice_detected=True,
            grain_count=sample_summary.get("total_rice_grains", n_grains),
            whole_count=whole_count,
            broken_count=broken_count,
            undetermined_count=undetermined_count,
            broken_percent=sample_summary.get("broken_percent"),
            segmentation_method=phase1_meta.get("segmentation_method_used") or phase1_meta.get("source"),
            reference_source=sample_summary.get("reference_source"),
            total_processing_ms=int((time.monotonic() - t0) * 1000),
        )

        return result_dict

    def _build_no_rice_response(
        self,
        image_info: Any,
        image_rgb: np.ndarray,
        rice_gate: Dict[str, Any],
        warnings: List[str],
        start_time: float,
        message: Optional[str] = None,
        sample_overrides: Optional[Dict[str, Any]] = None,
        phase1_meta: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        CASE 1 response: no valid rice-class detection, so the rice-analysis
        pipeline (grain segmentation, per-grain parameter extraction, defect
        classification, grading, admixture) is NOT executed.

        Foreign matter is still reported separately — foreign matter is NOT rice,
        and the foreign-matter channel must stay available for later work.
        """
        no_rice_message = message or (
            "No rice grains detected. Please upload an image containing rice grains."
        )

        # Foreign matter is detected on the full image (no rice masks exist here)
        # so the non-rice objects remain identifiable even though rice analysis stops.
        fm_result = merge_gate_foreign_objects(
            detect_foreign_matter(image_rgb), rice_gate
        )
        fm_summary = fm_result.to_dict()
        fm_objects = fm_summary.pop("objects")

        sample: Dict[str, Any] = {
            "total_detected": rice_gate.get("total_detections", 0),
            "analysed": 0,
            "uncertain": 0,
            "rejected": 0,
            "rice_detections": rice_gate.get("rice_detections", 0),
            "foreign_matter_detections": rice_gate.get("foreign_matter_detections", 0),
        }
        if sample_overrides:
            sample.update(sample_overrides)

        annotated_b64 = self._create_annotated_image_b64(
            image_rgb=image_rgb,
            grains=[],
            geometries=[],
            foreign_objects=fm_result.objects,
        )

        unique_warnings = list(
            dict.fromkeys(
                warnings
                + list(rice_gate.get("warnings", []))
                + [
                    no_rice_message,
                    "Rice analysis stopped before grain segmentation: no grain "
                    "measurements, defect classification, grading or admixture were computed.",
                ]
            )
        )

        return {
            "success": True,
            "rice_detected": False,
            "message": no_rice_message,
            "image": image_info.to_dict(),
            "sample": sample,
            "calibration": {
                "calibrated": False,
                "mode": "not_evaluated",
                "pixels_per_mm": None,
                "marker_count": 0,
                "message": "Calibration not performed — rice analysis was not executed.",
            },
            "quality": None,
            "grains": [],
            "foreign_matter": fm_objects,
            "foreign_matter_summary": fm_summary,
            "admixture": {
                "admixture_status": "not_applicable",
                "reason": "No rice grains detected — rice analysis was not executed.",
            },
            "summary": {},
            "standards": {"status": "Not evaluated — no rice grains detected"},
            "rice_gate": rice_gate,
            "annotated_image_base64": annotated_b64,
            "original_image_base64": self._encode_rgb_to_jpeg_b64(image_rgb=image_rgb),
            "warnings": unique_warnings,
            "processing_time_seconds": round(time.time() - start_time, 3),
        }

    def _encode_rgb_to_jpeg_b64(self, image_rgb: np.ndarray) -> str:
        """
        Encode an HxWx3 uint8 RGB numpy array to a data URI string suitable for
        ``<img src="...">`` embedding in the dashboard / demo frontend.

        Used by both the Phase 1 renderer (primary) and the classical CV
        fallback annotated renderer.
        """
        pil_img = Image.fromarray(image_rgb)
        h, w = image_rgb.shape[:2]
        if max(w, h) > 1600:
            pil_img.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=85)
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"

    def _create_annotated_image_b64(
        self,
        image_rgb: np.ndarray,
        grains: List[GrainInstance],
        geometries: List[Any],
        foreign_objects: List[Any],
    ) -> str:
        """
        Classical-CV fallback annotated renderer — called only from the
        NOT_RICE fast path (where no segmented grains exist yet and we just
        need FM-only boxes drawn) OR if the user disables Phase 1 renderer.

        For the main (rice-detected) path, the dashboard now uses
        ``render_phase1_overlay`` via the call site immediately above Step 13
        in ``analyze()``.
        """
        # Fast path: empty grains list and we just need FM boxes — delegate to
        # the shared renderer with empty grains list, which still draws
        # FM boxes correctly (identical output to the old function for zero
        # grains, no-op mask drawing).
        if not grains:
            fm_for_render: List[Dict[str, Any]] = []
            for idx, fo in enumerate(list(foreign_objects or [])):
                try:
                    bbox_attr = getattr(fo, "bbox", None)
                    if bbox_attr is None and isinstance(fo, dict):
                        bbox_attr = fo.get("bbox")
                    if not isinstance(bbox_attr, (list, tuple)) or len(bbox_attr) != 4:
                        continue
                    class_name = getattr(fo, "class_name", None) or (
                        fo.get("class_name") if isinstance(fo, dict) else "FM"
                    ) or "FM"
                    conf = float(getattr(fo, "confidence", 0.0) or 0.0)
                    fm_for_render.append(
                        {
                            "id": int(idx + 1),
                            "class": str(class_name),
                            "confidence": conf,
                            "bbox": [int(v) for v in bbox_attr],
                        }
                    )
                except Exception:
                    continue
            overlay = render_phase1_overlay(
                img_rgb=image_rgb,
                grains=[],
                foreign_matter=fm_for_render,
                include_legend=False,
                legend_method_label=None,
            )
            return self._encode_rgb_to_jpeg_b64(overlay)

        # Slow path: grains exist AND a caller explicitly used this function
        # (should not happen in normal dashboard flow).  Keep the original
        # cycle-palette renderer alive only so any third-party code that
        # imports this method directly keeps working.
        overlay = image_rgb.copy()
        h, w = image_rgb.shape[:2]

        colors = [
            (34, 197, 94),   # Green
            (59, 130, 246),  # Blue
            (236, 72, 153),  # Pink
            (168, 85, 247),  # Purple
            (20, 184, 166),  # Teal
            (249, 115, 22),  # Orange
        ]

        # Draw grain masks and IDs
        for i, (grain, geom) in enumerate(zip(grains, geometries)):
            color = colors[i % len(colors)]
            # The mask is a crop; blend it into the matching image region.
            ox, oy = grain.mask_offset
            mh, mw = grain.mask.shape[:2]
            region = overlay[oy : oy + mh, ox : ox + mw]
            mask_bool = grain.mask > 0

            # Blend color mask
            region[mask_bool] = (
                region[mask_bool] * 0.6 + np.array(color) * 0.4
            ).astype(np.uint8)

            # Draw contour if present
            if grain.contour is not None:
                cv2.drawContours(overlay, [grain.contour], -1, color, 1)

            # Draw Grain #ID text
            cx, cy = int(grain.centroid[0]), int(grain.centroid[1])
            # Only draw label if reasonable size
            if geom.area_pixels >= 40:
                text = f"#{grain.grain_id}"
                font_scale = 0.4 if min(w, h) < 800 else 0.5
                thickness = 1
                (tw, th), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                cv2.rectangle(
                    overlay,
                    (cx - tw // 2 - 2, cy - th // 2 - 2),
                    (cx + tw // 2 + 2, cy + th // 2 + 2),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    overlay,
                    text,
                    (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

        # Draw Foreign Matter objects (in Red / Yellow with bounding box)
        for fo in foreign_objects:
            fx, fy, fw, fh = fo.bbox
            cv2.rectangle(overlay, (fx, fy), (fx + fw, fy + fh), (239, 68, 68), 2)
            label = f"FM:{fo.class_name}"
            cv2.putText(
                overlay,
                label,
                (fx, max(12, fy - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (239, 68, 68),
                1,
                cv2.LINE_AA,
            )

        # Encode to JPEG base64
        pil_img = Image.fromarray(overlay)
        # Limit preview resolution to avoid giant responses if > 2000px
        if max(w, h) > 1600:
            pil_img.thumbnail((1600, 1600), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=85)
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"
