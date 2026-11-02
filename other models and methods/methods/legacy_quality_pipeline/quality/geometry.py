"""
Per-grain geometry analysis module.

Computes: area, perimeter, centroid, orientation, bounding box,
fitted ellipse, minAreaRect, major/minor axes, solidity, aspect ratio,
Length, Breadth, L/B ratio, and contour-based effective length.

Converts to mm only if calibration exists, otherwise uses pixels.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from ml.quality.profiles import GrainProfile, get_default_grain_profile, load_grain_profile

logger = logging.getLogger(__name__)


@dataclass
class GrainGeometry:
    """Per-grain geometric measurements."""
    grain_id: int
    area_pixels: float
    perimeter_pixels: float
    centroid: Tuple[float, float]
    orientation_deg: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    fitted_ellipse: Optional[Tuple] = None  # ((cx,cy), (MA,ma), angle)
    min_area_rect: Optional[Tuple] = None   # ((cx,cy), (w,h), angle)
    major_axis_pixels: float = 0.0
    minor_axis_pixels: float = 0.0
    solidity: float = 0.0
    aspect_ratio: float = 0.0
    length_pixels: float = 0.0
    breadth_pixels: float = 0.0
    effective_length_pixels: float = 0.0
    length_mm: Optional[float] = None
    breadth_mm: Optional[float] = None
    effective_length_mm: Optional[float] = None
    lb_ratio: Optional[float] = None
    measurement_quality: str = "good"
    is_anomalous: bool = False
    anomaly_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grain_id": self.grain_id,
            "area_pixels": round(self.area_pixels, 2),
            "perimeter_pixels": round(self.perimeter_pixels, 2),
            "centroid": [round(c, 2) for c in self.centroid],
            "orientation_deg": round(self.orientation_deg, 2),
            "bbox": list(self.bbox),
            "major_axis_pixels": round(self.major_axis_pixels, 2),
            "minor_axis_pixels": round(self.minor_axis_pixels, 2),
            "solidity": round(self.solidity, 4),
            "aspect_ratio": round(self.aspect_ratio, 4),
            "length_pixels": round(self.length_pixels, 2),
            "breadth_pixels": round(self.breadth_pixels, 2),
            "effective_length_pixels": round(self.effective_length_pixels, 2),
            "length_mm": round(self.length_mm, 3) if self.length_mm is not None else None,
            "breadth_mm": round(self.breadth_mm, 3) if self.breadth_mm is not None else None,
            "effective_length_mm": round(self.effective_length_mm, 3) if self.effective_length_mm is not None else None,
            "lb_ratio": round(self.lb_ratio, 4) if self.lb_ratio is not None else None,
            "measurement_quality": self.measurement_quality,
            "is_anomalous": self.is_anomalous,
            "anomaly_reasons": self.anomaly_reasons,
        }


def compute_grain_geometry(
    mask: Optional[np.ndarray] = None,
    grain_id: int = 1,
    pixels_per_mm: Optional[float] = None,
    offset: Tuple[int, int] = (0, 0),
    **kwargs,
) -> GrainGeometry:
    """
    Compute geometric properties for a single grain mask.

    Args:
        mask: Binary mask (uint8, 0/255) for one grain.  May be a tight
            bounding-box crop of the image instead of a full-frame mask.
        grain_id: Unique grain identifier
        pixels_per_mm: If available, convert to mm
        offset: (x, y) position of the mask crop in the image.  Positions
            (bbox, centroid) are reported in image coordinates.

    Returns:
        GrainGeometry object
    """
    if mask is None and "grain_id" in kwargs:
        grain_id = kwargs["grain_id"]
    if mask is None:
        mask = kwargs.get("mask", np.zeros((10, 10), dtype=np.uint8))

    def _empty_geometry(reasons: List[str]) -> GrainGeometry:
        return GrainGeometry(
            grain_id=grain_id,
            area_pixels=0.0,
            perimeter_pixels=0.0,
            centroid=(0.0, 0.0),
            orientation_deg=0.0,
            bbox=(0, 0, 0, 0),
            major_axis_pixels=0.0,
            minor_axis_pixels=0.0,
            solidity=0.0,
            aspect_ratio=0.0,
            length_pixels=0.0,
            breadth_pixels=0.0,
            effective_length_pixels=0.0,
            length_mm=None,
            breadth_mm=None,
            effective_length_mm=None,
            lb_ratio=None,
            measurement_quality="poor",
            is_anomalous=True,
            anomaly_reasons=reasons,
        )

    if mask is None or mask.size == 0 or np.sum(mask) == 0:
        return _empty_geometry(["Empty mask"])

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return _empty_geometry(["No contours found in mask"])
    
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < 5:
        return _empty_geometry(["Contour area < 5 pixels"])
    
    perimeter = cv2.arcLength(contour, True)
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return _empty_geometry(["Zero moment contour"])
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    
    # Bounding box
    x, y, w, h = cv2.boundingRect(contour)
    
    # Fitted ellipse (requires >= 5 points)
    fitted_ellipse = None
    major_axis = 0.0
    minor_axis = 0.0
    orientation = 0.0
    
    if len(contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(contour)
            fitted_ellipse = ellipse
            (ecx, ecy), (ma, MA), angle = ellipse
            major_axis = max(MA, ma)
            minor_axis = min(MA, ma)
            orientation = angle
        except cv2.error:
            major_axis = max(w, h)
            minor_axis = min(w, h)
            orientation = 0.0
    else:
        major_axis = max(w, h)
        minor_axis = min(w, h)
    
    # Min area rectangle
    min_rect = None
    rect_extent = major_axis
    if len(contour) >= 5:
        try:
            min_rect = cv2.minAreaRect(contour)
            rect_w, rect_h = min_rect[1]
            rect_extent = max(rect_w, rect_h)
        except Exception:
            min_rect = None
    
    # Robust principal-axis end-to-end projection measurement
    # Rather than solely relying on fitted ellipse (which assumes elliptical shape),
    # project actual contour points onto the grain's principal inertial orientation.
    pts = contour.reshape(-1, 2).astype(np.float64)
    mu20 = M["mu20"]
    mu02 = M["mu02"]
    mu11 = M["mu11"]
    
    # Inertial orientation angle in radians
    theta = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    # Centered contour points projected onto principal and transverse axes
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    proj_u = dx * cos_t + dy * sin_t
    proj_v = -dx * sin_t + dy * cos_t
    
    principal_span = float(np.max(proj_u) - np.min(proj_u)) if len(proj_u) > 0 else major_axis
    transverse_span = float(np.max(proj_v) - np.min(proj_v)) if len(proj_v) > 0 else minor_axis
    
    # Contour caliper length along principal axis
    caliper_length = max(principal_span, transverse_span)
    
    # Effective length reconciles principal projection, minAreaRect, and ellipse major axis
    # It reflects true physical tip-to-tip extent without boundary distortion.
    candidates = [v for v in [caliper_length, rect_extent, major_axis] if v > 0]
    effective_length_pixels = float(np.median(candidates)) if candidates else float(major_axis)

    # Convex hull for solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    
    # Length = major dimension, Breadth = minor dimension
    length_pixels = major_axis
    breadth_pixels = minor_axis
    
    # Aspect ratio
    aspect_ratio = length_pixels / breadth_pixels if breadth_pixels > 0 else 0.0
    
    # L/B ratio (protected against division by zero)
    lb_ratio = None
    if breadth_pixels > 0:
        lb_ratio = length_pixels / breadth_pixels
    
    # Convert to mm if calibration exists
    length_mm = None
    breadth_mm = None
    effective_length_mm = None
    if pixels_per_mm is not None and pixels_per_mm > 0:
        length_mm = length_pixels / pixels_per_mm
        breadth_mm = breadth_pixels / pixels_per_mm
        effective_length_mm = effective_length_pixels / pixels_per_mm

    # Report positions in image coordinates (the mask may be a crop).
    off_x, off_y = int(offset[0]), int(offset[1])
    cx = cx + off_x
    cy = cy + off_y
    x = x + off_x
    y = y + off_y
    if fitted_ellipse is not None:
        (ecx, ecy), (ma, MA), angle = fitted_ellipse
        fitted_ellipse = ((ecx + off_x, ecy + off_y), (ma, MA), angle)
    if min_rect is not None:
        (rcx, rcy), (rw, rh), angle = min_rect
        min_rect = ((rcx + off_x, rcy + off_y), (rw, rh), angle)
    
    # Anomaly detection
    is_anomalous = False
    anomaly_reasons = []
    
    if lb_ratio is not None and lb_ratio > 10:
        is_anomalous = True
        anomaly_reasons.append(f"Extreme L/B ratio: {lb_ratio:.2f}")
    
    if solidity < 0.5:
        is_anomalous = True
        anomaly_reasons.append(f"Very low solidity: {solidity:.3f}")
    
    if area < 20:
        is_anomalous = True
        anomaly_reasons.append(f"Very small area: {area:.0f} px")
    
    # Measurement quality
    measurement_quality = "good"
    if area < 100:
        measurement_quality = "limited"
    if area < 25:
        measurement_quality = "poor"
    if is_anomalous:
        measurement_quality = "unreliable"
    
    return GrainGeometry(
        grain_id=grain_id,
        area_pixels=area,
        perimeter_pixels=perimeter,
        centroid=(cx, cy),
        orientation_deg=orientation,
        bbox=(x, y, w, h),
        fitted_ellipse=fitted_ellipse,
        min_area_rect=min_rect,
        major_axis_pixels=major_axis,
        minor_axis_pixels=minor_axis,
        solidity=solidity,
        aspect_ratio=aspect_ratio,
        length_pixels=length_pixels,
        breadth_pixels=breadth_pixels,
        effective_length_pixels=effective_length_pixels,
        length_mm=length_mm,
        breadth_mm=breadth_mm,
        effective_length_mm=effective_length_mm,
        lb_ratio=lb_ratio,
        measurement_quality=measurement_quality,
        is_anomalous=is_anomalous,
        anomaly_reasons=anomaly_reasons,
    )


def resolve_whole_kernel_reference(
    lengths: List[float],
    profile: Optional[Union[GrainProfile, Dict[str, Any], str]] = None,
    geometries: Optional[List[GrainGeometry]] = None,
    measurement_unit: str = "pixels",
    calibration: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], str, str, Dict[str, Any]]:
    """
    Resolve whole-kernel reference length using the reference hierarchy:
    
    1. Explicit calibrated physical profile (mm) + valid calibration
    2. Explicit valid same-image/sample reference
    3. Sample-derived whole reference from intact population in same image
    4. Explicit legacy pixel profile (demo-only, proxy)
    5. No valid reference -> undetermined
    
    Returns:
        (reference_length, reference_source, reference_status, metadata_dict)
        where:
        - reference_source: 'profile', 'sample_derived', or 'unavailable'
        - reference_status: 'reliable', 'limited', or 'undetermined'
        - metadata includes: reference_data_status, production_eligible,
          calibration_required, calibration_status, reference_unit, etc.
    """
    metadata: Dict[str, Any] = {
        "reference_unit": measurement_unit,
        "calibration_required": False,
        "calibration_status": "unavailable",
        "calibration_validity": "unavailable",
        "calibration_validity_reason": "",
    }

    calibration_status = "unavailable"
    calibration_validity = "unavailable"
    calibration_validity_reason = ""
    if calibration:
        if calibration.get("calibrated"):
            calibration_status = calibration.get("mode", "unavailable")
        calibration_validity = calibration.get("validity", "unavailable")
        calibration_validity_reason = calibration.get("validity_reason", "")
    metadata["calibration_status"] = calibration_status
    metadata["calibration_validity"] = calibration_validity
    metadata["calibration_validity_reason"] = calibration_validity_reason

    # -------------------------------------------------------------
    # TIER 1: Configured / Trusted Profile
    # -------------------------------------------------------------
    active_profile: Optional[GrainProfile] = None
    if profile is not None:
        if isinstance(profile, GrainProfile):
            active_profile = profile
        else:
            active_profile = load_grain_profile(profile)

    if active_profile is not None:
        is_valid, err_msg = active_profile.validate()
        if not is_valid:
            logger.warning(f"Profile '{active_profile.profile_name}' is invalid: {err_msg}")
            active_profile = None

    if active_profile is not None:
        profile_unit = active_profile.reference_unit
        profile_data_status = active_profile.data_status or "Proxy"
        production_eligible = bool(active_profile.production_eligible)

        # Physical mm profile requires valid calibration
        if profile_unit == "mm":
            metadata["calibration_required"] = True
            if calibration_validity != "valid":
                logger.warning(
                    f"Profile '{active_profile.profile_name}' is a physical mm profile, "
                    f"but this image calibration is {calibration_validity}. "
                    f"Cannot use mm reference without valid metric calibration."
                )
                return (
                    None,
                    "unavailable",
                    "undetermined",
                    {
                        **metadata,
                        "profile_name": active_profile.profile_name,
                        "reference_unit": profile_unit,
                        "reference_data_status": profile_data_status,
                        "production_eligible": production_eligible,
                        "reason": (
                            "Physical profile supplied but calibration is not valid "
                            f"({calibration_validity}: {calibration_validity_reason})."
                            if calibration_validity_reason
                            else "Physical profile supplied but calibration is not valid for this image."
                        ),
                    },
                )
            # mm profile + valid calibration -> production eligible
            ref_len = float(active_profile.whole_kernel_length)
            metadata.update({
                "profile_name": active_profile.profile_name,
                "profile_source": active_profile.source,
                "reference_unit": profile_unit,
                "reference_count": active_profile.reference_count,
                "reference_data_status": profile_data_status,
                "production_eligible": True,
                "data_status": profile_data_status,
            })
            return ref_len, "profile", "reliable", metadata

        # Legacy pixel profile: demo-only, proxy, NOT production eligible
        if profile_unit == "pixels":
            ref_len = float(active_profile.whole_kernel_length)

            # Scale-compatibility guard for pixel-unit profiles.
            if measurement_unit == "pixels" and lengths and len(lengths) >= 3:
                sample_median = float(np.median(lengths))
                ratio = ref_len / sample_median if sample_median > 0 else 0.0
                SCALE_LO, SCALE_HI = 0.3, 3.0
                if not (SCALE_LO <= ratio <= SCALE_HI):
                    logger.warning(
                        f"Profile '{active_profile.profile_name}' reference length {ref_len:.1f} px "
                        f"is scale-incompatible with this image (sample median={sample_median:.1f} px, "
                        f"ratio={ratio:.2f} outside [{SCALE_LO},{SCALE_HI}]). "
                        f"Falling back to sample-derived reference."
                    )
                    # Fall through to Tier 3 below
                else:
                    metadata.update({
                        "profile_name": active_profile.profile_name,
                        "profile_source": active_profile.source,
                        "reference_unit": profile_unit,
                        "reference_count": active_profile.reference_count,
                        "scale_ratio": round(ratio, 3),
                        "reference_data_status": profile_data_status,
                        "production_eligible": production_eligible,
                        "data_status": profile_data_status,
                    })
                    # Pixel profile is NEVER labeled "reliable" — it is a scale-specific proxy
                    return ref_len, "profile", "limited", metadata
            else:
                metadata.update({
                    "profile_name": active_profile.profile_name,
                    "profile_source": active_profile.source,
                    "reference_unit": profile_unit,
                    "reference_count": active_profile.reference_count,
                    "reference_data_status": profile_data_status,
                    "production_eligible": production_eligible,
                    "data_status": profile_data_status,
                })
                # Pixel profile is NEVER labeled "reliable" — it is a scale-specific proxy
                return ref_len, "profile", "limited", metadata

    # -------------------------------------------------------------
    # TIER 2/3: Validated Sample-Derived Reference Fallback
    # -------------------------------------------------------------
    # A sample can only derive a reference if there is sufficient evidence
    # of intact whole grains.
    if not lengths or len(lengths) < 3:
        return (
            None,
            "unavailable",
            "undetermined",
            {
                **metadata,
                "reason": f"Sample size ({len(lengths) if lengths else 0}) <= 2 without trusted profile.",
            },
        )

    arr_lengths = np.array(lengths, dtype=np.float64)
    n = len(arr_lengths)

    max_len = float(np.max(arr_lengths))
    min_len = float(np.min(arr_lengths))
    span = max_len - min_len

    if span < 0.25 * max_len:
        if geometries and len(geometries) == n:
            med_lb = float(np.median([g.lb_ratio or 1.0 for g in geometries]))
            med_solidity = float(np.median([g.solidity for g in geometries]))
            if med_lb >= 2.2 and med_solidity >= 0.88:
                candidate_whole = arr_lengths
            else:
                return (
                    None,
                    "unavailable",
                    "undetermined",
                    {
                        **metadata,
                        "reason": "Uniform sample lacks elongation evidence of intact whole grains; reference unavailable.",
                    },
                )
        else:
            if max_len >= 85.0 or (measurement_unit == "mm" and max_len >= 6.0):
                candidate_whole = arr_lengths
            else:
                return (
                    None,
                    "unavailable",
                    "undetermined",
                    {
                        **metadata,
                        "reason": f"Sample grains are uniformly short ({max_len:.1f} < 85 px) without trusted profile; reference unavailable.",
                    },
                )
    else:
        cutoff = max(min_len + 0.35 * span, 0.75 * max_len)
        candidate_whole = arr_lengths[arr_lengths >= cutoff]
        if len(candidate_whole) >= 3:
            cand_med = float(np.median(candidate_whole))
            candidate_whole = candidate_whole[
                (candidate_whole >= 0.85 * cand_med) & (candidate_whole <= 1.20 * cand_med)
            ]

    if len(candidate_whole) < 3:
        return (
            None,
            "unavailable",
            "undetermined",
            {
                **metadata,
                "reason": f"Only {len(candidate_whole)} candidate whole grains found; minimum 3 required.",
            },
        )

    cand_mean = float(np.mean(candidate_whole))
    cand_std = float(np.std(candidate_whole))
    cv = cand_std / cand_mean if cand_mean > 0 else 1.0

    if cv > 0.15:
        return (
            None,
            "unavailable",
            "undetermined",
            {
                **metadata,
                "reason": f"Candidate whole-grain population has high length variance (CV={cv:.2f} > 0.15).",
            },
        )

    ref_len = float(np.median(candidate_whole))
    status = "reliable" if len(candidate_whole) >= 10 else "limited"
    metadata.update({
        "candidate_count": int(len(candidate_whole)),
        "candidate_cv": round(cv, 4),
        "method": "upper_population_clustering",
        "reference_data_status": "Sample-Derived",
        "production_eligible": False,
        "data_status": "Sample-Derived",
    })
    return ref_len, "sample_derived", status, metadata


def compute_robust_whole_kernel_length(
    grain_lengths: List[float],
    threshold_fraction: float = 0.75,
    robust_filter_fraction: float = 0.85,
    robust_iterations: int = 3,
    profile: Optional[Union[GrainProfile, Dict[str, Any], str]] = None,
) -> Tuple[Optional[float], str]:
    """
    Backwards-compatible wrapper around resolve_whole_kernel_reference.
    
    Returns:
        (whole_kernel_length, status) — status is 'reliable', 'limited', or 'undetermined'
    """
    ref_len, source, status, _ = resolve_whole_kernel_reference(
        lengths=grain_lengths,
        profile=profile,
    )
    return ref_len, status


def classify_broken(
    length: float,
    whole_kernel_length: Optional[float],
    threshold_fraction: float = 0.75,
    small_broken_fraction: float = 0.25,
    reference_source: str = "profile",
    reference_status: str = "reliable",
    measurement_quality: str = "good",
) -> Dict[str, Any]:
    """
    Classify whether a grain is broken based on its effective length relative to the whole-kernel reference.
    
    FSSAI & Standards Definition: A grain/fragment is broken if its length is less than three-fourths (0.75)
    of the whole-kernel length.
    
    Args:
        length: Grain effective length (calibrated mm or pixels)
        whole_kernel_length: Trusted or sample-derived whole-kernel reference length
        threshold_fraction: Broken if length < this fraction (default 0.75)
        small_broken_fraction: Sub-category for small broken fragments (default 0.25)
        reference_source: 'profile', 'sample_derived', or 'unavailable'
        reference_status: 'reliable', 'limited', or 'undetermined'
        measurement_quality: 'good', 'limited', 'poor', 'unreliable'
        
    Returns:
        Structured dict with broken status, ratio, and reference provenance.
    """
    if (
        whole_kernel_length is None
        or whole_kernel_length <= 0
        or reference_status == "undetermined"
        or reference_source == "unavailable"
    ):
        return {
            "broken_label": "undetermined",
            "effective_length": round(length, 2),
            "whole_kernel_length_ref": None,
            "length_ratio": None,
            "broken_ratio": None,
            "is_small_broken": None,
            "confidence": 0.0,
            "method": reference_source,
            "reference_source": reference_source,
            "reference_status": "undetermined",
            "threshold_fraction": threshold_fraction,
            "measurement_quality": measurement_quality,
            "classification_reason": "Insufficient whole-kernel reference (cannot determine without trusted profile or intact population)",
        }
    
    ratio = length / whole_kernel_length
    is_broken = ratio < threshold_fraction
    is_small_broken = ratio < small_broken_fraction
    
    # Base confidence based on margin from 0.75 decision boundary
    margin = abs(ratio - threshold_fraction)
    confidence = min(1.0, margin / 0.20 + 0.60)
    if measurement_quality in ("poor", "unreliable"):
        confidence = min(confidence, 0.45)
    elif measurement_quality == "limited":
        confidence = min(confidence, 0.75)
    
    label = "broken" if is_broken else "whole"
    
    return {
        "broken_label": label,
        "effective_length": round(length, 2),
        "whole_kernel_length_ref": round(whole_kernel_length, 2),
        "length_ratio": round(ratio, 4),
        "broken_ratio": round(ratio, 4),  # backwards compatibility
        "is_small_broken": is_small_broken,
        "confidence": round(confidence, 4),
        "method": reference_source,
        "reference_source": reference_source,
        "reference_status": reference_status,
        "threshold_fraction": threshold_fraction,
        "measurement_quality": measurement_quality,
        "classification_reason": (
            f"Length ratio {ratio:.3f} {'<' if is_broken else '>='} threshold {threshold_fraction:.2f} "
            f"against {reference_source} reference ({whole_kernel_length:.1f})"
        ),
    }
