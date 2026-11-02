"""
Image quality assessment module.

Computes image-quality indicators AFTER segmentation:
- Blur score (Laplacian variance)
- Median grain size in pixels
- Segmentation confidence
- Background separation quality
- Illumination uniformity
- Proportion of uncertain grains

Produces quality tiers: GOOD, FAIR, POOR, UNRELIABLE
These are engineering heuristics, NOT scientifically validated scores.
"""

import logging
from typing import Dict, List, Optional

import cv2
import numpy as np

from ml.config import get_threshold

logger = logging.getLogger(__name__)


def assess_image_quality(
    image_rgb: np.ndarray,
    grain_areas: List[float],
    grain_confidences: List[float],
    uncertain_count: int,
    total_count: int,
    segmentation_qualities: Optional[List[str]] = None,
    grain_mask: Optional[np.ndarray] = None,
) -> Dict:
    """
    Compute image quality assessment.
    
    This happens AFTER grain detection/segmentation, not before.
    Total megapixels is used for logging only — not for rejection.
    
    Args:
        image_rgb: The input image
        grain_areas: List of grain areas in pixels
        grain_confidences: List of segmentation confidences
        uncertain_count: Number of uncertain/merged grain instances
        total_count: Total accepted grain instances
    """
    h, w = image_rgb.shape[:2]
    megapixels = (h * w) / 1_000_000
    valid_region_mask = (
        grain_mask > 0
        if grain_mask is not None
        and grain_mask.shape == (h, w)
        and np.any(grain_mask)
        else None
    )
    
    # 1. Blur score (Laplacian variance)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blur_gray = gray.copy()
    if valid_region_mask is not None:
        blur_gray[~valid_region_mask] = 0
    laplacian = cv2.Laplacian(blur_gray, cv2.CV_64F)
    blur_pixels = laplacian[valid_region_mask] if valid_region_mask is not None else laplacian
    blur_score = float(np.var(blur_pixels))
    
    # 2. Median grain size
    median_grain_pixels = float(np.median(grain_areas)) if grain_areas else 0
    
    # 3. Mean segmentation confidence
    mean_confidence = float(np.mean(grain_confidences)) if grain_confidences else 0
    
    # 4. Uncertain grain fraction
    uncertain_fraction = uncertain_count / total_count if total_count > 0 else 0
    segmentation_qualities = segmentation_qualities or []
    low_quality_segmentation_count = sum(
        quality in ("poor", "unreliable", "uncertain")
        for quality in segmentation_qualities
    )
    low_quality_segmentation_fraction = (
        low_quality_segmentation_count / total_count if total_count > 0 else 0
    )
    
    # 5. Illumination uniformity (std of block means)
    block_size = max(32, min(h, w) // 8)
    block_means = []
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            if valid_region_mask is not None:
                block = block[valid_region_mask[i:i+block_size, j:j+block_size]]
            if block.size:
                block_means.append(float(np.mean(block)))
    illumination_uniformity = 1.0 - min(1.0, float(np.std(block_means)) / 50.0) if block_means else 0.5
    
    # 6. Saturation/clipping check
    valid_grain_pixels = gray[valid_region_mask] if valid_region_mask is not None else gray.reshape(-1)
    clipping_basis = "accepted grain-mask pixels" if valid_region_mask is not None else "full image (no valid grain mask supplied)"
    clipped_dark = float(np.mean(valid_grain_pixels < 5))
    clipped_bright = float(np.mean(valid_grain_pixels > 250))
    
    # Quality thresholds
    blur_good = get_threshold("image_quality", "blur_threshold_good", 100)
    blur_fair = get_threshold("image_quality", "blur_threshold_fair", 50)
    blur_poor = get_threshold("image_quality", "blur_threshold_poor", 20)
    
    grain_px_good = get_threshold("image_quality", "min_grain_pixels_good", 400)
    grain_px_fair = get_threshold("image_quality", "min_grain_pixels_fair", 100)
    grain_px_poor = get_threshold("image_quality", "min_grain_pixels_poor", 25)
    
    uncertain_warning = get_threshold("image_quality", "uncertain_grain_fraction_warning", 0.2)
    segmentation_confidence_floor = get_threshold("segmentation", "confidence_threshold", 0.5)
    uncertain_unreliable = get_threshold("image_quality", "unreliable_uncertain_grain_fraction", 0.5)
    low_quality_unreliable = get_threshold("image_quality", "unreliable_low_quality_segmentation_fraction", 0.5)
    clipping_unreliable = get_threshold("image_quality", "unreliable_clipped_pixel_fraction", 0.5)
    illumination_unreliable = get_threshold("image_quality", "unreliable_illumination_uniformity_below", 0.15)
    
    # Determine quality tier
    reasons = []
    scores = []
    
    # Blur
    if blur_score >= blur_good:
        scores.append(1.0)
    elif blur_score >= blur_fair:
        scores.append(0.7)
        reasons.append("Image appears slightly blurry.")
    elif blur_score >= blur_poor:
        scores.append(0.4)
        reasons.append("Image is noticeably blurry.")
    else:
        scores.append(0.1)
        reasons.append("Image is very blurry — measurements may be unreliable.")
    
    # Grain pixel size
    if median_grain_pixels >= grain_px_good:
        scores.append(1.0)
    elif median_grain_pixels >= grain_px_fair:
        scores.append(0.7)
        reasons.append("Many grains occupy relatively few pixels.")
    elif median_grain_pixels >= grain_px_poor:
        scores.append(0.4)
        reasons.append("Many grains occupy very few pixels.")
    else:
        scores.append(0.1)
        reasons.append("Grains are extremely small in the image — measurements unreliable.")
    
    # Uncertain grains
    if uncertain_fraction > uncertain_warning:
        scores.append(0.5)
        reasons.append(f"Several grains appear merged ({uncertain_count} uncertain).")
    else:
        scores.append(1.0)
    
    # Illumination
    if illumination_uniformity < 0.5:
        scores.append(0.5)
        reasons.append("Lighting is highly uneven.")
    else:
        scores.append(min(1.0, illumination_uniformity + 0.2))
    
    # Clipping
    if clipped_dark > 0.1 or clipped_bright > 0.1:
        scores.append(0.6)
        reasons.append("Image has significant clipping (very dark or very bright regions).")
    else:
        scores.append(1.0)
    
    # Overall score
    overall_score = float(np.mean(scores))
    
    unreliable_reasons = []
    if blur_score < blur_poor:
        unreliable_reasons.append("Laplacian blur variance is below the configured poor-image floor.")
    if median_grain_pixels < grain_px_poor:
        unreliable_reasons.append("Median grain area is below the configured poor-image floor.")
    if total_count > 0 and mean_confidence < segmentation_confidence_floor:
        unreliable_reasons.append("Mean segmentation confidence is below the segmentation acceptance threshold.")
    if uncertain_fraction >= uncertain_unreliable:
        unreliable_reasons.append("At least half of the accepted grains are uncertain or merged.")
    if low_quality_segmentation_fraction >= low_quality_unreliable:
        unreliable_reasons.append("At least half of accepted instances have poor segmentation quality.")
    if max(clipped_dark, clipped_bright) >= clipping_unreliable:
        unreliable_reasons.append("At least half of image pixels are clipped at the dark or bright end.")
    if illumination_uniformity < illumination_unreliable:
        unreliable_reasons.append("Block illumination uniformity is below the severe-variation floor.")

    if unreliable_reasons:
        quality_level = "UNRELIABLE"
        reasons.extend(unreliable_reasons)
    elif overall_score >= 0.8:
        quality_level = "GOOD"
    elif overall_score >= 0.6:
        quality_level = "FAIR"
    else:
        quality_level = "POOR"
    
    return {
        "quality_level": quality_level,
        "quality_score": round(overall_score, 4),
        "blur_score": round(blur_score, 2),
        "median_grain_pixels": round(median_grain_pixels, 1),
        "mean_segmentation_confidence": round(mean_confidence, 4),
        "uncertain_grain_fraction": round(uncertain_fraction, 4),
        "illumination_uniformity": round(illumination_uniformity, 4),
        "clipped_dark_fraction": round(clipped_dark, 4),
        "clipped_bright_fraction": round(clipped_bright, 4),
        "clipping_basis": clipping_basis,
        "megapixels": round(megapixels, 2),
        "image_dimensions": [w, h],
        "reasons": reasons,
        "thresholds_used": {
            "blur_laplacian_variance": {
                "good_min": blur_good,
                "fair_min": blur_fair,
                "poor_min": blur_poor,
            },
            "median_grain_area_pixels": {
                "good_min": grain_px_good,
                "fair_min": grain_px_fair,
                "poor_min": grain_px_poor,
            },
            "uncertain_grain_fraction_warning_above": uncertain_warning,
            "segmentation_confidence_floor": segmentation_confidence_floor,
            "unreliable_hard_failures": {
                "blur_below": blur_poor,
                "median_grain_area_below_pixels": grain_px_poor,
                "mean_segmentation_confidence_below": segmentation_confidence_floor,
                "uncertain_fraction_at_least": uncertain_unreliable,
                "low_quality_segmentation_fraction_at_least": low_quality_unreliable,
                "clipped_pixel_fraction_at_least": clipping_unreliable,
                "illumination_uniformity_below": illumination_unreliable,
            },
            "low_quality_segmentation_count": low_quality_segmentation_count,
            "low_quality_segmentation_fraction": round(low_quality_segmentation_fraction, 4),
            "quality_score_tiers": {
                "good_min": 0.8,
                "fair_min": 0.6,
                "poor_below": 0.6,
                "unreliable_uses_explicit_hard_failures": True,
            },
            "mean_segmentation_confidence_used_for_tier": True,
            "segmentation_quality_used_for_tier": True,
            "megapixel_rejection_threshold": None,
            "provenance": "Engineering threshold — requires calibration against labeled image-quality judgments.",
        },
        "unreliable_reasons": unreliable_reasons,
        "_note": "Quality tiers and hard-failure thresholds are engineering heuristics requiring calibration against labeled human image-quality judgments; they are not scientifically established.",
    }
