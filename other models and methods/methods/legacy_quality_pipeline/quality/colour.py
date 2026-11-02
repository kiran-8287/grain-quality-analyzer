"""
Colour analysis module for rice grain quality parameters.

Implements:
- Red grain detection (LAB + HSV colour analysis)
- Discoloured grain detection (DeltaE colour distance)
- Dehusked grain estimation (visual bran-coverage proxy)

All colour analysis preserves original colour information.
Never converts to grayscale before colour analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ml.config import get_threshold

logger = logging.getLogger(__name__)


def extract_grain_crop(
    image_rgb: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the grain region using mask.
    Returns (cropped_rgb, cropped_mask) — both cropped to bounding box.
    """
    coords = cv2.findNonZero(mask)
    if coords is None:
        return image_rgb, mask
    x, y, w, h = cv2.boundingRect(coords)
    crop_rgb = image_rgb[y:y+h, x:x+w].copy()
    crop_mask = mask[y:y+h, x:x+w].copy()
    return crop_rgb, crop_mask


def _masked_pixels(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Get pixels where mask is nonzero."""
    return image[mask > 0]


# ============================================================
# RED GRAIN DETECTION
# ============================================================

def analyze_red(
    grain_rgb: np.ndarray,
    grain_mask: np.ndarray,
) -> Dict:
    """
    Detect red grain using LAB and HSV colour analysis.
    
    Uses:
    - Red/pigmented area fraction
    - a* statistics from LAB
    - Hue/saturation statistics from HSV
    - Spatial distribution
    
    Official concept: red grain classification concerns surface area
    covered by red cuticle.
    
    Thresholds are configurable. Does not claim to exactly reproduce
    the official laboratory method.
    """
    crop_rgb, crop_mask = extract_grain_crop(grain_rgb, grain_mask)
    
    if crop_mask.sum() == 0:
        return _red_unavailable("No valid pixels in mask")
    
    # Load thresholds
    thresholds = get_threshold("red", "a_star_mean_threshold", 10.0)
    red_area_thresh = get_threshold("red", "red_area_fraction_threshold", 0.15)
    
    # Convert to LAB
    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB)
    grain_lab = lab[crop_mask > 0]
    
    a_star = grain_lab[:, 1].astype(float) - 128  # a* channel (centered)
    a_mean = float(np.mean(a_star))
    a_std = float(np.std(a_star))
    
    # Convert to HSV
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    grain_hsv = hsv[crop_mask > 0]
    
    # Red hue masks (HSV red wraps around 0/180)
    h, s, v = grain_hsv[:, 0], grain_hsv[:, 1], grain_hsv[:, 2]
    
    red_low = get_threshold("red", "hue_red_low", [0, 50, 50])
    red_high = get_threshold("red", "hue_red_high", [10, 255, 255])
    red_low2 = get_threshold("red", "hue_red_low2", [170, 50, 50])
    red_high2 = get_threshold("red", "hue_red_high2", [180, 255, 255])
    
    red_mask1 = (h >= red_low[0]) & (h <= red_high[0]) & (s >= red_low[1]) & (v >= red_low[2])
    red_mask2 = (h >= red_low2[0]) & (h <= red_high2[0]) & (s >= red_low2[1]) & (v >= red_low2[2])
    red_pixels = np.sum(red_mask1 | red_mask2)
    total_pixels = len(grain_hsv)
    
    red_area_fraction = red_pixels / total_pixels if total_pixels > 0 else 0.0
    
    # Combined confidence
    a_score = max(0, min(1, (a_mean - 5) / 15))  # Normalize a* contribution
    area_score = min(1.0, red_area_fraction / red_area_thresh) if red_area_thresh > 0 else 0.0
    confidence = 0.5 * a_score + 0.5 * area_score
    
    is_red = red_area_fraction >= red_area_thresh and a_mean > thresholds
    
    return {
        "red_label": "red" if is_red else "not_red",
        "red_area_fraction": round(red_area_fraction, 4),
        "red_confidence": round(confidence, 4),
        "a_star_mean": round(a_mean, 2),
        "a_star_std": round(a_std, 2),
        "red_pixel_count": int(red_pixels),
        "total_grain_pixels": int(total_pixels),
        "red_method": "image-based colour proxy",
        "_source": "engineering_heuristic",
    }


def _red_unavailable(reason: str) -> Dict:
    return {
        "red_label": "unavailable",
        "red_area_fraction": None,
        "red_confidence": 0.0,
        "red_method": "image-based colour proxy",
        "reason": reason,
    }


# ============================================================
# DISCOLOURED GRAIN DETECTION
# ============================================================

def compute_reference_lab(
    all_grain_labs: List[np.ndarray],
) -> Optional[np.ndarray]:
    """
    Build a reference LAB profile from all grains in the sample.
    Uses the median LAB values as the 'normal' reference.
    """
    if not all_grain_labs:
        return None
    all_pixels = np.concatenate(all_grain_labs, axis=0)
    return np.median(all_pixels, axis=0).astype(float)


def analyze_discoloured(
    grain_rgb: np.ndarray,
    grain_mask: np.ndarray,
    reference_lab: Optional[np.ndarray] = None,
) -> Dict:
    """
    Detect discoloured grain using LAB colour-distance analysis (DeltaE).
    
    Prefers reference-adaptive behaviour rather than hard-coded HSV bands.
    Because lighting changes across photographs, avoids brittle absolute
    RGB thresholds.
    """
    crop_rgb, crop_mask = extract_grain_crop(grain_rgb, grain_mask)
    
    if crop_mask.sum() == 0:
        return _discoloured_unavailable("No valid pixels")
    
    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB)
    grain_lab = lab[crop_mask > 0].astype(float)
    
    grain_mean_lab = np.mean(grain_lab, axis=0)
    
    if reference_lab is None:
        # Use the grain's own mean as a rough self-reference
        # This means discolouration can't be detected without a reference
        return {
            "discoloured_label": "undetermined",
            "colour_distance": None,
            "discoloured_confidence": 0.0,
            "method": "lab_colour_distance",
            "reason": "No reference profile available for comparison",
        }
    
    # Compute DeltaE (CIE76 — Euclidean distance in LAB space)
    delta_e = float(np.sqrt(np.sum((grain_mean_lab - reference_lab) ** 2)))
    
    threshold = get_threshold("discoloured", "delta_e_threshold", 15.0)
    area_thresh = get_threshold("discoloured", "discoloured_area_fraction_threshold", 0.2)
    
    # Per-pixel DeltaE for spatial analysis
    pixel_deltas = np.sqrt(np.sum((grain_lab - reference_lab) ** 2, axis=1))
    discoloured_pixels = np.sum(pixel_deltas > threshold)
    discoloured_fraction = discoloured_pixels / len(grain_lab) if len(grain_lab) > 0 else 0
    
    is_discoloured = discoloured_fraction >= area_thresh
    confidence = min(1.0, discoloured_fraction / area_thresh) if area_thresh > 0 else 0.0
    
    return {
        "discoloured_label": "discoloured" if is_discoloured else "normal",
        "colour_distance": round(delta_e, 2),
        "discoloured_area_fraction": round(float(discoloured_fraction), 4),
        "discoloured_confidence": round(confidence, 4),
        "method": "lab_colour_distance",
        "_source": "engineering_heuristic",
    }


def _discoloured_unavailable(reason: str) -> Dict:
    return {
        "discoloured_label": "unavailable",
        "colour_distance": None,
        "discoloured_confidence": 0.0,
        "method": "lab_colour_distance",
        "reason": reason,
    }


# ============================================================
# DEHUSKED GRAIN ESTIMATION
# ============================================================

def analyze_dehusked(
    grain_rgb: np.ndarray,
    grain_mask: np.ndarray,
) -> Dict:
    """
    Estimate dehusked grain status using visual bran-coverage proxy.
    
    IMPORTANT SCIENTIFIC LIMITATION:
    The official dehusked-grain test involves staining/chemical analysis.
    A normal RGB internet image CANNOT reproduce that laboratory test exactly.
    
    Uses:
    - Brown/bran-like surface coverage
    - Colour distribution (HSV brown hue range)
    - Texture features
    
    Official conceptual threshold: more than one-quarter of the kernel
    surface area covered with bran.
    """
    crop_rgb, crop_mask = extract_grain_crop(grain_rgb, grain_mask)
    
    if crop_mask.sum() == 0:
        return _dehusked_unavailable("No valid pixels")
    
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    grain_hsv = hsv[crop_mask > 0]
    
    # Brown/bran hue detection
    brown_low = get_threshold("dehusked", "brown_hue_low", [10, 40, 40])
    brown_high = get_threshold("dehusked", "brown_hue_high", [30, 255, 200])
    bran_thresh = get_threshold("dehusked", "bran_coverage_threshold", 0.25)
    
    h, s, v = grain_hsv[:, 0], grain_hsv[:, 1], grain_hsv[:, 2]
    brown_mask = (
        (h >= brown_low[0]) & (h <= brown_high[0]) &
        (s >= brown_low[1]) & (s <= brown_high[1]) &
        (v >= brown_low[2]) & (v <= brown_high[2])
    )
    
    brown_pixels = np.sum(brown_mask)
    total_pixels = len(grain_hsv)
    surface_fraction = brown_pixels / total_pixels if total_pixels > 0 else 0.0
    
    is_dehusked = surface_fraction >= bran_thresh
    confidence = min(1.0, surface_fraction / bran_thresh) if bran_thresh > 0 else 0.0
    
    return {
        "dehusked_label": "dehusked" if is_dehusked else "not_dehusked",
        "dehusked_surface_fraction": round(surface_fraction, 4),
        "dehusked_confidence": round(confidence, 4),
        "method": "visual proxy",
        "limitation": "Image-based visual estimate — not equivalent to the official staining test",
        "_source": "official concept threshold; method is experimental visual proxy",
    }


def _dehusked_unavailable(reason: str) -> Dict:
    return {
        "dehusked_label": "unavailable",
        "dehusked_surface_fraction": None,
        "dehusked_confidence": 0.0,
        "method": "visual proxy",
        "reason": reason,
    }


def extract_grain_lab_pixels(
    grain_rgb: np.ndarray,
    grain_mask: np.ndarray,
) -> Optional[np.ndarray]:
    """Extract LAB pixels for a grain (used for building reference profiles)."""
    crop_rgb, crop_mask = extract_grain_crop(grain_rgb, grain_mask)
    if crop_mask.sum() == 0:
        return None
    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB)
    return lab[crop_mask > 0].astype(float)
