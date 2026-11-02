"""
Texture feature extraction for rice grain quality analysis.

Implements GLCM (Gray-Level Co-occurrence Matrix) features for chalkiness detection
and general texture analysis.

Features extracted:
- GLCM contrast, homogeneity, energy, correlation
- LAB brightness features (mean L*, std L*, bright-pixel fraction)
- Chalky pixel fraction
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ml.config import get_model_info, get_project_root, get_threshold

logger = logging.getLogger(__name__)
_CHALKY_ARTIFACTS = None


def compute_glcm(
    gray_image: np.ndarray,
    distances: List[int] = None,
    angles: List[float] = None,
    levels: int = 256,
    mask: np.ndarray = None,
) -> np.ndarray:
    """
    Compute GLCM manually (avoiding skimage dependency for basic version).
    Uses a simplified approach for the key features.
    
    For production, use skimage.feature.graycomatrix.
    """
    if distances is None:
        distances = [1, 3]
    if angles is None:
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    valid_mask = mask > 0 if mask is not None else np.ones(gray_image.shape, dtype=bool)
    valid_pixels = gray_image[valid_mask]
    if valid_pixels.size == 0:
        return np.zeros((min(levels, 64), min(levels, 64)), dtype=np.float64)

    # Quantize from valid pixels only so background intensity cannot set the scale.
    n_levels = min(levels, 64)
    min_value = float(valid_pixels.min())
    value_range = float(valid_pixels.max()) - min_value
    if value_range > 0:
        quantized = (
            (gray_image.astype(float) - min_value) / value_range * (n_levels - 1)
        ).clip(0, n_levels - 1).astype(np.uint8)
    else:
        quantized = np.zeros(gray_image.shape, dtype=np.uint8)
    
    h, w = quantized.shape
    glcm = np.zeros((n_levels, n_levels), dtype=np.float64)
    
    for d in distances:
        for angle in angles:
            dx = int(round(d * np.cos(angle)))
            dy = int(round(-d * np.sin(angle)))
            
            y1, y2 = max(0, -dy), min(h, h - dy)
            x1, x2 = max(0, -dx), min(w, w - dx)
            if y2 <= y1 or x2 <= x1:
                continue
                
            m1 = valid_mask[y1:y2, x1:x2]
            m2 = valid_mask[y1 + dy:y2 + dy, x1 + dx:x2 + dx]
            valid = m1 & m2
            if np.any(valid):
                r_vals = quantized[y1:y2, x1:x2][valid]
                c_vals = quantized[y1 + dy:y2 + dy, x1 + dx:x2 + dx][valid]
                np.add.at(glcm, (r_vals, c_vals), 1.0)
    
    # Normalize
    total = glcm.sum()
    if total > 0:
        glcm = glcm / total
    
    return glcm


def glcm_features(glcm: np.ndarray) -> Dict[str, float]:
    """Extract GLCM properties: contrast, homogeneity, energy, correlation."""
    n = glcm.shape[0]
    i_indices, j_indices = np.meshgrid(range(n), range(n), indexing='ij')
    i_indices = i_indices.astype(float)
    j_indices = j_indices.astype(float)
    
    # Contrast
    contrast = float(np.sum(glcm * (i_indices - j_indices) ** 2))
    
    # Homogeneity (Inverse Difference Moment)
    homogeneity = float(np.sum(glcm / (1.0 + np.abs(i_indices - j_indices))))
    
    # Energy (Angular Second Moment)
    energy = float(np.sum(glcm ** 2))
    
    # Correlation
    mu_i = float(np.sum(i_indices * glcm))
    mu_j = float(np.sum(j_indices * glcm))
    sigma_i = float(np.sqrt(np.sum(glcm * (i_indices - mu_i) ** 2)))
    sigma_j = float(np.sqrt(np.sum(glcm * (j_indices - mu_j) ** 2)))
    
    correlation = 0.0
    if sigma_i > 0 and sigma_j > 0:
        correlation = float(
            np.sum(glcm * (i_indices - mu_i) * (j_indices - mu_j)) / (sigma_i * sigma_j)
        )
    
    return {
        "glcm_contrast": round(contrast, 4),
        "glcm_homogeneity": round(homogeneity, 4),
        "glcm_energy": round(energy, 6),
        "glcm_correlation": round(correlation, 4),
    }


def extract_chalky_features(
    grain_rgb: np.ndarray,
    grain_mask: np.ndarray,
) -> Dict[str, float]:
    """
    Extract features for chalkiness detection.
    
    Features:
    - LAB: mean L*, std L*, mean a*, mean b*
    - Brightness: bright-pixel fraction, local brightness stats
    - GLCM: contrast, homogeneity, energy, correlation
    - Mask: chalky pixel fraction, internal texture stats
    """
    from ml.quality.colour import extract_grain_crop
    
    crop_rgb, crop_mask = extract_grain_crop(grain_rgb, grain_mask)
    
    if crop_mask.sum() == 0:
        return _empty_chalky_features()
    
    # LAB features
    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB)
    grain_lab = lab[crop_mask > 0].astype(float)
    
    l_star = grain_lab[:, 0]
    a_star = grain_lab[:, 1] - 128  # Center a*
    b_star = grain_lab[:, 2] - 128  # Center b*
    
    mean_l = float(np.mean(l_star))
    std_l = float(np.std(l_star))
    mean_a = float(np.mean(a_star))
    mean_b = float(np.mean(b_star))
    
    # Brightness statistics are descriptive features, not a chalkiness decision.
    bright_threshold = get_threshold("chalky", "bright_pixel_l_threshold", 85)
    bright_pixels = np.sum(l_star > bright_threshold)
    total_pixels = len(l_star)
    bright_fraction = bright_pixels / total_pixels if total_pixels > 0 else 0.0
    
    # GLCM pairs are counted only when both pixels belong to the grain.
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    glcm = compute_glcm(gray, mask=crop_mask)
    glcm_feats = glcm_features(glcm)
    
    # Chalky pixel fraction estimate
    # Chalky regions appear white/opaque — high L*, low saturation
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    grain_hsv = hsv[crop_mask > 0]
    low_sat = grain_hsv[:, 1] < 40
    high_val = grain_hsv[:, 2] > 180
    chalky_pixels = np.sum(low_sat & high_val)
    chalky_fraction = chalky_pixels / total_pixels if total_pixels > 0 else 0.0
    
    features = {
        "mean_l_star": round(mean_l, 4),
        "std_l_star": round(std_l, 4),
        "mean_a_star": round(mean_a, 4),
        "mean_b_star": round(mean_b, 4),
        "bright_pixel_fraction": round(bright_fraction, 4),
        "chalky_pixel_fraction": round(float(chalky_fraction), 4),
        **glcm_feats,
    }
    
    return features


def _empty_chalky_features() -> Dict[str, float]:
    return {
        "mean_l_star": 0.0,
        "std_l_star": 0.0,
        "mean_a_star": 0.0,
        "mean_b_star": 0.0,
        "bright_pixel_fraction": 0.0,
        "chalky_pixel_fraction": 0.0,
        "glcm_contrast": 0.0,
        "glcm_homogeneity": 0.0,
        "glcm_energy": 0.0,
        "glcm_correlation": 0.0,
    }


def heuristic_chalky_classification(features: Dict[str, float]) -> Dict:
    """
    Predict only from a real-data-trained artifact; otherwise abstain.

    Brightness alone is not evidence of chalkiness. Model inputs are the
    mask-normalized grain-level features produced above.
    """
    global _CHALKY_ARTIFACTS
    model_info = get_model_info("chalky")
    weights_path = Path(model_info.get("weights_path", "models/chalky/model.joblib"))
    model_dir = get_project_root() / weights_path.parent
    metadata_path = model_dir / "metadata.json"
    try:
        if _CHALKY_ARTIFACTS is None:
            import joblib

            with metadata_path.open(encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            if metadata.get("status") != "trained":
                raise ValueError("No validated real-grain chalky model is available")
            with (model_dir / "feature_schema.json").open(encoding="utf-8") as schema_file:
                feature_names = json.load(schema_file)["features"]
            _CHALKY_ARTIFACTS = (
                joblib.load(model_dir / "model.joblib"),
                joblib.load(model_dir / "scaler.joblib"),
                feature_names,
                metadata,
            )

        classifier, scaler, feature_names, metadata = _CHALKY_ARTIFACTS
        class_mapping = metadata.get("classes", {})
        positive_ids = [
            str(class_id)
            for class_id, label in class_mapping.items()
            if str(label).strip().lower() == "chalky"
        ]
        if len(positive_ids) != 1:
            raise ValueError("Model metadata must map exactly one class to chalky")

        feature_vector = np.asarray([[features[name] for name in feature_names]], dtype=float)
        probabilities = classifier.predict_proba(scaler.transform(feature_vector))[0]
        positive_index = next(
            index
            for index, class_id in enumerate(classifier.classes_)
            if str(class_id) == positive_ids[0]
        )
        chalky_probability = float(probabilities[positive_index])
        threshold = float(metadata["decision_threshold"])
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Chalky decision threshold must be between 0 and 1")
        label = "chalky" if chalky_probability >= threshold else "not_chalky"
        return {
            "chalky_label": label,
            "chalky_probability": round(chalky_probability, 4),
            "confidence": round(max(chalky_probability, 1.0 - chalky_probability), 4),
            "probability_calibrated": False,
            "decision_threshold": threshold,
            "threshold_source": metadata.get("threshold_source", "validation set"),
            "chalky_model_version": metadata.get("version", "unknown"),
            "training_dataset": metadata.get("dataset"),
            "classes": class_mapping,
            "method": "grain-level Logistic Regression on instance-mask LAB/GLCM features",
            "_source": "trained_grain_level_classifier",
            "limitation": "Raw model probability is not calibrated; use as an experimental grain-level prediction.",
        }
    except (FileNotFoundError, KeyError, ValueError, OSError, AttributeError) as error:
        logger.info("Chalky classifier abstained: %s", error)

    return {
        "chalky_label": "undetermined",
        "chalky_probability": None,
        "confidence": None,
        "probability_calibrated": False,
        "chalky_model_version": "unvalidated_synthetic_model_disabled",
        "method": "unavailable — validated grain-level classifier required",
        "_source": "unvalidated_model_disabled",
        "limitation": (
            "The available Logistic Regression was trained on synthetic feature vectors, "
            "not labeled rice-grain images; its probability is not reported. "
            "Mask-normalized features are retained for future validated training."
        ),
    }
