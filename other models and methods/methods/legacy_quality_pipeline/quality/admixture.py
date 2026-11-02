"""
Geometry outlier diagnostic for rice grain samples.

This is a SAMPLE-LEVEL parameter — NOT classified per individual grain.

Uses statistical outlier detection on grain geometry:
- Length, Breadth, L/B, Area, Shape descriptors
- Robust Mahalanobis distance
- Excludes broken grains and uncertain segmentation first

This is NOT a lower-class grain classifier. A geometric outlier does not
establish variety/class identity, so the official/project admixture output
remains unsupported until labeled definitions and data are available.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from ml.config import get_threshold

logger = logging.getLogger(__name__)


def detect_admixture(
    grain_geometries: List[Dict],
    broken_labels: List[str],
    segmentation_qualities: List[str],
) -> Dict:
    """
    Compute a diagnostic geometry-outlier score without asserting admixture.
    
    Sample-level parameter. Runs only when there are enough valid grains.
    First excludes broken grains and uncertain segmentation.
    
    Args:
        grain_geometries: List of grain geometry dicts
        broken_labels: List of broken status per grain ('broken', 'whole', 'undetermined')
        segmentation_qualities: List of segmentation quality per grain
        
    Returns an unsupported admixture status and a separate outlier diagnostic.
    """
    min_grains = get_threshold("admixture", "min_grains_for_analysis", 10)
    mahalanobis_thresh = get_threshold("admixture", "mahalanobis_threshold", 3.0)
    
    # Filter to whole, well-segmented grains only
    valid_indices = []
    for i, (broken, seg_q) in enumerate(zip(broken_labels, segmentation_qualities)):
        if broken in ("whole", "undetermined") and seg_q in ("good", "limited"):
            valid_indices.append(i)
    
    n_valid = len(valid_indices)
    
    if n_valid < min_grains:
        return {
            "admixture_status": "unsupported",
            "admixture_count": None,
            "admixture_percentage": None,
            "admixture_confidence": None,
            "geometry_outlier_status": "insufficient_data",
            "geometry_outlier_count": None,
            "geometry_outlier_fraction": None,
            "valid_grains_used": n_valid,
            "min_required": min_grains,
            "method": "geometry_outlier_diagnostic",
            "_source": "engineering_heuristic",
            "reason": "Geometric outliers do not identify lower-class grains; labeled class evidence is unavailable.",
            "diagnostic_reason": f"Too few valid whole grains ({n_valid}) for the geometry diagnostic; configured project minimum is {min_grains}.",
            "scope": "sample_level",
        }
    
    # Extract features for valid grains
    features = []
    for i in valid_indices:
        g = grain_geometries[i]
        features.append([
            g.get("length_pixels", 0),
            g.get("breadth_pixels", 0),
            g.get("lb_ratio", 0) or 0,
            g.get("area_pixels", 0),
            g.get("solidity", 0),
        ])
    
    features = np.array(features, dtype=float)
    
    if features.shape[0] < 3 or features.shape[1] < 2:
        return {
            "admixture_status": "unsupported",
            "admixture_count": None,
            "admixture_percentage": None,
            "admixture_confidence": None,
            "geometry_outlier_status": "insufficient_data",
            "geometry_outlier_count": None,
            "geometry_outlier_fraction": None,
            "method": "geometry_outlier_diagnostic",
            "reason": "Geometric outliers do not identify lower-class grains; labeled class evidence is unavailable.",
            "diagnostic_reason": "Insufficient feature dimensions for geometry outlier diagnostic.",
            "scope": "sample_level",
        }
    
    # Compute robust Mahalanobis distance
    try:
        mean = np.mean(features, axis=0)
        cov = np.cov(features.T)
        
        # Regularize covariance
        cov += np.eye(cov.shape[0]) * 1e-6
        
        cov_inv = np.linalg.inv(cov)
        
        distances = []
        for f in features:
            diff = f - mean
            d = float(np.sqrt(diff @ cov_inv @ diff))
            distances.append(d)
        
        distances = np.array(distances)
        
        # Outliers are geometry diagnostics, not identified lower-class grains.
        outlier_mask = distances > mahalanobis_thresh
        outlier_count = int(np.sum(outlier_mask))
        outlier_percentage = (outlier_count / n_valid * 100) if n_valid > 0 else 0.0
        
        # Confidence based on sample size and separation
        confidence = min(1.0, n_valid / 50.0) * min(1.0, np.mean(distances[outlier_mask]) / mahalanobis_thresh if outlier_count > 0 else 0.5)
        
        return {
            "admixture_status": "unsupported",
            "admixture_count": None,
            "admixture_percentage": None,
            "admixture_confidence": None,
            "geometry_outlier_status": "diagnostic_only",
            "geometry_outlier_count": outlier_count,
            "geometry_outlier_fraction": round(outlier_percentage / 100.0, 4),
            "geometry_outlier_indices": [valid_indices[i] for i in range(len(outlier_mask)) if outlier_mask[i]],
            "geometry_outlier_diagnostic_score": round(float(confidence), 4),
            "valid_grains_used": n_valid,
            "total_grains": len(grain_geometries),
            "mahalanobis_threshold": mahalanobis_thresh,
            "method": "geometry_outlier_diagnostic",
            "_source": "engineering_heuristic",
            "basis": "Geometric outlier diagnostic only; not lower-class admixture evidence",
            "reason": "No supported mapping from geometric outlier to lower-class grain is available.",
            "scope": "sample_level",
        }
        
    except np.linalg.LinAlgError:
        return {
            "admixture_status": "unsupported",
            "admixture_count": None,
            "admixture_percentage": None,
            "admixture_confidence": None,
            "geometry_outlier_status": "insufficient_data",
            "geometry_outlier_count": None,
            "geometry_outlier_fraction": None,
            "method": "geometry_outlier_diagnostic",
            "reason": "Geometric outliers do not identify lower-class grains; labeled class evidence is unavailable.",
            "diagnostic_reason": "Covariance matrix is singular; geometry diagnostic is unavailable.",
            "scope": "sample_level",
        }
