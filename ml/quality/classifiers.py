"""
Defect classification module for rice grains.

Implements ML-based and heuristic fallback classifiers for:
- Damaged / Slightly Damaged (VGG-19 transfer learning or heuristic)
- Sprouted / Weevilled (ResNet-18 transfer learning or heuristic)
- Immature / Shrunken / Shrivelled (geometry + texture proxy)
- Chalky (Logistic Regression or heuristic — see texture.py)

Each classifier clearly labels its method and limitations.
"""

import logging
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from ml.config import get_model_info, get_project_root, get_threshold
from ml.segmentation.preprocessing import masked_grain_square_crop, pad_to_square

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy torch / torchvision import — imported only once, not per grain call.
# This avoids the ~0.5 s import overhead per grain during heuristic fallback.
# ---------------------------------------------------------------------------
_torch = None
_T = None


def _get_torch():
    """Return (torch, torchvision.transforms) importing once lazily."""
    global _torch, _T
    if _torch is None:
        import torch as _torch_mod
        import torchvision.transforms as _T_mod
        try:
            _torch_mod.set_num_threads(1)
        except Exception:
            pass
        _torch = _torch_mod
        _T = _T_mod
    return _torch, _T


# ============================================================
# DAMAGED / SLIGHTLY DAMAGED
# ============================================================

def classify_damaged(
    grain_rgb: np.ndarray,
    grain_mask: np.ndarray,
) -> Dict:
    """
    Classify grain as damaged/slightly damaged using VGG-19 transfer
    learning or heuristic fallback.
    
    VGG-19 uses 224x224 input with aspect-ratio-preserving padding
    (NOT stretching).
    """
    model_info = get_model_info("damaged")
    model_path = get_project_root() / model_info.get("weights_path", "")
    
    if model_path.exists() and model_info.get("status") in ("trained", "experimental"):
        return _classify_damaged_ml(grain_rgb, grain_mask, model_path)
    else:
        return _classify_damaged_heuristic(grain_rgb, grain_mask)


_LOADED_MODELS: Dict[str, Any] = {}  # module-level model cache

def _get_cached_model(key: str, model_path: Path, device):
    """Cache loaded models in memory for fast batched inference."""
    torch, _ = _get_torch()
    if key not in _LOADED_MODELS:
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        _LOADED_MODELS[key] = model
    return _LOADED_MODELS[key]


def _classify_damaged_ml(
    grain_rgb: np.ndarray,
    grain_mask: np.ndarray,
    model_path: Path,
) -> Dict:
    """ML-based damaged classification using VGG-19."""
    model_info = get_model_info("damaged")
    try:
        torch, T = _get_torch()
        
        # Match synthetic training's black exterior while excluding background pixels.
        padded = masked_grain_square_crop(grain_rgb, grain_mask, target_size=224)
        if padded is None:
            return _damaged_unavailable("No valid grain pixels")
        
        # To tensor
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor = transform(padded).unsqueeze(0)
        
        # Load cached model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _get_cached_model("damaged_vgg19", model_path, device)
        
        with torch.no_grad():
            output = model(tensor.to(device))
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            pred_prob = probs[0, pred_class].item()
        
        # Load class mapping
        import json
        mapping_path = model_path.parent / "class_mapping.json"
        if mapping_path.exists():
            with open(mapping_path) as f:
                class_mapping = json.load(f)
            label = class_mapping.get(str(pred_class), f"class_{pred_class}")
        else:
            label = "damaged" if pred_class == 1 else "normal"
        
        return {
            "damaged_label": label,
            "damaged_probability": round(pred_prob, 4),
            "confidence": round(pred_prob, 4),
            "confidence_basis": "softmax score for predicted class; not calibrated",
            "class_mapping": class_mapping,
            "model_status": model_info.get("status", "experimental"),
            "training_dataset": model_info.get("dataset"),
            "damaged_model_version": model_info.get("version", "1.0.0"),
            "method": "vgg19_transfer_learning",
            "_source": "experimental_synthetic_training",
            "limitation": "Checkpoint was trained on synthetic generated grain illustrations, not labeled rice grains; prediction and score are experimental and uncalibrated.",
        }
    except Exception as e:
        logger.warning(f"VGG-19 inference failed: {e}. Using heuristic fallback.")
        return _classify_damaged_heuristic(grain_rgb, grain_mask)


def _classify_damaged_heuristic(
    grain_rgb: np.ndarray,
    grain_mask: np.ndarray,
) -> Dict:
    """
    Heuristic fallback for damaged grain detection.
    Uses colour variance, dark spot ratio, and texture irregularity.
    Clearly labelled as heuristic.
    """
    from ml.quality.colour import extract_grain_crop
    crop_rgb, crop_mask = extract_grain_crop(grain_rgb, grain_mask)
    
    if crop_mask.sum() == 0:
        return _damaged_unavailable("No valid grain pixels")
    
    # Analyse colour variance as proxy for damage
    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB)
    grain_lab = lab[crop_mask > 0].astype(float)
    
    l_channel = grain_lab[:, 0]
    l_std = float(np.std(l_channel))
    l_mean = float(np.mean(l_channel))
    
    # Dark spots (potential damage)
    dark_threshold = max(50, l_mean - 2 * l_std)
    dark_fraction = float(np.sum(l_channel < dark_threshold) / len(l_channel))
    
    # Brown/discoloured patches
    a_channel = grain_lab[:, 1] - 128
    b_channel = grain_lab[:, 2] - 128
    colour_variance = float(np.std(a_channel) + np.std(b_channel))
    
    # Score
    score = 0.3 * min(1.0, l_std / 30.0) + 0.4 * dark_fraction + 0.3 * min(1.0, colour_variance / 20.0)
    
    label = "damaged" if score > 0.5 else "normal"
    
    return {
        "damaged_label": label,
        "damaged_probability": round(score, 4),
        "damaged_model_version": "heuristic_fallback_v1",
        "method": "heuristic — colour variance and dark spot analysis",
        "_source": "engineering_heuristic",
        "limitation": "Heuristic fallback — VGG-19 model not available. "
                      "Run: python training/train_damaged.py",
    }


def _damaged_unavailable(reason: str) -> Dict:
    return {
        "damaged_label": "unavailable",
        "damaged_probability": None,
        "damaged_model_version": None,
        "method": "unavailable",
        "reason": reason,
    }


def batch_classify_damaged(
    image_rgb: np.ndarray,
    grain_masks: List[np.ndarray],
    grain_crops: Optional[List[np.ndarray]] = None,
) -> List[Dict]:
    """
    Run damaged classification for all grains in a single batched forward pass.
    Falls back to per-grain heuristic if ML model is not available.

    This is ~50x faster than calling classify_damaged() per grain when a
    VGG-19 model is loaded, because it avoids 50 separate GPU/CPU dispatches.

    ``grain_crops`` optionally provides the per-grain image region matching
    each (cropped) mask.  When omitted, the masks are assumed to be
    full-frame masks aligned with ``image_rgb``.
    """

    def _image_for(index: int) -> np.ndarray:
        if grain_crops is not None and index < len(grain_crops):
            return grain_crops[index]
        return image_rgb

    model_info = get_model_info("damaged")
    model_path = get_project_root() / model_info.get("weights_path", "")

    if not (
        model_path.exists()
        and model_info.get("status") in ("trained", "experimental")
    ):
        return [
            _classify_damaged_heuristic(_image_for(i), m)
            for i, m in enumerate(grain_masks)
        ]

    try:
        torch, T = _get_torch()
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _get_cached_model("damaged_vgg19", model_path, device)

        # Load class mapping once
        import json
        mapping_path = model_path.parent / "class_mapping.json"
        class_mapping = {}
        if mapping_path.exists():
            with open(mapping_path) as f:
                class_mapping = json.load(f)

        # Preprocess all grains
        tensors = []
        valid_indices = []
        for i, mask in enumerate(grain_masks):
            padded = masked_grain_square_crop(_image_for(i), mask, target_size=224)
            if padded is None:
                continue
            tensors.append(transform(padded))
            valid_indices.append(i)

        if not tensors:
            return [_damaged_unavailable("No valid grain pixels")] * len(grain_masks)

        # Single batched forward pass
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            output = model(batch)
            probs = torch.softmax(output, dim=1)
            pred_classes = torch.argmax(probs, dim=1).tolist()
            pred_probs = probs[range(len(tensors)), pred_classes].tolist()

        # Build results
        results = [_damaged_unavailable("No valid grain pixels")] * len(grain_masks)
        for out_i, grain_i in enumerate(valid_indices):
            pred_class = pred_classes[out_i]
            pred_prob = pred_probs[out_i]
            label = class_mapping.get(str(pred_class), "damaged" if pred_class == 1 else "normal")
            results[grain_i] = {
                "damaged_label": label,
                "damaged_probability": round(pred_prob, 4),
                "confidence": round(pred_prob, 4),
                "confidence_basis": "softmax score for predicted class; not calibrated",
                "class_mapping": class_mapping,
                "model_status": model_info.get("status", "experimental"),
                "training_dataset": model_info.get("dataset"),
                "damaged_model_version": model_info.get("version", "1.0.0"),
                "method": "vgg19_transfer_learning",
                "_source": "experimental_synthetic_training",
                "limitation": "Checkpoint was trained on synthetic generated grain illustrations, not labeled rice grains; prediction and score are experimental and uncalibrated.",
            }
        return results
    except Exception as e:
        logger.warning(f"VGG-19 batch inference failed: {e}. Using per-grain heuristic.")
        return [
            _classify_damaged_heuristic(_image_for(i), m)
            for i, m in enumerate(grain_masks)
        ]


# ============================================================
# SPROUTED / WEEVILLED
# ============================================================

def _load_sprouted_class_mapping(model_path: Path) -> Dict[str, str]:
    """Load and log the class mapping stored alongside the checkpoint."""
    mapping_path = model_path.parent / "class_mapping.json"
    with mapping_path.open(encoding="utf-8") as mapping_file:
        mapping = json.load(mapping_file)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Sprouted/weevilled class mapping is empty or invalid")

    metadata_path = model_path.parent / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as metadata_file:
            metadata_mapping = json.load(metadata_file).get("classes")
        if metadata_mapping and metadata_mapping != mapping:
            raise ValueError("Checkpoint class mapping disagrees with model metadata")

    logger.info("Sprouted/weevilled checkpoint class mapping: %s", mapping)
    return {str(class_id): str(class_name) for class_id, class_name in mapping.items()}


def _normalize_sprouted_class(class_name: str) -> Optional[str]:
    normalized = re.sub(r"[^a-z0-9]+", "_", class_name.strip().lower()).strip("_")
    if normalized in {"normal", "no", "clean", "not_sprouted", "not_weevilled"}:
        return "normal"
    if normalized in {
        "sprouted_weevilled", "sprouted", "weevilled", "yes", "positive"
    }:
        return "sprouted_weevilled"
    return None


def _sprouted_decision_threshold(model_path: Path) -> float:
    metadata_path = model_path.parent / "metadata.json"
    if not metadata_path.exists():
        return 0.5
    with metadata_path.open(encoding="utf-8") as metadata_file:
        threshold = json.load(metadata_file).get("decision_threshold", 0.5)
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Sprouted/weevilled decision threshold must be between 0 and 1")
    return threshold


def _choose_sprouted_class(probabilities, class_mapping, threshold):
    positive_ids = [
        int(class_id)
        for class_id, name in class_mapping.items()
        if _normalize_sprouted_class(name) == "sprouted_weevilled"
    ]
    negative_ids = [
        int(class_id)
        for class_id, name in class_mapping.items()
        if _normalize_sprouted_class(name) == "normal"
    ]
    if len(positive_ids) != 1 or len(negative_ids) != 1:
        raise ValueError("Class mapping must identify exactly one positive and one normal class")

    positive_id = positive_ids[0]
    negative_id = negative_ids[0]
    positive_probability = float(probabilities[positive_id])
    predicted_id = positive_id if positive_probability >= threshold else negative_id
    confidence = float(probabilities[predicted_id])
    return predicted_id, confidence

def classify_sprouted_weevilled(
    grain_rgb: np.ndarray,
    grain_mask: np.ndarray,
) -> Dict:
    """
    Classify grain as sprouted/weevilled using ResNet-18 transfer
    learning or heuristic fallback.
    """
    model_info = get_model_info("sprouted_weevilled")
    model_path = get_project_root() / model_info.get("weights_path", "")
    
    if model_path.exists() and model_info.get("status") in ("trained", "experimental"):
        return _classify_sprouted_ml(grain_rgb, grain_mask, model_path)
    else:
        return _classify_sprouted_heuristic(grain_rgb, grain_mask)


def _classify_sprouted_ml(
    grain_rgb: np.ndarray,
    grain_mask: np.ndarray,
    model_path: Path,
) -> Dict:
    """ML-based sprouted/weevilled classification."""
    model_info = get_model_info("sprouted_weevilled")
    try:
        torch, T = _get_torch()
        class_mapping = _load_sprouted_class_mapping(model_path)
        
        padded = masked_grain_square_crop(grain_rgb, grain_mask, target_size=224)
        if padded is None:
            return _sprouted_unavailable("No valid pixels")
        
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor = transform(padded).unsqueeze(0)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _get_cached_model("sprouted_resnet18", model_path, device)
        
        with torch.no_grad():
            output = model(tensor.to(device))
            probs = torch.softmax(output, dim=1)
            decision_threshold = _sprouted_decision_threshold(model_path)
            pred_class, pred_prob = _choose_sprouted_class(
                probs[0].cpu().numpy(), class_mapping, decision_threshold
            )
        
        mapped_class = class_mapping.get(str(pred_class))
        label = _normalize_sprouted_class(mapped_class) if mapped_class else None
        if label is None:
            return _sprouted_unavailable(
                f"Checkpoint class {pred_class} has no recognized label mapping"
            )
        
        return {
            "sprouted_weevilled_label": label,
            "probability": round(pred_prob, 4),
            "confidence": round(pred_prob, 4),
            "confidence_basis": "softmax score for predicted class; not calibrated",
            "predicted_class_index": pred_class,
            "class_mapping": class_mapping,
            "decision_threshold": decision_threshold,
            "model_status": "experimental / uncalibrated",
            "model_version": model_info.get("version", "1.0.0"),
            "confidence_level": "low" if pred_prob < 0.7 else "moderate",
            "method": "resnet18_transfer_learning",
            "_source": "experimental_synthetic_training",
            "training_dataset": model_info.get("dataset"),
            "limitation": "Model was trained on synthetic demonstration crops. "
                         "Prediction and confidence are experimental and uncalibrated.",
        }
    except Exception as e:
        logger.warning(f"ResNet-18 inference failed: {e}. Using heuristic fallback.")
        return _classify_sprouted_heuristic(grain_rgb, grain_mask)


def _classify_sprouted_heuristic(
    grain_rgb: np.ndarray,
    grain_mask: np.ndarray,
) -> Dict:
    """
    Heuristic fallback for sprouted/weevilled detection.
    Uses texture irregularity and hole/cavity detection.
    """
    from ml.quality.colour import extract_grain_crop
    crop_rgb, crop_mask = extract_grain_crop(grain_rgb, grain_mask)
    
    if crop_mask.sum() == 0:
        return _sprouted_unavailable("No valid pixels")
    
    # Look for holes/cavities (weevil damage)
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    gray_masked = gray.copy()
    gray_masked[crop_mask == 0] = 128  # Neutral
    
    # Dark spots within grain (potential weevil holes)
    grain_pixels = gray[crop_mask > 0]
    mean_brightness = float(np.mean(grain_pixels))
    
    very_dark = np.sum(grain_pixels < mean_brightness * 0.5)
    dark_fraction = very_dark / len(grain_pixels) if len(grain_pixels) > 0 else 0
    
    # Texture irregularity
    laplacian = cv2.Laplacian(gray_masked, cv2.CV_64F)
    texture_var = float(np.var(laplacian[crop_mask > 0])) if crop_mask.sum() > 0 else 0
    
    score = 0.5 * min(1.0, dark_fraction * 5) + 0.5 * min(1.0, texture_var / 5000)
    
    return {
        "sprouted_weevilled_label": "sprouted_weevilled" if score > 0.5 else "normal",
        "probability": round(score, 4),
        "model_version": "heuristic_fallback_v1",
        "confidence_level": "low",
        "method": "heuristic — texture and dark spot analysis",
        "_source": "engineering_heuristic",
        "limitation": "Heuristic fallback — ResNet-18 model not available. "
                      "Run: python training/train_sprouted.py",
    }


def _sprouted_unavailable(reason: str) -> Dict:
    return {
        "sprouted_weevilled_label": "unavailable",
        "probability": None,
        "model_version": None,
        "confidence_level": None,
        "method": "unavailable",
        "reason": reason,
    }


def batch_classify_sprouted_weevilled(
    image_rgb: np.ndarray,
    grain_masks: List[np.ndarray],
    grain_crops: Optional[List[np.ndarray]] = None,
) -> List[Dict]:
    """
    Run sprouted/weevilled classification for all grains in a single batch pass.
    Falls back to per-grain heuristic if ML model is not available.

    ``grain_crops`` optionally provides the per-grain image region matching
    each (cropped) mask.
    """

    def _image_for(index: int) -> np.ndarray:
        if grain_crops is not None and index < len(grain_crops):
            return grain_crops[index]
        return image_rgb

    model_info = get_model_info("sprouted_weevilled")
    model_path = get_project_root() / model_info.get("weights_path", "")

    if not (
        model_path.exists()
        and model_info.get("status") in ("trained", "experimental")
    ):
        return [
            _classify_sprouted_heuristic(_image_for(i), m)
            for i, m in enumerate(grain_masks)
        ]

    try:
        torch, T = _get_torch()
        class_mapping = _load_sprouted_class_mapping(model_path)
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _get_cached_model("sprouted_resnet18", model_path, device)

        decision_threshold = _sprouted_decision_threshold(model_path)
        tensors = []
        valid_indices = []
        for i, mask in enumerate(grain_masks):
            padded = masked_grain_square_crop(_image_for(i), mask, target_size=224)
            if padded is None:
                continue
            tensors.append(transform(padded))
            valid_indices.append(i)

        if not tensors:
            return [_sprouted_unavailable("No valid grain pixels")] * len(grain_masks)

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            output = model(batch)
            probs = torch.softmax(output, dim=1)
            predictions = [
                _choose_sprouted_class(probability, class_mapping, decision_threshold)
                for probability in probs.cpu().numpy()
            ]

        results = [_sprouted_unavailable("No valid grain pixels")] * len(grain_masks)
        for out_i, grain_i in enumerate(valid_indices):
            pred_class, pred_prob = predictions[out_i]
            mapped_class = class_mapping.get(str(pred_class))
            label = _normalize_sprouted_class(mapped_class) if mapped_class else None
            if label is None:
                results[grain_i] = _sprouted_unavailable(
                    f"Checkpoint class {pred_class} has no recognized label mapping"
                )
                continue
            results[grain_i] = {
                "sprouted_weevilled_label": label,
                "probability": round(pred_prob, 4),
                "confidence": round(pred_prob, 4),
                "confidence_basis": "softmax score for predicted class; not calibrated",
                "predicted_class_index": pred_class,
                "class_mapping": class_mapping,
                "decision_threshold": decision_threshold,
                "model_status": "experimental / uncalibrated",
                "model_version": model_info.get("version", "1.0.0"),
                "confidence_level": "low" if pred_prob < 0.7 else "moderate",
                "method": "resnet18_transfer_learning",
                "_source": "experimental_synthetic_training",
                "training_dataset": model_info.get("dataset"),
                "limitation": "Model was trained on synthetic demonstration crops. "
                              "Prediction and confidence are experimental and uncalibrated.",
            }
        return results
    except Exception as e:
        logger.warning(f"ResNet-18 batch inference failed: {e}. Using per-grain heuristic.")
        return [
            _classify_sprouted_heuristic(_image_for(i), m)
            for i, m in enumerate(grain_masks)
        ]


# ============================================================
# IMMATURE / SHRUNKEN / SHRIVELLED
# ============================================================

def classify_immature_shrunken(
    grain_geometry: Dict,
    population_stats: Optional[Dict] = None,
) -> Dict:
    """
    Classify grain as immature/shrunken/shrivelled.
    
    Uses geometry + texture as an experimental proxy:
    - Low breadth relative to reference
    - Low area
    - Low solidity
    - Abnormal L/B
    
    Thresholds are configurable.
    Does NOT claim to be a government-certified image threshold.
    """
    if population_stats is None:
        return {
            "immature_shrunken_status": "undetermined",
            "proxy_score": None,
            "confidence": 0.0,
            "method": "experimental geometry/texture proxy",
            "reason": "No population reference available",
        }
    
    breadth = grain_geometry.get("breadth_pixels", 0)
    area = grain_geometry.get("area_pixels", 0)
    solidity = grain_geometry.get("solidity", 1.0)
    lb_ratio = grain_geometry.get("lb_ratio", 0)
    
    ref_breadth = population_stats.get("median_breadth", 1)
    ref_area = population_stats.get("median_area", 1)
    
    breadth_thresh = get_threshold("immature_shrunken", "breadth_ratio_threshold", 0.7)
    area_thresh = get_threshold("immature_shrunken", "area_ratio_threshold", 0.5)
    solidity_thresh = get_threshold("immature_shrunken", "solidity_threshold", 0.85)
    
    # Compute ratios
    breadth_ratio = breadth / ref_breadth if ref_breadth > 0 else 1.0
    area_ratio = area / ref_area if ref_area > 0 else 1.0
    
    # Score
    score = 0.0
    reasons = []
    
    if breadth_ratio < breadth_thresh:
        score += 0.35
        reasons.append(f"Low breadth ratio: {breadth_ratio:.2f}")
    
    if area_ratio < area_thresh:
        score += 0.35
        reasons.append(f"Low area ratio: {area_ratio:.2f}")
    
    if solidity < solidity_thresh:
        score += 0.15
        reasons.append(f"Low solidity: {solidity:.3f}")
    
    if lb_ratio and lb_ratio > 5.0:
        score += 0.15
        reasons.append(f"Abnormal L/B: {lb_ratio:.2f}")
    
    status = "immature_shrunken" if score > 0.5 else "normal"
    
    return {
        "immature_shrunken_status": status,
        "proxy_score": round(score, 4),
        "confidence": round(min(score, 1.0), 4),
        "method": "experimental geometry/texture proxy",
        "_source": "engineering_heuristic",
        "reasons": reasons,
        "limitation": "Experimental image-based parameter — no government-certified threshold exists",
    }
