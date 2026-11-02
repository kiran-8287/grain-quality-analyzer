"""
Configuration loader for the rice quality analysis system.
Loads JSON configs from the configs/ directory.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

_CONFIG_CACHE: Dict[str, Any] = {}

CONFIDENCE_HIGH_THRESHOLD: float = 0.80
CONFIDENCE_MEDIUM_THRESHOLD: float = 0.50
CONFIDENCE_THRESHOLDS: Dict[str, float] = {
    "high": CONFIDENCE_HIGH_THRESHOLD,
    "medium": CONFIDENCE_MEDIUM_THRESHOLD,
}

def _get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent

def load_config(name: str, force_reload: bool = False) -> Dict[str, Any]:
    """
    Load a JSON config file by name (without extension).
    
    Args:
        name: Config file name without .json extension (e.g., 'app', 'thresholds')
        force_reload: If True, bypass cache
        
    Returns:
        Parsed config dictionary
    """
    if name in _CONFIG_CACHE and not force_reload:
        return _CONFIG_CACHE[name]
    
    config_path = _get_project_root() / "configs" / f"{name}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    _CONFIG_CACHE[name] = config
    logger.debug(f"Loaded config: {name} from {config_path}")
    return config

def load_standards(name: str = "india_kms_2026_27_raw_rice") -> Dict[str, Any]:
    """Load a standards JSON file."""
    standards_path = _get_project_root() / "standards" / f"{name}.json"
    if not standards_path.exists():
        raise FileNotFoundError(
            f"Standards file not found: {standards_path}. "
            "Please supply the standards JSON manually."
        )
    
    with open(standards_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_threshold(category: str, key: str, default: Any = None) -> Any:
    """Get a specific threshold value."""
    thresholds = load_config("thresholds")
    return thresholds.get(category, {}).get(key, default)

def get_inference_config(model_name: str) -> Dict[str, Any]:
    """Get inference config for a specific model."""
    inference = load_config("inference")
    return inference.get(model_name, {})

def get_model_registry() -> Dict[str, Any]:
    """Get the full model registry."""
    return load_config("models")

def load_model_registry() -> Dict[str, Any]:
    """Alias for get_model_registry."""
    return get_model_registry()

def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get registry info for a specific model."""
    registry = get_model_registry()
    return registry.get(model_name, {})

def get_project_root() -> Path:
    """Public accessor for project root."""
    return _get_project_root()
