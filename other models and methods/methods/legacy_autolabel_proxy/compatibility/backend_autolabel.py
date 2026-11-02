"""Compatibility import; implementation lives in rice_count_pipeline/methods/autolabel_proxy."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from rice_count_pipeline.methods.autolabel_proxy.autolabel import draw, label_image

__all__ = ["draw", "label_image"]
