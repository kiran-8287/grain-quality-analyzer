"""Compatibility entry point for the shared rice-photo audit tool."""
from pathlib import Path
import runpy
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(PROJECT_ROOT / "rice_count_pipeline" / "tools" / "audit_photos_v2.py"), run_name="__main__")
