"""Compatibility entry point for shared layout grouping."""
from pathlib import Path
import runpy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(PROJECT_ROOT / "rice_count_pipeline" / "tools" / "find_layout_groups.py"), run_name="__main__")
