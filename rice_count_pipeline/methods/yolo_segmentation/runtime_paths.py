"""Keep Python caches and temporary files on the project drive."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
WORK = PROJECT_ROOT / "data" / "rice_count" / "work"
CACHE = WORK / "runtime_cache"
for key, path in {
    "TEMP": WORK / "tmp", "TMP": WORK / "tmp",
    "TORCH_HOME": CACHE / "torch", "YOLO_CONFIG_DIR": CACHE / "ultralytics",
    "MPLCONFIGDIR": CACHE / "matplotlib",
}.items():
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
