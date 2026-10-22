"""Keep Python caches and temporary files on the project drive."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
for key, path in {
    "TEMP": ROOT / "tmp", "TMP": ROOT / "tmp",
    "TORCH_HOME": CACHE / "torch", "YOLO_CONFIG_DIR": CACHE / "ultralytics",
    "MPLCONFIGDIR": CACHE / "matplotlib",
}.items():
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
