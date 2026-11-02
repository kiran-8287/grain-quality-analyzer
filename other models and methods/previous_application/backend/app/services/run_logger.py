"""
Run Logger for Rice Quality Analysis.

Creates a per-run folder under data/runs/ and stores:
- input_image.{ext}
- masked_image.jpg
- metrics.json
- result.json (complete backend response)
"""

import base64
import json
import logging
import os
import re
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RUNS_DIR = PROJECT_ROOT / "data" / "runs"


def _is_test_run() -> bool:
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _sanitize_filename(name: str) -> str:
    """Remove unsafe characters from filename."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _get_next_run_folder() -> Path:
    """Create a new run folder with sequential ID and current date."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    date_str = now.strftime("%d_%m_%y")

    existing = []
    if RUNS_DIR.exists():
        for entry in RUNS_DIR.iterdir():
            if entry.is_dir():
                m = re.match(r"(\d+)_", entry.name)
                if m:
                    existing.append(int(m.group(1)))

    next_id = max(existing, default=0) + 1
    folder_name = f"{next_id:03d}_{date_str}"
    run_path = RUNS_DIR / folder_name

    counter = 1
    while run_path.exists():
        run_path = RUNS_DIR / f"{next_id:03d}_{date_str}_{counter}"
        counter += 1

    run_path.mkdir(parents=True, exist_ok=True)
    return run_path


def log_run(
    input_bytes: bytes,
    original_filename: str,
    result: Dict[str, Any],
    grade: str = "grade_a",
) -> Optional[Path]:
    """
    Save a complete analysis run to a new per-run folder.

    Args:
        input_bytes: Original uploaded image bytes.
        original_filename: Original filename (used to infer extension).
        result: Full analysis result dict from pipeline / job manager.
        grade: Analysis grade used ('grade_a' or 'common').

    Returns:
        Path to the created run folder, or None on failure.
    """
    try:
        if _is_test_run():
            return None

        run_path = _get_next_run_folder()
        logger.info("Saving analysis run to %s", run_path)

        ext = Path(original_filename).suffix.lower()
        if not ext or len(ext) > 5:
            ext = ".jpg"

        input_path = run_path / f"input_image{ext}"
        with open(input_path, "wb") as f:
            f.write(input_bytes)

        annotated_b64 = result.get("annotated_image_base64", "")
        if annotated_b64 and annotated_b64.startswith("data:image/jpeg;base64,"):
            raw_b64 = annotated_b64.split(",", 1)[1]
            masked_bytes = base64.b64decode(raw_b64)
            masked_path = run_path / "masked_image.jpg"
            with open(masked_path, "wb") as f:
                f.write(masked_bytes)
        else:
            logger.warning("No annotated image available for run %s", run_path.name)

        metrics = {
            "run_id": run_path.name,
            "grade": grade,
            "success": result.get("success", False),
            "rice_detected": result.get("rice_detected", False),
            "message": result.get("message"),
            "image": result.get("image"),
            "sample": result.get("sample"),
            "calibration": result.get("calibration"),
            "quality": result.get("quality"),
            "grains": result.get("grains", []),
            "foreign_matter": result.get("foreign_matter", []),
            "admixture": result.get("admixture"),
            "summary": result.get("summary"),
            "standards": result.get("standards"),
            "rice_gate": result.get("rice_gate"),
            "warnings": result.get("warnings", []),
            "processing_time_seconds": result.get("processing_time_seconds"),
        }
        metrics_path = run_path / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

        clean_result = {k: v for k, v in result.items() if k != "annotated_image_base64"}
        result_path = run_path / "result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(clean_result, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

        logger.info(
            "Run %s saved: %d grains, %d foreign objects",
            run_path.name,
            len(result.get("grains", [])),
            len(result.get("foreign_matter", [])),
        )
        return run_path

    except Exception as e:
        logger.error("Failed to save analysis run: %s", e, exc_info=True)
        return None


class StructuredLogger:
    """Structured JSON logger for pipeline observability."""

    def __init__(self, name: str = "grain_structured"):
        self._logger = logging.getLogger(name)
        self._logger.propagate = False

    @property
    def logger(self):
        return self._logger

    def _emit(
        self,
        level: int,
        event: str,
        job_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        now = time.time()
        ts = time.strftime(
            "%Y-%m-%dT%H:%M:%S.", time.gmtime(now)
        ) + f"{int(now * 1000) % 1000:03d}Z"
        record: Dict[str, Any] = {
            "timestamp": ts,
            "level": logging.getLevelName(level),
            "event": event,
            "job_id": job_id,
            "message": kwargs.pop("message", ""),
        }
        for k, v in kwargs.items():
            if v is not None:
                record[k] = v
        clean = {k: v for k, v in record.items() if v is not None}
        self._logger.log(level, json.dumps(clean, default=str))

    def info(self, event: str, job_id: Optional[str] = None, **kwargs: Any) -> None:
        self._emit(logging.INFO, event, job_id, **kwargs)

    def error(self, event: str, job_id: Optional[str] = None, **kwargs: Any) -> None:
        self._emit(logging.ERROR, event, job_id, **kwargs)

    def warning(self, event: str, job_id: Optional[str] = None, **kwargs: Any) -> None:
        self._emit(logging.WARNING, event, job_id, **kwargs)

    def debug(self, event: str, job_id: Optional[str] = None, **kwargs: Any) -> None:
        self._emit(logging.DEBUG, event, job_id, **kwargs)


structured_logger = StructuredLogger()
