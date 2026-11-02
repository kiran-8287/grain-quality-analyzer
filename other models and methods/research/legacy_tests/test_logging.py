"""
Tests for structured application logging and processing-time tracking.
"""

import json
import logging
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from backend.app.services.run_logger import structured_logger
from backend.app.services.job_manager import job_manager


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))


class _LogCapture:
    """Capture structured log records for assertions."""

    def __init__(self):
        self.records = []
        self.handler = logging.Handler()
        self.handler.emit = lambda record: self.records.append(record)

    def __enter__(self):
        structured_logger.logger.addHandler(self.handler)
        structured_logger.logger.setLevel(logging.DEBUG)
        return self

    def __exit__(self, *args):
        structured_logger.logger.removeHandler(self.handler)

    @property
    def events(self):
        return [json.loads(r.getMessage()) for r in self.records]


def _make_grain_image(size: int = 300) -> bytes:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.ellipse(img, (size // 2, size // 2), (40, 15), 30, 0, 360, (230, 230, 230), -1)
    _, enc = cv2.imencode(".jpg", img)
    return enc.tobytes()


def test_successful_analysis_emits_analysis_started():
    with _LogCapture() as cap:
        job_manager.process_sync(_make_grain_image(), "test_start.jpg")

    assert any(e["event"] == "analysis_started" for e in cap.events)


def test_successful_analysis_emits_analysis_completed():
    with _LogCapture() as cap:
        job_manager.process_sync(_make_grain_image(), "test_complete.jpg")

    assert any(e["event"] == "analysis_completed" for e in cap.events)


def test_job_id_consistent_throughout_analysis():
    with _LogCapture() as cap:
        result = job_manager.process_sync(_make_grain_image(), "test_jobid.jpg")

    events = cap.events
    job_ids = [e["job_id"] for e in events if "job_id" in e]
    assert len(set(job_ids)) == 1
    assert result["job_id"] == job_ids[0]


def test_total_processing_ms_present_in_completion():
    with _LogCapture() as cap:
        job_manager.process_sync(_make_grain_image(), "test_total_ms.jpg")

    completed = [e for e in cap.events if e["event"] == "analysis_completed"]
    assert completed
    assert "total_processing_ms" in completed[0]
    assert isinstance(completed[0]["total_processing_ms"], int)


def test_stage_timing_fields_present_in_completed_events():
    with _LogCapture() as cap:
        job_manager.process_sync(_make_grain_image(), "test_stage_timing.jpg")

    stage_completed = [e for e in cap.events if e["event"].endswith("_completed") and e["event"] != "analysis_completed"]
    for e in stage_completed:
        assert "duration_ms" in e, f"Missing duration_ms in {e['event']}"
        assert isinstance(e["duration_ms"], int)


def test_failed_analysis_emits_analysis_failed():
    from ml.segmentation import pipeline as pipeline_module

    def _bad_load_image(*args, **kwargs):
        raise ValueError("controlled test failure")

    with patch.object(pipeline_module, "load_image", _bad_load_image):
        with _LogCapture() as cap:
            job_manager.process_sync(_make_grain_image(), "test_fail.jpg")

    assert any(e["event"] == "analysis_failed" for e in cap.events)


def test_failed_stage_recorded():
    from ml.segmentation import pipeline as pipeline_module

    def _bad_load_image(*args, **kwargs):
        raise ValueError("controlled test failure")

    with patch.object(pipeline_module, "load_image", _bad_load_image):
        with _LogCapture() as cap:
            job_manager.process_sync(_make_grain_image(), "test_failed_stage.jpg")

    failed = [e for e in cap.events if e["event"] == "analysis_failed"]
    assert failed
    assert "failed_stage" in failed[0]
    assert failed[0]["failed_stage"] == "image_decode"


def test_error_type_and_message_recorded():
    from ml.segmentation import pipeline as pipeline_module

    def _bad_load_image(*args, **kwargs):
        raise ValueError("controlled test failure")

    with patch.object(pipeline_module, "load_image", _bad_load_image):
        with _LogCapture() as cap:
            job_manager.process_sync(_make_grain_image(), "test_error_fields.jpg")

    failed = [e for e in cap.events if e["event"] == "analysis_failed"]
    assert failed
    assert failed[0]["error_type"] == "DecodeError"
    assert "controlled test failure" in failed[0]["error_message"]


def test_no_image_bytes_or_base64_in_logs():
    with _LogCapture() as cap:
        job_manager.process_sync(_make_grain_image(), "test_no_bytes.jpg")

    forbidden_keys = [
        "annotated_image_base64",
        "original_image_base64",
        "image_bytes",
    ]
    for record in cap.events:
        for key in forbidden_keys:
            assert key not in record, f"Forbidden key {key} in log record"
        msg = record.get("message", "")
        assert "data:image/jpeg;base64" not in msg


def test_whole_broken_summary_in_successful_completion():
    with _LogCapture() as cap:
        job_manager.process_sync(_make_grain_image(), "test_summary.jpg")

    completed = [e for e in cap.events if e["event"] == "analysis_completed"]
    assert completed
    c = completed[0]
    assert "whole_count" in c
    assert "broken_count" in c


def test_duration_fields_are_numeric():
    with _LogCapture() as cap:
        job_manager.process_sync(_make_grain_image(), "test_numeric.jpg")

    for record in cap.events:
        if "duration_ms" in record:
            assert isinstance(record["duration_ms"], int)
        if "total_processing_ms" in record:
            assert isinstance(record["total_processing_ms"], int)


def test_no_duplicate_events_for_single_job():
    with _LogCapture() as cap:
        job_manager.process_sync(_make_grain_image(), "test_dedup.jpg")

    job_id = cap.events[0]["job_id"]
    job_events = [e for e in cap.events if e.get("job_id") == job_id]
    from collections import Counter
    counts = Counter(e["event"] for e in job_events)
    for event, count in counts.items():
        assert count == 1, f"Event {event} appears {count} times"


def test_stage_durations_are_individual_not_cumulative():
    with _LogCapture() as cap:
        job_manager.process_sync(_make_grain_image(), "test_stage_individual.jpg")

    completed = [e for e in cap.events if e["event"].endswith("_completed") and e["event"] != "analysis_completed"]
    durations = [e["duration_ms"] for e in completed]
    total = sum(durations)
    final = next((e["total_processing_ms"] for e in cap.events if e["event"] == "analysis_completed"), None)
    assert final is not None
    assert total <= final
    assert final < total + 500
