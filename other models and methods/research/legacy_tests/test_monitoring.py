"""
Tests for monitoring log parser.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from monitoring.log_parser import (
    AnalysisRecord,
    LogRecord,
    ParseResult,
    RequestRecord,
    StageTiming,
    _extract_json,
    _parse_line,
    compute_analysis_stats,
    compute_endpoint_stats,
    compute_error_summary,
    compute_stage_timings,
    get_last_log_time,
    get_memory_metrics,
    parse_file,
    parse_logs,
    reconstruct_analyses,
    reconstruct_requests,
)

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "monitoring_sample.log"


def test_parse_pure_json_line() -> None:
    line = '{"timestamp": "2026-10-07T12:00:00.000Z", "event": "analysis_started"}'
    rec = _parse_line(line)
    assert not rec.parse_error
    assert rec.event == "analysis_started"


def test_parse_logging_prefix_json_line() -> None:
    line = '2026-10-08 03:07:48,214 [INFO] grain_structured: {"timestamp": "2026-10-07T21:37:48.214Z", "event": "request_started"}'
    rec = _parse_line(line)
    assert not rec.parse_error
    assert rec.event == "request_started"


def test_parse_request_started_from_prefixed_log() -> None:
    line = '2026-10-08 03:07:48,214 [INFO] grain_structured: {"timestamp": "2026-10-07T21:37:48.214Z", "level": "INFO", "event": "request_started", "request_id": "req-1", "method": "GET", "path": "/health", "route": "/health"}'
    rec = _parse_line(line)
    assert not rec.parse_error
    assert rec.event == "request_started"
    assert rec.raw["request_id"] == "req-1"


def test_parse_request_finished_from_prefixed_log() -> None:
    line = '2026-10-08 03:07:48,226 [INFO] grain_structured: {"timestamp": "2026-10-07T21:37:48.226Z", "level": "INFO", "event": "request_finished", "request_id": "req-1", "method": "GET", "path": "/health", "route": "/health", "status_code": 200, "ok": true, "duration_ms": 10}'
    rec = _parse_line(line)
    assert not rec.parse_error
    assert rec.event == "request_finished"
    assert rec.raw["status_code"] == 200


def test_parse_request_failed_from_prefixed_log() -> None:
    line = '2026-10-08 03:07:48,226 [ERROR] grain_structured: {"timestamp": "2026-10-07T21:37:48.226Z", "level": "ERROR", "event": "request_failed", "request_id": "req-1", "method": "GET", "path": "/health", "route": "/health", "status_code": 500, "duration_ms": 10, "error_type": "TypeError", "error_message": "coroutine"}'
    rec = _parse_line(line)
    assert not rec.parse_error
    assert rec.event == "request_failed"
    assert rec.raw["error_type"] == "TypeError"


def test_parse_analysis_started_from_prefixed_log() -> None:
    line = '2026-10-08 03:07:49,870 [INFO] grain_structured: {"timestamp": "2026-10-07T21:37:49.870Z", "event": "analysis_started", "job_id": "job-1"}'
    rec = _parse_line(line)
    assert not rec.parse_error
    assert rec.event == "analysis_started"
    assert rec.raw["job_id"] == "job-1"


def test_parse_analysis_completed_from_prefixed_log() -> None:
    line = '2026-10-08 03:07:51,914 [INFO] grain_structured: {"timestamp": "2026-10-07T21:37:51.914Z", "event": "analysis_completed", "job_id": "job-1", "status": "success", "total_processing_ms": 1000}'
    rec = _parse_line(line)
    assert not rec.parse_error
    assert rec.event == "analysis_completed"
    assert rec.raw["total_processing_ms"] == 1000


def test_parse_analysis_failed_from_prefixed_log() -> None:
    line = '2026-10-08 03:07:51,914 [ERROR] grain_structured: {"timestamp": "2026-10-07T21:37:51.914Z", "event": "analysis_failed", "job_id": "job-1", "failed_stage": "segmentation", "error_type": "ValueError", "error_message": "boom"}'
    rec = _parse_line(line)
    assert not rec.parse_error
    assert rec.event == "analysis_failed"
    assert rec.raw["failed_stage"] == "segmentation"


def test_malformed_line_does_not_crash_parser() -> None:
    rec = _parse_line("this is not valid json at all")
    assert rec.parse_error


def test_extract_json_returns_none_when_no_brace() -> None:
    assert _extract_json("no json here") is None


def test_reconstruct_request_started_plus_finished() -> None:
    result = parse_file(FIXTURE_PATH, max_lines=100_000)
    requests = reconstruct_requests(result.records)
    req = next((r for r in requests if r.request_id == "req-001"), None)
    assert req is not None
    assert req.finished_at is not None
    assert req.duration_ms == 1120
    assert req.ok is True


def test_reconstruct_request_started_plus_failed() -> None:
    result = parse_file(FIXTURE_PATH, max_lines=100_000)
    requests = reconstruct_requests(result.records)
    req = next((r for r in requests if r.request_id == "req-003"), None)
    assert req is not None
    assert req.ok is False
    assert req.error_type == "ValueError"


def test_reconstruct_analysis_started_plus_completed() -> None:
    result = parse_file(FIXTURE_PATH, max_lines=100_000)
    analyses = reconstruct_analyses(result.records)
    analysis = next((a for a in analyses if a.job_id == "job-001"), None)
    assert analysis is not None
    assert analysis.status == "success"
    assert analysis.duration_ms == 1210


def test_reconstruct_analysis_started_plus_failed() -> None:
    result = parse_file(FIXTURE_PATH, max_lines=100_000)
    analyses = reconstruct_analyses(result.records)
    analysis = next((a for a in analyses if a.job_id == "job-003"), None)
    assert analysis is not None
    assert analysis.status == "failed"
    assert analysis.failed_stage == "segmentation"


def test_analysis_reconstruction_works_without_request_id() -> None:
    result = parse_file(FIXTURE_PATH, max_lines=100_000)
    analyses = reconstruct_analyses(result.records)
    job_ids = [a.job_id for a in analyses]
    assert "job-001" in job_ids
    assert "job-003" in job_ids


def test_api_failure_after_successful_analysis_keeps_analysis_success() -> None:
    result = parse_file(FIXTURE_PATH, max_lines=100_000)
    requests = reconstruct_requests(result.records)
    analyses = reconstruct_analyses(result.records)
    analysis_map = {a.job_id: a for a in analyses}
    failed_reqs = [r for r in requests if not r.ok]
    assert any(r.job_id in analysis_map and analysis_map[r.job_id].status == "success" for r in failed_reqs)


def test_rotated_files_are_merged() -> None:
    import tempfile

    temp_dir = Path(tempfile.mkdtemp())
    base = temp_dir / "app.log"
    rotated = temp_dir / "app.log.1"
    base.write_text('{"timestamp": "2026-10-07T12:00:00.000Z", "event": "request_started", "request_id": "rot-1", "method": "GET", "path": "/health", "route": "/health"}\n', encoding="utf-8")
    rotated.write_text('{"timestamp": "2026-10-07T12:00:01.000Z", "event": "request_finished", "request_id": "rot-1", "method": "GET", "path": "/health", "route": "/health", "status_code": 200, "ok": true, "duration_ms": 5}\n', encoding="utf-8")
    try:
        result = parse_logs(log_dir=temp_dir, max_lines=100_000)
        assert result.files_read >= 2
    finally:
        base.unlink(missing_ok=True)
        rotated.unlink(missing_ok=True)
        temp_dir.rmdir()


def test_records_are_chronologically_sorted() -> None:
    result = parse_file(FIXTURE_PATH, max_lines=100_000)
    timestamps = [r.timestamp for r in result.records if r.timestamp is not None]
    assert timestamps == sorted(timestamps)


def test_duplicate_lifecycle_events_do_not_double_count() -> None:
    result = parse_file(FIXTURE_PATH, max_lines=100_000)
    requests = reconstruct_requests(result.records)
    request_ids = [r.request_id for r in requests]
    assert len(request_ids) == len(set(request_ids))


def test_explicit_duration_ms_is_respected() -> None:
    result = parse_file(FIXTURE_PATH, max_lines=100_000)
    requests = reconstruct_requests(result.records)
    req = next((r for r in requests if r.request_id == "req-001"), None)
    assert req is not None
    assert req.duration_ms == 1120


def test_parser_never_returns_zero_records_when_prefixed_json_exists() -> None:
    result = parse_file(FIXTURE_PATH, max_lines=100_000)
    assert result.records_loaded > 0


def test_fixture_contains_prefixed_lines() -> None:
    with open(FIXTURE_PATH, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    assert "grain_structured:" in first_line


def test_request_started_finished_equals_one_request() -> None:
    result = parse_file(FIXTURE_PATH, max_lines=100_000)
    requests = reconstruct_requests(result.records)
    completed = [r for r in requests if r.finished_at is not None]
    assert len(completed) >= 1


def test_malformed_line_does_not_prevent_other_records() -> None:
    result = parse_file(FIXTURE_PATH, max_lines=100_000)
    assert result.records_loaded > 0
    assert result.malformed_lines >= 1


def test_endpoint_aggregation() -> None:
    result = parse_file(FIXTURE_PATH, max_lines=100_000)
    requests = reconstruct_requests(result.records)
    stats = compute_endpoint_stats(requests)
    routes = {s["route"]: s for s in stats}
    assert "/analyze" in routes or "/health" in routes


def test_stage_timing_aggregation() -> None:
    result = parse_file(FIXTURE_PATH, max_lines=100_000)
    timings = compute_stage_timings(result.records)
    assert len(timings) >= 1
