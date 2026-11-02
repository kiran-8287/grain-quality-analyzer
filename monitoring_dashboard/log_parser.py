"""Parse current and legacy application logs without retaining image data."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Iterable


_FIELDS = (
    "method",
    "path",
    "status",
    "ok",
    "duration_ms",
    "duration_s",
    "rss_mb",
    "process_cpu_percent",
    "in_flight",
    "content_length",
    "received_body_bytes",
    "file_bytes",
    "processing_ms",
    "queue_wait_ms",
    "image_width",
    "image_height",
    "rice_count",
    "foreign_matter_count",
    "whole_grain_count",
    "broken_grain_count",
    "whole_percent",
    "reference_length",
    "response_objects",
    "error_type",
    "reason",
)


def _first(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return None


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        if isinstance(value, (float, int)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def parse_log_line(line: str) -> dict[str, Any] | None:
    """Read a JSON line or a legacy logging prefix followed by a JSON object."""
    start = line.find("{")
    if start < 0:
        return None
    try:
        source = json.loads(line[start:])
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(source, dict):
        return None

    event = _first(source, "event", "type")
    if not event:
        return None
    timestamp = _parse_timestamp(
        _first(source, "timestamp", "startedAt", "started_at", "finishedAt", "finished_at")
    )
    normalized: dict[str, Any] = {
        "event": str(event).strip().lower(),
        "timestamp": timestamp,
        "request_id": _first(source, "request_id", "requestId", "requestID"),
    }
    aliases = {
        "status": ("status", "status_code", "statusCode"),
        "duration_ms": ("duration_ms", "durationMs"),
        "duration_s": ("duration_s", "duration_s", "durationSeconds"),
        "rss_mb": ("rss_mb", "rssMb"),
        "in_flight": ("in_flight", "inFlight"),
        "content_length": ("content_length", "contentLength"),
        "processing_ms": ("processing_ms", "processingMs"),
        "queue_wait_ms": ("queue_wait_ms", "queueWaitMs"),
        "rice_count": ("rice_count", "riceCount"),
        "foreign_matter_count": ("foreign_matter_count", "foreignMatterCount", "foreign_count"),
        "whole_grain_count": ("whole_grain_count", "wholeGrainCount"),
        "broken_grain_count": ("broken_grain_count", "brokenGrainCount"),
        "reference_length": ("reference_length", "referenceLength"),
    }
    for field in _FIELDS:
        value = _first(source, *aliases.get(field, (field,)))
        if value is not None:
            normalized[field] = value
    if normalized.get("duration_ms") is None and normalized.get("duration_s") is not None:
        try:
            normalized["duration_ms"] = float(normalized["duration_s"]) * 1000
        except (TypeError, ValueError):
            pass
    if normalized.get("ok") is None and normalized.get("status") is not None:
        try:
            normalized["ok"] = int(normalized["status"]) < 400
        except (TypeError, ValueError):
            pass
    return normalized


def parse_log_lines(lines: Iterable[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return normalized events and data-quality counts for a stream of lines."""
    events: list[dict[str, Any]] = []
    total = blank = malformed = 0
    for line in lines:
        total += 1
        if not line.strip():
            blank += 1
            continue
        parsed = parse_log_line(line)
        if parsed is None:
            malformed += 1
        else:
            events.append(parsed)
    return events, {"lines": total, "blank": blank, "malformed": malformed, "parsed": len(events)}


def build_request_rows(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join start, finish, and analysis events by request ID."""
    requests: dict[str, dict[str, Any]] = {}
    for index, event in enumerate(events):
        name = event.get("event", "")
        if name not in {
            "request_started",
            "request_finished",
            "request_failed",
            "analysis_started",
            "analysis_completed",
            "analysis_rejected",
            "analysis_failed",
            "analysis_queue_full",
        }:
            continue

        request_id = event.get("request_id") or f"unidentified-{index}"
        row = requests.setdefault(request_id, {"request_id": request_id})
        timestamp = event.get("timestamp")

        if name == "request_started":
            row["start_time"] = timestamp
            row["method"] = event.get("method", row.get("method"))
            row["path"] = event.get("path", row.get("path"))
            row["content_length"] = event.get("content_length", row.get("content_length"))
            row["rss_start_mb"] = event.get("rss_mb", row.get("rss_start_mb"))
            row["request_started"] = True
        elif name in {"request_finished", "request_failed"}:
            row["end_time"] = timestamp
            for field in ("method", "path", "status", "ok", "duration_ms", "duration_s", "in_flight", "rss_mb", "error_type"):
                if event.get(field) is not None:
                    row[field] = event[field]
            # Some older logs wrote zero at finish even when the start event
            # contained the actual HTTP Content-Length.
            if event.get("content_length") not in (None, 0, "0") or row.get("content_length") is None:
                row["content_length"] = event.get("content_length", row.get("content_length"))
            row["rss_end_mb"] = event.get("rss_mb", row.get("rss_end_mb"))
            row["request_finished"] = True
        else:
            if timestamp is not None:
                row["analysis_timestamp"] = timestamp
            for field in (
                "file_bytes",
                "processing_ms",
                "queue_wait_ms",
                "image_width",
                "image_height",
                "rice_count",
                "foreign_matter_count",
                "whole_grain_count",
                "broken_grain_count",
                "whole_percent",
                "reference_length",
                "response_objects",
                "error_type",
                "reason",
            ):
                if event.get(field) is not None:
                    row[field] = event[field]
            row["analysis_event"] = name
            if name in {"analysis_completed", "analysis_rejected", "analysis_failed", "analysis_queue_full"}:
                row["analysis_finished"] = True
            if name == "analysis_started":
                row["analysis_started"] = True

        if timestamp and not row.get("start_time"):
            row["start_time"] = timestamp

    rows = list(requests.values())
    for row in rows:
        start = row.get("start_time")
        end = row.get("end_time")
        if row.get("duration_ms") is None and start and end:
            row["duration_ms"] = max(0.0, (end - start).total_seconds() * 1000)
        row["timestamp"] = start or end
        row["rss_max_mb"] = max(
            [value for value in (row.get("rss_start_mb"), row.get("rss_end_mb")) if isinstance(value, (int, float))],
            default=None,
        )
        if row.get("status") is not None:
            try:
                row["status"] = int(row["status"])
            except (TypeError, ValueError):
                row["status"] = None
        if row.get("ok") is None and row.get("status") is not None:
            row["ok"] = row["status"] < 400
    return rows
