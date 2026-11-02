"""
Robust parser for Grain Quality Analyzer structured JSON Lines logs.

Reads rotating log files, tolerates malformed lines, reconstructs logical
requests and analysis jobs, and produces metrics for the monitoring dashboard.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = PROJECT_ROOT / "monitoring" / "logs"
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "app.log"
MAX_PARSE_ERRORS = 50
DEFAULT_MAX_LINES = 200_000
IST_OFFSET_HOURS = 5
IST_OFFSET_MINUTES = 30


def _utc_to_ist(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    offset_minutes = IST_OFFSET_HOURS * 60 + IST_OFFSET_MINUTES
    return dt.astimezone(timezone.utc).replace(tzinfo=timezone.utc).timestamp() + offset_minutes * 60


def _ist_label(dt: Optional[datetime]) -> str:
    if dt is None:
        return "N/A"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ist = dt.astimezone(timezone.utc)
    offset = IST_OFFSET_HOURS * 60 + IST_OFFSET_MINUTES
    ist = datetime.fromtimestamp(ist.timestamp() + offset * 60, tz=timezone.utc)
    return ist.strftime("%Y-%m-%d %H:%M:%S IST")


@dataclass
class LogRecord:
    raw: Dict[str, Any]
    timestamp_str: str = ""
    event: str = ""
    parse_error: bool = False
    parse_error_msg: str = ""

    def __post_init__(self):
        self.timestamp_str = str(self.raw.get("timestamp", ""))
        self.event = str(self.raw.get("event", ""))

    @property
    def timestamp(self) -> Optional[datetime]:
        ts = self.timestamp_str
        if not ts:
            return None
        ts = ts.replace("Z", "+00:00")
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
        ):
            try:
                return datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    @property
    def ist_timestamp(self) -> str:
        dt = self.timestamp
        return _ist_label(dt)


@dataclass
class RequestRecord:
    request_id: str
    method: str = ""
    route: str = ""
    path: str = ""
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    status_code: Optional[int] = None
    duration_ms: Optional[int] = None
    ok: bool = False
    error_type: str = ""
    error_message: str = ""
    job_id: str = ""
    analysis_duration_ms: Optional[int] = None
    analysis_status: str = ""
    rice_detected: bool = False
    grain_count: int = 0
    whole_count: int = 0
    broken_count: int = 0
    undetermined_count: int = 0


@dataclass
class AnalysisRecord:
    job_id: str
    request_id: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    status: str = ""
    rice_detected: bool = False
    grain_count: int = 0
    whole_count: int = 0
    broken_count: int = 0
    undetermined_count: int = 0
    failed_stage: str = ""
    error_type: str = ""
    error_message: str = ""
    reference_source: str = ""
    total_processing_ms: Optional[int] = None
    filename: str = ""
    segmentation_method: str = ""


@dataclass
class StageTiming:
    stage: str
    duration_ms: int = 0
    job_id: str = ""


@dataclass
class ParseResult:
    records: List[LogRecord] = field(default_factory=list)
    parse_errors: int = 0
    malformed_lines: int = 0
    files_read: int = 0
    lines_read: int = 0
    records_loaded: int = 0


def _parse_line(line: str) -> LogRecord:
    line = line.strip()
    if not line:
        return LogRecord(raw={}, parse_error=True, parse_error_msg="empty line")
    data = _extract_json(line)
    if data is None:
        return LogRecord(raw={}, parse_error=True, parse_error_msg="no JSON object found")
    if not isinstance(data, dict):
        return LogRecord(raw={}, parse_error=True, parse_error_msg="not a JSON object")
    return LogRecord(raw=data)


def _extract_json(line: str) -> Any:
    first = line.find("{")
    if first == -1:
        return None
    candidate = line[first:]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _iter_log_files(log_dir: Path) -> List[Tuple[Path, int]]:
    files: List[Tuple[Path, int]] = []
    if not log_dir.exists():
        return files
    for entry in log_dir.iterdir():
        if entry.is_file() and entry.name.startswith("app.log"):
            suffix = entry.suffixes[-1] if entry.suffixes else ""
            seq = 0
            if suffix.lstrip(".").isdigit():
                seq = int(suffix.lstrip("."))
            files.append((entry, seq))
    files.sort(key=lambda x: (x[1], x[0].name))
    return files


def parse_file(path: Path, max_lines: int = DEFAULT_MAX_LINES) -> ParseResult:
    result = ParseResult()
    all_lines: List[Tuple[datetime, LogRecord]] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                result.lines_read += 1
                if len(all_lines) >= max_lines:
                    break
                rec = _parse_line(line)
                if rec.parse_error:
                    result.malformed_lines += 1
                    if result.parse_errors < MAX_PARSE_ERRORS:
                        result.parse_errors += 1
                    continue
                result.records_loaded += 1
                ts = rec.timestamp
                if ts is None:
                    ts = datetime.min.replace(tzinfo=timezone.utc)
                all_lines.append((ts, rec))
    except OSError:
        return result
    all_lines.sort(key=lambda x: x[0])
    result.records = [rec for _, rec in all_lines]
    result.files_read = 1
    return result


def parse_logs(
    log_dir: Path = DEFAULT_LOG_DIR,
    max_lines: int = DEFAULT_MAX_LINES,
) -> ParseResult:
    log_files = _iter_log_files(log_dir)
    if not log_files:
        return ParseResult()

    result = ParseResult()
    all_lines: List[Tuple[datetime, LogRecord]] = []

    for log_path, _ in log_files:
        result.files_read += 1
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    result.lines_read += 1
                    if len(all_lines) >= max_lines:
                        break
                    rec = _parse_line(line)
                    if rec.parse_error:
                        result.malformed_lines += 1
                        if result.parse_errors < MAX_PARSE_ERRORS:
                            result.parse_errors += 1
                        continue
                    result.records_loaded += 1
                    ts = rec.timestamp
                    if ts is None:
                        ts = datetime.min.replace(tzinfo=timezone.utc)
                    all_lines.append((ts, rec))
        except OSError:
            continue

    all_lines.sort(key=lambda x: x[0])
    result.records = [rec for _, rec in all_lines]
    return result


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def reconstruct_requests(records: List[LogRecord]) -> List[RequestRecord]:
    requests: Dict[str, RequestRecord] = {}
    analysis_events: Dict[str, Dict[str, Any]] = defaultdict(dict)

    for rec in records:
        ev = rec.event
        if ev in ("analysis_started", "analysis_completed", "analysis_failed"):
            jid = rec.raw.get("job_id", "")
            rid = rec.raw.get("request_id", "")
            if jid:
                analysis_events[jid][ev] = rec.raw
            if rid and jid:
                analysis_events[rid]["_request_id"] = rid

    request_to_job: Dict[str, str] = {}
    for jid, evts in analysis_events.items():
        for ev_name in ("analysis_started", "analysis_completed", "analysis_failed"):
            if ev_name in evts:
                rid = evts[ev_name].get("request_id", "")
                if rid:
                    request_to_job[rid] = jid
                    break

    for rec in records:
        ev = rec.event
        rid = rec.raw.get("request_id")
        if not rid:
            continue
        if ev == "request_started":
            req = RequestRecord(
                request_id=rid,
                method=rec.raw.get("method", ""),
                route=rec.raw.get("route", ""),
                path=rec.raw.get("path", ""),
                started_at=rec.timestamp,
            )
            requests[rid] = req
        elif ev == "request_finished":
            if rid in requests:
                requests[rid].finished_at = rec.timestamp
                requests[rid].status_code = rec.raw.get("status_code")
                requests[rid].duration_ms = rec.raw.get("duration_ms")
                requests[rid].ok = bool(rec.raw.get("ok", False))
                requests[rid].job_id = request_to_job.get(rid, "")
        elif ev == "request_failed":
            if rid in requests:
                requests[rid].finished_at = rec.timestamp
                requests[rid].status_code = rec.raw.get("status_code")
                requests[rid].duration_ms = rec.raw.get("duration_ms")
                requests[rid].ok = False
                requests[rid].error_type = rec.raw.get("error_type", "")
                requests[rid].error_message = rec.raw.get("error_message", "")
                requests[rid].job_id = request_to_job.get(rid, "")

    for req in requests.values():
        jid = req.job_id
        if jid and jid in analysis_events:
            evts = analysis_events[jid]
            if "analysis_completed" in evts:
                a = evts["analysis_completed"]
                req.analysis_status = a.get("status", "")
                req.analysis_duration_ms = a.get("total_processing_ms")
                req.rice_detected = bool(a.get("rice_detected", False))
                req.grain_count = a.get("grain_count", 0)
                req.whole_count = a.get("whole_count", 0)
                req.broken_count = a.get("broken_count", 0)
                req.undetermined_count = a.get("undetermined_count", 0)
            elif "analysis_failed" in evts:
                a = evts["analysis_failed"]
                req.analysis_status = "failed"
                req.analysis_duration_ms = a.get("total_processing_ms") or a.get("duration_ms")
                req.failed_stage = a.get("failed_stage", "")
                req.error_type = a.get("error_type", req.error_type)
                req.error_message = a.get("error_message", req.error_message)

    return list(requests.values())


def reconstruct_analyses(records: List[LogRecord]) -> List[AnalysisRecord]:
    events: Dict[str, Dict[str, Any]] = defaultdict(dict)
    stage_events: List[Dict[str, Any]] = []

    for rec in records:
        ev = rec.event
        if ev in ("analysis_started", "analysis_completed", "analysis_failed"):
            jid = rec.raw.get("job_id", "")
            if jid:
                events[jid][ev] = rec.raw
        elif ev.endswith("_started") or ev.endswith("_completed"):
            stage_events.append(rec.raw)

    analyses: List[AnalysisRecord] = []
    for jid, evts in events.items():
        if "analysis_started" not in evts:
            continue
        start_raw = evts["analysis_started"]
        start_ts = _timestamp_from_record(start_raw)
        record = AnalysisRecord(
            job_id=jid,
            request_id=start_raw.get("request_id", ""),
            started_at=start_ts,
            filename=start_raw.get("filename", ""),
        )
        if "analysis_completed" in evts:
            end_raw = evts["analysis_completed"]
            record.completed_at = _timestamp_from_record(end_raw)
            record.duration_ms = end_raw.get("total_processing_ms")
            record.status = end_raw.get("status", "success")
            record.rice_detected = bool(end_raw.get("rice_detected", False))
            record.grain_count = end_raw.get("grain_count", 0)
            record.whole_count = end_raw.get("whole_count", 0)
            record.broken_count = end_raw.get("broken_count", 0)
            record.undetermined_count = end_raw.get("undetermined_count", 0)
            record.total_processing_ms = end_raw.get("total_processing_ms")
            record.reference_source = end_raw.get("reference_source", "")
            record.segmentation_method = end_raw.get("segmentation_method", "")
        elif "analysis_failed" in evts:
            end_raw = evts["analysis_failed"]
            record.completed_at = _timestamp_from_record(end_raw)
            record.duration_ms = end_raw.get("total_processing_ms") or end_raw.get("duration_ms")
            record.status = "failed"
            record.failed_stage = end_raw.get("failed_stage", "")
            record.error_type = end_raw.get("error_type", "")
            record.error_message = end_raw.get("error_message", "")
            record.total_processing_ms = end_raw.get("total_processing_ms") or end_raw.get("duration_ms")
        analyses.append(record)

    analyses.sort(key=lambda a: a.started_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return analyses


def _timestamp_from_record(raw: Dict[str, Any]) -> Optional[datetime]:
    ts = str(raw.get("timestamp", ""))
    if not ts:
        return None
    ts = ts.replace("Z", "+00:00")
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
    ):
        try:
            return datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def compute_stage_timings(records: List[LogRecord]) -> List[StageTiming]:
    stage_durations: Dict[str, List[int]] = defaultdict(list)
    for rec in records:
        ev = rec.event
        if ev.endswith("_completed") and ev != "analysis_completed":
            stage = ev[: -len("_completed")]
            dur = rec.raw.get("duration_ms")
            if isinstance(dur, int):
                stage_durations[stage].append(dur)
    timings: List[StageTiming] = []
    for stage, durs in stage_durations.items():
        timings.append(StageTiming(stage=stage, duration_ms=int(sum(durs) / len(durs)) if durs else 0))
    return timings


def compute_endpoint_stats(requests: List[RequestRecord]) -> List[Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "route": "",
            "request_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "durations": [],
            "max_duration_ms": 0,
        }
    )
    for req in requests:
        route = req.route or "/unknown"
        s = stats[route]
        s["route"] = route
        s["request_count"] += 1
        if req.ok and req.status_code is not None and 200 <= req.status_code < 400:
            s["success_count"] += 1
        else:
            s["failure_count"] += 1
        if req.duration_ms is not None:
            s["durations"].append(req.duration_ms)
            s["max_duration_ms"] = max(s["max_duration_ms"], req.duration_ms)

    result: List[Dict[str, Any]] = []
    for route, s in stats.items():
        durs = sorted(s["durations"])
        result.append(
            {
                "route": route,
                "request_count": s["request_count"],
                "success_count": s["success_count"],
                "failure_count": s["failure_count"],
                "success_rate": round(s["success_count"] / s["request_count"] * 100, 2) if s["request_count"] else 0.0,
                "avg_duration_ms": round(sum(durs) / len(durs), 2) if durs else 0.0,
                "p50_duration_ms": round(_percentile(durs, 50), 2) if durs else 0.0,
                "p95_duration_ms": round(_percentile(durs, 95), 2) if durs else 0.0,
                "max_duration_ms": s["max_duration_ms"],
            }
        )
    result.sort(key=lambda x: x["request_count"], reverse=True)
    return result


def compute_analysis_stats(analyses: List[AnalysisRecord]) -> Dict[str, Any]:
    total = len(analyses)
    succeeded = sum(1 for a in analyses if a.status == "success")
    failed = total - succeeded
    durations = sorted([a.duration_ms for a in analyses if a.duration_ms is not None])
    return {
        "total_analyses": total,
        "successful_analyses": succeeded,
        "failed_analyses": failed,
        "avg_duration_ms": round(sum(durations) / len(durations), 2) if durations else 0.0,
        "p50_duration_ms": round(_percentile(durations, 50), 2) if durations else 0.0,
        "p95_duration_ms": round(_percentile(durations, 95), 2) if durations else 0.0,
        "max_duration_ms": max(durations) if durations else 0,
    }


def compute_error_summary(records: List[LogRecord]) -> Dict[str, Any]:
    errors = [r for r in records if r.event in ("request_failed", "analysis_failed")]
    total = len(errors)
    by_endpoint: Counter = Counter()
    by_stage: Counter = Counter()
    by_type: Counter = Counter()
    for r in records:
        if r.event == "request_failed":
            route = r.raw.get("route", r.raw.get("path", "unknown"))
            by_endpoint[route] += 1
            by_type[r.raw.get("error_type", "Unknown")] += 1
        elif r.event == "analysis_failed":
            by_stage[r.raw.get("failed_stage", "unknown")] += 1
            by_type[r.raw.get("error_type", "Unknown")] += 1
    return {
        "total_errors": total,
        "by_endpoint": dict(by_endpoint.most_common(10)),
        "by_stage": dict(by_stage.most_common(10)),
        "by_type": dict(by_type.most_common(10)),
        "recent_errors": [
            {
                "timestamp": r.ist_timestamp,
                "event": r.event,
                "request_id": r.raw.get("request_id", ""),
                "job_id": r.raw.get("job_id", ""),
                "method": r.raw.get("method", ""),
                "route": r.raw.get("route", r.raw.get("path", "")),
                "status": r.raw.get("status_code", ""),
                "failed_stage": r.raw.get("failed_stage", ""),
                "error_type": r.raw.get("error_type", ""),
                "error_message": r.raw.get("error_message", ""),
                "duration_ms": r.raw.get("duration_ms", ""),
            }
            for r in errors[-50:]
        ],
    }


def get_memory_metrics(records: List[LogRecord]) -> Dict[str, Any]:
    rss_vals: List[float] = []
    for r in records:
        rss = r.raw.get("rss_mb")
        pid = r.raw.get("process_id")
        if rss is not None and pid is not None:
            try:
                rss_vals.append(float(rss))
            except (TypeError, ValueError):
                continue
    if not rss_vals:
        return {"available": False, "current_mb": None, "avg_mb": None, "peak_mb": None}
    return {
        "available": True,
        "current_mb": rss_vals[-1],
        "avg_mb": round(sum(rss_vals) / len(rss_vals), 2),
        "peak_mb": max(rss_vals),
    }


def get_last_log_time(records: List[LogRecord]) -> Optional[datetime]:
    for rec in reversed(records):
        ts = rec.timestamp
        if ts is not None:
            return ts
    return None
