"""
Grain Quality Analyzer — Internal Monitoring Dashboard.

Run with:
    C:\\Python314\\python.exe -m streamlit run monitoring/app.py

This dashboard is read-only. It never clears, rotates, truncates,
or deletes application logs.
"""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from log_parser import (
    DEFAULT_LOG_DIR,
    DEFAULT_LOG_FILE,
    compute_analysis_stats,
    compute_endpoint_stats,
    compute_error_summary,
    compute_stage_timings,
    get_last_log_time,
    get_memory_metrics,
    parse_logs,
    reconstruct_analyses,
    reconstruct_requests,
)

st.set_page_config(
    page_title="Grain Quality Analyzer — Monitoring",
    page_icon="🌾",
    layout="wide",
)

RESOLVED_LOG_DIR = Path(os.environ.get("GRAIN_LOG_DIR", DEFAULT_LOG_DIR))


@st.cache_data(ttl=5, show_spinner="Loading logs...")
def load_data(max_lines: int = 200_000) -> dict:
    parse = parse_logs(RESOLVED_LOG_DIR, max_lines=max_lines)
    records = parse.records
    requests = reconstruct_requests(records)
    analyses = reconstruct_analyses(records)
    return {
        "records": records,
        "requests": requests,
        "analyses": analyses,
        "parse_errors": parse.parse_errors,
        "malformed_lines": parse.malformed_lines,
        "files_read": parse.files_read,
        "lines_read": parse.lines_read,
        "records_loaded": parse.records_loaded,
        "stage_timings": compute_stage_timings(records),
        "endpoint_stats": compute_endpoint_stats(requests),
        "analysis_stats": compute_analysis_stats(analyses),
        "error_summary": compute_error_summary(records),
        "memory": get_memory_metrics(records),
        "last_log_time": get_last_log_time(records),
    }


def _ist_label(dt: datetime | None) -> str:
    if dt is None:
        return "N/A"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    offset_minutes = 5 * 60 + 30
    ist = datetime.fromtimestamp(dt.timestamp() + offset_minutes * 60, tz=timezone.utc)
    return ist.strftime("%Y-%m-%d %H:%M:%S IST")


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _health_state(last_log_time: datetime | None, error_count: int, req_count: int, parsed_count: int) -> tuple[str, str]:
    if parsed_count == 0:
        return "RED", "No log activity detected."
    if last_log_time is None:
        return "RED", "Logs could not be parsed."
    now = datetime.now(timezone.utc)
    age_seconds = (now - last_log_time).total_seconds()
    if age_seconds > 300:
        return "RED", f"Logs stale — last event {int(age_seconds)}s ago."
    if req_count > 0 and error_count / max(req_count, 1) > 0.1:
        return "YELLOW", f"Elevated error rate: {error_count} errors / {req_count} requests."
    return "GREEN", f"Backend healthy — last event {_ist_label(last_log_time)}."


def overview_tab(data: dict) -> None:
    requests = data["requests"]
    analyses = data["analyses"]
    last_log_time = data["last_log_time"]
    error_count = data["error_summary"]["total_errors"]
    req_count = len(requests)
    parsed_count = data["records_loaded"]
    failed_reqs = sum(1 for r in requests if not r.ok)
    succeeded_reqs = req_count - failed_reqs
    durations = [r.duration_ms for r in requests if r.duration_ms is not None]
    durs_sorted = sorted(durations)
    avg_dur = sum(durs_sorted) / len(durs_sorted) if durs_sorted else 0.0
    p50_dur = _percentile(durs_sorted, 50) if durs_sorted else 0.0
    p95_dur = _percentile(durs_sorted, 95) if durs_sorted else 0.0
    max_dur = max(durs_sorted) if durs_sorted else 0

    peak_concurrency = 0
    cur_concurrency = 0
    events_by_ts: dict = defaultdict(int)
    for r in requests:
        if r.started_at:
            events_by_ts[r.started_at] += 1
        if r.finished_at:
            events_by_ts[r.finished_at] -= 1
    con = 0
    for ts in sorted(events_by_ts):
        con += events_by_ts[ts]
        peak_concurrency = max(peak_concurrency, con)
        cur_concurrency = con

    state, detail = _health_state(last_log_time, error_count, req_count, parsed_count)
    color_map = {"GREEN": "green", "YELLOW": "orange", "RED": "red"}
    st.markdown(
        f"### Backend Status: :{color_map.get(state, 'gray')}[{state}] — {detail}"
    )

    mem = data["memory"]
    peak_rss = mem.get("peak_mb") if mem.get("available") else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total API Requests", req_count)
    c2.metric("Successful Requests", succeeded_reqs)
    c3.metric("Failed Requests", failed_reqs)
    c4.metric("Success Rate", f"{(succeeded_reqs / max(req_count, 1) * 100):.1f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Avg Response Time", f"{avg_dur:.1f} ms")
    c6.metric("P50 Response Time", f"{p50_dur:.1f} ms")
    c7.metric("P95 Response Time", f"{p95_dur:.1f} ms")
    c8.metric("Max Response Time", f"{max_dur} ms")

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Peak Concurrency", peak_concurrency)
    c10.metric("Current Concurrency", cur_concurrency)
    c11.metric("Last Log Event", _ist_label(last_log_time))
    c12.metric("Peak Process RSS", f"{peak_rss:.1f} MB" if peak_rss is not None else "Not available")


def system_health_tab(data: dict) -> None:
    requests = data["requests"]
    last_log_time = data["last_log_time"]
    mem = data["memory"]

    st.subheader("Request activity over time")
    if requests:
        df = pd.DataFrame(
            [
                {
                    "timestamp": r.started_at,
                    "duration_ms": r.duration_ms or 0,
                    "ok": r.ok,
                }
                for r in requests
                if r.started_at is not None
            ]
        )
        if not df.empty:
            df = df.sort_values("timestamp")
            df["minute"] = df["timestamp"].dt.floor("min")
            per_min = df.groupby("minute").size().reset_index(name="requests")
            fig = px.bar(per_min, x="minute", y="requests", title="Requests per minute")
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("No request data available yet.")

    st.subheader("Process RSS Memory")
    if mem.get("available"):
        st.metric("Current RSS", f"{mem['current_mb']:.1f} MB")
        st.metric("Average RSS", f"{mem['avg_mb']:.1f} MB")
        st.metric("Peak RSS", f"{mem['peak_mb']:.1f} MB")
    else:
        st.info("Process RSS memory not available in logs.")

    st.subheader("Log Freshness")
    if last_log_time:
        st.write(f"Last log event: {_ist_label(last_log_time)}")
        now = datetime.now(timezone.utc)
        age = (now - last_log_time).total_seconds()
        if age < 60:
            st.success(f"Logs are fresh ({int(age)}s old)")
        elif age < 300:
            st.warning(f"Logs slightly stale ({int(age)}s old)")
        else:
            st.error(f"Logs are stale ({int(age)}s old)")
    else:
        st.error("No recent log activity detected.")


def api_performance_tab(data: dict) -> None:
    stats = data["endpoint_stats"]
    if not stats:
        st.info("No API request data yet.")
        return

    st.subheader("Endpoint Statistics")
    df = pd.DataFrame(stats)
    cols = ["route", "request_count", "success_count", "failure_count", "success_rate",
            "avg_duration_ms", "p50_duration_ms", "p95_duration_ms", "max_duration_ms"]
    df = df[cols]
    st.dataframe(df, width="stretch")

    st.subheader("Latency distribution")
    plot_df = df.melt(
        id_vars=["route"],
        value_vars=["avg_duration_ms", "p50_duration_ms", "p95_duration_ms", "max_duration_ms"],
        var_name="metric",
        value_name="ms",
    )
    fig = px.bar(plot_df, x="route", y="ms", color="metric", barmode="group", title="Latency by endpoint")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Highest volume endpoints")
    top_vol = df.nlargest(5, "request_count")
    st.dataframe(top_vol[["route", "request_count", "success_rate"]], width="stretch")

    st.subheader("Highest failure endpoints")
    top_fail = df[df["failure_count"] > 0].nlargest(5, "failure_count")
    if not top_fail.empty:
        st.dataframe(top_fail[["route", "failure_count", "success_rate"]], width="stretch")
    else:
        st.success("No endpoint failures recorded.")


def analysis_performance_tab(data: dict) -> None:
    analyses = data["analyses"]
    stage_timings = data["stage_timings"]
    stats = data["analysis_stats"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Analyses", stats["total_analyses"])
    c2.metric("Successful", stats["successful_analyses"])
    c3.metric("Failed", stats["failed_analyses"])
    c4.metric("Analysis Success Rate", f"{(stats['successful_analyses'] / max(stats['total_analyses'], 1) * 100):.1f}%")

    c5, c6, c7 = st.columns(3)
    c5.metric("Avg Analysis Time", f"{stats['avg_duration_ms']:.1f} ms")
    c6.metric("P95 Analysis Time", f"{stats['p95_duration_ms']:.1f} ms")
    c7.metric("Max Analysis Time", f"{stats['max_duration_ms']} ms")

    st.subheader("Analysis Stage Timing (average)")
    if stage_timings:
        stage_df = pd.DataFrame(
            [
                {
                    "stage": t.stage.replace("_", " ").title(),
                    "avg_duration_ms": t.duration_ms,
                }
                for t in stage_timings
            ]
        )
        st.bar_chart(stage_df.set_index("stage")["avg_duration_ms"])
    else:
        st.info("No stage timing data available.")

    st.subheader("API response time vs Analysis processing time")
    requests = data["requests"]
    if requests:
        analysis_map = {a.job_id: a for a in analyses}
        rows = []
        for r in requests:
            jid = r.job_id
            if jid and jid in analysis_map:
                a = analysis_map[jid]
                rows.append(
                    {
                        "job_id": jid,
                        "api_duration_ms": r.duration_ms or 0,
                        "analysis_duration_ms": a.duration_ms or 0,
                        "analysis_status": a.status,
                    }
                )
        if rows:
            comp_df = pd.DataFrame(rows)
            fig = px.scatter(comp_df, x="api_duration_ms", y="analysis_duration_ms",
                             color="analysis_status", title="API response time vs Analysis time", trendline="ols")
            st.plotly_chart(fig, width="stretch")


def analysis_history_tab(data: dict) -> None:
    analyses = data["analyses"]
    if not analyses:
        st.info("No analysis history yet.")
        return

    st.subheader("Filter")
    status_filter = st.selectbox("Status", ["All", "success", "failed"], index=0)
    filtered = analyses
    if status_filter != "All":
        filtered = [a for a in analyses if a.status == status_filter]

    rows = []
    for a in filtered:
        rows.append(
            {
                "Timestamp (IST)": _ist_label(a.started_at),
                "job_id": a.job_id,
                "request_id": a.request_id or "-",
                "filename": a.filename or "-",
                "status": a.status,
                "processing_ms": a.duration_ms or 0,
                "rice_detected": a.rice_detected,
                "grain_count": a.grain_count,
                "whole_count": a.whole_count,
                "broken_count": a.broken_count,
                "undetermined_count": a.undetermined_count,
                "reference_source": a.reference_source or "-",
                "segmentation_method": a.segmentation_method or "-",
                "failed_stage": a.failed_stage or "-",
                "error_type": a.error_type or "-",
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch")


def errors_tab(data: dict) -> None:
    es = data["error_summary"]
    st.metric("Total Errors", es["total_errors"])

    requests = data["requests"]
    failed_reqs = sum(1 for r in requests if not r.ok)
    analyses = data["analyses"]
    failed_analyses = sum(1 for a in analyses if a.status == "failed")
    c1, c2, c3 = st.columns(3)
    c1.metric("Failed API Requests", failed_reqs)
    c2.metric("Failed Analyses", failed_analyses)
    c3.metric("Error Rate", f"{(es['total_errors'] / max(len(requests), 1) * 100):.1f}%")

    if es["by_endpoint"]:
        st.subheader("HTTP failures by endpoint")
        st.bar_chart(pd.Series(es["by_endpoint"]))

    if es["by_stage"]:
        st.subheader("Analysis failures by pipeline stage")
        st.bar_chart(pd.Series(es["by_stage"]))

    if es["by_type"]:
        st.subheader("Errors by exception type")
        st.bar_chart(pd.Series(es["by_type"]))

    if es["recent_errors"]:
        st.subheader("Recent Errors")
        err_df = pd.DataFrame(es["recent_errors"])
        st.dataframe(err_df, width="stretch")


def logs_tab(data: dict) -> None:
    records = data["records"]
    st.subheader("Live Log View")
    filters: dict = {
        "event": st.text_input("Event filter", ""),
        "level": st.text_input("Level filter", ""),
        "route": st.text_input("Route filter", ""),
        "job_id": st.text_input("Job ID filter", ""),
        "request_id": st.text_input("Request ID filter", ""),
        "errors_only": st.checkbox("Errors only"),
    }
    filtered = records
    if filters["event"]:
        filtered = [r for r in filtered if filters["event"].lower() in r.event.lower()]
    if filters["level"]:
        filtered = [r for r in filtered if filters["level"].lower() in r.raw.get("level", "").lower()]
    if filters["route"]:
        filtered = [r for r in filtered if filters["route"].lower() in (r.raw.get("route", "") or r.raw.get("path", "")).lower()]
    if filters["job_id"]:
        filtered = [r for r in filtered if filters["job_id"] in str(r.raw.get("job_id", ""))]
    if filters["request_id"]:
        filtered = [r for r in filtered if filters["request_id"] in str(r.raw.get("request_id", ""))]
    if filters["errors_only"]:
        filtered = [r for r in filtered if r.event in ("request_failed", "analysis_failed")]

    display = []
    for r in filtered[-500:]:
        display.append(
            {
                "Timestamp": r.ist_timestamp,
                "event": r.event,
                "level": r.raw.get("level", ""),
                "request_id": r.raw.get("request_id") or None,
                "job_id": r.raw.get("job_id") or None,
                "route": r.raw.get("route", r.raw.get("path", "")),
                "stage": r.raw.get("stage", None),
                "status": r.raw.get("status", None),
                "duration_ms": r.raw.get("duration_ms", None),
            }
        )
    if display:
        st.dataframe(pd.DataFrame(display), width="stretch")
    else:
        st.info("No log events match the current filters.")


def diagnostics_expander(data: dict) -> None:
    with st.expander("Log Parser Diagnostics"):
        st.write(f"Resolved log directory: `{RESOLVED_LOG_DIR}`")
        st.write(f"Log file pattern: `{DEFAULT_LOG_FILE.name}`")
        st.write(f"Files found/read: {data['files_read']}")
        st.write(f"Lines read: {data['lines_read']}")
        st.write(f"Structured records parsed: {data['records_loaded']}")
        st.write(f"Malformed lines: {data['malformed_lines']}")
        st.write(f"Parse errors (capped): {data['parse_errors']}")
        st.write(f"Latest structured timestamp: {_ist_label(data['last_log_time'])}")
        latest_event = "N/A"
        if data["records"]:
            latest_event = data["records"][-1].event
        st.write(f"Latest event: {latest_event}")
        st.caption("Parser supports both pure JSONL and Python logging-prefixed JSON.")


def main() -> None:
    st.title("Grain Quality Analyzer")
    st.caption("Backend Monitoring Dashboard — Read-Only")

    with st.sidebar:
        st.header("Controls")
        max_lines = st.slider("Max log lines to load", 1000, 500_000, 200_000, 1000)
        if st.button("Refresh Now"):
            st.cache_data.clear()
            st.rerun()
        st.write(f"Log directory: `{RESOLVED_LOG_DIR}`")
        st.write(f"Log file: `{DEFAULT_LOG_FILE.name}`")
        st.markdown("---")
        st.markdown("Auto-refresh: Off (use Refresh Now)")

    try:
        data = load_data(max_lines=max_lines)
    except Exception as exc:
        st.error(f"Failed to load logs: {exc}")
        return

    tab_overview, tab_health, tab_api, tab_analysis, tab_history, tab_errors, tab_logs = st.tabs(
        [
            "Overview",
            "System Health",
            "API Performance",
            "Analysis Performance",
            "Analysis History",
            "Errors & Failures",
            "Live Log View",
        ]
    )

    with tab_overview:
        overview_tab(data)

    with tab_health:
        system_health_tab(data)

    with tab_api:
        api_performance_tab(data)

    with tab_analysis:
        analysis_performance_tab(data)

    with tab_history:
        analysis_history_tab(data)

    with tab_errors:
        errors_tab(data)

    with tab_logs:
        logs_tab(data)

    diagnostics_expander(data)

    st.markdown("---")
    st.caption(
        "Dashboard is read-only. Application-side log rotation may still occur independently."
    )


if __name__ == "__main__":
    main()
