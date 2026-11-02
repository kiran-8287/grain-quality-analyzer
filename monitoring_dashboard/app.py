from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from log_parser import build_request_rows, parse_log_lines


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
WINDOWS = {
    "Last hour": pd.Timedelta(hours=1),
    "Last 24 hours": pd.Timedelta(hours=24),
    "Last 7 days": pd.Timedelta(days=7),
    "Last 30 days": pd.Timedelta(days=30),
    "All available": None,
}

_chart_template = go.layout.Template(pio.templates["plotly_dark"].to_plotly_json())
_chart_template.layout.update(
    paper_bgcolor="#121b1d",
    plot_bgcolor="#1a2729",
    font={"color": "#d4dfdb", "family": "Inter, sans-serif"},
    colorway=["#8fc6a5", "#8bb5dc", "#d8b06f", "#dd8982", "#b1a0d4", "#71bac0", "#b5c985"],
    xaxis={"gridcolor": "#344547", "linecolor": "#506264", "zerolinecolor": "#46595b"},
    yaxis={"gridcolor": "#344547", "linecolor": "#506264", "zerolinecolor": "#46595b"},
)
pio.templates["grain_quality_dark"] = _chart_template
pio.templates.default = "grain_quality_dark"

st.set_page_config(
    page_title="Grain Quality Analyzer - Operations",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {
        color-scheme: dark;
        --rice-bg: #121b1d;
        --rice-panel: #1a2729;
        --rice-panel-raised: #223234;
        --rice-border: #34484a;
        --rice-text: #d4dfdb;
        --rice-muted: #b1bfbb;
        --rice-accent: #9bc9a9;
      }
      .stApp { background: var(--rice-bg); color: var(--rice-text); }
      [data-testid="stAppViewContainer"] { background: var(--rice-bg); }
      [data-testid="stHeader"] { background: rgba(18, 27, 29, .94); }
      [data-testid="stSidebar"] { background: #182426; border-right: 1px solid var(--rice-border); }
      .block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; }
      h1, h2, h3 { color: #c3d9cc !important; }
      [data-testid="stMarkdownContainer"] p,
      [data-testid="stWidgetLabel"] p,
      [data-testid="stCaptionContainer"] p,
      label { color: var(--rice-text); }
      [data-testid="stCaptionContainer"] p, .small-muted { color: var(--rice-muted); font-size: .9rem; }
      [data-testid="stMetric"] {
        background: var(--rice-panel);
        border: 1px solid var(--rice-border);
        padding: .9rem 1rem;
        border-radius: 12px;
        box-shadow: 0 5px 18px rgba(0, 0, 0, .16);
      }
      [data-testid="stMetricLabel"] { color: #c0cdc8 !important; }
      [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
        color: var(--rice-accent) !important;
      }
      [data-testid="stMetricDelta"] { color: #dfbd7c !important; }
      [data-testid="stExpander"] {
        background: var(--rice-panel);
        border: 1px solid var(--rice-border);
        border-radius: 10px;
      }
      [data-testid="stFileUploader"] section {
        background: var(--rice-panel);
        border: 1px dashed #526d61;
        border-radius: 12px;
      }
      input, textarea, [data-baseweb="select"] > div {
        background: #151f21 !important;
        color: var(--rice-text) !important;
        border-color: #405658 !important;
      }
      .stButton > button {
        background: #29483d;
        color: #d7e7dc;
        border: 1px solid #52745f;
        border-radius: 9px;
      }
      .stButton > button:hover { background: #365b49; border-color: var(--rice-accent); color: #e0ebe3; }
      [data-testid="stDataFrame"] { border: 1px solid var(--rice-border); border-radius: 9px; }
      hr { border-color: var(--rice-border); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Grain Quality Analyzer Operations Dashboard")
st.caption("Operations and analysis performance, sourced from application event logs")

with st.sidebar:
    st.header("Log source")
    log_dir_value = st.text_input("Local log folder", value=str(DEFAULT_LOG_DIR))
    uploaded_logs = st.file_uploader(
        "Or upload exported logs",
        type=["log", "txt", "jsonl"],
        accept_multiple_files=True,
        help="Uploaded log files are read in memory and are not saved by this dashboard.",
    )
    source_mode = st.radio(
        "Read from",
        ["Local folder", "Uploaded files", "Both"],
        index=0,
        horizontal=False,
    )
    window_label = st.selectbox("Time window", list(WINDOWS), index=1)
    with st.expander("User-configured SLO targets"):
        apdex_target_ms = st.number_input("Apdex satisfied threshold T (ms)", min_value=1, value=1000, step=100)
        error_objective_pct = st.number_input("5xx-rate objective (%)", min_value=0.01, max_value=100.0, value=1.0, step=0.1)
        st.caption("These are editable example objectives, not assumed regulatory or industry limits.")
    auto_refresh = st.toggle("Refresh every 15 seconds", value=False)
    if st.button("Refresh now", width="stretch"):
        st.rerun()
    st.divider()
    st.caption("Metrics describe observed requests only. Quiet periods do not prove service availability.")


def _read_sources() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    all_events: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    total_counts = Counter()

    if source_mode in {"Local folder", "Both"}:
        log_dir = Path(log_dir_value).expanduser()
        if not log_dir.is_absolute():
            log_dir = (PROJECT_ROOT / log_dir).resolve()
        files = (
            sorted({
                path
                for pattern in ("app.log*", "app.legacy.*.log")
                for path in log_dir.glob(pattern)
                if path.is_file()
            })
            if log_dir.exists()
            else []
        )
        for path in files:
            try:
                with path.open("r", encoding="utf-8", errors="replace") as stream:
                    events, counts = parse_log_lines(stream)
                all_events.extend(events)
                source_rows.append({"Source": str(path), **counts})
                total_counts.update(counts)
            except OSError as exc:
                source_rows.append({"Source": str(path), "Read error": type(exc).__name__})

    if source_mode in {"Uploaded files", "Both"}:
        for upload in uploaded_logs or []:
            try:
                text = upload.getvalue().decode("utf-8", errors="replace")
                events, counts = parse_log_lines(text.splitlines())
                all_events.extend(events)
                source_rows.append({"Source": upload.name, **counts})
                total_counts.update(counts)
            except Exception as exc:
                source_rows.append({"Source": upload.name, "Read error": type(exc).__name__})

    return all_events, source_rows, dict(total_counts)


def _as_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for field in ("timestamp", "analysis_timestamp"):
        if field in frame:
            frame[field] = pd.to_datetime(frame[field], utc=True, errors="coerce")
    return frame


def _show_chart(fig: go.Figure, key: str, explanation: str, *, height: int = 430) -> None:
    """Render one full-width chart with consistent sizing and an interpretation note."""
    fig.update_layout(
        height=height,
        margin={"l": 32, "r": 28, "t": 72, "b": 48},
        title_x=0.02,
        title_font_size=19,
        xaxis_automargin=True,
        yaxis_automargin=True,
    )
    st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False})
    st.caption(explanation)


def _bucket_frequency(span_seconds: float) -> str:
    if span_seconds <= 6 * 60 * 60:
        return "1min"
    if span_seconds <= 3 * 24 * 60 * 60:
        return "15min"
    if span_seconds <= 60 * 24 * 60 * 60:
        return "1h"
    return "1D"


@st.fragment(run_every="15s" if auto_refresh else None)
def render_dashboard() -> None:
    events, sources, parse_counts = _read_sources()
    if not events:
        st.warning("No readable application events were found. Check the log folder or upload a JSON/legacy log file.")
        if sources:
            st.dataframe(pd.DataFrame(sources), width="stretch", hide_index=True)
        return

    event_frame = _as_frame(events)
    if {"event", "request_id", "timestamp"}.issubset(event_frame.columns):
        event_frame = event_frame.drop_duplicates(
            subset=["event", "request_id", "timestamp"], keep="last"
        )
    request_frame = _as_frame(build_request_rows(events))
    if request_frame.empty:
        st.warning("Logs contain events, but no request lifecycle records were found.")
        st.dataframe(pd.DataFrame(sources), width="stretch", hide_index=True)
        return

    request_frame["timestamp"] = pd.to_datetime(request_frame.get("timestamp"), utc=True, errors="coerce")
    newest = request_frame["timestamp"].max()
    if pd.isna(newest):
        filtered = request_frame.copy()
        window_events = event_frame.copy()
        excluded_untimed = 0
    else:
        window = WINDOWS[window_label]
        if window is None:
            filtered = request_frame.copy()
            window_events = event_frame.copy()
            excluded_untimed = 0
        else:
            cutoff = newest - window
            filtered = request_frame[request_frame["timestamp"].notna() & (request_frame["timestamp"] >= cutoff)].copy()
            window_events = event_frame[
                event_frame.get("timestamp", pd.Series(index=event_frame.index, dtype="datetime64[ns, UTC]")).notna()
                & (event_frame["timestamp"] >= cutoff)
            ].copy()
            excluded_untimed = int(request_frame["timestamp"].isna().sum())

    if filtered.empty:
        st.info("There are no requests in this time window.")
        return

    analysis_frame = request_frame.copy()
    if "analysis_timestamp" not in analysis_frame:
        analysis_frame["analysis_timestamp"] = analysis_frame["timestamp"]
    else:
        analysis_frame["analysis_timestamp"] = pd.to_datetime(
            analysis_frame["analysis_timestamp"], utc=True, errors="coerce"
        ).combine_first(analysis_frame["timestamp"])
    if not pd.isna(newest) and WINDOWS[window_label] is not None:
        analysis_cutoff = newest - WINDOWS[window_label]
        analysis_frame = analysis_frame[
            analysis_frame["analysis_timestamp"].notna()
            & (analysis_frame["analysis_timestamp"] >= analysis_cutoff)
        ].copy()
    analysis_frame["timestamp"] = analysis_frame["analysis_timestamp"]
    for field in (
        "processing_ms",
        "file_bytes",
        "queue_wait_ms",
        "rice_count",
        "foreign_matter_count",
        "whole_grain_count",
        "broken_grain_count",
        "whole_percent",
        "reference_length",
    ):
        if field not in analysis_frame:
            analysis_frame[field] = pd.NA
        analysis_frame[field] = pd.to_numeric(analysis_frame[field], errors="coerce")
    no_reference = analysis_frame["reference_length"].isna() | (analysis_frame["reference_length"] <= 0)
    analysis_frame.loc[no_reference, ["whole_grain_count", "broken_grain_count"]] = pd.NA

    if "status" not in filtered:
        filtered["status"] = pd.NA
    filtered["status"] = pd.to_numeric(filtered["status"], errors="coerce")
    if "duration_ms" not in filtered:
        filtered["duration_ms"] = pd.NA
    filtered["duration_ms"] = pd.to_numeric(filtered["duration_ms"], errors="coerce")
    for field in ("rss_max_mb", "content_length", "file_bytes", "processing_ms", "queue_wait_ms", "in_flight"):
        if field not in filtered:
            filtered[field] = pd.NA
        filtered[field] = pd.to_numeric(filtered[field], errors="coerce")

    completed = filtered[filtered["status"].notna()].copy()
    failed = completed[completed["status"] >= 400]
    server_errors = completed[completed["status"] >= 500]
    client_errors = completed[(completed["status"] >= 400) & (completed["status"] < 500)]
    latencies = completed["duration_ms"].dropna()
    health = completed[completed.get("path", pd.Series(index=completed.index, dtype=str)).astype(str).str.endswith("/health")]
    health_success = health[health["status"] < 400]
    request_count = len(filtered)
    completed_count = len(completed)
    success_rate = (100 * (completed_count - len(failed)) / completed_count) if completed_count else None
    health_rate = (100 * len(health_success) / len(health)) if len(health) else None
    if len(completed):
        apdex_status = pd.to_numeric(completed["status"], errors="coerce")
        apdex_latency = pd.to_numeric(completed["duration_ms"], errors="coerce")
        satisfied = int(((apdex_status < 400) & (apdex_latency <= apdex_target_ms)).sum())
        tolerating = int(((apdex_status < 400) & (apdex_latency > apdex_target_ms) & (apdex_latency <= 4 * apdex_target_ms)).sum())
        apdex_value = (satisfied + 0.5 * tolerating) / len(completed)
    else:
        apdex_value = None
    server_error_rate = (100 * len(server_errors) / completed_count) if completed_count else None
    error_budget_pct = (server_error_rate / error_objective_pct * 100) if server_error_rate is not None else None

    if len(filtered) > 1 and filtered["timestamp"].notna().sum() > 1:
        span_seconds = (filtered["timestamp"].max() - filtered["timestamp"].min()).total_seconds()
        req_per_min = request_count / (span_seconds / 60.0) if span_seconds > 0 else None
    else:
        req_per_min = None
    p95 = float(latencies.quantile(0.95)) if len(latencies) else None
    p99 = float(latencies.quantile(0.99)) if len(latencies) else None
    rss_series = pd.to_numeric(window_events.get("rss_mb", pd.Series(dtype=float)), errors="coerce").dropna()
    cpu_series = pd.to_numeric(window_events.get("process_cpu_percent", pd.Series(dtype=float)), errors="coerce").dropna()
    concurrency_series = pd.to_numeric(window_events.get("in_flight", pd.Series(dtype=float)), errors="coerce").dropna()

    if pd.isna(newest):
        st.caption("No parseable request timestamps were found; time filtering is unavailable.")
    else:
        st.caption(
            f"Window is anchored to the newest request timestamp in the logs: {newest.strftime('%Y-%m-%d %H:%M UTC')}"
            + (f". Excluded {excluded_untimed} request(s) without timestamps." if excluded_untimed else "")
        )

    st.subheader("Request volume and outcomes")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Requests", f"{request_count:,}", help="Reconstructed from request start/finish events")
    metric_columns[1].metric("Success rate", f"{success_rate:.2f}%" if success_rate is not None else "N/A")
    metric_columns[2].metric("4xx rate", f"{100 * len(client_errors) / completed_count:.2f}%" if completed_count else "N/A", delta_color="inverse")
    metric_columns[3].metric("5xx rate", f"{100 * len(server_errors) / completed_count:.2f}%" if completed_count else "N/A", delta_color="inverse")

    st.subheader("Request latency and throughput")
    latency_columns = st.columns(4)
    latency_columns[0].metric("P50 latency", f"{latencies.quantile(.50):,.0f} ms" if len(latencies) else "N/A")
    latency_columns[1].metric("P95 latency", f"{p95:,.0f} ms" if p95 is not None else "N/A")
    latency_columns[2].metric("P99 latency", f"{p99:,.0f} ms" if p99 is not None else "N/A")
    latency_columns[3].metric(
        "Avg requests/min",
        f"{req_per_min:.2f}" if req_per_min is not None else "N/A",
        help="Request starts divided by elapsed time between the earliest and latest request timestamps in this window.",
    )

    st.subheader("Runtime and capacity")
    runtime_columns = st.columns(4)
    runtime_columns[0].metric("P95 analysis time", f"{analysis_frame['processing_ms'].dropna().quantile(.95):,.0f} ms" if analysis_frame["processing_ms"].notna().any() else "N/A")
    runtime_columns[1].metric("Peak logged RSS", f"{rss_series.max():,.1f} MB" if len(rss_series) else "N/A")
    runtime_columns[2].metric("P95 logged RSS", f"{rss_series.quantile(.95):,.1f} MB" if len(rss_series) else "N/A")
    runtime_columns[3].metric("P95 process CPU", f"{cpu_series.quantile(.95):.1f}%" if len(cpu_series) else "N/A")
    runtime_columns = st.columns(3)
    runtime_columns[0].metric("Peak process CPU", f"{cpu_series.max():.1f}%" if len(cpu_series) else "N/A")
    runtime_columns[1].metric("Max in-flight", f"{concurrency_series.max():.0f}" if len(concurrency_series) else "N/A")
    runtime_columns[2].metric("Health-check success", f"{health_rate:.2f}%" if health_rate is not None else "N/A")

    st.markdown("**Configured service objectives**")
    slo_columns = st.columns(3)
    slo_columns[0].metric("Apdex", f"{apdex_value:.3f}" if apdex_value is not None else "N/A", help="(Satisfied + half of tolerating) / completed requests; HTTP errors count as dissatisfied")
    slo_columns[1].metric("Error budget consumed", f"{error_budget_pct:.1f}%" if error_budget_pct is not None else "N/A", help="Observed 5xx rate divided by the configured 5xx-rate objective; 4xx responses do not consume this budget")
    slo_columns[2].metric("Within Apdex target T", f"{100 * satisfied / len(completed):.1f}%" if len(completed) else "N/A")
    st.caption(f"Apdex T = {apdex_target_ms:,} ms; configured 5xx-rate objective = {error_objective_pct:.2f}%; health checks are included in request-level SLO calculations.")

    st.divider()
    st.subheader("Request performance")
    dated = filtered[filtered["timestamp"].notna()].sort_values("timestamp").copy()
    if not dated.empty:
        span = (dated["timestamp"].max() - dated["timestamp"].min()).total_seconds()
        bucket = _bucket_frequency(span)
        dated["bucket"] = dated["timestamp"].dt.floor(bucket)
        per_bucket = dated.groupby("bucket").agg(requests=("request_id", "count"))
        bucket_start = dated["bucket"].min()
        bucket_end = dated["bucket"].max()
        all_buckets = pd.date_range(bucket_start, bucket_end, freq=bucket)
        per_bucket = per_bucket.reindex(all_buckets, fill_value=0).rename_axis("timestamp").reset_index()

        request_fig = px.bar(
            per_bucket,
            x="timestamp",
            y="requests",
            title=f"Requests started per {bucket} interval",
            labels={"timestamp": "Time (UTC)", "requests": "Requests"},
        )
        request_fig.update_traces(hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>Requests: %{y}<extra></extra>")
        _show_chart(
            request_fig,
            "request-volume-over-time",
            "Each bar counts request starts in that interval. Empty intervals are shown as zero; missing log data can still make an interval look quiet.",
        )

        latency_buckets = dated.dropna(subset=["duration_ms"]).groupby("bucket").agg(
            p50_ms=("duration_ms", lambda values: values.quantile(.50)),
            p95_ms=("duration_ms", lambda values: values.quantile(.95)),
            sample_count=("duration_ms", "count"),
        ).reset_index()
        if not latency_buckets.empty:
            latency_fig = go.Figure()
            for field, label, color in (
                ("p50_ms", "P50", "#8bb5dc"),
                ("p95_ms", "P95", "#d8b06f"),
            ):
                latency_fig.add_trace(go.Scatter(
                    x=latency_buckets["bucket"],
                    y=latency_buckets[field],
                    name=label,
                    mode="lines+markers",
                    line={"color": color, "width": 2},
                    customdata=latency_buckets["sample_count"],
                    hovertemplate=f"%{{x|%Y-%m-%d %H:%M UTC}}<br>{label}: %{{y:,.0f}} ms<br>Requests in interval: %{{customdata}}<extra></extra>",
                ))
            latency_fig.update_layout(
                title=f"Request latency by {bucket} interval",
                xaxis_title="Time (UTC)",
                yaxis_title="Latency (ms)",
                legend={"orientation": "h", "y": 1.12},
            )
            _show_chart(
                latency_fig,
                "request-latency-over-time",
                "P50 is the median and P95 is the latency below which 95% of completed requests fall in each interval. Hover to see the sample count; percentiles from very small samples can vary sharply.",
            )

        status_data = completed.dropna(subset=["status"]).copy()
        if not status_data.empty:
            status_data["Outcome"] = status_data["status"].map(
                lambda status: "Successful (under 400)" if status < 400 else "Client error (4xx)" if status < 500 else "Server error (5xx)"
            )
            status_counts = (
                status_data.groupby("Outcome").size().reindex(
                    ["Successful (under 400)", "Client error (4xx)", "Server error (5xx)"], fill_value=0
                ).rename("requests").reset_index()
            )
            status_fig = px.bar(
                status_counts,
                x="requests",
                y="Outcome",
                orientation="h",
                color="Outcome",
                color_discrete_map={
                    "Successful (under 400)": "#8fc6a5",
                    "Client error (4xx)": "#d8b06f",
                    "Server error (5xx)": "#dd8982",
                },
                title="Completed requests by HTTP outcome",
                labels={"requests": "Completed requests"},
            )
            status_fig.update_layout(showlegend=False, yaxis={"categoryorder": "array", "categoryarray": ["Server error (5xx)", "Client error (4xx)", "Successful (under 400)"]})
            status_fig.update_traces(text=status_counts["requests"], textposition="outside", cliponaxis=False)
            _show_chart(
                status_fig,
                "http-outcome-counts",
                "This groups completed responses by status class. It is a count of logged responses in the selected window, not an availability percentage over wall-clock time.",
            )
        else:
            st.info("No completed HTTP responses with status codes are available in this time window.")

        latency_data = completed.dropna(subset=["duration_ms"])
        if not latency_data.empty:
            latency_fig = px.histogram(
                latency_data,
                x="duration_ms",
                nbins=40,
                title="Distribution of completed-request latency",
                labels={"duration_ms": "Request latency (ms)", "count": "Requests"},
            )
            latency_fig.update_layout(yaxis_title="Requests", bargap=0.06)
            latency_fig.add_vline(x=latency_data["duration_ms"].quantile(.50), line_dash="dash", line_color="#8bb5dc", annotation_text="P50")
            latency_fig.add_vline(x=latency_data["duration_ms"].quantile(.95), line_dash="dash", line_color="#d8b06f", annotation_text="P95")
            _show_chart(
                latency_fig,
                "request-latency-distribution",
                f"Each bar counts completed requests in a latency range. The dashed lines mark P50 and P95; based on {len(latency_data):,} requests with recorded durations.",
            )

        if not completed.empty and "path" in completed:
            by_endpoint = completed.groupby("path").agg(requests=("request_id", "count")).sort_values("requests").reset_index()
            endpoint_fig = px.bar(
                by_endpoint,
                x="requests",
                y="path",
                orientation="h",
                text="requests",
                title="Completed request volume by endpoint",
                labels={"path": "API endpoint", "requests": "Completed requests"},
            )
            endpoint_fig.update_traces(textposition="outside", cliponaxis=False)
            _show_chart(
                endpoint_fig,
                "endpoint-request-volume",
                "Compares completed request counts per route. Health-check routes are included, so separate them from user-facing routes when interpreting traffic.",
            )

            timed_by_endpoint = completed.dropna(subset=["duration_ms"])
            if not timed_by_endpoint.empty:
                endpoint_latency = timed_by_endpoint.groupby("path").agg(
                    p95_ms=("duration_ms", lambda values: values.quantile(.95)),
                    samples=("duration_ms", "count"),
                ).sort_values("p95_ms").reset_index()
                endpoint_latency_fig = px.bar(
                    endpoint_latency,
                    x="p95_ms",
                    y="path",
                    orientation="h",
                    color="p95_ms",
                    color_continuous_scale=["#8bb5dc", "#d8b06f", "#dd8982"],
                    title="P95 completed-request latency by endpoint",
                    labels={"path": "API endpoint", "p95_ms": "P95 latency (ms)"},
                    hover_data={"samples": True},
                )
                _show_chart(
                    endpoint_latency_fig,
                    "endpoint-p95-latency",
                    "Compares the 95th-percentile response time per route. Hover for sample counts; compare cautiously when endpoints have few requests.",
                )
    else:
        st.info("Request timestamps are missing, so time-series charts are unavailable.")

    st.subheader("Runtime sample trends")
    memory_events = window_events.copy()
    if "rss_mb" in memory_events:
        memory_events["rss_mb"] = pd.to_numeric(memory_events["rss_mb"], errors="coerce")
        memory_events = memory_events.dropna(subset=["timestamp", "rss_mb"]).sort_values("timestamp")
    else:
        memory_events = pd.DataFrame()
    if not memory_events.empty:
        memory_fig = px.line(
            memory_events,
            x="timestamp",
            y="rss_mb",
            color="event",
            markers=True,
            title="Process memory (RSS) samples",
            labels={"timestamp": "Time (UTC)", "rss_mb": "Resident memory (MiB)", "event": "Log event"},
        )
        _show_chart(
            memory_fig,
            "process-memory-over-time",
            "Each point is resident memory recorded when a request or lifecycle event was logged. Short peaks between samples are not captured; this is not a continuous memory profile.",
        )
    else:
        st.info("RSS samples are not present in this time window.")

    cpu_events = window_events.copy()
    if "process_cpu_percent" in cpu_events:
        cpu_events["process_cpu_percent"] = pd.to_numeric(cpu_events["process_cpu_percent"], errors="coerce")
        cpu_events = cpu_events.dropna(subset=["timestamp", "process_cpu_percent"]).sort_values("timestamp")
    else:
        cpu_events = pd.DataFrame()
    if not cpu_events.empty:
        cpu_fig = px.line(
            cpu_events,
            x="timestamp",
            y="process_cpu_percent",
            color="event",
            markers=True,
            title="Process CPU samples",
            labels={"timestamp": "Time (UTC)", "process_cpu_percent": "Process CPU (%)", "event": "Log event"},
        )
        _show_chart(
            cpu_fig,
            "process-cpu-over-time",
            "CPU is sampled at request/lifecycle log events, not continuously. Process utilization can exceed 100% when multiple CPU cores are used; brief peaks between samples may be missed.",
        )
    else:
        st.info("Process CPU samples are not present in this time window.")

    upload_sizes = analysis_frame["file_bytes"].dropna() / (1024 * 1024)
    if len(upload_sizes):
        upload_fig = px.histogram(
            x=upload_sizes,
            nbins=30,
            title="Uploaded image size distribution by request",
            labels={"x": "Image size (MiB)", "count": "Analyses"},
        )
        upload_fig.update_layout(yaxis_title="Analyses", bargap=0.06)
        _show_chart(
            upload_fig,
            "upload-size-distribution",
            f"Each bar counts requests whose uploaded image size falls in that range. Based on {len(upload_sizes):,} logged image sizes; multipart request overhead is excluded.",
        )
    else:
        st.info("Image byte counts are not present in analysis events for this window.")

    queue_wait = analysis_frame["queue_wait_ms"].dropna()
    if len(queue_wait):
        queue_fig = px.histogram(
            x=queue_wait,
            nbins=30,
            title="Analysis queue-wait distribution",
            labels={"x": "Queue wait (ms)", "count": "Analyses"},
        )
        queue_fig.update_layout(yaxis_title="Analyses", bargap=0.06)
        _show_chart(
            queue_fig,
            "queue-wait-distribution",
            f"Each bar counts analyses by time spent waiting for an available processing slot. Based on {len(queue_wait):,} logged queue-wait values; zero means processing began without waiting.",
        )
    else:
        st.info("Analysis queue wait is not recorded in this time window.")

    st.divider()
    st.subheader("Analysis output trends")
    analyzed = analysis_frame[analysis_frame["rice_count"].notna()].copy()
    if not analyzed.empty:
        count_columns = st.columns(3)
        count_columns[0].metric("Analyzed images", f"{len(analyzed):,}")
        count_columns[1].metric("Avg rice grains", f"{analyzed['rice_count'].mean():.1f}")
        count_columns[2].metric("Avg foreign objects", f"{analyzed['foreign_matter_count'].mean():.1f}" if analyzed["foreign_matter_count"].notna().any() else "N/A")
        whole_broken_columns = st.columns(2)
        whole_broken_columns[0].metric("Avg whole grains", f"{analyzed['whole_grain_count'].mean():.1f}" if analyzed["whole_grain_count"].notna().any() else "N/A")
        whole_broken_columns[1].metric("Avg broken grains", f"{analyzed['broken_grain_count'].mean():.1f}" if analyzed["broken_grain_count"].notna().any() else "N/A")

        count_fields = [field for field in ("rice_count", "whole_grain_count", "broken_grain_count") if analyzed[field].notna().any()]
        dated_analyses = analyzed.dropna(subset=["timestamp"]).sort_values("timestamp")
        if count_fields and not dated_analyses.empty:
            count_long = dated_analyses.melt(
                id_vars=["timestamp"],
                value_vars=count_fields,
                var_name="Grain measure",
                value_name="Grains",
            )
            count_long["Grain measure"] = count_long["Grain measure"].map({
                "rice_count": "Rice grains (total)",
                "whole_grain_count": "Whole grains",
                "broken_grain_count": "Broken grains",
            })
            grain_count_fig = px.line(
                count_long,
                x="timestamp",
                y="Grains",
                color="Grain measure",
                markers=True,
                title="Rice-grain counts per analysis",
                labels={"timestamp": "Analysis time (UTC)", "Grains": "Detected grains"},
            )
            grain_count_fig.update_traces(connectgaps=False)
            _show_chart(
                grain_count_fig,
                "grain-counts-per-analysis",
                "Each point is one completed analysis. Total, whole, and broken rice counts use the same scale. Whole/broken values are omitted when the logged reference length is missing or non-positive; undetermined grains can make whole plus broken differ from total.",
            )

        foreign_analyses = dated_analyses.dropna(subset=["foreign_matter_count"])
        if not foreign_analyses.empty:
            foreign_fig = px.line(
                foreign_analyses,
                x="timestamp",
                y="foreign_matter_count",
                markers=True,
                title="Foreign objects detected per analysis",
                labels={"timestamp": "Analysis time (UTC)", "foreign_matter_count": "Foreign objects"},
            )
            foreign_fig.update_traces(line={"color": "#dd8982", "width": 2}, connectgaps=False)
            _show_chart(
                foreign_fig,
                "foreign-counts-per-analysis",
                "Each point is the foreign-object count returned for one completed analysis. This is a prediction trend, not a measure of detection accuracy.",
            )
    else:
        st.info("No completed analysis count events are present in this time window.")

    with st.expander("Model quality metrics requiring labeled ground truth"):
        st.warning(
            "Precision, recall, F1, IoU/mAP, false-positive/false-negative rates, count MAE, and drift cannot be computed from request logs alone. "
            "They need image-level reviewed labels (and instance masks for segmentation metrics). These are intentionally shown as unavailable rather than estimated from predictions."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {"Metric": name, "Value": "Not available", "Required data": requirement}
                    for name, requirement in [
                        ("Precision / Recall / F1", "Reviewed object labels per image"),
                        ("Mask IoU / AP", "Reviewed ground-truth instance masks"),
                        ("Count MAE / bias", "Trusted expected counts per test image"),
                        ("Data or model drift", "Stable reference set and feature/prediction baselines"),
                    ]
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    with st.expander("Errors, request details, and log health"):
        if not failed.empty:
            st.markdown("**Failed or rejected requests**")
            columns = [field for field in ("timestamp", "request_id", "method", "path", "status", "duration_ms", "error_type", "reason") if field in failed]
            st.dataframe(failed.sort_values("timestamp", ascending=False)[columns], width="stretch", hide_index=True)
        else:
            st.success("No failed HTTP requests in this time window.")

        request_end_count = int(filtered.get("request_finished", pd.Series(False, index=filtered.index)).fillna(False).sum())
        request_start_count = int(filtered.get("request_started", pd.Series(False, index=filtered.index)).fillna(False).sum())
        integrity = pd.DataFrame(
            [
                {"Check": "Parsed events", "Value": parse_counts.get("parsed", 0)},
                {"Check": "Malformed/non-JSON lines", "Value": parse_counts.get("malformed", 0)},
                {"Check": "Requests with start events", "Value": request_start_count},
                {"Check": "Requests with finish events", "Value": request_end_count},
                {"Check": "Unmatched/incomplete requests", "Value": int((~filtered.get("request_started", pd.Series(False, index=filtered.index)).fillna(False) | ~filtered.get("request_finished", pd.Series(False, index=filtered.index)).fillna(False)).sum())},
            ]
        )
        st.markdown("**Log integrity**")
        st.dataframe(integrity, width="stretch", hide_index=True)
        st.markdown("**Loaded sources**")
        st.dataframe(pd.DataFrame(sources), width="stretch", hide_index=True)
        st.markdown("**Recent requests**")
        visible_columns = [field for field in ("timestamp", "request_id", "method", "path", "status", "duration_ms", "processing_ms", "queue_wait_ms", "rss_max_mb", "file_bytes", "rice_count", "foreign_matter_count", "whole_grain_count", "broken_grain_count") if field in filtered]
        st.dataframe(filtered.sort_values("timestamp", ascending=False)[visible_columns].head(200), width="stretch", hide_index=True)


render_dashboard()
