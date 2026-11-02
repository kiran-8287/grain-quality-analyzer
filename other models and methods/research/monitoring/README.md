# Grain Quality Analyzer — Monitoring Dashboard

## Overview

Internal read-only Streamlit dashboard for observing the Grain Quality Analyzer backend.

The dashboard reads structured JSON Lines logs written by the application. It never clears, rotates, truncates, or deletes logs.

## Log Location

- Primary log file: `monitoring/logs/app.log`
- Rotated files: `monitoring/logs/app.log.1`, `monitoring/logs/app.log.2`, etc.
- Log rotation: size-based, 10 MB max, 5 backups (configured in `backend/app/main.py`)

## Startup

Install dashboard dependencies (if not already installed):

    pip install streamlit pandas plotly

Start the dashboard:

    C:\Python314\python.exe -m streamlit run monitoring/app.py

> **Note:** On this machine, the `streamlit` command may resolve to a Python 3.12
> installation with an incompatible `starlette` version. Using the Python 3.14
> module invocation above avoids that issue.

Environment variables:
- `GRAIN_LOG_DIR` — optional override for the log directory (default: `monitoring/logs/`)

## Dashboard Tabs

### Overview
Top-level health state and key metrics:
- Total API Requests
- Success Rate
- Failed Requests
- Average Response Time
- P95 Response Time
- Peak / Current Concurrency
- Peak Process RSS Memory
- Last Log Event

Health states:
- GREEN = Healthy
- YELLOW = Warning (elevated error rate > 10%)
- RED = Critical (stale logs or no activity)

### System Health
- Request activity over time (per minute)
- Process RSS Memory (current / average / peak)
- Log freshness indicator
- Backend activity state

### API Performance
Per-endpoint statistics:
- route
- request_count
- success_count / failure_count
- success_rate
- avg / p50 / p95 / max duration_ms

Charts:
- Average vs P95 vs Maximum latency
- Highest volume endpoints
- Highest failure endpoints

### Analysis Performance
Grain Quality Analyzer specific metrics:
- total / successful / failed analyses
- avg / p50 / p95 / max analysis duration

Stage timing (average):
- image decode
- rice gate
- segmentation
- geometry
- broken classification
- annotation

### Analysis History
Recent analysis jobs table:
- timestamp (IST)
- job_id
- request_id
- status
- processing_time_ms
- rice_detected
- grain_count / whole_count / broken_count / undetermined_count
- reference_source
- failed_stage

Filterable by status and date.

### Errors & Failures
- total errors
- error rate
- failed API requests
- failed analyses

Charts:
- failures by endpoint
- failures by pipeline stage
- failures over time

Recent errors table with all fields visible.

### Live Log View
Recent structured log events with filters:
- event
- stage
- job_id
- request_id
- errors only
- endpoint / route

Newest first, limited to last 500 matching events for performance.

## Refresh

- Refresh Now button (re-reads logs)
- Auto-refresh: Off (use Refresh Now manually)
- Cached for 5 seconds by Streamlit

**Refresh never modifies, deletes, rotates, or truncates logs.**

## Data Retention

- Logs are persistent historical records
- Dashboard is read-only
- Application-side rotation controls disk growth
- No destructive admin controls

## Log Format

Structured JSON Lines. Each line is a JSON object, optionally prefixed by the
Python standard logger format.

### Supported formats

**FORMAT A — pure JSONL:**
```json
{"timestamp":"...","event":"request_started",...}
```

**FORMAT B — Python logging prefix + JSON:**
```
2026-10-08 03:07:48,214 [INFO] grain_structured: {"timestamp":"...","event":"request_started",...}
```

The parser extracts the JSON starting from the first `{` on the line. This
allows it to handle both the clean JSONL test fixtures and the real production
log lines emitted by Python's `logging` module.

### Minimum fields for request records
- `timestamp`
- `event` (`request_started` / `request_finished` / `request_failed`)
- `request_id`
- `method`
- `path` / `route`
- `status_code`
- `ok`
- `duration_ms`
- `error_type` / `error_message` (on failure)

### Minimum fields for analysis records
- `timestamp`
- `event` (`analysis_started` / `analysis_completed` / `analysis_failed`)
- `job_id`
- `request_id` (linked when available; optional for analysis records)
- `status`
- `total_processing_ms`
- `failed_stage` / `error_type` / `error_message` (on failure)

### Stage timing records
- `duration_ms`
- Stage-specific fields (`grain_count`, `rice_detected`, etc.)

### System records (when available)
- `process_id`
- `rss_mb`

## Request vs Analysis Duration

- **Request duration**: HTTP request lifecycle (serialization, network, etc.)
- **Analysis duration**: actual image processing pipeline time

These are not identical. The dashboard shows both separately.

## Timezone

Timestamps stored internally in UTC. Displayed in IST (UTC+5:30).

## Security

The dashboard and logger never log:
- uploaded image bytes
- base64 image data
- secrets / passwords
- .env contents
- authorization headers

## Testing

Unit tests for the log parser and dashboard metrics:

    pytest tests/test_monitoring.py -v

Test fixture (synthetic data, clearly labeled):
- `tests/fixtures/monitoring_sample.log`

## Known Limitations

- This is an internal monitoring dashboard built on persistent structured logs
- Not a centralized observability platform
- Memory metrics require explicit `rss_mb` / `process_id` fields in log records
- Rotated log discovery follows `app.log*` naming convention
