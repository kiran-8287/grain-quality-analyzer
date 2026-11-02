# Grain Quality Analyzer Operations Dashboard

The Streamlit dashboard reads Grain Quality Analyzer backend logs and presents operational metrics, analysis-count trends, charts, failed requests, and log-integrity details. It reads logs only: it does not connect to the analysis API, run image analysis, or retain uploaded log files.

## Start the dashboard

Run these commands from the project root. Dashboard packages are separate from the backend requirements:

```powershell
python -m pip install -r monitoring_dashboard/requirements.txt
python -m streamlit run monitoring_dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501). The app uses the project theme at [`.streamlit/config.toml`](../.streamlit/config.toml) and a matching Plotly theme. It has a charcoal canvas, slate-green panels, sage metric values, and blue, amber, and coral series. Charts take the full available width. KPI rows use at most four cards so values are not squeezed into ellipses.

## Choose log sources

The sidebar provides three source modes:

- **Local folder:** reads files matching `app.log*` from the folder field. The default is the project's `logs/` directory; rotated logs and preserved legacy logs are included.
- **Uploaded files:** reads one or more `.log`, `.txt`, or `.jsonl` files selected from the computer. File bytes are processed in memory and are not saved by the dashboard.
- **Both:** combines the local folder and uploaded files. Requests are reconstructed by request ID, and duplicate raw events are removed for resource charts.

The parser accepts current JSON Lines events and the previous timestamp-prefixed JSON log format. Each line must contain a JSON object with an `event` or `type` field. Non-JSON lines are counted as malformed in the log-integrity section. The dashboard does not fetch Render logs; download/export them from the hosting service and upload them here when needed.

Choose a time range in the sidebar: last hour, last 24 hours, last 7 days, last 30 days, or all available. Bounded ranges are anchored to the newest parseable request timestamp in the loaded logs, not the computer's current clock. Requests without timestamps are excluded from bounded windows and reported in the header note; All available includes them. Analysis charts use the analysis event time when available, falling back to the request time. Use **Refresh now** for an immediate reread or enable the 15-second refresh toggle while the backend is writing logs.

## KPI metrics

Request and runtime metrics are grouped into readable rows above the charts:

| Metric | Meaning |
| --- | --- |
| Requests | Request rows reconstructed from start, finish, and analysis events in the selected time window. |
| Success rate | Completed HTTP responses with status below 400 divided by completed responses. |
| 4xx rate / 5xx rate | Client/server error responses divided by completed responses. |
| P50 / P95 / P99 latency | Percentiles of completed request durations with a logged duration. |
| Average requests/min | Request starts divided by the elapsed time between the earliest and latest request timestamps in the window. Shows N/A if a rate cannot be inferred from multiple timestamps. |
| P95 analysis time | 95th-percentile analysis processing time from analysis events in the selected window. |
| Peak / P95 logged RSS | Peak and 95th-percentile process resident memory from logged samples, in MiB. |
| Peak / P95 process CPU | Peak and 95th-percentile process CPU samples. This is process utilization, not whole-host CPU. |
| Max in-flight | Maximum in-flight HTTP request sample in the selected window. |
| Health-check success | Successful logged `/health` responses divided by logged health responses. This is not uptime. |
| Apdex | `(satisfied + 0.5 * tolerating) / completed requests`. Satisfied requests are successful and at or below T; tolerating requests are successful and above T but at or below 4T. Errors and missing durations are dissatisfied. |
| Error budget consumed | Observed 5xx rate divided by the configured 5xx-rate objective. 4xx responses do not consume this budget. |
| Within Apdex target T | Successful completed requests at or below T divided by completed requests. |
| Average grain/object counts | Mean counts across analyses with the relevant result field logged. Whole/broken means exclude analyses whose logged reference length is missing or non-positive; those objects are undetermined rather than whole or broken. |

The Apdex target defaults to 1,000 ms and the error-rate objective defaults to 1%. Both are editable in the sidebar. These are example configuration values, not prescribed or certified service objectives. Health checks are included in the request-level Apdex and outcome rates.

## Graph guide

Every chart is full-width and has an explanation directly beneath it. Hover over points or bars for exact timestamps and values where available.

| Graph | What it shows | How to read it and limits |
| --- | --- | --- |
| **Reconstructed requests per interval** | Number of request rows in each time bucket. | Bucket size is 1 minute for spans up to 6 hours, 15 minutes up to 3 days, 1 hour up to 60 days, and 1 day for longer spans. Timestamps use request start when available and fall back to a finish/analysis event for incomplete legacy rows. Empty intervals between the first and last observed row are filled with zero; there are no leading/trailing empty buckets. Missing logs can still make a quiet interval look like zero traffic. |
| **Request latency by interval** | P50 and P95 completed-request duration for each bucket. Hover also shows the number of requests in that bucket. | A percentile based on a small number of requests can move sharply. Requests without a logged duration are omitted. |
| **Completed requests by HTTP outcome** | Counts grouped as successful (status below 400), client error (4xx), and server error (5xx). | This is a response count, not uptime or an availability percentage over wall-clock time. |
| **Distribution of completed-request latency** | Histogram of request durations; dashed vertical lines mark P50 and P95. | Each bar counts responses in a latency range. Only completed requests with a duration are included. |
| **Completed request volume by endpoint** | Completed request counts by route. | Health-check routes are included and can contribute substantial traffic; compare route labels when interpreting user-facing load. |
| **P95 completed-request latency by endpoint** | P95 response time for each route; hover shows sample count. | Compare routes cautiously when an endpoint has few requests. Requests without durations are excluded. |
| **Process memory (RSS) samples** | Resident memory in MiB over time, colored by log-event type. | Samples are recorded at request/lifecycle event boundaries. This is not continuous memory telemetry and can miss peaks between events. |
| **Process CPU samples** | Process CPU percentage over time, colored by log-event type. | CPU is sampled at request/lifecycle event boundaries, not continuously. Process utilization can exceed 100% when multiple cores are used; brief peaks between samples can be missed. |
| **Uploaded image size distribution by request** | Histogram of logged image byte counts, converted to MiB. | Counts requests with a recorded image size. It excludes multipart request overhead and only includes events that logged the image byte count. |
| **Analysis queue-wait distribution** | Histogram of time spent waiting for an analysis slot, in milliseconds. | Zero means processing began without waiting. Only logged queue-wait values are included. |
| **Rice-grain counts per analysis** | Total rice, whole-grain, and broken-grain counts for each analysis over time. | Uses the analysis event timestamp when available. Whole plus broken can differ from total when whole/broken classification is undetermined. These are predictions, not validated counts. |
| **Foreign objects detected per analysis** | Logged foreign-object count for each analysis over time. | This shows prediction counts only; it does not show false positives or detection accuracy. |

## Request details and log integrity

The **Errors, request details, and log health** expander contains:

- Failed or rejected HTTP requests with timestamps, request IDs, routes, status, duration, and available error/reason fields.
- Parsed-event and malformed-line counts.
- Request-start, request-finish, and unmatched/incomplete lifecycle counts.
- Loaded source files and a table of recent request/analysis fields.

## What the dashboard cannot calculate

Request logs do not contain reviewed expected counts or ground-truth instance masks. Therefore precision, recall, F1, false-positive/negative rates, mask IoU/AP, count MAE/bias, and model/data drift are shown as unavailable. Those require a reviewed evaluation set; segmentation metrics require object masks. Predicted rice/foreign counts and their trends are not measures of model accuracy.

## Log behavior and retention

Backend application, Uvicorn, request, analysis, and lifecycle events are written as JSON Lines to the rotating `GRAIN_LOG_DIR/app.log` file and are not echoed to the terminal. The logger retains up to five 10 MiB backups. Uploaded filenames and image contents are not logged. Local hosted filesystems can be ephemeral; configure persistent storage and `GRAIN_LOG_DIR` if logs must survive restarts or service replacement. Old logs may not contain CPU, upload-size, queue-wait, or grain-count fields, so affected values and graphs can show N/A or be absent.
