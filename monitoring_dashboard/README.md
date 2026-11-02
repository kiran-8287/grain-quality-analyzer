# Operations dashboard

This optional Streamlit app reads Grain Quality Analyzer's local or exported logs. It does not call the analysis API or store uploaded log files.

## Run it

From the repository root, install dashboard dependencies into the active Python environment and start Streamlit:

```powershell
python -m pip install -r monitoring_dashboard/requirements.txt
python -m streamlit run monitoring_dashboard/app.py
```

By default it reads `logs/app.log*`, including rotated logs and the preserved legacy log. Use the sidebar to change the folder, upload exported `.log`/`.txt`/`.jsonl` files, choose a time range, or enable 15-second refresh. The dashboard expects the backend's structured request events; it can also parse the previous timestamp-prefixed JSON log format. The dashboard uses a dark charcoal and sage palette: canvas `#121B1D`, panels `#1A2729`, sage values `#9BC9A9`, and muted blue/amber/coral chart accents. Metric cards are grouped into at most four columns so long values have room to render. The palette uses Streamlit theme settings and dashboard color-contrast guidance ([Streamlit theme docs](https://docs.streamlit.io/develop/concepts/configuration/theming-customize-colors-and-borders), [Michelin dashboard color principles](https://designsystem.michelin.com/data-visualization/design-guidelines/principles-of-a-dashboard)).

## Metrics

- Request volume, success rate, 4xx/5xx rates, throughput, endpoint volume, and HTTP status distribution.
- Request latency P50/P95/P99, analysis latency, queue wait, and latency histograms.
- RSS and process CPU samples, in-flight requests, uploaded image sizes, and health-check success rate.
- Rice, whole/broken, and foreign-object count trends from completed analyses.
- Log parse health, unmatched lifecycle events, and recent failed requests.

Charts use the selected log window, render full-width, and include a note explaining the measure and its limits. Request volume is bucketed by interval; latency charts use completed requests with recorded durations; endpoint comparisons include sample counts; runtime lines show event-boundary samples rather than continuous monitoring. Analysis-count charts use the analysis event timestamp when it is logged. Empty intervals are explicit zeroes only for request-start volume, and untimed requests are excluded from bounded time windows.

Precision, recall, F1, mask IoU/AP, count MAE, and drift are explicitly reported as unavailable because request logs do not contain reviewed ground-truth labels. Add a reviewed evaluation set to calculate those quality metrics.

Logs contain operational metadata, not image bytes or source filenames. Backend application, Uvicorn, request, and lifecycle logs are written only to the rotating `GRAIN_LOG_DIR/app.log` file; they are not echoed to the terminal. Duplicate Uvicorn access lines are disabled because request middleware records request activity. File logs are local and may be ephemeral on a hosted service; use persistent storage for retention across host replacement.
