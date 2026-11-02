# Grain Quality Analyzer - Backend API

The active FastAPI API exposes `GET /health` and `POST /api/rice-quality/analyze`. The upload route calls the classical computer-vision method in [`../rice_analysis_method/`](../rice_analysis_method/README.md).

Run commands and deployment configuration are documented in the repository [README](../README.md). The API accepts one JPEG, PNG, WebP, TIFF, or BMP image up to 25 MB and 16 megapixels. It validates dimensions before decoding, processes without writing a separate application temp file, and returns lossless source-resolution mask imagery. It analyzes one image at a time and permits two queued analysis requests; further concurrent requests receive HTTP `503`. Request and analysis events, including RSS and process-CPU samples, are logged as rotating JSON Lines in `logs/app.log` by default; the uploaded bytes and filename are excluded from logs. Backend application and Uvicorn logs are written to this file and are not echoed to the terminal.
