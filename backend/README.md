# Backend API

The active FastAPI API exposes `GET /health` and `POST /api/rice-quality/analyze`. The upload route calls the classical computer-vision method in [`../rice_analysis_method/`](../rice_analysis_method/README.md).

Run commands and deployment configuration are documented in the repository [README](../README.md). The API accepts one JPEG, PNG, WebP, TIFF, or BMP image up to 25 MB.
