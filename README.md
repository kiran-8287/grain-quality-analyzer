# Grain Quality Analyzer

Grain Quality Analyzer is a browser application for inspecting rice sample images. Upload a photo or capture one with a camera to see detected rice grains, foreign objects, masks, counts, and per-grain measurements.

## Current application

- **Frontend:** React, TypeScript, Vite, and Tailwind in `frontend/`.
- **Backend:** FastAPI in `backend/app/`.
- **Active image method:** classical computer vision in `rice_analysis_method/rice_analysis.py`.
- **Optional operations dashboard:** Streamlit in `monitoring_dashboard/`.

The active application does not load trained model weights. The backend calls the active method's `label_image` function to find rice and foreign matter and to split touching grains. Retired models, experiments, and the previous application are kept in [`other models and methods/`](other%20models%20and%20methods/README.md) and are not part of the live analysis path.

## Image analysis and results

The upload page accepts JPEG, PNG, WebP, TIFF, and BMP images. It presents the input image, then reports:

- Total rice grain and foreign-object counts.
- Whole and broken grain counts and the whole-grain percentage.
- An annotated image with masks for whole grains, broken grains, and foreign matter.
- IDs and measurements for each detected object, plus an isolated mask preview when an object is selected.
- Downloadable annotated image and grain-data CSV.

Mask colors are blue for whole grains, yellow for broken grains, red for foreign matter, and gray for rice when whole/broken status cannot be determined. The image viewer supports selection, optional IDs and boxes, mask visibility, pointer-centered zoom, and optional pan mode. IDs and boxes start hidden; masks start visible and pan starts off.

The current CV method analyzes at 40% of the source image's width and height, then maps detected masks to source-image coordinates for display. Very fine details can be lost at the working resolution. The original upload is returned unchanged for formats the browser can display; TIFF is converted for browser display. Masks and annotated output are encoded as lossless PNG. Camera captures are JPEG encoded at quality 98.

Whole/broken status is an image-based estimate. The method estimates a whole-grain reference length from the uploaded sample image and labels a rice object whole when its measured length is at least 75% of that reference. Measurements are in pixels and depend on image scale. This does not replace a laboratory or weight-based regulatory grade.

## Requirements

- Python 3.10 or newer.
- Node.js 18 or newer and npm.
- Windows PowerShell commands are shown below. Use the equivalent virtual-environment commands for other shells.

## Run locally

From the repository root, install the backend and frontend dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
npm install
npm --prefix frontend install
```

If PowerShell blocks environment activation, call `.venv\Scripts\python.exe` directly for Python commands.

Start both development servers:

```powershell
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). Vite serves the frontend and forwards `/api` and `/health` requests to the backend on port `10000`. Backend application and Uvicorn logs are written to `logs/app.log`; frontend Vite/npm messages remain in the terminal.

The servers can also be started separately:

```powershell
npm run dev:backend
npm run dev:frontend
```

Create a production frontend bundle with:

```powershell
npm run build
```

The static site is written to `frontend/dist/`.

## Image and processing limits

- Maximum uploaded image file: 25 MiB.
- Maximum multipart request body: 26 MiB, including multipart overhead.
- Maximum decoded image dimensions: 16 megapixels total.
- Minimum image side: 8 pixels.
- Accepted formats: JPEG, PNG, WebP, TIFF, and BMP.
- The backend processes one analysis at a time and allows two analyses to wait. Additional concurrent analysis requests receive HTTP `503` and can be retried.

The backend checks image dimensions before decoding pixels. It handles EXIF orientation, grayscale, alpha, and 16-bit input (normalized to 8-bit for the active CV method). Analysis happens in memory; uploaded filenames and image contents are not written to the application log. Chunked request bodies may be temporarily spooled by the request middleware and the temporary spool is closed after the request.

## API

The FastAPI service provides:

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Health status and service name. |
| `POST` | `/api/rice-quality/analyze` | Analyze one multipart image in the `file` field. |

For compatibility, the analysis route and health route are also available without the `/api` prefix at `/rice-quality/analyze` and `/health`. OpenAPI documentation is available at `/docs` while the backend is running.

The analysis response includes `original_image`, `masked_image`, `image_width`, `image_height`, `rice_count`, `foreign_matter_count`, an `objects` list with IDs/categories/masks/measurements, and a `summary` object with grain counts and aggregate measurements. Images are returned as data URLs.

## Operations dashboard

The optional Streamlit dashboard reads local and rotated backend logs or exported log files. Install its dependencies separately from the backend and start it from the repository root:

```powershell
python -m pip install -r monitoring_dashboard/requirements.txt
python -m streamlit run monitoring_dashboard/app.py
```

It opens at [http://localhost:8501](http://localhost:8501). The dashboard includes request/analysis latency, HTTP outcomes, endpoint activity, runtime samples, upload and queue distributions, image-count trends, and log-integrity views. It can refresh every 15 seconds. It reports model-quality measures such as precision, recall, mask IoU, and count error as unavailable because logs do not include reviewed ground truth. See the [full dashboard guide](monitoring_dashboard/README.md) for every metric, graph, source option, and interpretation note.

## Logs, privacy, and configuration

Backend settings are read from environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `PORT` | Backend listening port. | `10000` |
| `FRONTEND_ORIGIN` | Comma-separated browser origins allowed by CORS. | `http://localhost:5173,http://localhost:3000` |
| `LOG_LEVEL` | Minimum level for regular backend/Uvicorn logs. Request/lifecycle events remain enabled. | `INFO` |
| `GRAIN_LOG_DIR` | Directory for rotating JSON Lines application logs. | `logs/` |

Every HTTP request writes start and finish events with a request ID, method, path, status, duration, in-flight count, process RSS, and CPU sample. Image analyses also log queue wait, processing time, dimensions, result counts, and upload size. File contents and original filenames are not logged. Uvicorn access lines are disabled because the middleware records request activity without client IP addresses.

Application, Uvicorn, request, analysis, and lifecycle events are written to `GRAIN_LOG_DIR/app.log` and are not echoed to the terminal. The file rotates at 10 MiB and keeps up to five backups. Existing non-JSON `app.log` files are preserved as `app.legacy.<timestamp>.log` at startup; new events are JSON Lines. Local and hosted filesystems may be ephemeral, so configure `GRAIN_LOG_DIR` to persistent storage if logs must survive service replacement. Uvicorn/platform stdout logs are not a substitute for a persistent log directory.

## Local image sets

The local image folders are:

```text
data/rice_count/raw/   # 860 raw images
data/rice_count/test/  # 20 test images
```

They are ignored by Git and are not required to run the application. Historical datasets and experiment outputs are preserved under `other models and methods/data_and_runs/`.

## Deployment

[`render.yaml`](render.yaml) defines the Python backend service with the Render slug `grain-quality-analyzer`. The current blueprint deploys the backend only. Deploy the Vite frontend separately as a static site:

1. Set the backend's `FRONTEND_ORIGIN` to the deployed frontend's exact HTTPS origin.
2. Set `VITE_API_BASE_URL` in the frontend build environment to the backend origin. The local template is [`frontend/.env.example`](frontend/.env.example).
3. Build with `npm run build` and publish `frontend/dist/`.
4. If log retention is needed, configure persistent storage and point `GRAIN_LOG_DIR` to it. Hosted instance files may otherwise disappear on restart or replacement.

The backend can serve `frontend/dist/` when the bundle is present in its deployment. The Render blueprint does not currently build or publish the frontend as part of the backend service.

## Project layout

```text
backend/                    FastAPI app, routes, logging, and backend dependencies
frontend/                   React, TypeScript, and Vite website
rice_analysis_method/       Active classical image-analysis implementation
monitoring_dashboard/       Optional Streamlit log and performance dashboard
.streamlit/config.toml      Dashboard theme
data/rice_count/raw/         Local raw image set (860 images)
data/rice_count/test/        Local test image set (20 images)
docs/                       Current developer-facing project context
other models and methods/   Archived methods, models, datasets, and previous app
```

## Troubleshooting

- **Backend fails to start:** Activate the intended Python environment and install `backend/requirements.txt`.
- **Frontend cannot reach the API:** Confirm the backend is running; for separate hosting, check `VITE_API_BASE_URL` and `FRONTEND_ORIGIN`.
- **Image is rejected:** Confirm its format, file size (at most 25 MiB), pixel count (at most 16 megapixels), and minimum side length (8 pixels).
- **Analysis is busy:** One analysis is active and up to two wait; retry after a `503` response.
- **Dashboard shows no logs:** Confirm `GRAIN_LOG_DIR` or the sidebar folder contains `app.log*`, or upload exported `.log`, `.txt`, or `.jsonl` files.
- **Dashboard values show `N/A`:** Older log formats may not contain CPU, upload, queue, or grain-count fields. Newer backend events populate those values.
- **Masks are incomplete or objects touch:** Use a well-lit image with grains spread apart on a green background. Review the masks because touching objects and details smaller than the working resolution can affect results.

## Further reading

- [Backend API notes](backend/README.md)
- [Active analysis method](rice_analysis_method/README.md)
- [Current project context](docs/PROJECT_CONTEXT.md)
- [Archived methods and application](other%20models%20and%20methods/README.md)
