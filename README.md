# Rice Grain Quality Analysis

A browser-based tool for analyzing a photo of rice on a green background. Upload a photo or capture one with a camera; the application segments rice grains and foreign matter, displays a color-masked result, and reports image-based counts and measurements.

The live application uses a **classical computer-vision method**, not a trained neural-network model. Its implementation is in [`rice_analysis_method/rice_analysis.py`](rice_analysis_method/rice_analysis.py). Retired models, pipelines, evaluations, and previous application code are preserved under [`other models and methods/`](other%20models%20and%20methods/README.md) and are not imported by the current app.

## What the application shows

- Total detected rice grains and foreign objects.
- Whole and broken grain counts, plus the whole-grain percentage.
- An annotated image with separate masks for whole rice, broken rice, and foreign matter.
- Per-grain measurements, object IDs, a grain inspector, and downloadable masked-image and CSV outputs.
- Image controls for selecting objects, showing or hiding masks and boxes, zooming, and optional panning.

Mask colors are blue for whole grains, yellow for broken grains, red for foreign matter, and gray when a grain's whole/broken status cannot be determined.

Whole/broken status is an image-based estimate: the method estimates a reference length from the uploaded photo and marks rice objects at least 75% of that length as whole. Lengths are in pixels and depend on image scale. This count-based estimate is not a laboratory measurement or a certified weight-based grade.

## Requirements

- Python 3.10 or newer.
- Node.js 18 or newer and npm.
- Windows PowerShell commands are shown below; the Python and npm commands also work in other shells with equivalent virtual-environment activation.

## Run locally

From the repository root, create and activate a virtual environment if needed, then install the backend and frontend dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
npm install
npm --prefix frontend install
```

If PowerShell blocks virtual-environment activation, run the Python commands using `.venv\Scripts\python.exe` directly.

Start both development servers:

```powershell
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). The frontend runs on port `5173`; the backend runs on port `10000`. Vite forwards `/api` and `/health` requests to the backend. To run only one server, use `npm run dev:frontend` or `npm run dev:backend`.

Create a production frontend build with:

```powershell
npm run build
```

The generated static site is in `frontend/dist/`.

## Analyze an image

1. Open the site and choose an image, drag and drop it, or use the camera option.
2. Submit the photo for analysis. Supported formats are JPEG, PNG, WebP, TIFF, and BMP; the upload limit is 25 MB.
3. Review the count cards and annotated result. Select a detected object to inspect its ID and individual mask; use the image controls to adjust the view.
4. Download the masked image or grain data CSV from the results page.

For best results, photograph grains on a green background, spread them apart when possible, and avoid harsh shadows. Touching or overlapping objects may be separated by the image-processing method, so review the resulting masks for difficult samples.

## API

The FastAPI service provides:

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Service health status. |
| `POST` | `/api/rice-quality/analyze` | Analyze an uploaded image. Send it as multipart form data in a field named `file`. |

The analysis response includes `original_image`, `masked_image`, image dimensions, `rice_count`, `foreign_matter_count`, per-object IDs and measurements, and summary statistics. The server also exposes the same router without the `/api` prefix for compatibility (`/rice-quality/analyze` and `/api/health`). Interactive API documentation is available at `/docs` while the backend is running.

## Data folders

The local image sets are available at:

```text
data/rice_count/raw/   # 860 raw images
data/rice_count/test/  # 20 test images
```

These folders are ignored by Git because they contain local data. They are useful for local evaluation and are not required to start the web application. Other historical data and experiment outputs remain in `other models and methods/data_and_runs/`.

## Configuration

Backend settings are read from environment variables:

| Variable | Purpose | Local default |
| --- | --- | --- |
| `PORT` | Backend listening port. | `10000` |
| `FRONTEND_ORIGIN` | Comma-separated allowed browser origins for CORS. | `http://localhost:5173,http://localhost:3000` |
| `LOG_LEVEL` | Backend log level. | `INFO` |
| `GRAIN_LOG_DIR` | Directory for the rotating application log. | `logs/` |

For a separately hosted frontend, set `VITE_API_BASE_URL` in the frontend build environment to the backend origin, for example `https://your-api.example.com`. The frontend template is [`frontend/.env.example`](frontend/.env.example). For local development it can remain blank because Vite uses its development proxy.

## Deployment

[`render.yaml`](render.yaml) defines the Python backend web service. It installs `backend/requirements.txt` and starts Uvicorn on the platform-provided port. Set `FRONTEND_ORIGIN` to the deployed frontend's exact origin (scheme and hostname) so browser requests pass CORS checks. Deploy the Vite frontend separately as a static site using `npm run build` with `frontend/dist/` as the publish directory, and set its `VITE_API_BASE_URL` to the backend origin before building.

The backend can serve the built frontend from `frontend/dist/` when that directory is present in the deployed backend bundle. The current Render blueprint configures only the backend service, so a separately hosted frontend is the documented deployment setup.

## Project structure

```text
backend/                    FastAPI routes, app setup, and backend dependencies
frontend/                   React, TypeScript, and Vite website
rice_analysis_method/       Active rice segmentation and counting implementation
data/rice_count/raw/         Local raw image set (860 images)
data/rice_count/test/        Local test image set (20 images)
docs/                       Current project context and documentation
progress/                   Project work log
other models and methods/   Archived methods, models, data, and prior app code
```

The analysis route imports `label_image` from `rice_analysis_method`; it does not load model weights. The archive is kept for reference and is outside the active application path.

## Troubleshooting

- **Backend fails to start:** Confirm the active virtual environment is selected and install `backend/requirements.txt`.
- **Frontend cannot reach the API:** Check that the backend is running on port `10000`; for separate hosting, verify `VITE_API_BASE_URL` and the backend's `FRONTEND_ORIGIN`.
- **Upload rejected:** Use a supported image format and keep the file at or below 25 MB.
- **No masks or poor separation:** Use a well-lit photo with a green background and reduce grain overlap. The estimates depend on image conditions and should be reviewed visually.

## Further documentation

- [Active analysis method](rice_analysis_method/README.md)
- [Backend API notes](backend/README.md)
- [Current project context](docs/PROJECT_CONTEXT.md)
- [Archived project material](other%20models%20and%20methods/README.md)
