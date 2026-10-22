# Testing, Environment, and Runbook

Audit date: 2026-10-21.

## Runtime environments observed

| Environment | Python / package evidence | Use and limitations |
|---|---|---|
| Global `python` | Python 3.14.5; FastAPI 0.142.4, Uvicorn 0.54.0, Ultralytics 8.4.171, Torch 2.13.0, torchvision 0.28.0, OpenCV 4.13.0.92, NumPy 2.4.6, Pillow 12.2.0, Pydantic 2.13.5, pytest 9.1.1, python-multipart 0.0.32, SciPy 1.18.0, scikit-image 0.26.0, scikit-learn 1.9.0. | This is the environment used for pytest, app import and frontend-triggering commands. |
| `backend/.venv` | Python 3.11.9; FastAPI 0.156.6, Uvicorn 0.54.0, Ultralytics 8.4.183, Torch 2.6.0+cu124, torchvision 0.21.0+cu124, OpenCV 5.0.0.93, NumPy 2.4.6, Pillow 13.0.0, Pydantic 2.14.1, pytest 9.1.1 plus CV packages. | Contains web and YOLO11 experiment dependencies. No training/inference run was done in this audit. |
| Root `.venv` | Python 3.12.4; Ultralytics 8.4.183, Torch 2.14.1+cu130, torchvision 0.29.1+cu130, OpenCV 5.0.0.93, NumPy 2.5.2, Pillow 12.3.0. FastAPI/Uvicorn/Pydantic/pytest not installed. | Used only for the safe local checkpoint load. |

Declared `backend/requirements.txt` dependencies are minimum ranges rather than a complete lock: `ultralytics>=8.3.0`, `fastapi>=0.110.0`, `imagehash>=4.3.1`, `matplotlib>=3.7.0`, `numpy>=1.24.0`, `opencv-python>=4.8.0`, `pillow>=10.0.0`, `pydantic>=2.6.0`, `pytest>=8.0.0`, `python-multipart>=0.0.9`, `scikit-image>=0.21.0`, `scikit-learn>=1.3.0`, `scipy>=1.10.0`, `uvicorn>=0.28.0`, `PyYAML>=6.0`. The historical `backend/old models/requirements-ml.txt` is separate (Torch, torchvision, pycocotools, OpenCV, NumPy, Ultralytics). No `requirements.txt` pins Python itself. Frontend versions are pinned by package locks; `frontend/package.json` declares React/ReactDOM `^18.3.1`, Vite `^6.1.0`, TypeScript `^5.7.3`, Tailwind `^3.4.17`; build resolved Vite 6.4.3.

## Environment variables

| Name | Source/use | Required? |
|---|---|---|
| `LOG_LEVEL` | `backend/app/main.py`, default INFO. | Optional. |
| `FRONTEND_ORIGIN` | `backend/app/main.py`, comma-separated CORS origin list. | Optional; default localhost:5173 and localhost:3000. |
| `PORT` | `backend/app/main.py` script entry, default 10000; deployment uses `$PORT`. | Optional. |
| `GRAIN_LOG_DIR` | `monitoring/app.py`, overrides local log directory. | Optional. |
| `PYTEST_CURRENT_TEST` | Internal test-mode detection in `run_logger.py`. | Set by pytest. |
| `VITE_API_BASE_URL` | Frontend `App.tsx`, `ExportControls.tsx`, `Phase1Demo.tsx`; empty means relative URL. | Optional, but set to backend origin if Vite is not proxying API. |
| `GRAIN_MASKRCNN_CHECKPOINT`, `GRAIN_YOLO_WEIGHTS`, `GRAIN_DEVICE`, `GRAIN_TILE_OVERLAP`, `GRAIN_MASK_SCORE_THRESHOLD`, `GRAIN_YOLO_IMGSZ`, `GRAIN_YOLO_CONF`, `GRAIN_INFER_TIMEOUT_SECONDS`, `GRAIN_MAX_UPLOAD_BYTES`, `GRAIN_MAX_IMAGE_PIXELS`, `GRAIN_MASK_NMS_IOU`, `GRAIN_SUPPRESS_FOREIGN_OVERLAP`, `GRAIN_FOREIGN_MASK_IOU` | Read by `backend/old models/grain_infer.py` experimental runtime. | Optional there; model weights are required to run it. Current runtime import path is broken, so these variables are not operational through the API as checked out. |

Root and `backend/.env.example` include `BACKEND_HOST`, `BACKEND_PORT`, `DEBUG`, `MAX_UPLOAD_SIZE_MB`, `UPLOAD_DIR`, `ALLOWED_ORIGINS`, `MODELS_DIR`, `USE_GPU`, `GPU_DEVICE`, `MAX_IMAGE_PIXELS`, `TILE_SIZE`, `BATCH_SIZE`, `VITE_API_URL`; active source does not read most of them. There is a confirmed frontend configuration-name mismatch: code reads `VITE_API_BASE_URL`, examples name `VITE_API_URL`. The backend script defaults to port 10000, while examples list 8000. Do not copy secret values into the handover; no actual `.env` file was present in the inspected root listings. `.env.example` is safe after checking it contains placeholders only.

## Commands and verification

### Commands executed during this audit

| Exact command | Result |
|---|---|
| `python --version` | Completed, exit 0: Python 3.14.5. |
| `node --version` | Completed, exit 0: v24.15.0. |
| `npm --version` | Completed, exit 0: 11.6.1. |
| `.venv\Scripts\python.exe --version` | Completed, exit 0: Python 3.12.4. |
| `python -m pytest -q` | Failed, exit 1 during collection: `tests/test_grain_infer.py` imports nonexistent `backend.models.grain_infer` (`ModuleNotFoundError`). No tests were executed in this full-suite invocation. |
| `python -m pytest -q --ignore=tests/test_grain_infer.py` | Failed, exit 1: 80 passed, 2 failed, 1 warning. Failing tests: `test_case2a_only_rice_boxes_are_kept_after_gate_filter`, `test_case3_rice_plus_foreign_matter`; gate reports rice present (one rice detection), but segmentation reports zero grain instances and final `rice_detected=false`. Warning: Starlette TestClient/httpx deprecation. |
| `npm run build` | Passed, exit 0: `tsc` succeeded; Vite 6.4.3 transformed 1,596 modules and built `frontend/dist`. |
| `python -c "from backend.app.main import app; print(app.title); print(sorted(app.openapi()['paths']))"` | Passed, exit 0: FastAPI imports as `GRAIN QUALITY ANALYZER API`; OpenAPI includes root and `/api` forms of the documented routes. Import success does not load experimental model runtime (lifespan runs on startup). |
| `.venv\Scripts\python.exe -c "from ultralytics import YOLO; ... YOLO(local_path) ..."` | Passed, exit 0, about 20 s: all three local files load; each is `task=detect` with 80 COCO classes. No rice inference was performed. |

No application code was modified by this audit. No model training, package installation, network download, test-image inference, external upload, or standards verification was performed.

## Running the application (source-grounded, not end-to-end verified)

Current source registers FastAPI at port 10000 when run directly, not the 8000 described in some `.env.example`/README text. Root `package.json` declares:

```powershell
npm run dev
```

It starts `python backend/app/main.py` and `cd frontend && npm run dev`. This exact command was not launched. Frontend uses same-origin relative `/api/...` unless `VITE_API_BASE_URL` is set; `vite.config.ts` has no configured API proxy. Therefore set `VITE_API_BASE_URL=http://localhost:10000` for this source layout or configure a proxy before expecting the dev UI to reach FastAPI. The README's documented default at port 8000 is stale versus `main.py`'s port 10000.

Backend only (source-grounded):

```powershell
python backend/app/main.py
```

Equivalent deployment-style command from `render.yaml`:

```powershell
uvicorn backend.app.main:app --host 0.0.0.0 --port $env:PORT
```

No production deployment start or live HTTP session was exercised. Root `npm run build` is the verified frontend build command.

## Test inventory and smoke tests

- `tests/test_api.py`: FastAPI health/models/standards/analyze and JSON/CSV route tests.
- `tests/test_rice_gate.py`: synthetic object fixtures for non-rice, rice, rice+stone and gate/segmentation integration; currently contains two confirmed failures described above.
- `tests/test_whole_vs_broken.py`: reference hierarchy, calibration, threshold and physical-profile utility logic.
- `tests/test_logging.py`: lifecycle, timing, no image bytes/base64 and event consistency.
- `tests/test_monitoring.py`: JSON/prefixed log parsing, event reconstruction and aggregation.
- `tests/test_grain_infer.py`: experimental tiling/NMS/measurement/API tests, currently uncollectable because implementation import is missing.
- `tests/grain/generate_synthetic.py`: creates synthetic COCO training images; not a test of real rice performance.
- `test_pipeline.py`, `scripts/validate_*`: legacy/manual validation scripts and image sets. Not run as part of this audit.

The existing test suite uses synthetic generated inputs in several places. Passing unit tests do not establish field accuracy, precision/recall, mask IoU, physical measurement validity, or standards compliance.

## Safe next smoke commands (not run)

After restoring/fixing the experimental module path and confirming both required checkpoints locally, the repository's intended experimental test would be:

```powershell
python -m pytest tests/test_grain_infer.py -q
```

The referenced checkpoint defaults in the old guide (`runs/grain/maskrcnn/best.pth` and `runs/grain/yolo/exp/weights/best.pt`) were not found in the current folder listing; a weight location must be supplied if intentionally running that experiment. Do not run training merely to satisfy this audit.

To start only the main API in a configured environment:

```powershell
python backend/app/main.py
```

Then check `http://localhost:10000/health` and `/docs`. These exact live HTTP steps were not run here.

## Troubleshooting and blockers

1. Full pytest is blocked at collection by missing `backend.models.grain_infer`; similarly named code under `backend/old models/` is not importable at the expected module path.
2. Excluding that test yields two real integration failures in the current gate/segmentation contract. Debug the earliest failing segmentation stage with the same fixtures; preserve gate behavior.
3. Frontend calls same-origin paths and has no Vite proxy; set the actual `VITE_API_BASE_URL` env name to the API host when running on separate ports.
4. The primary configured model registry points to missing placeholder checkpoints. Normal analysis falls back to classical CV.
5. There is no complete reviewed training export or current rice checkpoint. Do not report synthetic/demo metrics as real-data validation.
6. Check local `monitoring/logs/`/`runs/` artifacts before sharing externally; they may contain user image files, result images or filenames. Never share `.env` values, credentials, private images or raw dataset without explicit curation.
