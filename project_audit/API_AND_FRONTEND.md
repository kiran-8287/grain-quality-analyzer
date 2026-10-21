# API and Frontend Audit

Audit date: 2026-10-21. Routes are registered twice by `backend/app/main.py`: at `/...` and `/api/...`. The UI primarily uses `/api/...`.

## API endpoint inventory

| Method + path | Purpose and input | Response / status notes | Caller and verification |
|---|---|---|---|
| `GET /health`, `/api/health` | Health check; no input. | `HealthResponse`: status/version/service/gpu_available/device. Code hardcodes GPU false/device CPU. | `App.tsx` polls `/health` every 10 seconds. API tests cover. |
| `GET /models`, `/api/models` | Returns `configs/models.json` model registry. | `{ "models": ... }`; current paths/statuses are placeholder. | No primary UI caller found. API tests cover. |
| `GET /standards`, `/api/standards` | Loads named local standards JSON, default `india_kms_2026_27_raw_rice`. | Raw JSON; 404 if missing. No authoritative-source check here. | API tests cover local shape, not compliance. |
| `POST /analyze`, `/api/analyze` | Multipart `file` required, `grade` default `grade_a`, optional `reference_pixels`, `reference_mm`, `profile`. | Analysis response dictionary: `success`, `rice_detected`, `message`, `image`, `sample`, `calibration`, `quality`, `grains[]`, `foreign_matter[]`, `admixture`, `summary`, `standards`, `annotated_image_base64`, `warnings[]`, `processing_time_seconds`, plus `job_id` on success. Empty file 400; pipeline error 400. No route `response_model`. | `frontend/src/App.tsx`; `tests/test_api.py` covers health, models, standards, empty/valid/non-rice and exports. |
| `POST /analyze-ml`, `/api/analyze-ml` | Multipart `file`, optional `px_per_mm`. Experimental Mask R-CNN + YOLO. | Schema `experimental: true`, dimensions, counts/percentages, grains, foreign matter, aggregates, model versions, device, timings. Intended status codes 400/413/415/422/500/503/504. Current startup runtime is unavailable because module import target is absent, so expect 503. | `frontend/src/components/Phase1Demo.tsx` tries `/api/phase1/analyze` then `/api/analyze`; it is not imported by `App.tsx`. The dedicated `tests/test_grain_infer.py` cannot collect due missing import. |
| `GET /analysis/{job_id}`, `/api/analysis/{job_id}` | Job status record. | `JobStatusResponse`: id/status/progress stage/percent/timestamps/error; 404 missing. | No frontend polling found; synchronous route populates a completed RAM record. |
| `GET /analysis/{job_id}/result`, `/api/analysis/{job_id}/result` | Retrieve stored response. | Full result; 404 if missing, 400 unless completed. RAM only; process restart loses it. | No frontend caller found. |
| `GET /analysis/{job_id}/image`, `/api/analysis/{job_id}/image` | Retrieves JPEG bytes decoded from stored annotated base64. | `image/jpeg`; 404 when missing/unavailable. | No current primary UI caller; UI renders base64 in analyze JSON. |
| `GET /analysis/{job_id}/export/json`, `/api/analysis/{job_id}/export/json` | Export stored result JSON. | `application/json`, attachment; 404 if result absent. | `ExportControls.tsx` uses `/api/...`. |
| `GET /analysis/{job_id}/export/csv`, `/api/analysis/{job_id}/export/csv` | Export grain table CSV. | `text/csv`, attachment; 404 if result absent. | `ExportControls.tsx` uses `/api/...`. |

### Main request and response details

`POST /api/analyze` expects browser `multipart/form-data`:

```text
file: UploadFile (required; any content type reaches decoder)
grade: string (optional; default "grade_a")
reference_pixels: float (optional)
reference_mm: float (optional)
profile: string (optional profile name/path)
```

Manual scale is created only when both scale values exist and `reference_pixels > 0`; the route does not itself validate `reference_mm > 0` or finite values. The pipeline/calibration layer must reject invalid data or treat it as unavailable. Actual success response varies for rice/no-rice/error; no strict route schema guarantees optional keys. Per-grain objects contain `id`, `bbox`, `centroid`, `confidence`, segmentation quality/touching/contour, `geometry`, `defects`, and sample-level parameters. `summary` holds total/whole/broken/foreign/defect counts and percentages/reference metadata; see `frontend/src/types.ts` for frontend types and `ml/segmentation/pipeline.py` for assembly. No example below is fabricated as a live sample response.

`POST /api/analyze-ml` expected response types live in `backend/app/schemas/models.py`: `ExperimentalGrainInstance` (class, score, px/mm dimensions, bbox, area, orientation, measurable, flags), foreign detections, aggregate values and timing. Runtime behavior cannot be verified because `backend/app/main.py` imports `backend.models.grain_infer`, while the only matching implementation is `backend/old models/grain_infer.py`; the current path does not resolve.

## Request flow and execution

`App.tsx` builds `FormData`, posts to `/api/analyze`, parses JSON and selects one of: regular result, non-rice result, or error. It does not use job status/result endpoints. `routes.py` awaits the upload read but then directly executes synchronous `job_manager.process_sync` inside an async endpoint. Image analysis is therefore not offloaded to a thread for the production route and there is no background task/queue. Concurrent CPU-heavy uploads may block request processing. Results and jobs are in-process memory only; `run_logger.py` separately writes local artifacts.

The experimental route has safer byte-size/content-type checks, `px_per_mm` validation, a threadpool call and timeout wrapper. Those protections are not shared by `/analyze`. Authentication is absent on all routes. CORS origins are configured through `FRONTEND_ORIGIN`, defaulting to localhost dev origins; methods and headers are wildcard and credentials are enabled.

## Frontend structure

| File/component | Role/data mapping |
|---|---|
| `frontend/src/main.tsx` | React 18 root; renders `App` under StrictMode. |
| `frontend/src/App.tsx` | Main single-page screen and state: health poll, upload handler, result/non-rice/error, selected grain, saved display preferences. No router/history. |
| `UploadSection.tsx` | File selection, drag/drop and camera capture, preview, profile selection and upload control. |
| `ProcessingState.tsx` | Loading animation/progress text; progress is presentational, not server job polling. |
| `GlobalMetrics.tsx` | Maps `summary` total/whole/broken counts and percentages, references and foreign count to cards. Explicitly describes count-based result and standards limitations. |
| `AnnotatedViewer.tsx` | Displays original/annotated base64 images, mask polygons/boxes/IDs and Whole/Broken overlay interaction. |
| `GrainTable.tsx`, `GrainDetailPanel.tsx` | Map `grains[]`, `geometry`, and `defects.broken` into rows/details. |
| `QualityWarningsPanel.tsx` | Displays warnings/parameter status caveats. |
| `ExportControls.tsx` | Requests JSON/CSV by `job_id`. |
| `Navbar.tsx` | Top-level navigation/reset/backend status. |
| `Phase1Demo.tsx` | Separate demo/experimental canvas and request implementation; not imported by `App.tsx`. It tries a nonexistent `/api/phase1/analyze` then falls back to `/api/analyze`. |
| `types.ts` | TypeScript response and metric declarations, including optional legacy/experimental model fields. These types are compile-time only; no runtime response validation. |

The primary UI uses same-origin relative API URLs by default. Components look for `VITE_API_BASE_URL`; `frontend/.env.example`/root `.env.example` instead list `VITE_API_URL`, a configuration name mismatch. Vite dev config should be checked before relying on proxy behavior; `vite.config.ts` defines standard React plugin and no evidenced API proxy. The root `npm run dev` starts backend and Vite; local development likely relies on relative URLs and proxy/deployment arrangement, but live browser integration was not run in this audit.

## API/reliability/security observations

- Confirmed: production upload has no file-size limit, MIME allowlist, or route-level image pixel bound; it reads the whole upload into memory. Decoder validation is downstream.
- Confirmed: synchronous analysis runs directly from async route and may block event loop; no inference timeout applies to `/analyze`.
- Confirmed: job store has no eviction, persistence, auth, or ownership check. Job IDs are UUIDs but endpoints are unauthenticated.
- Confirmed: run logger persists image artifacts and results under local `runs/`; retention/cleanup is not evident. It filters base64 from structured JSON but decodes/saves the annotated image separately.
- Confirmed: `GET /health` says GPU unavailable/CPU even if an experimental model runtime could pick CUDA; it is static, not runtime telemetry.
- Confirmed: registry endpoint reveals configured relative checkpoint paths/status.
- Potential: development CORS configuration is permissive for methods/headers and allows credentials; tighten only with explicit deployment-origin requirements.
- Potential: frontend base URL env var mismatch can misdirect requests in deployed/non-same-origin setups.
- Confirmed: Experimental error handler strips details only partially; request logging middleware includes path/filename in structured analysis logs, so avoid sensitive filenames.
- Not established: reverse-proxy body/time limits, deployment TLS/auth, or production filesystem controls; deployment config is not a complete production security specification.
