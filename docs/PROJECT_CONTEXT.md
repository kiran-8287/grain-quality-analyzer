# Rice Quality Analysis — Current Project Context

## Active application

- Frontend: React, TypeScript, Vite, and Tailwind in `frontend/`.
- Backend: FastAPI in `backend/app/`.
- Analysis: the classical computer-vision method in `rice_analysis_method/`.
- Image sets: local raw and test photos are available at `data/rice_count/raw/` and `data/rice_count/test/`.
- Live flow: upload or capture one image → `POST /api/rice-quality/analyze` → show masked image, rice and foreign-object counts, grain measurements, and Whole/Broken estimates.

The active method is image processing, not a trained ML model. It segments the green background, detects rice and foreign matter, separates touching rice components, and estimates grain geometry. Whole/Broken is an image-based length rule using a reference estimated from the uploaded image; it is not a certified weight-based grade.

## Run locally

1. Install Python dependencies with `python -m pip install -r backend/requirements.txt`.
2. Install web dependencies with `npm install` and `npm --prefix frontend install`.
3. Start both servers with `npm run dev`.

The backend listens on port 10000 by default. Vite proxies `/api` and `/health` to it. For a separately hosted frontend, set `VITE_API_BASE_URL` to the backend origin.

## Scope

The active website uses only `rice_analysis_method`. The raw and test rice image sets remain available under `data/rice_count/`; the other datasets, retired pipelines, reports, and prior UI implementations are stored in `other models and methods/` for reference and are not imported by the live application.
