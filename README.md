# GRAIN QUALITY ANALYZER

Grain Quality Analyzer is an image-based rice grain inspection application for uploading grain images, detecting rice material, segmenting individual grains, and computing mask-based geometric measurements and Whole vs Broken classification for each detected grain.

## Current Phase

**Phase 1 — Rice Grain Detection, Segmentation, Geometry, and Whole/Broken Classification**

The system is currently performing:
- Rice/non-rice detection (classical CV rice gate)
- Individual grain segmentation and masking (classical CV)
- Per-grain geometric measurements (length, breadth, L/B ratio, area, solidity)
- Total grain count from actual segmentation instances
- Whole vs Broken classification (rule-based geometric classification)
- CSV/JSON export of measurements

Future phases will add defect classification, foreign matter detection, admixture analysis, and official standards screening once validated models and datasets are approved.

## Project Structure

Only files that are part of the shipped application are tracked in this
repository. Editor/IDE and AI-assistant working files (e.g. `AGENTS.md`,
`currenttask.md`, `.kilo/`) are kept locally only and are ignored by Git.

```
A:\grain\
├── .pytest_cache/            # Python test cache (ignored)
├── backend/                  # FastAPI backend
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py     # API endpoints
│   │   ├── schemas/
│   │   │   └── models.py     # Pydantic models
│   │   ├── services/
│   │   │   ├── export.py     # JSON/CSV export
│   │   │   ├── job_manager.py
│   │   │   └── run_logger.py
│   │   └── main.py           # FastAPI entry point
│   ├── .env.example
│   ├── package-lock.json
│   └── requirements.txt
├── configs/                  # Configuration files
│   ├── app.json
│   ├── inference.json
│   ├── models.json
│   ├── parameters.json
│   └── thresholds.json
├── data/                     # Data and audit files (ignored)
│   ├── audit/
│   ├── runs/
│   └── datasets/
├── docs/                     # Documentation
├── frontend/                 # React + Vite frontend
│   ├── src/
│   │   ├── components/       # UI components
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   └── types.ts
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   └── tsconfig.json
├── grain_profiles/           # Grain reference profiles
├── logs/                     # Application logs (ignored)
├── ml/                       # Machine learning modules
│   ├── config.py
│   ├── quality/              # Quality classification modules
│   │   ├── geometry.py
│   │   ├── calibration.py
│   │   └── profiles.py
│   ├── segmentation/         # Segmentation pipeline modules
│   │   ├── inference.py
│   │   ├── pipeline.py
│   │   ├── postprocessing.py
│   │   ├── preprocessing.py
│   │   ├── rice_gate.py
│   │   └── segmentation.py
│   └── standards/            # Standards comparison modules
├── models/                   # Model artifacts and mappings (ignored binaries)
│   ├── foreign_matter/
│   └── segmentation/
├── node_modules/             # Frontend dependencies (ignored)
├── package.json              # Root package.json (concurrently)
├── package-lock.json
├── progress/                  # Progress and work logs (ignored)
├── render.yaml               # Render deployment config
├── scripts/                  # Utility and audit scripts
├── standards/                # Standards data files
├── test_images/              # Test and validation images
├── tests/                    # Backend tests
├── .env.example              # Environment variables template
├── .gitignore
├── PROJECT_CONTEXT.md
└── README.md
```

## Quick Start

From the repository root (`A:\grain\`):

```bash
# Install root dependencies (concurrently)
npm install

# Start both backend and frontend
npm run dev
```

This starts:
- **Backend**: FastAPI server at `http://localhost:8000`
- **Frontend**: Vite dev server at `http://localhost:5173`

### Run services separately

```bash
# Backend only
npm run dev:backend

# Frontend only
npm run dev:frontend
```

### Build frontend

```bash
npm run build
```

## Backend

```bash
python backend/app/main.py
```

API docs available at: `http://localhost:8000/docs`

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: `http://localhost:5173`

## Current Capabilities

### Phase 1 — Detection, Segmentation, Geometry, and Whole/Broken
- Upload rice grain images or use camera capture
- Rice/non-rice detection via classical CV rice gate
- Individual grain segmentation and masking (classical CV)
- Per-grain mask visualization with unique IDs
- Total grain count from segmentation results
- Per-grain geometric measurements:
  - Length (pixels or mm with calibration)
  - Breadth (pixels or mm with calibration)
  - L/B Ratio
  - Mask area, solidity, confidence
- Whole vs Broken classification (rule-based relative-length classification, 0.75 threshold)
- CSV/JSON export of current measurements

### Production Pipeline Notes
- The current production pipeline uses classical computer vision, not YOLO or deep learning models.
- Previous YOLO experiments were intentionally removed and are not part of the production baseline.
- Any future model experiments (e.g. YOLOv8, Qwen) must be explicitly approved and isolated from production.

### Foreign Matter
- Foreign matter is detected as a heuristic (color/contour) fallback within the rice/non-rice gate.
- It is reported separately and never treated as rice, but the detector is untrained — treat as experimental.
- Trained foreign-matter detection is not implemented.

### Calibration
- Physical millimetre measurements require calibration (e.g., ArUco-based calibration).
- Without calibration, measurements are reported in pixels.
- A pixel reference is not universal across different camera distances.

### Not Yet Implemented (validated)
- Defect classification (damaged, discolored, chalky, red, dehusked, immature, sprouted/weevilled)
- Admixture analysis
- Official standards screening and grading

## Notes

- No training pipeline is committed in this branch.
- No dataset downloads or model checkpoints are included.
- Future dataset selection and model training will be implemented in a separate phase after manifest review and source-group safety checks.
- Test images are validation/demo assets, not training data.
- Logs are runtime artifacts and remain ignored by Git.
