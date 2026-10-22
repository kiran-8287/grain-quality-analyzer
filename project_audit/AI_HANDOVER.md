# AI Handover — Grain Quality Analyzer

**Audit snapshot:** 2026-10-21, workspace `A:\grain`. This briefing is self-contained, based on files actually inspected and commands actually run. The current checkout is authoritative; prior work-log entries and older docs conflict with it.

## Post-audit workspace update — 2026-10-23

The rice-count experiments are organized under `rice_count_pipeline/`, with shared source photos and datasets in the repository-level `data/rice_count/` folder. Separate method code remains in `rice_count_pipeline/methods/{autolabel_proxy,area_count,weak_count_model,yolo_segmentation}`. The single proxy implementation is `methods/autolabel_proxy/autolabel.py`; `backend/models/autolabel.py` is only a compatibility import. This remains an isolated experiment and does not activate YOLO in production. See `progress/WORK_LOG.md` for cleanup history.

## What the project does

It is a React/TypeScript + Vite frontend and a FastAPI/Python service for image-based rice-grain segmentation, geometry, and Whole/Broken estimation. The primary UI uploads or captures an image, posts it to `/api/analyze`, and renders the returned annotated JPEG, grain details, summary, warnings and CSV/JSON download controls. The main production path in the current checkout is classical CV because no matching rice model checkpoint is configured/present.

## Important current-state conflicts

- `PROJECT_CONTEXT.md`, `README.md`, `docs/CURRENT_STATUS.md`, and `configs/models.json` describe a clean classical-CV baseline with experiments removed/placeholder.
- `progress/WORK_LOG.md` has later entries claiming YOLO26 integration and several isolated YOLO11/Mask R-CNN experiments. Some later backend YOLO11 dataset scripts and raw images do exist, but no YOLO26 application code/checkpoint path is present. Current model config still points to placeholder weights.
- Current `backend/app/main.py` and `routes.py` do register experimental `/analyze-ml`; startup tries `from backend.models.grain_infer import ...`, but that file does not exist. Similar code is located in `backend/old models/grain_infer.py`, a different non-package path. The experimental tests import the missing module and make full pytest collection fail.
- Three local checkpoint files load with Ultralytics, but all are `task=detect` with the standard 80 COCO labels. They are not rice-grain instance-segmentation weights.
- `frontend/src/App.tsx` describes `default_rice` as trusted, while the profile JSON labels it Proxy and the geometry resolver keeps pixel references limited/non-production. Treat as scale-specific proxy only.

Do not “restore YOLO26” based only on a work-log paragraph. First inspect active paths, git status and configured checkpoints.

## Architecture and full image flow

```text
UploadSection / App.tsx
  -> POST /api/analyze (multipart image, grade, optional profile)
  -> routes.py reads all bytes and calls job_manager.process_sync inline
  -> RiceQualityPipeline.analyze decodes image as RGB
  -> detect_rice_presence -> rice_gate.py + shared foreground extraction
       -> non-rice: no-rice response; stop before grain measurements
       -> rice: calibration -> phase1_analyze_image
  -> inference.py checks absent legacy YOLOv8/Mask R-CNN paths, selects classical CV
  -> segmentation.py extracts Otsu foreground, removes noise, connected components,
     optional watershed/pinch splitting, cropped instance masks + offsets
  -> pipeline filters grains using gate results; keeps foreign detections separate
  -> geometry.py computes pixel geometry; optional valid calibration yields mm
  -> profile/reference resolver + 0.75 length ratio gives Whole/Broken or Undetermined
  -> heuristic/optional defect, quality, admixture and standards helpers run
  -> response + annotated base64 JPEG; job is held in RAM and artifacts/logs written
  -> App.tsx displays result and ExportControls requests CSV/JSON
```

The API call is synchronous from the UI's perspective and the main route runs analysis inline; the job object is an in-memory record, not a queue. No persistent database/history store was found. `run_logger.py` writes local run artifacts, including images.

## Main modules and files

- API: `backend/app/main.py`, `backend/app/api/routes.py`, `backend/app/schemas/models.py`.
- App job handling/logging/export: `backend/app/services/job_manager.py`, `run_logger.py`, `export.py`.
- Pipeline: `ml/segmentation/pipeline.py`, `inference.py`, `rice_gate.py`, `segmentation.py`, `preprocessing.py`, `postprocessing.py`, `mask_ops.py`.
- Geometry/profile/calibration: `ml/quality/geometry.py`, `profiles.py`, `calibration.py`.
- Other Experimental/Proxy modules: `ml/quality/{classifiers,colour,texture,foreign_matter,admixture,quality}.py`.
- UI: `frontend/src/App.tsx`, `types.ts`, `components/UploadSection.tsx`, `AnnotatedViewer.tsx`, `GlobalMetrics.tsx`, `GrainTable.tsx`, `GrainDetailPanel.tsx`, `QualityWarningsPanel.tsx`, `ExportControls.tsx`.
- Isolated current YOLO11 experiment: `backend/models/{autolabel,build_dataset,train_yolo,predict_count,evaluate_counts}.py`, `backend/README.md`.
- Older experiment: `backend/old models/` including `grain_infer.py` and training docs; it is not the current import path.

### Model folders vs. audit outputs — keep separate

- **Root `models/`:** current app-side folders, but only class-mapping JSON files are present; registry checkpoint paths are placeholders. No active trained rice checkpoint is stored here.
- **`backend/models/`:** the newer isolated YOLO11-seg toolchain (`autolabel.py`, `build_dataset.py`, `train_yolo.py`, `predict_count.py`, `evaluate_counts.py`). It is training/data tooling, not the production runtime. Autolabels are candidate proxy labels for human review; the prepared YOLO dataset config and trained rice weights are absent.
- **`backend/old models/`:** the earlier Mask R-CNN/YOLO experiment. Its `grain_infer.py` is where the similar experimental runtime file actually sits, but the route imports `backend.models.grain_infer`; the path mismatch is why that module is missing to Python.
- **`backend/audit_out/`:** generated first-pass proxy audit artifacts (CSV, contact sheet, histograms, masks and overlays).
- **`backend/audit2_out/`:** generated second-pass proxy audit artifacts (CSV and overlays). The historical run reports 546 photos; the current `backend/data/raw/` count is 860, so treat these outputs as potentially stale. Neither audit directory contains trained model weights or model accuracy evidence.

Both audit folders are sibling output directories under `backend/`; neither is a subfolder of root `models/` or `backend/models/`.

## Segmentation, rice gate and measurement facts

`rice_gate.py` independently evaluates whether image objects resemble rice. It uses the shared Otsu/polarity foreground mask, object-level color/shape/size features and configured rice class mapping. Current mapping says class 1 `rice_grain`; no trained application checkpoint establishes an ML class. Hard-negative foreign objects must remain non-rice.

Classical segmentation performs grayscale + blur + Otsu polarity logic using border pixels, a 3x3 opening, connected components with a configured 50 px minimum, and conditional watershed/pinch-point split. Instances retain cropped masks and offsets. Masks feed geometry; a segmentation defect can yield a zero grain count even if the gate found rice.

`compute_grain_geometry` derives area, perimeter, centroid, bbox, ellipse axes, orientation, solidity, aspect/LB ratio and effective length from mask contours. Values are in px/px² except with valid calibration. The profile `grain_profiles/default_rice.json` has 154.1 px length, 48.2 px breadth, `data_status=Proxy`; use only at matching capture scale and only when explicitly selected. Whole/Broken decision is `effective_length/reference_length < 0.75` => broken; exactly 0.75 is whole. No sound reference yields undetermined. This is a per-grain image/count estimate, not mass-percent or official lot certification.

Foreign object, colour/texture, damaged, immature/shrunken, sprouted/weevilled, admixture and image-quality algorithms exist, but code presence is not evidence of scientific validation. Use their returned provenance/limitation fields and label unsupported output Experimental, Proxy or Undetermined.

## YOLO / model status

Main inference lookup only names legacy YOLOv8-seg and Mask R-CNN paths, then classical fallback. Registry is placeholder. No YOLO26 symbols/settings/checkpoint configured in active source. Generic local `yolo11n.pt`, `yolo11s.pt`, `weights/yolo26n.pt` loaded locally as 80-class COCO detection checkpoints. No rice class map, rice masks, actual trained rice checkpoint or inference metrics available.

An experimental YOLO11-seg training path exists under `backend/models/`; it expects `backend/data/yolo/data.yaml`, but that directory is empty. Training scripts may download initial weights if run and must not be invoked as audit/testing. Dataset labeling uses proxy heuristics and the small reviewed COCO file is only two images / 224 annotations, so no accuracy claim follows.

## Data and validation available

- `backend/data/raw/`: 860 JPGs (~4.1 GiB). Provenance, count assumptions and complete labels are not independently established.
- `backend/data/work/review/coco.json`: 2 images, 224 polygon annotations, classes rice/foreign. No complete held-out split verified.
- `backend/data/test/`, `backend/data/yolo/`, and `data/grain/{raw_images,annotations,dataset,overlays}` are empty.
- `test_images/`: 256 actual image files over three test/validation groups plus many result artifacts; filenames do not equal verified labels. Do not use as training automatically.
- Synthetic generator under `tests/grain/` is a demo/test fixture source only.
- Ground truth for precision, recall, mask IoU, scale measurement and train/test leakage is unavailable.

Request CVAT-reviewed COCO export, source/group manifest, explicit split, class mapping, checkpoint provenance, training config/logs and independent real-image evaluation before evaluating an ML claim.

## Verified tests/build and current failures

- `npm run build`: passed; TypeScript and Vite 6.4.3 build completed.
- `python -m pytest -q`: failed collection because `tests/test_grain_infer.py` cannot import `backend.models.grain_infer`.
- `python -m pytest -q --ignore=tests/test_grain_infer.py`: 80 passed, 2 failed. Failures: `test_case2a_only_rice_boxes_are_kept_after_gate_filter`, `test_case3_rice_plus_foreign_matter`; both gate rice+stone synthetic input, then segmentation yields zero grains. Cause after gate remains unresolved.
- FastAPI module import/OpenAPI path generation passed; startup lifespan and live `/analyze-ml` request were not tested. Missing runtime implies expected experimental 503.
- Safe local YOLO checkpoint load passed for three generic COCO detectors. No rice inference was run.

Global environment used tests: Python 3.14.5, FastAPI 0.142.4, pytest 9.1.1. `backend/.venv` is Python 3.11.9 with web + YOLO11 dependencies; root `.venv` is Python 3.12.4 with ML dependencies but lacks FastAPI. See `TESTING_AND_ENVIRONMENT.md` for full versions and commands.

## Known risks / current bugs

1. Experimental import/test path is missing; its route is unavailable.
2. Two rice+foreign segmentation integration tests fail after the gate accepts rice. Preserve gate behavior while debugging the first failed segmentation stage.
3. Production `/analyze` reads unbounded upload bytes and runs synchronous CV inside an async endpoint without timeout/pixel limit.
4. Frontend reads `VITE_API_BASE_URL`; env examples set `VITE_API_URL`. Frontend uses relative `/api` by default with no Vite proxy; backend direct run uses port 10000 while examples mention 8000.
5. RAM job map has no eviction/persistence; local `runs/` stores images/results and has no evident retention policy.
6. Main API has no auth. Deployment exposure must be checked before treating it as public-safe.
7. `App.tsx` calls default pixel profile trusted despite JSON `Proxy` status; correct semantics before promoting that profile.
8. Whole/Broken percentages divide by total detected grains once any grain is classifiable, so undetermined grains remain in denominator; clarify product meaning and expose coverage.
9. Local standards JSON has not been checked against authoritative current source and equivalent basis. Do not claim official compliance.

## Recommended order for future work

1. Decide whether experimental `/analyze-ml` is supported; reconcile import path, model paths and tests only if retaining it.
2. Reproduce/debug the two gate-to-segmentation fixture failures at the earliest segmentation stage; do not change gate without regression evidence.
3. Align frontend API variable/port/proxy and verify a live browser request.
4. Define image upload limits, execution/concurrency model and retention/security requirements.
5. Review the raw image provenance and label coverage; complete group-safe reviewed split and independent real-photo evaluation.
6. Update docs to distinguish production classical CV from isolated YOLO11 experiments and absent YOLO26 integration.
7. Resolve profile wording and percentage denominator semantics; validate physical calibration/profile before stronger measurements.

## Instructions to the next AI/developer

Before proposing changes, re-read current `PROJECT_CONTEXT.md`, inspect current code and `git status`, and verify which experiment is actually active. Preserve rice/non-rice gate and segmentation behavior unless regression evidence requires a minimal change. Do not infer implementation from `WORK_LOG.md` or older docs. Do not download weights, install dependencies, train, or upload source/images. Keep model experiments isolated. Do not treat synthetic/demo tests or proxy audit summaries as real-data metrics. Do not claim KMS/current official compliance without verified authoritative source and equivalent measurement basis. Ask for missing COCO splits, checkpoint, provenance and logs when needed. Do not delete or restore historical files without explicit task scope. When coding, append a dated `progress/WORK_LOG.md` entry with summary, paths, tests/results, blockers and next steps.
