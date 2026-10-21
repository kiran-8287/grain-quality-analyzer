# Relevant File Inventory

Audit date: 2026-10-21. Generated artifacts and dependency trees are summarized by directory; no secrets, image contents, or dependency dumps are included.

## Application source and configuration

| Path | Role / relationships | Verification and notes |
|---|---|---|
| `backend/app/main.py` | FastAPI entry, lifespan, logging/CORS, route mounts, built frontend serving. | Import/OpenAPI succeeded. Experimental runtime points at missing module. Direct script default port is 10000. |
| `backend/app/api/routes.py` | Main API endpoints; calls `job_manager`; routes mounted both root and `/api`. | API tests partly pass; experimental route has missing runtime; main upload is unbounded. |
| `backend/app/schemas/models.py` | Pydantic health/job/analysis and experimental response schemas. | Used for route response models where configured. |
| `backend/app/services/job_manager.py` | Runs primary pipeline synchronously, stores jobs in RAM, logs run. | Confirmed process-local; no DB/queue. |
| `backend/app/services/run_logger.py` | Structured lifecycle logger and local image/result artifacts. | `tests/test_logging.py` in passing remainder; images saved below `runs/`. |
| `backend/app/services/export.py` | JSON/CSV serializers called by export routes. | Export API tests included in passing 80. |
| `backend/requirements.txt` | Backend dependency minimums. | No full lock; Python version not constrained. |
| `configs/{app,inference,models,parameters,thresholds}.json` | Runtime configuration and thresholds. | Model registry says placeholder; no app segmentation checkpoint configured. |
| `ml/config.py` | Loads JSON configuration and standards/model registry. | Used by segmentation, calibration and quality code. |
| `ml/segmentation/pipeline.py` | Main end-to-end image analysis and response assembly. | Current primary implementation. |
| `ml/segmentation/inference.py` | Phase-1 model lookup, inference adapters, postprocessing and classical fallback. | Legacy YOLOv8/Mask R-CNN paths; no matching weights. |
| `ml/segmentation/rice_gate.py` | Rice/non-rice object classification and gate. | Two integration test failures occur after gate success. Preserve semantics. |
| `ml/segmentation/segmentation.py` | Foreground mask, connected components and classical instance masks; optional legacy model paths. | Current fallback. |
| `ml/segmentation/{preprocessing,postprocessing,mask_ops}.py` | Input decoding, image tiles/detection filters/coordinate transforms. | Used by main inference path. |
| `ml/quality/geometry.py` | Mask geometry, reference resolver and Whole/Broken classifier. | Source confirms pixel geometry and `<0.75` rule. |
| `ml/quality/{calibration,profiles}.py` | ArUco/manual calibration validity and profile loading. | Default profile is proxy; mm profiles require valid calibration in resolver. |
| `ml/quality/{quality,foreign_matter,admixture,colour,texture,classifiers}.py` | Image-quality score and optional/heuristic quality parameters. | Experimental/Proxy unless independently validated. |
| `models/segmentation/class_mapping.json` | Class 0 background, class 1 rice_grain. | Mapping only; no trained checkpoint alongside it. |
| `models/foreign_matter/class_mapping.json` | Foreign class mapping. | Mapping only; no trained checkpoint alongside it. |
| `grain_profiles/default_rice.json` | Default whole-kernel pixel reference. | 154.1 px, `Proxy`, camera-scale-specific; UI wording conflicts. |
| `standards/india_kms_2026_27_raw_rice.json` | Static local standards data served by API/used by comparator. | Not authoritative-source verified in this audit. |

## Frontend

| Path | Role / relationships | Verification and notes |
|---|---|---|
| `frontend/src/main.tsx` | React root; mounts `App`. | Build passed. |
| `frontend/src/App.tsx` | Primary upload/result state and `/api/analyze` request. | Reads `VITE_API_BASE_URL`; source comment describes profile as trusted incorrectly. |
| `frontend/src/types.ts` | Types for response, grains, metrics and experimental fields. | Build typecheck passed; no runtime response validation. |
| `frontend/src/components/UploadSection.tsx` | File/camera selection, preview, profile options, submit. | Camera/UI code present; live browser not tested. |
| `frontend/src/components/{AnnotatedViewer,GlobalMetrics,GrainTable,GrainDetailPanel,QualityWarningsPanel,ProcessingState,ExportControls,Navbar}.tsx` | Result visualization, metric summaries, grain table/detail, caveats, loading and exports. | Referenced by App; build passed. |
| `frontend/src/components/Phase1Demo.tsx` | Separate demo canvas/experimental upload path. | Not imported by App; tries nonexistent `/api/phase1/analyze` then `/api/analyze`. |
| `frontend/package.json`, `frontend/package-lock.json` | Frontend dependencies/scripts and resolved versions. | `npm run build` passed; Vite 6.4.3 resolved. |
| `frontend/vite.config.ts`, `tsconfig.json`, `tailwind.config.js`, `postcss.config.js`, `index.html` | Build and styling config/entry shell. | No API proxy observed in Vite config. |

## Experimental model training/inference

| Path | Role / relationships | Verification and notes |
|---|---|---|
| `backend/models/autolabel.py` | Classical proxy label generation for rice/foreign. | Existing local modifications; left untouched. Not ground truth. |
| `backend/models/build_dataset.py` | Grouping, COCO review package creation, reviewed dataset conversion. | Existing local modifications; current small review JSON has 2 images only. |
| `backend/models/train_yolo.py` | Trains YOLO11-seg with configured data YAML and fallback batches. | Needs `backend/data/yolo/data.yaml`; absent. Not run. |
| `backend/models/predict_count.py`, `evaluate_counts.py` | Tiled mask inference/count/measurement and count evaluation. | No rice checkpoint/eval set; not run. |
| `backend/old models/grain_infer.py` | Isolated Mask R-CNN + YOLO experimental inference implementation. | File exists under old folder; API expects different module path. |
| `backend/old models/{train_yolo.py,train_maskrcnn.py,prepare_data.py,auto_annotate.py}` | Older experimental training workflow. | Scripts not part of current API path. |
| `backend/old models/{GRAIN_INFERENCE.md,GRAIN_TRAINING.md,requirements-ml.txt}` | Experimental setup/usage docs and ML dependencies. | Historical docs; some commands reference missing scripts/paths. |
| `yolo11n.pt`, `yolo11s.pt`, `weights/yolo26n.pt` | Local Ultralytics checkpoints. | Safely loaded; all generic 80-class COCO `detect`, not rice masks. |
| `runs/` | Generated model outputs. | 43 files (~13.5 MiB) observed; do not infer provenance/metrics without inspecting metadata. |
| `tmp/grain-smoke-.../` | Generated training smoke diagnostics. | About 2.7 GiB; ignored/generated; synthetic metrics are not real-data validation. |

## Tests, docs, scripts and data

| Path | Role / relationships | Verification and notes |
|---|---|---|
| `tests/test_api.py` | HTTP/API route tests. | Included in 80 passing remainder. |
| `tests/test_rice_gate.py` | Gate + segmentation scenarios. | Two current failures after gate accepts rice and segmentation emits zero grains. |
| `tests/test_whole_vs_broken.py` | Classification/reference/calibration tests. | Included in 80 passing remainder. |
| `tests/test_logging.py`, `tests/test_monitoring.py` | Logger lifecycle and monitoring parser. | Included in 80 passing remainder. |
| `tests/test_grain_infer.py` | Isolated experimental runtime tests. | Collection fails importing absent `backend.models.grain_infer`. |
| `tests/grain/generate_synthetic.py` | Synthetic COCO illustration generator. | Test/demo data only. |
| `test_pipeline.py`, `scripts/validate_*.py` | Older/manual validation runners. | Not run in audit. |
| `scripts/build_physical_profile.py` | Physical profile utility. | Not run; no audited measurements supplied. |
| `scripts/{convert_heic_to_jpg.py,organize_raw_images.ps1}` | Image utilities. | Current local untracked additions; untouched. |
| `README.md`, `PROJECT_CONTEXT.md`, `AGENTS.md` | Project description, source-of-truth and work instructions. | Context docs conflict with later experiment/logs; current code used as authority. |
| `docs/{architecture,CURRENT_STATUS,FULL_PROJECT_EXPLANATION,edge-cases,cleanup_report}.md` | Architecture/status/history documentation. | Some stale; not overwritten. |
| `backend/README.md` | Current YOLO11 experiment setup. | Contains instructions not fully supported by current folders/data. |
| `progress/WORK_LOG.md` | Historical task log. | Read; appended audit entry, retained prior entries. |
| `monitoring/{app.py,log_parser.py,README.md}` | Read-only Streamlit log monitor. | Code and parser tests exist; live dashboard not run. |
| `backend/data/raw/` | Local raw photo corpus. | 860 JPGs, ~4.1 GiB; provenance/labels not established. |
| `backend/data/work/review/coco.json` | COCO review package. | 2 images, 224 annotations, classes rice/foreign; not full dataset. |
| `backend/data/work/{report.csv,groups.csv,layout_groups/groups.csv}` | Dataset/audit reports and group information. | 7 total files under `backend/data/work`; read status only, no full audit of CSV semantics. |
| `backend/data/test/`, `backend/data/yolo/` | Test/prepared YOLO data paths. | Empty. |
| `raw_images/` | Four local JPGs at repository root. | Local photos; not opened or used for prediction. |
| `test_images/` | Three validation/demo groups, 256 image files plus generated results/artifacts. | Not used as training data; no ground truth manifest. |
| `data/grain/{raw_images,annotations,dataset,overlays}/` | Older experiment storage locations. | Empty. |
| `logs/` | Runtime log files. | Contains `app.log` (~2 MiB); do not share unredacted without review. |

## Generated, large, and ignored folders

| Directory | Observed size/count | Notes |
|---|---:|---|
| `node_modules/` | Dependency tree; summarized, not expanded. | Generated; do not include in handover package. |
| `frontend/node_modules/` | 7,334 files, ~113.7 MiB, includes `dist/`. | Build output and installed dependencies; generated. |
| `.venv/` | Root ML venv. | Python 3.12.4, ML stack, missing web dependencies. |
| `backend/.venv/` | Python 3.11.9, ~26 GiB including packages. | Web + ML tools; avoid sharing. |
| `backend/audit_out/`, `backend/audit2_out/` | Generated proxy images/reports. | Worktree untracked; no source changes by this audit. |
| `backend/results/` | Generated autolabel preview outputs. | Images are proxy labels, not YOLO predictions. |
| `__pycache__/`, `.pytest_cache/`, `dist/`, `logs/`, `runs/`, `tmp/` | Generated caches/logs/build/model artifacts. | Summarize or omit in external handover; inspect privacy and provenance before sharing. |

## Pre-existing worktree state

Before report writing, `git status --short` showed modified `backend/models/autolabel.py`, `backend/models/build_dataset.py`; untracked `backend/audit2_out/`, `backend/audit_out/`, `backend/audit_photos.py`, `backend/audit_photos_v2.py`, `backend/find_layout_groups.py`, `scripts/convert_heic_to_jpg.py`, `scripts/organize_raw_images.ps1`. These were already present and were not changed during this audit. Audit changes are limited to this new directory and required `progress/WORK_LOG.md` append.
