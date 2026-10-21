# Bugs, Risks, and Roadmap

Audit date: 2026-10-21. Confirmed issues are backed by current source or commands; risks are labeled as needing investigation.

## Prioritized findings

| Priority | Finding | Evidence and impact | Recommended next step | Confidence |
|---|---|---|---|---|
| **Critical** | Experimental test/API implementation path is broken. | `backend/app/main.py` imports `backend.models.grain_infer`; that module is missing. A similarly named file exists in `backend/old models/grain_infer.py`. Full pytest collection stops with `ModuleNotFoundError`; experimental `/analyze-ml` startup runtime remains `None` and should return 503. | Decide whether this old experimental route should be restored, retired, or isolated. Keep it separate from core production analysis; fix import/tests in a separately scoped task. | Confirmed. |
| **High** | Gate accepts rice in rice+stone fixture but segmentation emits zero grains. | Actual `pytest -q --ignore=tests/test_grain_infer.py`: two failing integration tests; logs show gate `has_rice=true`, one rice detection, then segmentation `grain_count=0`, and final `rice_detected=false`. | Trace the earliest segmentation failure on the two deterministic fixture inputs, inspect threshold/mask/contour metrics, preserve gate logic, add regression coverage only after user authorizes implementation. | Confirmed on tests; cause unresolved. |
| **High** | Primary upload is unbounded and inference is synchronous. | `/analyze` calls `await file.read()` without byte/MIME/pixel-size caps, then runs `job_manager.process_sync()` inline from `async def`; no timeout/background queue. Large/parallel uploads can consume memory and block the service. | Establish supported image/file size and concurrency requirements, then implement bounded reads, decoded pixel limits, and an appropriate execution model. | Confirmed design risk. |
| **High** | Current documentation/work log/model config disagree with current source. | README/context/docs say clean classical baseline and model experiments removed; work log claims YOLO26 app integration and a later YOLO11 pipeline; current config is placeholder, YOLO26 code/checkpoint path is absent, scripts are present, and generic COCO checkpoints load as 80-class detection only. | Reconcile docs against current checkout after choosing the intended active/experimental scope. Do not resurrect YOLO26 on the basis of a log alone. | Confirmed. |
| **High** | Vite frontend's default API target is inconsistent with backend startup. | Frontend reads `VITE_API_BASE_URL` and defaults to relative `/api`; no Vite proxy is configured. Backend script defaults port 10000, env examples say `VITE_API_URL`/port 8000. The root dev script was not integration-tested. | Align env names, ports, and documented dev proxy/base URL, then run a live frontend-to-backend check. | Confirmed config mismatch; run-time outcome inferred. |
| **Medium** | Local job history and run artifacts have no evident retention. | `JobManager.jobs` holds full result dictionaries in a global in-memory map; `run_logger.log_run` writes images/results under local `runs/`. Restart loses jobs; long running processes can accumulate RAM/disk and user images. | Define retention/privacy requirements and cleanup limits. Avoid logging uploaded image bytes or sensitive filenames. | Confirmed storage behavior; operational impact depends on use. |
| **Medium** | Validation corpus has incomplete current reviewed labels and split evidence. | 860 raw JPGs (~4.1 GiB) are present, while work/review COCO contains two images and 224 annotations. Empty test/yolo dirs, no `data/grain` artifacts, no verified held-out real-data evaluation. | Audit provenance/groups, complete annotation review, establish leakage-safe train/val/test splits, and evaluate against an independent labeled set before discussing accuracy. | Confirmed local artifact counts; label correctness unknown. |
| **Medium** | Standards JSON is served as if a current standard but was not source-verified here. | `standards/india_kms_2026_27_raw_rice.json` is loaded directly by API and local comparator. No authoritative-source/equivalent-basis validation was done. | Verify source and measurement basis before exposing any compliance conclusion; otherwise use screening/proxy/undetermined wording. | Confirmed lack of verification in this audit. |
| **Medium** | UI text describes the legacy pixel profile as trusted despite its Proxy metadata. | `grain_profiles/default_rice.json` marks 154.1 px `data_status: Proxy` and scale-limited; `frontend/src/App.tsx` comments call it a "Tier-1 trusted profile" and option detail calls it a "trusted reference". Code resolver returns pixel profiles as limited/not production-eligible. | Align UI wording with profile provenance before presenting Whole/Broken output. | Confirmed inconsistency. |
| **Medium** | Whole/Broken percentage denominator includes undetermined grains. | In `pipeline.py`, percentages use `broken_count / n_grains` and `whole_count / n_grains`, while only requiring `broken_analyzed_count > 0`; if some grains are undetermined, displayed percentages include them in denominator and may sum below 100%. | Confirm intended product semantics (share of all detections vs classified detections), expose denominator/coverage, and test mixed determined/undetermined samples. | Confirmed formula; intended meaning needs decision. |
| **Medium** | Public API lacks authentication and data access controls. | All routes have no auth; UUID job IDs gate access only by possession. `runs/` persists images. Deployment is configured through `render.yaml`; production exposure/access policy was not verified. | Confirm whether API is private. If externally reachable, define auth, rate/body limits and storage access/retention. | Source fact; threat exposure depends on deployment. |
| **Low** | Example environment variable names/ports are stale. | `.env.example` includes many vars unused in source; frontend uses `VITE_API_BASE_URL`, example uses `VITE_API_URL`; backend direct port 10000 vs example 8000. | Update runbook and examples as a docs-only task after selecting supported local workflow. | Confirmed. |
| **Low** | Orphaned UI/demo code and historical scripts confuse active surface. | `frontend/src/components/Phase1Demo.tsx` is not imported by `App.tsx`; `test_pipeline.py`, multiple validation scripts and `backend/old models/` remain alongside current code. | Mark active/archived roles in docs; do not delete in audit. | Confirmed. |

## Confirmed defects vs possible causes

### Confirmed defects

- Missing experimental import target prevents the ML test module from collecting and leaves experimental inference unavailable.
- Two rice-plus-stone integration tests fail because gate acceptance is not followed by grain instances.
- API example/frontend configuration mismatch and backend port mismatch are visible in source/config.
- Current raw model files are not rice models: local YOLO load confirms `detect` task, 80 standard COCO classes. They cannot supply `rice_grain` segmentation masks.

### Unverified hypotheses / future checks

- Gate/segmentation fixture mismatch may be due to foreground polarity, Otsu foreground cleanup, small-object limits, cluster splitter or geometry filtering. Logs place failure after gate and during segmentation but do not identify the specific predicate.
- Dense real-image undercount/overcount, split/merge rates, edge clipping and mask quality require ground truth; screenshots/filename counts are insufficient for precision, recall or mask IoU.
- Arbitrary profile path handling should be reviewed for API exposure. `load_grain_profile` accepts path-like names and checks direct paths; the reachable impact depends on what profile metadata the API returns and deployment caller access.
- CORS and no authentication may be acceptable for private localhost use but are not established as safe for public deployment.

## Segmentation failure analysis status

The available deterministic failure is a synthetic test fixture where the gate found three objects and accepted one as rice. Classical segmentation found no grain instances after that point. This is evidence of a gate-to-segmentation integration defect for the fixture. It is not evidence of a YOLO defect, dataset defect, annotation defect, or official standards issue. No real labeled holdout is available to quantify masks. Existing audit outputs in `backend/audit_out/` and `backend/audit2_out/` came from proxy/blob methods and prior report notes explicitly say count assumptions calibrate those proxies; don't use them as independent model accuracy.

## Suggested order of next work

1. Resolve whether `backend/old models/grain_infer.py` and `/analyze-ml` are a supported experimental feature; bring the test import and runtime configuration into consistent state only if retaining it.
2. Investigate and fix the two rice-plus-stone segmentation failures at the first failing stage, with no change to the rice/non-rice gate unless a regression proves it necessary.
3. Align `VITE_API_BASE_URL`, Vite development proxy/origin, backend port, and docs; verify the browser can call `/api/health` and upload `/api/analyze`.
4. Review upload limits, synchronous workload and local image/log retention for the actual deployment requirements.
5. Review annotations/provenance for the 860 raw photos; complete a group-safe split and independent real-photo evaluation before any model claims.
6. Reconcile project documentation and progress log with code and label YOLO/foreign-matter/defect metrics Experimental, Proxy or Undetermined as appropriate.
7. Verify the local standards source and unit basis separately; do not advertise official lot compliance without equivalent measurement basis.

## Requested external evidence

To make the model/dataset audit complete, provide the intended trained rice segmentation checkpoint (if one is meant to exist), its training YAML/config and logs, reviewed COCO export including image list and groups, explicit class-ID map, train/validation/test manifest, ground-truth evaluation outputs, checkpoint provenance, and any real annotated application result pairs. Do not send private credentials, unrestricted raw images, or `.env` values.
