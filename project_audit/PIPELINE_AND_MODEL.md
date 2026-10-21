# Pipeline, Computer Vision, and Model Audit

Audit date: 2026-10-21. Findings below are from current source/configuration and commands actually run; historical claims are explicitly labeled.

## Primary `/analyze` image path

1. `frontend/src/components/UploadSection.tsx` accepts an image file or camera capture and calls `App.handleAnalyze(file)` in `frontend/src/App.tsx`. The UI submits multipart field `file`, `grade=grade_a`, and optional profile. It resets prior response state, displays loading/error states, and sends `POST /api/analyze`.
2. `backend/app/api/routes.py::analyze_rice_image` reads all uploaded bytes (`await file.read()`), converts optional manual scale fields, then calls `job_manager.process_sync`. It does not limit bytes or validate MIME type in this route.
3. `backend/app/services/job_manager.py::process_sync` creates an in-memory job ID and directly calls `RiceQualityPipeline.analyze`. There is no queue/background worker; inference blocks the request thread/event loop path. On success it stores result in process memory and calls `log_run`; on exception it logs failure and returns an error object.
4. `ml/segmentation/pipeline.py::RiceQualityPipeline.analyze` decodes with `load_image` (in `ml/segmentation/preprocessing.py`), produces RGB pixels and image metadata, and handles decode failures with a structured unsuccessful response.
5. `detect_rice_presence` in `ml/segmentation/segmentation.py` delegates to `ml/segmentation/rice_gate.py::evaluate_rice_presence`. Gate foreground uses `extract_foreground_mask` (perimeter background sampling, grayscale, Gaussian blur, Otsu, polarity choice, light morphology). Object features and the optional learned-feature artifact decide rice vs hard-negative/foreign. Current `models/segmentation/` contains class mapping only; `configs/models.json` says placeholder and its model weights file is absent, so the classical heuristic gate is active.
6. A negative/no-analysable-rice gate returns `_build_no_rice_response` early; it does not pass foreign objects through grain geometry/classification. For accepted rice scenes, `detect_calibration` checks ArUco then optional manual scale. Only valid calibration is used to produce mm fields.
7. `ml/segmentation/pipeline.py` invokes `phase1_analyze_image` from `ml/segmentation/inference.py`. `_load_model()` searches legacy `models/yolo_seg/run1_baseline/weights/best.pt`, then the `configs/models.json` Mask R-CNN path, and caches the first available model. Neither application checkpoint exists/configures in this checkout, so it selects `classical_cv`.
8. Classical path calls `ml/segmentation/segmentation.py::segment_grains`, which chooses `_segment_classical_cv`: background-polarity/Otsu foreground mask, 3x3 morphological opening, connected components, area filtering, then optional distance-transform watershed/pinch-point splitting via `_split_grain_cluster`. Each output `GrainInstance` stores a tight cropped mask plus `(x,y)` `mask_offset`, bbox, centroid, contour, confidence/quality metadata. It attempts instance-level separation; touching-object behavior is heuristic and known to be imperfect.
9. The phase-1 output masks are polygonized/rasterized by pipeline helpers and filtered against gate rice detections (`_filter_grains_to_rice`). Foreign detections from gate and `ml/quality/foreign_matter.py` stay in the separate foreign channel. Unresolved/uncertain detections are not equated with reliable grain instances.
10. `ml/quality/geometry.py::compute_grain_geometry` finds contours; computes area, perimeter, centroid, bounding box, ellipse major/minor axes, principal-axis/caliper length, breadth, solidity, aspect ratio, L/B, orientation, and optional calibrated mm. Pixel dimensions are not universally comparable between camera scales.
11. `resolve_whole_kernel_reference` selects an explicit profile, permitted sample-derived intact population, or undetermined. The default `grain_profiles/default_rice.json` is 154.1 px (proxy, capture-scale-specific), and frontend default selection is no explicit profile. `classify_broken` labels broken when effective length/reference length `< 0.75`; at `0.75` it is whole. Missing/untrusted reference yields `undetermined`.
12. Other per-grain defect helpers run after segmentation. Damaged and sprouted/weevilled helpers may use configured checkpoints or engineering heuristics; colour/texture/immature helpers are also heuristic/proxy. There is no basis to call these scientifically validated. Sample-level admixture/quality/standards results are computed by pipeline helpers; static standards data is not verified here against a current authoritative source.
13. Annotation helpers in `ml/segmentation/pipeline.py` produce JPEG base64 output; response includes image/sample/calibration/quality/grains/foreign/admixture/summary/standards/warnings/timing fields. `run_logger.py` saves run details/artifacts to local `runs/` and structured logs to `logs/`.
14. `App.tsx` sets result or a non-rice message. It renders `GlobalMetrics`, `AnnotatedViewer`, `GrainTable`, `GrainDetailPanel`, `QualityWarningsPanel`, and `ExportControls`. The viewer scales overlays over the returned image; export controls download JSON/CSV through API endpoints.

```mermaid
sequenceDiagram
  actor User
  participant UI as App.tsx / UploadSection.tsx
  participant API as routes.py /api/analyze
  participant Jobs as job_manager.py
  participant Pipe as RiceQualityPipeline.analyze
  participant Gate as rice_gate.py + segmentation.py
  participant Infer as inference.py + segmentation.py
  participant Metrics as geometry.py + quality modules
  participant Store as RAM jobs + run_logger.py
  UI->>API: multipart file, grade, optional profile
  API->>Jobs: process_sync(image_bytes,...)
  Jobs->>Pipe: analyze(bytes,...)
  Pipe->>Gate: decode RGB + rice-presence gate
  Gate-->>Pipe: rice/foreign detections and gate status
  alt gate rejects rice
    Pipe-->>Jobs: no-rice result
  else gate accepts rice
    Pipe->>Infer: phase1_analyze_image(image_rgb)
    Infer->>Infer: configured model lookup; classical CV fallback in current checkout
    Infer-->>Pipe: per-instance mask/polygon detections
    Pipe->>Metrics: geometry, reference, Whole/Broken, heuristic metrics
    Metrics-->>Pipe: per-grain/sample metrics
    Pipe->>Pipe: annotated JPEG + response assembly
  end
  Jobs->>Store: RAM job record + local run artifacts/logs
  Jobs-->>API: completed response
  API-->>UI: JSON result
  UI->>API: optional JSON/CSV download
```

## Segmentation path ambiguity and model lookup

There are two similar but not identical segmentation surfaces:

- `RiceQualityPipeline.analyze` calls `phase1_analyze_image` from `ml/segmentation/inference.py`; it contains tiling, model loading, post-processing, mask polygon generation, and classical fallback.
- The pipeline also retains a broad exception fallback to `segment_grains(image_rgb, method="auto")` from `ml/segmentation/segmentation.py`.
- `segment_grains(method="auto")` separately checks the legacy YOLOv8 path and Mask R-CNN registry, then uses `_segment_classical_cv`.
- The current main `configs/models.json` registry has `segmentation.status = placeholder`, `weights_path=models/segmentation/model.pth`; `models/segmentation/` only has `class_mapping.json`. No configured trained checkpoint is present.

`ml/segmentation/inference.py` legacy YOLO parser uses Ultralytics `model.predict` with segmentation confidence and IoU thresholds from `configs/thresholds.json`, converts output mask arrays to original `h×w` using nearest-neighbor resize, creates bbox `[x,y,w,h]`, then mask contours/polygons. The active current checkout does not execute this path because no matching checkpoint is configured. The expected class mapping file says class 1 `rice_grain`, but that does not establish the classes of arbitrary weights.

### YOLO26 claim audit

No YOLO26-specific code or configuration (`YOLO26_WEIGHTS_PATH`, YOLO26 confidence/UI slider, YOLO26 tests, or a rice YOLO26 checkpoint) was found in the current active `ml/segmentation/inference.py`, primary React flow, `configs/thresholds.json`, or `configs/models.json`. A root-level work-log entry dated 2026-10-20 claims a YOLO26 integration at `backend/models/weights.pt`; that path is absent. Three loadable files are present: `yolo11n.pt`, `yolo11s.pt`, and `weights/yolo26n.pt`. A safe local load in `.venv` identified all three as `task=detect`, 80 COCO object classes (including `person`, `bicycle`, ... `toothbrush`), not a rice-specific instance-segmentation class map. That smoke load verifies checkpoint readability only, not rice analysis or application integration.

No rice YOLO checkpoint class IDs, segmentation masks, trained classes, train resolution, NMS configuration, training framework compatibility, provenance, or metrics can be verified because the relevant rice checkpoint/configuration is missing. The root test path `tests/test_grain_infer.py` is not a YOLO26 test suite and cannot collect because the import target is missing.

## Classical CV details

| Stage | Implementation | Inputs / outputs and settings | Limits |
|---|---|---|---|
| Foreground extraction | `segmentation.py::extract_foreground_mask` | RGB to grayscale, Gaussian blur `(5,5)`, Otsu bright/dark threshold alternatives; border pixels estimate polarity and background gray. | Assumes background reaches borders and differs from foreground intensity. Shadows, colored/uneven backgrounds, cropped objects and strong highlights can mislead. |
| Morphology/object extraction | `rice_gate.py::detect_rice_detections`, `segmentation.py::_segment_classical_cv` | 3×3 elliptical morphological opening; contours or connected components; `min_grain_area_pixels` default 50. | Touching objects may merge; small grains/noisy fragments may be removed. |
| Gate classification | `rice_gate.py::_object_features`, `score_rice_confidence`, `evaluate_rice_presence` | Shape, color relative to border, geometry cues and optional learned feature vector; default rice confidence threshold 0.52; hard-negative stone/wheat cues retained. | Heuristic confidence is not calibrated probability; class mapping does not validate a trained model. |
| Cluster splitting | `segmentation.py::_split_grain_cluster`, `_partition_mask_by_markers` | Distance transform, marker/watershed and pinch-point rules; component area and shape thresholds in config. | May over-split, under-split or produce zero accepted grains even after gate acceptance; current rice+stone regression tests show the last failure mode. |
| Mask geometry | `geometry.py::compute_grain_geometry` | Contour, moments, fitEllipse, minAreaRect/principal-axis spans, convex hull. Outputs px, ratios, optional mm. | Sensitive to mask boundary and resolution; ellipse-based major/minor axes can differ from effective projected length. |
| Foreign matter | `foreign_matter.py` | Gate non-rice detections, color/contour heuristics, optional legacy YOLO path when configured. | Experimental/unvalidated; count reflects detections, not official lot impurity mass. |

Current production path is classical CV only given available configs/weights. Source retains legacy selectable/model-fallback code; it is not evidence of an active learned pipeline. No direct-vs-application trained rice model comparison is possible from current files.

## Metrics and semantics

| Metric | Current calculation/source | Units/status |
|---|---|---|
| Grain count | Count of accepted `GrainInstance` masks in pipeline summary. | Count of image detections, subject to missed/merged/split masks. |
| Rice/foreign count | Gate classification and separately merged foreign detections. | Detection counts; foreign heuristic is Experimental. |
| Area/perimeter | OpenCV contour area / arc length in `compute_grain_geometry`. | px² / px. |
| Length/breadth | Fit ellipse major/minor axes; effective length is median of positive caliper span, min-area-rectangle long side, ellipse major axis. | px, or mm only after valid calibration; no camera-distance normalization. |
| L/B, aspect ratio | `length_pixels / breadth_pixels` with zero guard. | Dimensionless. |
| Solidity | contour area / convex-hull area. | Dimensionless, usually [0,1]. |
| Whole/Broken | `effective_length / whole_kernel_reference_length`; `<0.75` Broken, else Whole; no reference => Undetermined. | Per-grain image/count classifier; not a mass-based official broken percentage. Default profile is Proxy. |
| Whole/Broken percentages | Count of classified whole/broken divided by classified rice population in the pipeline summary; null when there is no denominator/reference. | Image-derived count %. |
| Image quality score/tier | `ml/quality/quality.py::assess_image_quality`: average of heuristic sub-scores for blur, median mask area, uncertain fraction, illumination and clipping. | 0–1 score and GOOD/FAIR/POOR/UNRELIABLE engineering heuristic, not validated. |
| Defect/colour/texture/admixture | Per-module rules and/or optional checkpoints in `ml/quality/`. | Proxy/Experimental/Undetermined unless external validation evidence is supplied. |
| Standards screen | Static local JSON and comparison code. | Not verified against authoritative current standard or equivalent lab measurement; never characterize as official compliance. |

The UI includes some sample metrics beyond core Whole/Broken, with types in `frontend/src/types.ts`. Exact confidence/quality claims should be taken from response provenance/limitation fields, not inferred from field names.

## Dataset and training artifacts

- `backend/data/raw/`: 860 local JPGs (~4.1 GiB), current local raw image corpus. There is no proof in the repository that each image has correct rice/foreign labels or a fixed number of each object.
- `backend/data/work/review/coco.json`: 2 image records and 224 polygon annotations across `rice` and `foreign` categories. It is a small review package, not evidence of a finished export or split. Other work files include layout/group CSVs and report CSV.
- `backend/data/test/` and `backend/data/yolo/` are empty in this checkout.
- `data/grain/{raw_images,annotations,dataset,overlays}/` are empty. Current `backend/models/` training scripts describe a YOLO11-seg experiment and expect reviewed COCO data, but no corresponding current dataset YAML/splits/checkpoint is present.
- `backend/old models/` contains another synthetic training experiment (Mask R-CNN + YOLO) and setup docs. `tests/grain/generate_synthetic.py` creates synthetic illustrations with `whole_rice`, `broken_rice`, and `foreign_matter`; these do not establish real-image accuracy.
- Current checkpoints in root and `weights/` are generic 80-class COCO detection weights, not rice segmenters. No model provenance or real-image validation data is available for rice.
- `test_images/` contains 256 image files across three validation folders plus saved result/derived files. These are validation/demo assets, not automatically training data. The filenames and visuals are not ground truth annotations.

No train/validation/test split, image-level leakage analysis, authoritative label audit, or real-data precision/recall/mask IoU can be verified. Request CVAT-reviewed COCO export, raw-photo provenance, source-group mapping, exact held-out test set and trained rice checkpoint metadata before training or accuracy claims.

## Observed failure evidence

In the current test environment, `tests/test_rice_gate.py::test_case2a_only_rice_boxes_are_kept_after_gate_filter` and `test_case3_rice_plus_foreign_matter` both fail. Their synthetic rice-plus-stone input is accepted by the gate (`rice_detections=1`, `total_detections=3`), but segmentation emits `grain_count=0`; the final `rice_detected` becomes false. This is a confirmed gate-to-segmentation integration failure on those fixtures. The evidence does not establish whether the cause is thresholding, fixture/background interaction, or the splitter; the earliest failing stage is segmentation. Do not alter rice-gate logic to mask this downstream issue.

No directly available annotated production outputs or independent ground truth were audited for missed grain, foreign false positive, duplicate mask, mask IoU, or model-vs-app divergence. Existing `backend/audit_out` and `audit2_out` are historical generated proxy inspections documented in the work log; their calibration assumes image count priors, so those estimates are not independent accuracy metrics.
