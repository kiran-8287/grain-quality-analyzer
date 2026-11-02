# Rice counting experiments

The FastAPI app and production classical-CV pipeline remain unchanged. This workspace contains the isolated experimental methods; all shared source photos, review assets, reports, and prepared datasets live in the repository-level `A:\grain\data\rice_count` folder.

## Shared data

- `data/rice_count/raw/` — 860 source photos used across methods.
- `data/rice_count/test/` — separate filename-counted evaluation photos.
- `data/rice_count/work/` — audits, layout groups, review assets, and method caches.
- `data/rice_count/datasets/weak_count_model/` — the single prepared pseudo-label dataset for the experimental model.
- `data/rice_count/datasets/yolo_segmentation/` — prepared dataset from reviewed polygons, when built.
- `test_images/` remains the application's existing test-image folder and is separate.

Method code and method-specific outputs stay under `rice_count_pipeline/methods/`:

- `autolabel_proxy/` — HSV/morphology/KMeans rule-based proxy.
- `area_count/` — area-based estimator.
- `weak_count_model/` — experimental YOLO model trained on pseudo-masks.
- `yolo_segmentation/` — separate reviewed-polygon segmentation workflow.
- `../backend/models/autolabel.py` — compatibility import for the canonical proxy implementation.

The raw set is mostly one approximate composition. Proxy labels and pseudo-validation scores are not ground-truth accuracy; use fresh exact-count photos for independent evaluation.

## Setup (Windows, Python 3.11, NVIDIA GPU)

From PowerShell:

```powershell
Set-Location A:\grain\rice_count_pipeline
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

The setup script uses the shared `A:\grain\backend\.venv` Python 3.11 environment for the API and experiments, installs CUDA 12.4 PyTorch plus both requirement sets, then reports CUDA availability and the GPU name. Run commands below from `A:\grain\rice_count_pipeline` using `& '..\backend\.venv\Scripts\python.exe' ...`.

## Audit and layout grouping

```powershell
& '..\backend\.venv\Scripts\python.exe' 'tools\audit_photos_v2.py' '..\data\rice_count\raw' --out '..\data\rice_count\work\audit'
& '..\backend\.venv\Scripts\python.exe' 'tools\find_layout_groups.py' '..\data\rice_count\raw' --out '..\data\rice_count\work\layout_groups'
```

These outputs are exploratory image-processing results, not verified counts. Inspect the grouping and audit reports before using them for model splits.

## Compare no-training methods

```powershell
& '..\backend\.venv\Scripts\python.exe' 'methods\autolabel_proxy\evaluate_test_set.py' --source '..\data\rice_count\test'
& '..\backend\.venv\Scripts\python.exe' 'methods\area_count\area_count.py' calibrate
& '..\backend\.venv\Scripts\python.exe' 'methods\area_count\evaluate_area_test_set.py' --source '..\data\rice_count\test'
& '..\backend\.venv\Scripts\python.exe' 'methods\area_count\area_count.py' predict '..\data\rice_count\raw\1.jpg'
```

Both are experimental proxies. The area method estimates rice quantity from masked area; its foreign count is connected blobs, not necessarily individual stones.

## Weak-label YOLO experiment

```powershell
& '..\backend\.venv\Scripts\python.exe' 'methods\weak_count_model\build_pseudo_dataset.py'
& '..\backend\.venv\Scripts\python.exe' 'methods\weak_count_model\train_model.py' --epochs 50
& '..\backend\.venv\Scripts\python.exe' 'methods\weak_count_model\evaluate_test_set.py'
& '..\backend\.venv\Scripts\python.exe' 'methods\weak_count_model\predict_count.py' 'path\to\new_image.jpg'
```

The model learns pseudo-polygons from the rule-based labeler, not reviewed instance masks. Its pseudo-validation metrics measure agreement with proxy labels. Resume with `--name weak_count_yolo --resume`; use `--score-only` to score saved weights without retraining. Checkpoints, runs, and prediction outputs are under `..modelsexperimentsweak_count_model`; dataset reports and caches are in `..dataice_countworkweak_count_model`. These are not independent accuracy measurements.

## Reviewed instance-segmentation workflow

Review images and COCO proposals are in `data/rice_count/work/review/`; the Roboflow import package is in `data/rice_count/work/roboflow_upload/`. Export reviewed COCO polygons into `data/rice_count/work/reviewed/`, then run:

```powershell
& '..\backend\.venv\Scripts\python.exe' 'methods\yolo_segmentation\build_dataset.py' --source reviewed
& '..\backend\.venv\Scripts\python.exe' 'methods\yolo_segmentation\train_yolo.py' --name rice_stone_v1
& '..\backend\.venv\Scripts\python.exe' 'methods\yolo_segmentation\evaluate_counts.py' --weights '..\models\experiments\yolo_segmentation\rice_stone_v1.pt'
& '..\backend\.venv\Scripts\python.exe' 'methods\yolo_segmentation\predict_count.py' '..\data\rice_count\test\test_r150_f015_a.jpg' --weights '..\models\experiments\yolo_segmentation\rice_stone_v1.pt'
```

Training checkpoints and training runs are saved under `A:\grain\models\experiments\yolo_segmentation\`. Keep photos from one layout group in the same split.
