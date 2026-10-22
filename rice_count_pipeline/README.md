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

The setup script creates the workspace venv, installs CUDA 12.4 PyTorch wheels and project requirements, then reports CUDA availability and the GPU name. Run commands below from `A:\grain\rice_count_pipeline` using `& '.venv\Scripts\python.exe' ...`.

## Audit and layout grouping

```powershell
& '.venv\Scripts\python.exe' 'tools\audit_photos_v2.py' '..\data\rice_count\raw' --out '..\data\rice_count\work\audit'
& '.venv\Scripts\python.exe' 'tools\find_layout_groups.py' '..\data\rice_count\raw' --out '..\data\rice_count\work\layout_groups'
```

These outputs are exploratory image-processing results, not verified counts. Inspect the grouping and audit reports before using them for model splits.

## Compare no-training methods

```powershell
& '.venv\Scripts\python.exe' 'methods\autolabel_proxy\evaluate_test_set.py' --source '..\data\rice_count\test'
& '.venv\Scripts\python.exe' 'methods\area_count\area_count.py' calibrate
& '.venv\Scripts\python.exe' 'methods\area_count\evaluate_area_test_set.py' --source '..\data\rice_count\test'
& '.venv\Scripts\python.exe' 'methods\area_count\area_count.py' predict '..\data\rice_count\raw\1.jpg'
```

Both are experimental proxies. The area method estimates rice quantity from masked area; its foreign count is connected blobs, not necessarily individual stones.

## Weak-label YOLO experiment

```powershell
& '.venv\Scripts\python.exe' 'methods\weak_count_model\build_pseudo_dataset.py'
& '.venv\Scripts\python.exe' 'methods\weak_count_model\train_model.py' --epochs 50
& '.venv\Scripts\python.exe' 'methods\weak_count_model\evaluate_test_set.py'
& '.venv\Scripts\python.exe' 'methods\weak_count_model\predict_count.py' 'path\to\new_image.jpg'
```

The model learns pseudo-polygons from the rule-based labeler, not reviewed instance masks. Its pseudo-validation metrics measure agreement with proxy labels. Resume with `--name weak_count_yolo --resume`; use `--score-only` to score saved weights without retraining. These are not independent accuracy measurements.

## Reviewed instance-segmentation workflow

Review images and COCO proposals are in `data/rice_count/work/review/`; the Roboflow import package is in `data/rice_count/work/roboflow_upload/`. Export reviewed COCO polygons into `data/rice_count/work/reviewed/`, then run:

```powershell
& '.venv\Scripts\python.exe' 'methods\yolo_segmentation\build_dataset.py' --source reviewed
& '.venv\Scripts\python.exe' 'methods\yolo_segmentation\train_yolo.py' --name rice_stone_v1
& '.venv\Scripts\python.exe' 'methods\yolo_segmentation\evaluate_counts.py' --weights 'methods\yolo_segmentation\weights\rice_stone_v1.pt'
& '.venv\Scripts\python.exe' 'methods\yolo_segmentation\predict_count.py' '..\data\rice_count\test\test_r150_f015_a.jpg' --weights 'methods\yolo_segmentation\weights\rice_stone_v1.pt'
```

Training and evaluation outputs remain inside their method folder. Keep photos from one layout group in the same split.
