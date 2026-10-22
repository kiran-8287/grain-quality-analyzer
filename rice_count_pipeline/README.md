# Rice and stone instance counting experiment

This is an isolated project at `A:\grain\rice_count_pipeline`. It does not change the existing application's classical pipeline. The input photos are copied into `data/raw`; generated datasets, annotations, checkpoints, caches, and temporary files stay in this directory on A:.

## Goal and limits

Train an Ultralytics YOLO11 segmentation model to detect each visible rice grain (class 0) and stone (class 1), then count instances by class. Training images with about 100 rice and 10 stones are the starting set. Reliable counting at 150/15 requires labeled high-density examples and a separate test set of fresh 150/15 arrangements. Fully hidden grains cannot be guaranteed from a single image.

Each labeled object needs an instance polygon. The classical labeler can propose masks, but all annotations used for validation and test must be reviewed. Touching/overlapping grains need individual visible-region masks under one consistent rule.

The current 860-photo audit is only a proxy sanity check, not a verified count set. All 860 files are 3072×4096, not 1932×2576. The layout matcher produced provisional groups; inspect borderline pairs and unusually large groups before relying on validation scores. The audit found 535 PASS photos and 81 groups with a PASS representative. The first review round selects one representative per qualifying group to limit duplicate layouts. In the generated review package, 34 of 81 images have no auto-label proposals because the unchanged full-resolution labeler ran out of memory; their filenames are listed in `coco.json` under `info.images_without_proposals` and require manual annotation. No independent, freshly arranged 150-rice/15-foreign test photos are currently in `data/test`, so accuracy at that composition cannot yet be claimed.

## Setup (Windows, Python 3.11, NVIDIA CUDA)

From PowerShell:

```powershell
Set-Location A:\grain\rice_count_pipeline
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

`setup.ps1` creates the venv and all caches under this folder, installs CUDA 12.4 PyTorch wheels and the project requirements, then prints CUDA availability and GPU name.

## Run order

1. Refresh the photo audit and layout groups after the raw set is frozen:

   ```powershell
   .\.venv\Scripts\python.exe audit_photos_v2.py data\raw --out data\work\audit
   .\.venv\Scripts\python.exe find_layout_groups.py data\raw --out data\work\layout_groups
   ```

   Inspect borderline matches and confirm layout groups before splitting.

2. Run the no-polygon area-counting prototype:

   ```powershell
   .\.venv\Scripts\python.exe models\area_count.py calibrate
   .\.venv\Scripts\python.exe models\area_count.py predict data\raw\1.jpg
   ```

   This is experimental and calibrated using the assumption that reference photos contain 100 rice grains. A prediction on `data/raw` is an in-sample proxy, not accuracy validation. The estimate can fail when grains occlude one another or the rice/stone color masks are wrong. Wait for exact-count, freshly arranged images in `data/test` before judging whether the approach works for 150 grains.

3. Only if the area counter is not accurate enough, prepare a COCO review package:

   ```powershell
   .\.venv\Scripts\python.exe models\build_dataset.py --source classical --overlays 6
   ```

   This writes `data\work\report.csv` (area-proxy counts), `round1_selection.csv`, `review\coco.json`, and six overlay contact sheets under `data\work\overlays`. If we choose the segmentation fallback, create a Roboflow **Instance Segmentation** project with labels `rice` and `foreign`. Upload the review images and COCO annotations together. Correct every mask, manually annotate the 34 images with no proposals, then export reviewed COCO into `data\work\reviewed`. Check project privacy before uploading. Do not treat area-audit estimates or uncorrected proposals as instance labels.

4. Build the group-isolated YOLO dataset only if segmentation labeling is needed:

   ```powershell
   .\.venv\Scripts\python.exe models\build_dataset.py --source reviewed
   ```

5. Train and save the checkpoint:

   ```powershell
   .\.venv\Scripts\python.exe models\train_yolo.py --name rice_stone_v1
   ```

6. Evaluate only on held-out, freshly arranged photos with true counts in names such as `test_r150_f015_a.jpg`:

   ```powershell
   .\.venv\Scripts\python.exe models\evaluate_counts.py --weights models\weights\rice_stone_v1.pt
   .\.venv\Scripts\python.exe models\predict_count.py data\test\test_r150_f015_a.jpg --weights models\weights\rice_stone_v1.pt
   ```

Prediction defaults to full-frame at `imgsz=1280`, matching training. Use `--tile 1280` only if tile-sized training examples are added; tiled inference changes object scale. Keep every photo and all its tiles in one layout split. Review count MAE and per-class mask mAP50; neither metric alone establishes performance at 150/15. Agree on acceptable count errors before evaluating the final test set.

`predict_count.py` writes overlays to `data\work\results\` by default, keeping them out of `data\raw`.
