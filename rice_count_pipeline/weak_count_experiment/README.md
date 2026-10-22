# Weak-label instance-counting experiment

This workspace is isolated from the existing area estimator and application. It creates a YOLO11-seg student using the existing `models/autolabel.py` masks as automatic pseudo-labels. The 860 raw photos are not manually annotated.

The builder downsizes each frame to a 1536-pixel long side and scales the pseudo-labeler's 400 px minimum-area cutoff to 56 px. This preserves the shared labeler code while reducing its memory cost. It retains pseudo-labeled photos only when the predicted counts are within 80–130 rice and 6–14 foreign objects; extreme outliers create huge, noisy masks and are excluded. The pseudo-labeler can still split touching-grain clumps incorrectly. Therefore train/validation mAP and count-fit scores measure agreement with its labels, **not true accuracy**. The image-level 100-rice/10-foreign composition is only a plausibility gate; it is not used as pixel-level ground truth. A new exact-count image is still required for an independent count check.

## Run

To run the untrained proxy labeler on the count-named test photos, use:

```powershell
& '.venv\Scripts\python.exe' 'weak_count_experiment/evaluate_autolabel_test_set.py'
```

From `A:\grain\rice_count_pipeline` in PowerShell, using the existing Python 3.11 venv:

```powershell
$env:YOLO_CONFIG_DIR = 'A:\grain\rice_count_pipeline\weak_count_experiment\ultralytics_config'
$env:TORCH_HOME = 'A:\grain\rice_count_pipeline\weak_count_experiment\torch_cache'
$env:MPLCONFIGDIR = 'A:\grain\rice_count_pipeline\weak_count_experiment\mpl_cache'
& '.venv\Scripts\python.exe' 'weak_count_experiment\build_pseudo_dataset.py'
& '.venv\Scripts\python.exe' 'weak_count_experiment\train_model.py' --epochs 50
& '.venv\Scripts\python.exe' 'weak_count_experiment\predict_count.py' 'path\to\new_image.jpg'
& '.venv\Scripts\python.exe' 'weak_count_experiment\evaluate_test_set.py'
```

Resume an interrupted run from its `last.pt` checkpoint with:

```powershell
& '.venv\Scripts\python.exe' 'weak_count_experiment\train_model.py' --name weak_count_yolo --resume
```

Recompute count-fit scores against the pseudo-label counts without retraining:

```powershell
& '.venv\Scripts\python.exe' 'weak_count_experiment\train_model.py' --score-only
```

The data split is by provisional layout group (seed 0). Scaled images and YOLO labels are saved in `dataset_1536_curated/`. Skipped images and pseudo object counts are recorded in `skipped.csv` and `pseudo_counts.csv`; that CSV also marks count outliers excluded from training. Training saves its run, weights, count-fit CSV, and plots under this directory. Inference reports rice/foreign counts and percentages and saves a visual overlay.

`PSEUDO-VAL mask mAP50` is not ground-truth mAP. Count MAE is against pseudo-label counts. Treat all model outputs as experimental until evaluated on fresh images with exact independently known counts.

`evaluate_test_set.py` reads filenames of the form `115 whole, 15 broken, 15 stones.jpg`, treats whole+broken as total rice, compares predicted rice/foreign counts with those filename totals, and saves a CSV, box overlays, and colored mask overlays under `predictions/test_set/`.

`evaluate_autolabel_test_set.py` runs the unchanged rule-based labeler from `A:\grain\backend\models\autolabel.py` on those same filename-counted images. It saves proxy instance outlines and per-photo count errors under `predictions/test_set/autolabel_proxy/`; these are fixed-rule outputs, not trained-model predictions.
