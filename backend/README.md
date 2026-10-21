# Rice and foreign matter experiment

This is an isolated YOLO11 segmentation pipeline. It does not change the app's existing classical CV gate or segmentation. All commands below run from `backend/` in PowerShell. Keep test images out of `data/raw/`.

## Setup (Windows, Python 3.11, NVIDIA CUDA)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:TEMP = Join-Path (Get-Location) "tmp"
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
python -m pip install --no-cache-dir --upgrade pip
python -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install --no-cache-dir -r requirements.txt
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')"
```

Put training photos in `data/raw/`, filename-count test photos in `data/test/`, and the unchanged `models/autolabel.py` module (providing `label_image(path)`) in `models/`. That module supplies classical labels for training and review-package polygons.

## Run order

1. Make a CVAT review package from classical labels:

   ```powershell
   python models/build_dataset.py --source classical
   ```

   Review `data/work/review/coco.json` in CVAT using COCO 1.0. Correct reviewed objects and export COCO 1.0 to `data/work/reviewed/`.

2. Build the group-isolated YOLO dataset. Only reviewed photos in validation groups enter validation; other photos from those groups are excluded.

   ```powershell
   python models/build_dataset.py --source reviewed
   ```

3. Train and save the best checkpoint as `models/weights/final.pt`:

   ```powershell
   python models/train_yolo.py --name final
   ```

4. Evaluate filename-count test images, then count one image:

   ```powershell
   python models/evaluate_counts.py --weights models/weights/final.pt
   python models/predict_count.py data/test/test_r025_f000_a.jpg --weights models/weights/final.pt
   ```

Optional flags: `build_dataset.py --expected-rice 100 --expected-foreign 10 --tol 3 --overlays 4`; prediction/evaluation accept `--tile 1280 --overlap 0.2`. Prediction accepts `--px-per-mm` and `--broken-ratio`. Length measurements are pixels unless an explicit scale is provided. Broken flags are image-derived estimates.
