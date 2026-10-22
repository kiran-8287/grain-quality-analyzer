# Grain model experiment setup

This guide covers an isolated, experimental training workflow. These models are not part of the production analysis pipeline.

## Environment setup (Windows PowerShell)

From the repository root, create/use the project virtual environment and install the CUDA 13.0 PyTorch wheels selected for this machine (NVIDIA driver 581.86):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r backend\models\requirements-ml.txt
python -c "import torch, torchvision, cv2, pycocotools, ultralytics; print(torch.__version__, torch.cuda.is_available())"
```

Use the virtual environment for every command below. Never install these dependencies into the web application's Python environment. The CUDA wheel bundles the runtime; a separate CUDA toolkit is not needed. If CUDA is unavailable in the environment, install the CPU wheel instead with `python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

## Annotation and training order

**raw photos -> `auto_annotate.py` -> import to CVAT (COCO 1.0), fix every `needs_review` polygon, export COCO -> `prepare_data.py` -> `train_maskrcnn.py` and `train_yolo.py` (both use the same train/val split).**

In CVAT or Roboflow, use exactly `whole_rice`, `broken_rice`, and `foreign_matter`. Draw polygons for rice grains. Export as COCO 1.0 / COCO Segmentation at original image resolution; do not resize images. Delete or redraw every `needs_review` polygon into the correct individual labels. Check that each COCO `file_name` is relative to the images folder and resolves to the original image. Label every visible grain: unlabeled grains teach the model that touching grains are background.

Starting points (adjust after inspecting the dataset and resource use):

```powershell
python backend\models\auto_annotate.py --images data\grain\raw_images --out data\grain\annotations\auto.json --overlay-dir data\grain\overlays
python backend\models\prepare_data.py --coco data\grain\annotations\corrected.json --images data\grain\raw_images --out data\grain\dataset
python backend\models\train_maskrcnn.py --images data\grain\raw_images --train-json data\grain\dataset\maskrcnn\train.json --val-json data\grain\dataset\maskrcnn\val.json --output-dir runs\grain\maskrcnn --crop-size 512 --crops-per-image 3 --epochs 40 --batch-size 1 --workers 0 --amp
python backend\models\train_yolo.py --data data\grain\dataset\yolo\foreign.yaml --imgsz 1280 --batch 4 --workers 0 --epochs 40 --project (Join-Path (Get-Location) 'runs\grain\yolo')
```

These are conservative starting points verified to complete synthetic GPU smoke training on the 6 GB RTX 3050 Laptop GPU. Synthetic smoke runs demonstrate that the training loop executes; their metrics do not validate real-data accuracy. The YOLO command passes `project` through to Ultralytics. Use the absolute project path above so output stays under the repository's `runs\grain\yolo` folder; Ultralytics resolves relative project names under its configured `runs\detect` directory.

## Troubleshooting

- GPU memory pressure: use Mask R-CNN `--crop-size 512 --batch-size 1 --workers 0 --amp`. On this 6 GB GPU, training/evaluation completed at 512, but validation at 768 and 1024 ran out of CUDA memory. At 1024, training completed only in a diagnostic run with cuDNN benchmarking disabled; validation still ran out of memory. Do not treat larger crops as verified on this machine.
- Host memory pressure: monitor available RAM and C: free space; Windows is configured for a system-managed pagefile, which grew substantially during training. Close unused browser and editor sessions before a longer run if needed. Leave the pagefile system-managed and maintain disk space for it to grow.
- Inspect the printed grain-size statistics. If the median grain box is over 100 px, try `--anchor-sizes 32 64 128 256 512`.
- `foreign.yaml` contains an absolute `path:`. Re-run `prepare_data.py` if the dataset folder moves.
- An EXIF size-mismatch error means the decoded image size differs from the COCO dimensions. Re-export annotations against the original, unresized images; keep the training script's size guard enabled.

## Dataset and outputs

Keep source photos in `data\grain\raw_images`, corrected annotation exports in `data\grain\annotations`, prepared splits in `data\grain\dataset`, and experiment outputs in `runs\grain`. These locations and model weight files are ignored by Git. The prepared split is by image and is shared by both training scripts.
