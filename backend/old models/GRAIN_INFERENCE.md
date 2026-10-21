# Experimental grain inference API

The `POST /analyze-ml` and `POST /api/analyze-ml` routes run the isolated Mask R-CNN plus YOLO experiment. They do not change or call the existing `POST /analyze` rice gate or segmentation path. Every successful result and route-level error is marked `experimental: true`. When Torch, torchvision, Ultralytics, or either checkpoint is unavailable, the web app still starts and the route returns HTTP 503.

## Configure and start locally

Use the isolated ML virtual environment and install the regular backend dependencies into that environment. `backend/requirements.txt` is unchanged by this feature.

```powershell
\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
$env:GRAIN_MASKRCNN_CHECKPOINT = (Resolve-Path 'runs\grain\maskrcnn\best.pth').Path
$env:GRAIN_YOLO_WEIGHTS = (Resolve-Path 'runs\grain\yolo\exp\weights\best.pt').Path
\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 10000
```

The `Resolve-Path` commands require the checkpoints to exist. Paths can be absolute or relative to the repository root. Defaults are `runs/grain/maskrcnn/best.pth` and `runs/grain/yolo/exp/weights/best.pt`.

Send a multipart request from PowerShell (replace the sample image path; `px_per_mm` is optional):

```powershell
$file = (Resolve-Path 'data\grain\raw_images\sample.jpg').Path
curl.exe -X POST 'http://127.0.0.1:10000/analyze-ml' `
  -F "file=@$file;type=image/jpeg" `
  -F 'px_per_mm=10'
```

Useful environment settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `GRAIN_MASKRCNN_CHECKPOINT` | `runs/grain/maskrcnn/best.pth` | Mask R-CNN checkpoint |
| `GRAIN_YOLO_WEIGHTS` | `runs/grain/yolo/exp/weights/best.pt` | YOLO weights |
| `GRAIN_DEVICE` | `auto` | `auto`, `cpu`, or a Torch device such as `cuda:0`; unavailable CUDA falls back to CPU |
| `GRAIN_TILE_OVERLAP` | `128` | Tile overlap; values below 128 are rejected |
| `GRAIN_MASK_SCORE_THRESHOLD` | `0.5` | Mask R-CNN score cutoff |
| `GRAIN_YOLO_IMGSZ` | `1280` | YOLO prediction size |
| `GRAIN_YOLO_CONF` | `0.25` | YOLO confidence cutoff |
| `GRAIN_INFER_TIMEOUT_SECONDS` | `180` | Request timeout |
| `GRAIN_MAX_UPLOAD_BYTES` | `20971520` | Maximum upload size |
| `GRAIN_MAX_IMAGE_PIXELS` | `40000000` | Maximum decoded image pixel count |

## Response and measurement limits

Successful responses include image dimensions, class counts and percentages, per-grain pixel measurements and flags, foreign detections, aggregate pixel measurements, model checkpoint names, selected device, timings, and `experimental: true`. Physical length and breadth fields are `null` unless `px_per_mm` is supplied. The supplied scale must describe the same image plane and capture setup as the grains; otherwise those values are not meaningful. Missing scale is reported with `calibration_missing`.

Before interpreting millimetres, calibrate against a physical scale in the image plane and compare a representative sample of isolated detected grains with caliper measurements. Record that calibration protocol and its error on real images. Grains touching the image border or another predicted grain are marked unmeasurable. The model has only been exercised for runtime functionality on synthetic images here; its synthetic outputs are not evidence of real-photo accuracy, grading quality, or compliance with any official lot standard. Use `Experimental` outputs for review until representative real data and a documented validation study support stronger claims.
