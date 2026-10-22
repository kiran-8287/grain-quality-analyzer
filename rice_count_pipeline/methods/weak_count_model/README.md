# Weak-label YOLO experiment

This experimental YOLO-segmentation model uses pseudo-polygons from the shared rule-based proxy at `../autolabel_proxy/autolabel.py`. It does not use reviewed ground-truth instance masks. Pseudo-validation mAP and count-fit scores measure agreement with the proxy labels, not true accuracy.

From `A:\grain\rice_count_pipeline` in PowerShell:

```powershell
& '.venv\Scripts\python.exe' 'methods\weak_count_model\build_pseudo_dataset.py'
& '.venv\Scripts\python.exe' 'methods\weak_count_model\train_model.py' --epochs 50
& '.venv\Scripts\python.exe' 'methods\weak_count_model\evaluate_test_set.py'
& '.venv\Scripts\python.exe' 'methods\weak_count_model\predict_count.py' 'path\to\new_image.jpg'
```

Resume with `train_model.py --name weak_count_yolo --resume`; score the existing checkpoint against pseudo-label counts with `train_model.py --score-only`. Varied-count test overlays and CSVs are under `predictions/test_set/`. The prior proxy and area evaluation outputs have been moved to their respective method folders.
