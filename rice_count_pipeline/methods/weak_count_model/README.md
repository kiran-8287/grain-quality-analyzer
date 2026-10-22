# Weak-label YOLO experiment

This experimental YOLO-segmentation model uses pseudo-polygons from the shared rule-based proxy at `../autolabel_proxy/autolabel.py`. It does not use reviewed ground-truth instance masks. Pseudo-validation mAP and count-fit scores measure agreement with the proxy labels, not true accuracy.

From `A:\grain\rice_count_pipeline` in PowerShell, use the shared Python environment at `A:\grain\backend\.venv`:

```powershell
& '..\backend\.venv\Scripts\python.exe' 'methods\weak_count_model\build_pseudo_dataset.py'
& '..\backend\.venv\Scripts\python.exe' 'methods\weak_count_model\train_model.py' --epochs 50
& '..\backend\.venv\Scripts\python.exe' 'methods\weak_count_model\evaluate_test_set.py'
& '..\backend\.venv\Scripts\python.exe' 'methods\weak_count_model\predict_count.py' 'path\to\new_image.jpg'
```

Resume with `train_model.py --name weak_count_yolo --resume`; score the existing checkpoint against pseudo-label counts with `train_model.py --score-only`. Checkpoints, training metrics, and model prediction outputs are under `A:\grain\models\experiments\weak_count_model\`; pseudo-count reports and caches are under `A:\grain\data\rice_count\work\weak_count_model\`. The prior proxy and area evaluation outputs have been moved to their respective method folders.
