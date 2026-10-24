# Weak-label YOLO experiment

This experimental YOLO-segmentation model uses pseudo-polygons from the shared rule-based proxy at `../autolabel_proxy/autolabel.py`. The proxy estimates local felt color in Lab space, separates rice and brown/dark foreground, and splits connected rice clumps by area. The dataset builder keeps the largest contour per pseudo-instance so disconnected fragments do not become false extra objects. It does not use reviewed ground-truth instance masks. Pseudo-validation mAP and count-fit scores measure agreement with the proxy labels, not true accuracy.

From `A:\grain\rice_count_pipeline` in PowerShell, use the shared Python environment at `A:\grain\backend\.venv`:

```powershell
& '..\backend\.venv\Scripts\python.exe' 'methods\weak_count_model\build_all_raw_dataset.py'
& '..\backend\.venv\Scripts\python.exe' 'methods\weak_count_model\train_model.py' --name weak_count_lab_v3_30ep_768 --epochs 30 --imgsz 768
& '..\backend\.venv\Scripts\python.exe' 'methods\weak_count_model\evaluate_test_set.py'
& '..\backend\.venv\Scripts\python.exe' 'methods\weak_count_model\predict_count.py' 'path\to\new_image.jpg'
```

The all-raw dataset keeps every photo in training and uses 1536 px copies to limit Windows RAM use; source photos remain untouched. Training skips per-epoch validation because the only supplied labels are proxy labels; one final mAP pass on those same images is in-sample agreement, not independent validation accuracy. The 20 filename-counted test photos remain separate. Checkpoints and training metrics are under `A:\grain\models\experiments\weak_count_model\`; this method's test reports and prediction overlays are under `A:\grain\rice_count_pipeline\methods\weak_count_model\results\`. Pseudo-count reports and caches are under `A:\grain\data\rice_count\work\weak_count_model_all_raw\`.
