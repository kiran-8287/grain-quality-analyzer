# Fresh rice auto-label proxy

This folder contains the pasted classical rice/foreign-matter labeler and an evaluation runner for `A:\grain\data\rice_count\test`.

Run from this folder with:

```powershell
python -m pip install -r requirements.txt
python run_evaluation.py
```

The runner reads expected whole, broken, and stone counts from each filename. Rice ground truth is whole + broken. For memory and runtime on these large photos, the proxy processes images at 40% linear resolution and scales the minimum component area accordingly; overlays are saved at that processing resolution.

Outputs:

- `REPORT.md`, `metrics.json`: summary metrics and definitions
- `counts.csv`: expected and predicted counts for each image
- `actual_vs_predicted.png`: per-class actual-vs-predicted scatter plot
- `overlays/`: annotated images
