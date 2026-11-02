# Fresh auto-label proxy evaluation

- Images evaluated: 20
- Aggregate count accuracy: 96.24% (count-weighted; definition in `metrics.json`)
- Exact rice count: 2/20 (10.0%)
- Exact foreign count: 9/20 (45.0%)
- Both class counts exact: 1/20 (5.0%)
- Mean absolute error: rice 7.05 grains/image; foreign 0.55 items/image

Ground truth comes from the counts in each image filename. Rice ground truth is whole + broken. The supplied proxy emits rice and foreign instances; the reported exact rates measure exact class totals per image.

See `counts.csv` for per-image counts and `actual_vs_predicted.png` for the graph. Overlays are in `overlays/`.
