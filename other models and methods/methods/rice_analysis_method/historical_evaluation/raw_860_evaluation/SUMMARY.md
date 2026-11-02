# Fresh proxy raw-set count evaluation

## Scope and reference

- Images: **860/860** from `A:\grain\data\rice_count\raw`; all files were processed successfully.
- Reference used per your instruction: **100 rice grains and 10 foreign-matter objects per image**.
- Fresh proxy: current `rice_labeler.py`, 40% linear processing scale, with its minimum-area threshold scaled to match.
- This is an **in-sample training-set count-agreement evaluation** against the fixed reference. The raw-set README describes the composition as approximate, and the folder does not contain image-specific reviewed instance masks; confirm the fixed labels before treating the numbers as ground-truth accuracy or independent validation.

## Results

| Metric | Rice (reference 100) | Foreign matter (reference 10) |
|---|---:|---:|
| Mean predicted count | 106.68 | 9.94 |
| MAE [95% image-bootstrap CI] | 7.59 [6.57, 8.70] | 0.43 [0.40, 0.47] |
| RMSE | 17.80 | 0.73 |
| 95th percentile absolute error | 34 | 1 |
| Maximum absolute error | 155 | 4 |
| Mean signed bias | +6.68 | -0.06 |
| Exact count | 194/860 (22.6%) | 519/860 (60.3%) |
| Within ±1 count | 438/860 (50.9%) | 837/860 (97.3%) |
| Within practical tolerance | 627/860 (72.9%; ±5 grains) | 837/860 (97.3%; ±1 object) |
| WAPE | 7.59% | 4.34% |

- Both class counts exact: **135/860 (15.70%)**.
- Combined count accuracy (count-weighted): **92.70%**; total absolute error 6903 over 94600 reference objects.
- Mean rice-share error: **0.642 percentage points**; original share 90.91%, fresh mean 91.36%.
- Runtime mean: 1.43 s/image across workers (sum of per-image elapsed times; not wall-clock throughput).

Count accuracy is not object-level detection accuracy. Precision, recall, F1, and mask IoU cannot be measured without reviewed instance labels. Wilson intervals in the rate chart and image-bootstrap intervals are descriptive; captures may be correlated.

## Graphs

- `graphs/scorecard.png` — main KPI summary.
- `graphs/prediction_vs_reference_by_image.png` — count by image with nominal reference and tolerance band.
- `graphs/absolute_error_and_drift.png` — per-image error with rolling MAE.
- `graphs/count_residual_distributions.png` — signed error/bias distributions.
- `graphs/predicted_count_distributions.png` — count distributions against the nominal reference.
- `graphs/count_agreement_rates_ci.png` — exact/tolerance rates with 95% Wilson intervals.
- `graphs/rice_share_vs_original.png` — fresh rice share versus original 100:10 share.

`counts.csv` has per-image results; `graph_data.csv` is numerically ordered for plotting; `worst_20_images.csv` highlights the largest combined errors; `metrics.json` has full metrics and interval estimates. `failed_images.csv` records any unreadable or failed inputs (empty if none).
