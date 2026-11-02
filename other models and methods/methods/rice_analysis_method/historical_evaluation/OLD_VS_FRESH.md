# Old vs fresh auto-label proxy comparison

Compared **20 matching images** using filename counts as ground truth (rice = whole + broken; foreign = stones).

| Metric | Old proxy | Fresh proxy |
|---|---:|---:|
| Rice MAE (grains/image) | 2.95 | 7.05 |
| Foreign matter MAE (items/image) | 1.25 | 0.55 |
| Rice exact count | 1/20 (5.0%) | 2/20 (10.0%) |
| Foreign exact count | 2/20 (10.0%) | 9/20 (45.0%) |
| Both counts exact | 0/20 (0.0%) | 1/20 (5.0%) |
| Combined count accuracy | 97.92% | 96.24% |
| Combined absolute error | 84 counts | 152 counts |

**Overall, the old proxy performed better by total count error**: 84 vs 152 across the set, giving 97.92% vs 96.24% combined count accuracy. Its rice-count MAE was lower (2.95 vs 7.05). The fresh proxy performed better for foreign matter (MAE 0.55 vs 1.25) and had more exact matches on both classes together (1/20 vs 0/20).

Accuracy here is count accuracy, not per-object segmentation accuracy. Combined count accuracy is count-weighted, so the rice class (larger counts) has more influence. There are no instance-level ground-truth masks to compare segmentation precision or recall.

The prior old-folder summary and later per-image CSV disagreed. To avoid relying on stale mixed outputs, the old implementation was rerun from `rice_count_pipeline/methods/autolabel_proxy/autolabel.py` with its evaluator settings (long side 1536 px, rice minimum area 200 px, foreign minimum area 700 px, maximum 50 split clusters). Its new counts and overlays are in `old_proxy_rerun/`. The fresh results remain in `counts.csv` and `overlays/`.

See `old_vs_fresh_counts.csv`, `old_vs_fresh_metrics.json`, and `old_vs_fresh_comparison.png` for per-image and full metric details.
