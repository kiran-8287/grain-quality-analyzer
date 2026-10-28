# 860-image fresh proxy evaluation

Run the current fresh labeler on all numeric JPGs in `A:\grain\data\rice_count\raw` with the nominal reference of 100 rice + 10 foreign objects per image:

```powershell
python evaluate_raw_860.py --workers 4 --max-tasks-per-worker 1000
python analyze_raw_860.py
```

Outputs are count-only. Per-image counts, intervals, metrics, diagnostics, and graphs are in this folder. The run does not create overlays or modify the raw photos. The fixed 100/10 target came from the user; check `SUMMARY.md` for the validation scope and limits.
