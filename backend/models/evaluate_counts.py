import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from predict_count import count_image

ROOT = Path(__file__).resolve().parents[1]
COUNT_RE = re.compile(r"_r(\d+)_f(\d+)_", re.IGNORECASE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--tile", type=int)
    ap.add_argument("--overlap", type=float, default=0.2)
    args = ap.parse_args()
    test_dir = ROOT / "data" / "test"
    paths = sorted(p for p in test_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths: raise SystemExit(f"No test photos in {test_dir}")
    rows = []
    for path in paths:
        match = COUNT_RE.search(path.name)
        if not match:
            print(f"skip (filename needs _rN_fN_): {path.name}"); continue
        tr, tf = map(int, match.groups())
        pred = count_image(path, args.weights, tile=args.tile, overlap=args.overlap)
        pr, pf = pred["n_rice"], pred["n_foreign"]
        rows.append({"photo": path.name, "true_rice": tr, "pred_rice": pr,
                     "rice_abs_error": abs(pr - tr), "rice_pct_error": 100 * abs(pr - tr) / tr if tr else "",
                     "rice_signed_error": pr - tr, "true_foreign": tf, "pred_foreign": pf,
                     "foreign_abs_error": abs(pf - tf),
                     "foreign_pct_error": 100 * abs(pf - tf) / tf if tf else "",
                     "foreign_signed_error": pf - tf})
        print(f"{path.name}: rice {pr}/{tr}, foreign {pf}/{tf}")
    if not rows: raise SystemExit("No filenames matched the true-count pattern")
    work = ROOT / "data" / "work"; work.mkdir(parents=True, exist_ok=True)
    with (work / "eval_counts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    for cls in ("rice", "foreign"):
        abs_err = np.array([r[f"{cls}_abs_error"] for r in rows], float)
        true = np.array([r[f"true_{cls}"] for r in rows], float)
        valid = true > 0
        mape = float(np.mean(abs_err[valid] / true[valid]) * 100) if valid.any() else float("nan")
        print(f"{cls}: MAE={abs_err.mean():.3f} MAPE={mape:.2f}%")
        buckets = sorted(set(true.astype(int)))
        for bucket in buckets:
            vals = [r[f"{cls}_signed_error"] for r in rows if r[f"true_{cls}"] == bucket]
            print(f"  true {cls}={bucket}: mean signed error={np.mean(vals):+.3f}")
    plt.figure(figsize=(7, 7))
    plt.scatter([r["true_rice"] for r in rows], [r["pred_rice"] for r in rows], alpha=0.8)
    lim = max(max(r["true_rice"] for r in rows), max(r["pred_rice"] for r in rows), 1)
    plt.plot([0, lim], [0, lim], "k--"); plt.xlabel("True rice count"); plt.ylabel("Predicted rice count")
    plt.tight_layout(); plt.savefig(work / "counts_scatter.png", dpi=160); plt.close()
    model = YOLO(str(args.weights))
    metrics = model.val(data=str(ROOT / "data" / "yolo" / "data.yaml"), imgsz=1280, max_det=2000)
    names = model.names
    for i, name in names.items():
        def class_ap(metric):
            found = np.flatnonzero(np.asarray(metric.ap_class_index) == i)
            return f"{metric.ap50[found[0]]:.4f}" if len(found) else "n/a"
        print(f"{name}: box mAP50={class_ap(metrics.box)} mask mAP50={class_ap(metrics.seg)}")


if __name__ == "__main__":
    main()
