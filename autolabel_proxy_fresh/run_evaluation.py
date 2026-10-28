from pathlib import Path
import csv, json, re, sys
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rice_labeler

ROOT = Path(r'A:\grain')
INPUT = ROOT / 'data' / 'rice_count' / 'test'
OUT = Path(__file__).resolve().parent
OVERLAYS = OUT / 'overlays'
OVERLAYS.mkdir(parents=True, exist_ok=True)
files = sorted(INPUT.glob('*'))
rows = []
for index, path in enumerate(files, 1):
    expected = rice_labeler.expected_from_name(path)
    if expected is None:
        raise ValueError(f'Could not read expected counts from filename: {path.name}')
    im = cv2.imread(str(path))
    if im is None:
        raise ValueError(f'Could not read image: {path}')
    im = cv2.resize(im, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    overlay = im.copy()
    counts = {'rice': 0, 'foreign': 0}
    def draw_instance(mask, cls):
        counts[cls] += 1
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = (0, 0, 255) if cls == 'foreign' else (255, 0, 255)
        cv2.drawContours(overlay, contours, -1, color, max(2, im.shape[0] // 500))
    _, _, ref_area = rice_labeler.label_image(path, instance_callback=draw_instance)
    pred_rice, pred_foreign = counts['rice'], counts['foreign']
    actual_rice, actual_foreign = expected
    label = f'Rice {pred_rice}/{actual_rice}; foreign {pred_foreign}/{actual_foreign}'
    font_scale = max(0.7, im.shape[0] / 2500)
    cv2.putText(overlay, label, (24, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
    overlay_path = OVERLAYS / f'{path.stem}_overlay.jpg'
    cv2.imwrite(str(overlay_path), overlay)
    rows.append({
        'image': path.name,
        'actual_whole_rice': int(re.search(r'(\d+)\s*whole', path.name, re.I).group(1)),
        'actual_broken_rice': int(re.search(r'(\d+)\s*broken', path.name, re.I).group(1)),
        'actual_rice_total': actual_rice,
        'predicted_rice': pred_rice,
        'actual_foreign': actual_foreign,
        'predicted_foreign': pred_foreign,
        'rice_abs_error': abs(pred_rice - actual_rice),
        'foreign_abs_error': abs(pred_foreign - actual_foreign),
        'rice_exact': pred_rice == actual_rice,
        'foreign_exact': pred_foreign == actual_foreign,
        'both_classes_exact': pred_rice == actual_rice and pred_foreign == actual_foreign,
        'single_grain_reference_area_px': float(ref_area),
        'overlay': str(overlay_path.relative_to(OUT)),
    })
    print(f'{index:02d}/{len(files)} {path.name}: rice {pred_rice}/{actual_rice}, foreign {pred_foreign}/{actual_foreign}', flush=True)

if len(rows) != 20:
    raise ValueError(f'Expected 20 test images, found {len(rows)}')
with (OUT / 'counts.csv').open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)

n = len(rows)
mae_rice = float(np.mean([r['rice_abs_error'] for r in rows]))
mae_foreign = float(np.mean([r['foreign_abs_error'] for r in rows]))
exact_rice = sum(r['rice_exact'] for r in rows)
exact_foreign = sum(r['foreign_exact'] for r in rows)
exact_both = sum(r['both_classes_exact'] for r in rows)
actual_total = sum(r['actual_rice_total'] + r['actual_foreign'] for r in rows)
abs_error_total = sum(r['rice_abs_error'] + r['foreign_abs_error'] for r in rows)
aggregate_count_accuracy = 100 * (1 - abs_error_total / actual_total)
metrics = {
    'images': n,
    'rice_exact_count_images': exact_rice,
    'rice_exact_count_rate_percent': 100 * exact_rice / n,
    'foreign_exact_count_images': exact_foreign,
    'foreign_exact_count_rate_percent': 100 * exact_foreign / n,
    'both_classes_exact_images': exact_both,
    'both_classes_exact_rate_percent': 100 * exact_both / n,
    'rice_mean_absolute_error_grains_per_image': mae_rice,
    'foreign_mean_absolute_error_items_per_image': mae_foreign,
    'aggregate_count_accuracy_percent': aggregate_count_accuracy,
    'aggregate_absolute_count_error': abs_error_total,
    'aggregate_actual_count': actual_total,
    'accuracy_definition': '100 * (1 - sum of absolute rice and foreign count errors / sum of actual rice and foreign counts); also report exact-count rates and MAE.',
}
(OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, actual_key, pred_key, title, color in [
    (axes[0], 'actual_rice_total', 'predicted_rice', 'Rice count', '#3977b8'),
    (axes[1], 'actual_foreign', 'predicted_foreign', 'Foreign matter count', '#b87333'),
]:
    actual = np.array([r[actual_key] for r in rows])
    predicted = np.array([r[pred_key] for r in rows])
    ax.scatter(actual, predicted, s=55, color=color, alpha=.85, edgecolor='white', linewidth=.6)
    lo, hi = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    pad = max(2, (hi - lo) * .07)
    ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], '--', color='#555', linewidth=1, label='Perfect agreement')
    ax.set_xlim(lo-pad, hi+pad); ax.set_ylim(lo-pad, hi+pad)
    ax.set_title(title); ax.set_xlabel('Actual count'); ax.set_ylabel('Predicted count')
    ax.grid(alpha=.25); ax.legend(loc='best')
fig.suptitle(f'Actual vs predicted counts — {n} test images', fontsize=14)
fig.tight_layout()
fig.savefig(OUT / 'actual_vs_predicted.png', dpi=180, bbox_inches='tight')
plt.close(fig)

lines = [
    '# Fresh auto-label proxy evaluation', '',
    f'- Images evaluated: {n}',
    f'- Aggregate count accuracy: {aggregate_count_accuracy:.2f}% (count-weighted; definition in `metrics.json`)',
    f'- Exact rice count: {exact_rice}/{n} ({100*exact_rice/n:.1f}%)',
    f'- Exact foreign count: {exact_foreign}/{n} ({100*exact_foreign/n:.1f}%)',
    f'- Both class counts exact: {exact_both}/{n} ({100*exact_both/n:.1f}%)',
    f'- Mean absolute error: rice {mae_rice:.2f} grains/image; foreign {mae_foreign:.2f} items/image', '',
    'Ground truth comes from the counts in each image filename. Rice ground truth is whole + broken. The supplied proxy emits rice and foreign instances; the reported exact rates measure exact class totals per image.', '',
    'See `counts.csv` for per-image counts and `actual_vs_predicted.png` for the graph. Overlays are in `overlays/`.'
]
(OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')




