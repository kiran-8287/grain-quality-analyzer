from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
base=Path(__file__).resolve().parent
old_dir=base/'old_proxy_rerun'
old_rows=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((old_dir/'per_image').glob('*.json'))]
with (base/'counts.csv').open(newline='',encoding='utf-8-sig') as f: fresh_rows={r['image']:r for r in csv.DictReader(f)}
old_rows_by={r['image']:r for r in old_rows}
if len(old_rows_by)!=20 or set(old_rows_by)!=set(fresh_rows):
    raise RuntimeError(f'Need 20 matching rows: old={len(old_rows_by)} fresh={len(fresh_rows)}')
rows=[]
for name in sorted(fresh_rows):
    o,f=old_rows_by[name],fresh_rows[name]
    ar=int(f['actual_rice_total']); af=int(f['actual_foreign'])
    opr=int(o['predicted_rice']); ofr=int(o['predicted_foreign'])
    fpr=int(f['predicted_rice']); ffr=int(f['predicted_foreign'])
    rows.append({'image':name,'actual_rice':ar,'old_predicted_rice':opr,'fresh_predicted_rice':fpr,
                 'actual_foreign':af,'old_predicted_foreign':ofr,'fresh_predicted_foreign':ffr,
                 'old_rice_abs_error':abs(opr-ar),'fresh_rice_abs_error':abs(fpr-ar),
                 'old_foreign_abs_error':abs(ofr-af),'fresh_foreign_abs_error':abs(ffr-af),
                 'old_both_exact':opr==ar and ofr==af,'fresh_both_exact':fpr==ar and ffr==af})
with (base/'old_vs_fresh_counts.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def one(model,actual_key,pred_key):
    a=np.array([int(x[actual_key]) for x in rows]); p=np.array([int(x[pred_key]) for x in rows]); e=np.abs(p-a)
    return {'mae':float(e.mean()),'exact_images':int((e==0).sum()),'exact_rate_percent':float(100*(e==0).mean()),
            'within_1_images':int((e<=1).sum()),'wape_percent':float(100*e.sum()/a.sum()),
            'signed_bias':float((p-a).mean()),'absolute_error_total':int(e.sum()),'actual_count_total':int(a.sum())}
metrics={}
for model in ('old','fresh'):
    rice=one(model,'actual_rice',f'{model}_predicted_rice')
    foreign=one(model,'actual_foreign',f'{model}_predicted_foreign')
    error=rice['absolute_error_total']+foreign['absolute_error_total']; actual=rice['actual_count_total']+foreign['actual_count_total']
    both=sum(bool(r[f'{model}_both_exact']) for r in rows)
    metrics[model]={'rice':rice,'foreign':foreign,'both_classes_exact_images':both,
        'both_classes_exact_rate_percent':100*both/len(rows),'combined_abs_error_total':error,
        'combined_actual_total':actual,'combined_count_accuracy_percent':100*(1-error/actual),
        'mean_total_abs_error_per_image':error/len(rows)}
metrics.update({'images_compared':len(rows),
 'ground_truth':'Filename encoded counts; rice=whole+broken, foreign=stones.',
 'old_proxy_source':'rice_count_pipeline/methods/autolabel_proxy/autolabel.py, existing evaluator settings: longest side 1536 px, min rice area 200 px, min foreign area 700 px, max split clusters 50.',
 'fresh_proxy_source':'autolabel_proxy_fresh/rice_labeler.py, processing scale 0.4, min area scaled from 400 px.',
 'note':'Old historical summary_metrics.csv disagreed with the later per-image CSV. The old implementation was rerun on all same images; results here are from that isolated rerun.'})
metrics['definitions']={'MAE':'Mean absolute count error per image for a class.','WAPE':'Class total absolute error divided by the class total actual count.','combined_count_accuracy':'100% - total absolute rice+foreign count error / total actual rice+foreign count.','exact_rate':'Share of images with exact predicted count for the class.'}
(base/'old_vs_fresh_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
fig,axs=plt.subplots(2,2,figsize=(11,8)); labels=['Old proxy','Fresh proxy']; colors=['#527aa3','#df8a3b']
plots=[(axs[0,0],[metrics['old']['rice']['mae'],metrics['fresh']['rice']['mae']],'Rice MAE','Grains/image','{:.2f}'),
       (axs[0,1],[metrics['old']['foreign']['mae'],metrics['fresh']['foreign']['mae']],'Foreign matter MAE','Items/image','{:.2f}'),
       (axs[1,0],[metrics['old']['both_classes_exact_images'],metrics['fresh']['both_classes_exact_images']],'Both class counts exact','Images out of 20','{:.0f}/20'),
       (axs[1,1],[metrics['old']['combined_count_accuracy_percent'],metrics['fresh']['combined_count_accuracy_percent']],'Combined count accuracy','Percent','{:.2f}%')]
for ax,values,title,ylabel,fmt in plots:
    bars=ax.bar(labels,values,color=colors,width=.58);ax.set_title(title);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.25)
    for b,v in zip(bars,values): ax.text(b.get_x()+b.get_width()/2,b.get_height(),fmt.format(v),ha='center',va='bottom',fontsize=10)
fig.suptitle('Old vs fresh auto-label proxy — same 20 images',fontsize=14);fig.tight_layout()
fig.savefig(base/'old_vs_fresh_comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)
o=metrics['old'];f=metrics['fresh']
report=f'''# Old vs fresh auto-label proxy comparison

Compared **{len(rows)} matching images** using filename counts as ground truth (rice = whole + broken; foreign = stones).

| Metric | Old proxy | Fresh proxy |
|---|---:|---:|
| Rice MAE (grains/image) | {o['rice']['mae']:.2f} | {f['rice']['mae']:.2f} |
| Foreign matter MAE (items/image) | {o['foreign']['mae']:.2f} | {f['foreign']['mae']:.2f} |
| Rice exact count | {o['rice']['exact_images']}/20 ({o['rice']['exact_rate_percent']:.1f}%) | {f['rice']['exact_images']}/20 ({f['rice']['exact_rate_percent']:.1f}%) |
| Foreign exact count | {o['foreign']['exact_images']}/20 ({o['foreign']['exact_rate_percent']:.1f}%) | {f['foreign']['exact_images']}/20 ({f['foreign']['exact_rate_percent']:.1f}%) |
| Both counts exact | {o['both_classes_exact_images']}/20 ({o['both_classes_exact_rate_percent']:.1f}%) | {f['both_classes_exact_images']}/20 ({f['both_classes_exact_rate_percent']:.1f}%) |
| Combined count accuracy | {o['combined_count_accuracy_percent']:.2f}% | {f['combined_count_accuracy_percent']:.2f}% |
| Combined absolute error | {o['combined_abs_error_total']} counts | {f['combined_abs_error_total']} counts |

**Overall, the old proxy performed better by total count error**: {o['combined_abs_error_total']} vs {f['combined_abs_error_total']} across the set, giving {o['combined_count_accuracy_percent']:.2f}% vs {f['combined_count_accuracy_percent']:.2f}% combined count accuracy. Its rice-count MAE was lower ({o['rice']['mae']:.2f} vs {f['rice']['mae']:.2f}). The fresh proxy performed better for foreign matter (MAE {f['foreign']['mae']:.2f} vs {o['foreign']['mae']:.2f}) and had more exact matches on both classes together ({f['both_classes_exact_images']}/20 vs {o['both_classes_exact_images']}/20).

Accuracy here is count accuracy, not per-object segmentation accuracy. Combined count accuracy is count-weighted, so the rice class (larger counts) has more influence. There are no instance-level ground-truth masks to compare segmentation precision or recall.

The prior old-folder summary and later per-image CSV disagreed. To avoid relying on stale mixed outputs, the old implementation was rerun from `rice_count_pipeline/methods/autolabel_proxy/autolabel.py` with its evaluator settings (long side 1536 px, rice minimum area 200 px, foreign minimum area 700 px, maximum 50 split clusters). Its new counts and overlays are in `old_proxy_rerun/`. The fresh results remain in `counts.csv` and `overlays/`.

See `old_vs_fresh_counts.csv`, `old_vs_fresh_metrics.json`, and `old_vs_fresh_comparison.png` for per-image and full metric details.
'''
(base/'OLD_VS_FRESH.md').write_text(report,encoding='utf-8')
with (old_dir/'counts.csv').open('w',newline='',encoding='utf-8-sig') as fh:
    w=csv.DictWriter(fh,fieldnames=list(old_rows[0]));w.writeheader();w.writerows(old_rows)
print(report)
