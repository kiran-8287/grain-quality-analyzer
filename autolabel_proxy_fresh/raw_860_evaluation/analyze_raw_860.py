from pathlib import Path
import csv,json,math,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
GRAPH_DIR=ROOT/'graphs'
GRAPH_DIR.mkdir(exist_ok=True)
with (ROOT/'counts.csv').open(newline='',encoding='utf-8-sig') as f:
    rows=list(csv.DictReader(f))
if len(rows)!=860: raise ValueError(f'Expected exactly 860 successful records; found {len(rows)}')
if any(r.get('actual_rice')!='100' or r.get('actual_foreign')!='10' for r in rows):
    raise ValueError('Unexpected/non-constant reference counts in evaluation CSV')
def idx(r):
    try:return int(Path(r['image']).stem)
    except ValueError:return r['image']
rows.sort(key=idx)
n=len(rows)
image_no=np.array([int(Path(r['image']).stem) for r in rows])
actual_rice=np.array([int(r['actual_rice']) for r in rows]); pred_rice=np.array([int(r['predicted_rice']) for r in rows])
actual_foreign=np.array([int(r['actual_foreign']) for r in rows]); pred_foreign=np.array([int(r['predicted_foreign']) for r in rows])
rice_err=pred_rice-actual_rice; foreign_err=pred_foreign-actual_foreign
rice_abs=np.abs(rice_err); foreign_abs=np.abs(foreign_err)

def wilson(k,n,z=1.959963984540054):
    p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [100*(center-half),100*(center+half)]

def boot_mean_ci(values,seed):
    rng=np.random.default_rng(seed); reps=4000; batch=200
    samples=[]
    for i in range(0,reps,batch):
        ix=rng.integers(0,len(values),size=(min(batch,reps-i),len(values)))
        samples.append(values[ix].mean(axis=1))
    allv=np.concatenate(samples)
    return [float(np.percentile(allv,2.5)),float(np.percentile(allv,97.5))]

def class_stats(name,actual,pred,err,ab,tolerance):
    exact=int((ab==0).sum()); within1=int((ab<=1).sum()); within_tol=int((ab<=tolerance).sum())
    return {
        'reference_count_per_image':int(actual[0]), 'predicted_mean':float(pred.mean()),'predicted_sd_sample':float(pred.std(ddof=1)),
        'predicted_median':float(np.median(pred)),'predicted_min':int(pred.min()),'predicted_max':int(pred.max()),
        'absolute_error_p90':float(np.percentile(ab,90)),'absolute_error_p95':float(np.percentile(ab,95)),'absolute_error_p99':float(np.percentile(ab,99)),'absolute_error_max':int(ab.max()),
        'overcount_images':int((err>0).sum()),'undercount_images':int((err<0).sum()),'zero_bias_images':int((err==0).sum()),
        'signed_bias_mean':float(err.mean()),'signed_bias_bootstrap_95ci':boot_mean_ci(err,100+len(name)),
        'mae':float(ab.mean()),'mae_bootstrap_95ci':boot_mean_ci(ab,200+len(name)),
        'rmse':float(np.sqrt(np.mean(err**2))),'median_absolute_error':float(np.median(ab)),
        'mape_percent':float(np.mean(ab/actual)*100),'wape_percent':float(ab.sum()/actual.sum()*100),
        'count_accuracy_percent':float(100*(1-ab.sum()/actual.sum())),
        'absolute_error_total':int(ab.sum()),'exact_images':exact,'exact_rate_percent':100*exact/n,'exact_rate_wilson_95ci_percent':wilson(exact,n),
        'within_1_images':within1,'within_1_rate_percent':100*within1/n,'within_1_wilson_95ci_percent':wilson(within1,n),
        'practical_tolerance_count':int(tolerance),'within_tolerance_images':within_tol,
        'within_tolerance_rate_percent':100*within_tol/n,'within_tolerance_wilson_95ci_percent':wilson(within_tol,n),
    }

rs=class_stats('rice',actual_rice,pred_rice,rice_err,rice_abs,5) # 5% of reference 100
fs=class_stats('foreign',actual_foreign,pred_foreign,foreign_err,foreign_abs,1) # one item = 10% of reference
both_exact=int(((rice_abs==0)&(foreign_abs==0)).sum())
both_1=int(((rice_abs<=1)&(foreign_abs<=1)).sum())
combined_abs=rice_abs+foreign_abs
combined_actual=actual_rice+actual_foreign
share_actual=100*actual_rice/combined_actual
share_pred=100*pred_rice/(pred_rice+pred_foreign)
metrics={
 'dataset':'A:\\grain\\data\\rice_count\\raw',
 'images_evaluated':n,'images_failed':0,'reference_source':'User-provided fixed reference: 100 rice grains + 10 foreign-matter objects per image.',
 'fresh_proxy_processing':'rice_labeler.py at its current 0.40 linear image scale; min_area scaled by 0.40^2.',
 'rice':rs,'foreign':fs,
 'both_classes_exact_images':both_exact,'both_classes_exact_rate_percent':100*both_exact/n,
 'both_classes_exact_wilson_95ci_percent':wilson(both_exact,n),
 'both_classes_within_1_each_images':both_1,'both_classes_within_1_each_rate_percent':100*both_1/n,
 'combined_absolute_error_total':int(combined_abs.sum()),'combined_actual_count_total':int(combined_actual.sum()),
 'combined_count_accuracy_percent':float(100*(1-combined_abs.sum()/combined_actual.sum())),
 'mean_rice_share_absolute_error_percentage_points':float(np.mean(np.abs(share_pred-share_actual))),
 'mean_rice_share_signed_bias_percentage_points':float(np.mean(share_pred-share_actual)),
 'rice_share_original_percent':float(share_actual[0]),
 'rice_share_fresh_mean_percent':float(share_pred.mean()),
 'rice_share_fresh_sd_percentage_points':float(share_pred.std(ddof=1)),
 'runtime_mean_seconds_per_image':float(np.mean([float(r['seconds']) for r in rows])),
 'runtime_median_seconds_per_image':float(np.median([float(r['seconds']) for r in rows])),
 'limitations':['Evaluation uses fixed user-provided counts, not image-specific reviewed annotations.','Raw folder is training data; this is an in-sample count-agreement audit, not independent validation/generalization.','No instance-level ground-truth masks are available, so object precision, recall, F1, and IoU are not measurable.','Confidence intervals treat images as independent; repeated capture/layout correlations may make them optimistic.']
}
(ROOT/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')

# Plot style shared across outputs.
plt.rcParams.update({'font.size':10,'axes.titlesize':13,'axes.labelsize':10,'legend.fontsize':9,'figure.facecolor':'white','axes.facecolor':'#fbfcfe','axes.edgecolor':'#b8c2cc','axes.grid':True,'grid.alpha':.22,'grid.linewidth':.7,'savefig.facecolor':'white'})
blue='#3978a8'; orange='#ee762f'; navy='#243b53'; teal='#2a9d8f'; red='#c44536'; grey='#788896'

# 1. Count control/agreement by image index with target and practical bands.
fig,axs=plt.subplots(2,1,figsize=(15,9),sharex=True)
for ax,pred,target,title,color,tol in [(axs[0],pred_rice,100,'Rice grains',blue,5),(axs[1],pred_foreign,10,'Foreign matter objects',orange,1)]:
    ax.axhspan(target-tol,target+tol,color=teal,alpha=.10,label=f'Practical reference band ±{tol}')
    ax.axhline(target,color=navy,linestyle='--',linewidth=1.8,label=f'Original count = {target}')
    ax.scatter(image_no,pred,s=12,alpha=.48,color=color,edgecolors='none',label='Fresh proxy count',rasterized=True)
    rolling=np.convolve(pred,np.ones(25)/25,mode='valid')
    ax.plot(image_no[12:-12],rolling,color='#202a44',linewidth=1.7,label='25-image rolling mean')
    ax.set_ylabel('Predicted count');ax.set_title(title,loc='left',fontweight='bold');ax.legend(loc='upper right',ncol=3,frameon=True)
axs[-1].set_xlabel('Raw image number (numeric filename order)')
fig.suptitle('Fresh proxy predictions vs original fixed counts — 860 raw training images',fontsize=16,fontweight='bold',y=.995)
fig.tight_layout();fig.savefig(GRAPH_DIR/'prediction_vs_reference_by_image.png',dpi=220,bbox_inches='tight');plt.close(fig)

# 2. Absolute error by image and rolling MAE.
fig,axs=plt.subplots(2,1,figsize=(15,8.5),sharex=True)
for ax,ab,title,color,mean in [(axs[0],rice_abs,'Rice absolute count error',blue,rs['mae']),(axs[1],foreign_abs,'Foreign-matter absolute count error',orange,fs['mae'])]:
    ax.scatter(image_no,ab,s=11,alpha=.42,color=color,edgecolors='none',label='Per-image absolute error',rasterized=True)
    roll=np.convolve(ab,np.ones(50)/50,mode='valid')
    ax.plot(image_no[24:-25],roll,color=navy,lw=1.7,label='50-image rolling MAE')
    ax.axhline(mean,color=red,lw=1.5,ls='--',label=f'Overall MAE = {mean:.2f}')
    ax.set_ylabel('Absolute count error');ax.set_title(title,loc='left',fontweight='bold');ax.legend(loc='upper right',ncol=3)
axs[-1].set_xlabel('Raw image number (numeric filename order)')
fig.suptitle('Per-image error and drift diagnostics',fontsize=16,fontweight='bold',y=.995)
fig.tight_layout();fig.savefig(GRAPH_DIR/'absolute_error_and_drift.png',dpi=220,bbox_inches='tight');plt.close(fig)

# 3. Signed residual distributions, centered on zero.
fig,axs=plt.subplots(1,2,figsize=(13,5.3))
for ax,err,title,color,stats in [(axs[0],rice_err,'Rice count residuals (predicted − 100)',blue,rs),(axs[1],foreign_err,'Foreign count residuals (predicted − 10)',orange,fs)]:
    lo,hi=int(err.min()),int(err.max());bins=np.arange(lo-.5,hi+1.5,1)
    ax.hist(err,bins=bins,color=color,alpha=.82,edgecolor='white',linewidth=.5)
    ax.axvline(0,color=navy,ls='--',lw=1.8,label='No count bias')
    ax.axvline(err.mean(),color=red,ls='-',lw=1.8,label=f'Mean bias = {err.mean():+.2f}')
    ax.set_title(title,fontweight='bold');ax.set_xlabel('Signed count error');ax.set_ylabel('Number of images');ax.legend()
fig.suptitle('Count residual distributions',fontsize=16,fontweight='bold',y=1.02)
fig.tight_layout();fig.savefig(GRAPH_DIR/'count_residual_distributions.png',dpi=220,bbox_inches='tight');plt.close(fig)

# 4. Predicted count distributions with original reference lines.
fig,axs=plt.subplots(1,2,figsize=(13,5.3))
for ax,pred,target,title,color in [(axs[0],pred_rice,100,'Predicted rice count',blue),(axs[1],pred_foreign,10,'Predicted foreign count',orange)]:
    bins=np.arange(min(pred.min(),target)-.5,max(pred.max(),target)+1.5,1)
    ax.hist(pred,bins=bins,color=color,alpha=.8,edgecolor='white',linewidth=.5)
    ax.axvline(target,color=navy,ls='--',lw=2,label=f'Original reference = {target}')
    ax.axvline(np.mean(pred),color=red,lw=1.8,label=f'Fresh mean = {np.mean(pred):.2f}')
    ax.set_title(title,fontweight='bold');ax.set_xlabel('Predicted object count');ax.set_ylabel('Number of images');ax.legend()
fig.suptitle('Distribution of fresh proxy counts vs original reference',fontsize=16,fontweight='bold',y=1.02)
fig.tight_layout();fig.savefig(GRAPH_DIR/'predicted_count_distributions.png',dpi=220,bbox_inches='tight');plt.close(fig)

# 5. Exact and practical-tolerance rates with 95% Wilson intervals.
labels=['Rice exact','Rice within ±1','Rice within ±5%','Foreign exact','Foreign within ±1','Both exact']
ks=[rs['exact_images'],rs['within_1_images'],rs['within_tolerance_images'],fs['exact_images'],fs['within_1_images'],both_exact]
ci=[wilson(k,n) for k in ks]; vals=np.array(ks)/n*100
colors=[blue,blue,blue,orange,orange,navy]
fig,ax=plt.subplots(figsize=(13,6.2));x=np.arange(len(labels))
for i,(v,interval) in enumerate(zip(vals,ci)):
    ax.errorbar(i,v,yerr=[[v-interval[0]],[interval[1]-v]],fmt='o',ms=10,color=colors[i],ecolor=colors[i],capsize=5,lw=2)
    ax.text(i,v+4,f'{ks[i]}/{n}\n{v:.1f}%',ha='center',va='bottom',fontsize=9)
ax.set_xticks(x,labels,rotation=18,ha='right');ax.set_ylabel('Images meeting criterion (%)');ax.set_ylim(0,112);ax.set_title('Count agreement rates with 95% Wilson intervals',fontweight='bold')
ax.axhline(100,color=grey,lw=.8,ls=':');ax.grid(axis='x',visible=False)
fig.tight_layout();fig.savefig(GRAPH_DIR/'count_agreement_rates_ci.png',dpi=220,bbox_inches='tight');plt.close(fig)

# 6. Fresh rice share against the original count share.
fig,ax=plt.subplots(figsize=(15,5.8))
ax.axhline(share_actual[0],color=navy,ls='--',lw=2,label=f'Original rice share = {share_actual[0]:.2f}%')
ax.scatter(image_no,share_pred,s=13,alpha=.43,color=orange,edgecolors='none',label='Fresh proxy rice share',rasterized=True)
roll=np.convolve(share_pred,np.ones(50)/50,mode='valid')
ax.plot(image_no[24:-25],roll,color=blue,lw=1.7,label='50-image rolling mean')
ax.set_xlabel('Raw image number (numeric filename order)');ax.set_ylabel('Rice share of detected objects (%)')
ax.set_title('Fresh rice share vs original 100:10 composition',fontweight='bold');ax.legend(loc='best',ncol=3)
fig.tight_layout();fig.savefig(GRAPH_DIR/'rice_share_vs_original.png',dpi=220,bbox_inches='tight');plt.close(fig)

# Compact KPI card/table for quick review.
fig,ax=plt.subplots(figsize=(13,2.7));ax.axis('off')
cell_text=[
 ['Rice',f"{rs['mae']:.2f}",f"{rs['rmse']:.2f}",f"{rs['signed_bias_mean']:+.2f}",f"{rs['exact_images']}/{n} ({rs['exact_rate_percent']:.1f}%)",f"{rs['within_tolerance_images']}/{n} ({rs['within_tolerance_rate_percent']:.1f}%)"],
 ['Foreign matter',f"{fs['mae']:.2f}",f"{fs['rmse']:.2f}",f"{fs['signed_bias_mean']:+.2f}",f"{fs['exact_images']}/{n} ({fs['exact_rate_percent']:.1f}%)",f"{fs['within_1_images']}/{n} ({fs['within_1_rate_percent']:.1f}%)"],
 ['Both exact','—','—','—',f"{both_exact}/{n} ({100*both_exact/n:.1f}%)",'—']]
tbl=ax.table(cellText=cell_text,colLabels=['Class','MAE','RMSE','Mean bias','Exact count','Within practical tolerance'],cellLoc='center',loc='center',colWidths=[.18,.12,.12,.15,.20,.23])
tbl.auto_set_font_size(False);tbl.set_fontsize(11);tbl.scale(1,2.0)
for (r,c),cell in tbl.get_celld().items():
    if r==0:cell.set_facecolor(navy);cell.set_text_props(color='white',weight='bold')
    else:cell.set_facecolor('#eef3f7' if r%2 else 'white')
ax.set_title(f"Fresh proxy raw-set scorecard — {n} images | combined count accuracy {metrics['combined_count_accuracy_percent']:.2f}%",fontsize=15,fontweight='bold',pad=18)
fig.tight_layout();fig.savefig(GRAPH_DIR/'scorecard.png',dpi=240,bbox_inches='tight');plt.close(fig)

# Reordered graph data for numeric image order.
worst=sorted(rows,key=lambda r:(int(r['rice_abs_error'])+int(r['foreign_abs_error'])),reverse=True)[:20]
with (ROOT/'worst_20_images.csv').open('w',newline='',encoding='utf-8-sig') as f:
    fields=['image','actual_rice','predicted_rice','rice_signed_error','rice_abs_error','actual_foreign','predicted_foreign','foreign_signed_error','foreign_abs_error']
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for r in worst: w.writerow({k:r[k] for k in fields})

with (ROOT/'graph_data.csv').open('w',newline='',encoding='utf-8-sig') as f:
    fields=['image','actual_rice','predicted_rice','rice_signed_error','rice_abs_error','actual_foreign','predicted_foreign','foreign_signed_error','foreign_abs_error','original_rice_share_percent','fresh_rice_share_percent']
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for i,r in enumerate(rows):
        w.writerow({'image':r['image'],'actual_rice':actual_rice[i],'predicted_rice':pred_rice[i],'rice_signed_error':rice_err[i],'rice_abs_error':rice_abs[i],
            'actual_foreign':actual_foreign[i],'predicted_foreign':pred_foreign[i],'foreign_signed_error':foreign_err[i],'foreign_abs_error':foreign_abs[i],
            'original_rice_share_percent':round(share_actual[i],5),'fresh_rice_share_percent':round(share_pred[i],5)})

summary=f'''# Fresh proxy raw-set count evaluation

## Scope and reference

- Images: **{n}/860** from `A:\\grain\\data\\rice_count\\raw`; all files were processed successfully.
- Reference used per your instruction: **100 rice grains and 10 foreign-matter objects per image**.
- Fresh proxy: current `rice_labeler.py`, 40% linear processing scale, with its minimum-area threshold scaled to match.
- This is an **in-sample training-set count-agreement evaluation** against the fixed reference. The raw-set README describes the composition as approximate, and the folder does not contain image-specific reviewed instance masks; confirm the fixed labels before treating the numbers as ground-truth accuracy or independent validation.

## Results

| Metric | Rice (reference 100) | Foreign matter (reference 10) |
|---|---:|---:|
| Mean predicted count | {pred_rice.mean():.2f} | {pred_foreign.mean():.2f} |
| MAE [95% image-bootstrap CI] | {rs['mae']:.2f} [{rs['mae_bootstrap_95ci'][0]:.2f}, {rs['mae_bootstrap_95ci'][1]:.2f}] | {fs['mae']:.2f} [{fs['mae_bootstrap_95ci'][0]:.2f}, {fs['mae_bootstrap_95ci'][1]:.2f}] |
| RMSE | {rs['rmse']:.2f} | {fs['rmse']:.2f} |\n| 95th percentile absolute error | {rs['absolute_error_p95']:.0f} | {fs['absolute_error_p95']:.0f} |\n| Maximum absolute error | {rs['absolute_error_max']} | {fs['absolute_error_max']} |
| Mean signed bias | {rs['signed_bias_mean']:+.2f} | {fs['signed_bias_mean']:+.2f} |
| Exact count | {rs['exact_images']}/{n} ({rs['exact_rate_percent']:.1f}%) | {fs['exact_images']}/{n} ({fs['exact_rate_percent']:.1f}%) |
| Within ±1 count | {rs['within_1_images']}/{n} ({rs['within_1_rate_percent']:.1f}%) | {fs['within_1_images']}/{n} ({fs['within_1_rate_percent']:.1f}%) |
| Within practical tolerance | {rs['within_tolerance_images']}/{n} ({rs['within_tolerance_rate_percent']:.1f}%; ±5 grains) | {fs['within_tolerance_images']}/{n} ({fs['within_tolerance_rate_percent']:.1f}%; ±1 object) |
| WAPE | {rs['wape_percent']:.2f}% | {fs['wape_percent']:.2f}% |

- Both class counts exact: **{both_exact}/{n} ({100*both_exact/n:.2f}%)**.
- Combined count accuracy (count-weighted): **{metrics['combined_count_accuracy_percent']:.2f}%**; total absolute error {metrics['combined_absolute_error_total']} over {metrics['combined_actual_count_total']} reference objects.
- Mean rice-share error: **{metrics['mean_rice_share_absolute_error_percentage_points']:.3f} percentage points**; original share {metrics['rice_share_original_percent']:.2f}%, fresh mean {metrics['rice_share_fresh_mean_percent']:.2f}%.
- Runtime mean: {metrics['runtime_mean_seconds_per_image']:.2f} s/image across workers (sum of per-image elapsed times; not wall-clock throughput).

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
'''
(ROOT/'SUMMARY.md').write_text(summary,encoding='utf-8')
print(summary)
print('graphs',len(list(GRAPH_DIR.glob('*.png'))))


