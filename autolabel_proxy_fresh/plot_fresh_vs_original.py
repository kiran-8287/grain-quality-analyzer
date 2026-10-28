from pathlib import Path
import csv,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'fresh_vs_original_graphs'
OUT.mkdir(parents=True,exist_ok=True)
with (ROOT/'counts.csv').open(newline='',encoding='utf-8-sig') as f:
    rows=list(csv.DictReader(f))
if len(rows)!=20: raise ValueError(f'Expected 20 fresh results, found {len(rows)}')
labels=[]; actual_rice=[]; actual_foreign=[]; pred_rice=[]; pred_foreign=[]
replicate_counts={}
for r in rows:
    name=r['image']
    ar=int(r['actual_rice_total']); af=int(r['actual_foreign'])
    pr=int(r['predicted_rice']); pf=int(r['predicted_foreign'])
    # Images are four repeats of each filename-encoded scenario.
    scenario=(ar,af); replicate_counts[scenario]=replicate_counts.get(scenario,0)+1
    labels.append(f'{ar}/{af}\n{replicate_counts[scenario]}')
    actual_rice.append(ar); actual_foreign.append(af); pred_rice.append(pr); pred_foreign.append(pf)
actual_rice=np.array(actual_rice); actual_foreign=np.array(actual_foreign)
pred_rice=np.array(pred_rice); pred_foreign=np.array(pred_foreign)
foreign_error=np.abs(pred_foreign-actual_foreign)
actual_share=100*actual_rice/(actual_rice+actual_foreign)
pred_share=100*pred_rice/(pred_rice+pred_foreign)
x=np.arange(len(rows))

# Plot 1: only fresh proxy error against original/filename count truth.
fig,ax=plt.subplots(figsize=(15,6.4))
ax.bar(x,foreign_error,color='#ed7132',width=.58,label='Fresh autolabel proxy (absolute error)',zorder=3)
ax.axhline(0,color='black',linestyle='--',linewidth=1.8,label='Original count (ground truth; zero-error reference)',zorder=4)
ax.set_title('Absolute foreign-matter count error per image — fresh proxy vs original count',fontsize=14)
ax.set_xlabel('Test image (actual rice / foreign count; repeat)')
ax.set_ylabel('Absolute error (objects)')
ax.set_xticks(x,labels,fontsize=9)
ax.set_ylim(0,max(1.2,float(foreign_error.max())+.25))
ax.grid(axis='y',alpha=.25,zorder=0)
ax.legend(loc='upper left',frameon=True)
fig.tight_layout()
fig.savefig(OUT/'fresh_foreign_count_error_per_image.png',dpi=180,bbox_inches='tight')
plt.close(fig)

# Plot 2: original rice share vs fresh proxy share only.
fig,ax=plt.subplots(figsize=(15,6.4))
ax.plot(x,actual_share,color='black',linestyle='--',marker='o',linewidth=2.2,markersize=5,label='Original count (true rice share)',zorder=3)
ax.plot(x,pred_share,color='#ed7132',marker='o',linewidth=2.0,markersize=5,label='Fresh autolabel proxy',zorder=4)
ax.set_title('Rice percentage of detected objects — fresh proxy vs original count',fontsize=14)
ax.set_xlabel('Test image (actual rice / foreign count; repeat)')
ax.set_ylabel('Rice share (%)')
ax.set_xticks(x,labels,fontsize=9)
ax.set_ylim(min(actual_share.min(),pred_share.min())-1,max(actual_share.max(),pred_share.max())+1)
ax.grid(alpha=.25)
ax.legend(loc='best',frameon=True)
fig.tight_layout()
fig.savefig(OUT/'fresh_rice_share_vs_original.png',dpi=180,bbox_inches='tight')
plt.close(fig)

with (OUT/'fresh_vs_original_graph_data.csv').open('w',newline='',encoding='utf-8-sig') as f:
    fields=['image','actual_rice','actual_foreign','fresh_predicted_rice','fresh_predicted_foreign','fresh_foreign_absolute_error','original_rice_share_percent','fresh_rice_share_percent']
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for i,r in enumerate(rows):
        w.writerow({'image':r['image'],'actual_rice':actual_rice[i],'actual_foreign':actual_foreign[i],
            'fresh_predicted_rice':pred_rice[i],'fresh_predicted_foreign':pred_foreign[i],
            'fresh_foreign_absolute_error':foreign_error[i],
            'original_rice_share_percent':round(actual_share[i],4),'fresh_rice_share_percent':round(pred_share[i],4)})
(OUT/'README.md').write_text(
    '# Fresh proxy vs original count graphs\n\n'
    'Only the fresh autolabel proxy and filename-derived original counts are shown. The foreign-matter error plot compares the fresh proxy absolute error with a zero-error original-count reference. Rice share is count/(rice + foreign) for the original counts and fresh proxy predictions.\n\n'
    '- `fresh_foreign_count_error_per_image.png`\n'
    '- `fresh_rice_share_vs_original.png`\n'
    '- `fresh_vs_original_graph_data.csv`\n',encoding='utf-8')
print('Created:')
for p in sorted(OUT.iterdir()): print(p.name,p.stat().st_size)
print('foreign_error_max',int(foreign_error.max()),'foreign_exact',int((foreign_error==0).sum()),'/20')
print('original_share_range',round(actual_share.min(),2),round(actual_share.max(),2))
print('fresh_share_range',round(pred_share.min(),2),round(pred_share.max(),2))
