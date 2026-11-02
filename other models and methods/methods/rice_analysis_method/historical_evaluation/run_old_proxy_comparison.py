from pathlib import Path
import csv, re, sys, tempfile
import cv2
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
SOURCE = Path(r'A:\grain\data\rice_count\test')
OLD_CODE = Path(r'A:\grain\rice_count_pipeline\methods\autolabel_proxy')
sys.path.insert(0, str(OLD_CODE))
from autolabel import label_image

LONG_SIDE=1536
MIN_RICE_AREA=200
MIN_FOREIGN_AREA=700
old_out=OUT/'old_proxy_rerun'
overlays=old_out/'overlays'
overlays.mkdir(parents=True,exist_ok=True)
temp_root=OUT/'old_proxy_temp'
temp_root.mkdir(parents=True,exist_ok=True)
rows=[]
files=sorted(p for p in SOURCE.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'})
for i,path in enumerate(files,1):
    image=cv2.imread(str(path))
    if image is None: raise ValueError(f'Cannot read {path}')
    h,w=image.shape[:2]
    scale=min(1.0,LONG_SIDE/max(h,w))
    vis=cv2.resize(image,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA) if scale<1 else image.copy()
    counts={'rice':0,'foreign':0}
    def draw_instance(mask,cls):
        counts[cls]+=1
        color=(255,0,255) if cls=='rice' else (0,0,255)
        contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis,contours,-1,color,2,cv2.LINE_AA)
    with tempfile.TemporaryDirectory(prefix='img_',dir=temp_root) as td:
        tmp=Path(td)/path.name
        if not cv2.imwrite(str(tmp),vis): raise OSError(f'Could not write temp {tmp}')
        with threadpool_limits(limits=1):
            _,_,ref=label_image(tmp,min_area=MIN_RICE_AREA,min_foreign_area=MIN_FOREIGN_AREA,
                                instance_callback=draw_instance,max_split_clusters=50)
    truth=re.search(r'(\d+)\s*whole.*?(\d+)\s*broken.*?(\d+)\s*stones',path.name,re.I)
    if not truth: raise ValueError(f'Cannot parse truth from {path.name}')
    whole,broken,stones=map(int,truth.groups()); actual_rice=whole+broken
    pr,pf=counts['rice'],counts['foreign']
    cv2.putText(vis,f'OLD PROXY rice={pr}/{actual_rice}; foreign={pf}/{stones}',(16,38),cv2.FONT_HERSHEY_SIMPLEX,.75,(0,0,0),2,cv2.LINE_AA)
    op=overlays/(path.stem+'_overlay.jpg')
    if not cv2.imwrite(str(op),vis): raise OSError(f'Could not save {op}')
    rows.append({'image':path.name,'actual_whole_rice':whole,'actual_broken_rice':broken,'actual_rice_total':actual_rice,
                 'predicted_rice':pr,'actual_foreign':stones,'predicted_foreign':pf,
                 'rice_abs_error':abs(pr-actual_rice),'foreign_abs_error':abs(pf-stones),
                 'rice_exact':pr==actual_rice,'foreign_exact':pf==stones,
                 'both_classes_exact':pr==actual_rice and pf==stones,'single_grain_reference_area_px':float(ref),
                 'overlay':str(op.relative_to(OUT))})
    print(f'{i}/{len(files)} {path.name}: old rice={pr}/{actual_rice}, foreign={pf}/{stones}',flush=True)
with (old_out/'counts.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(f'OLD_PROXY_RUN_COMPLETE images={len(rows)} output={old_out}',flush=True)
