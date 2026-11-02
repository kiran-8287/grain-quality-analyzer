from pathlib import Path
import json, re, sys, tempfile
import cv2
from threadpoolctl import threadpool_limits
OUT=Path(__file__).resolve().parent
OLD_CODE=Path(r'A:\grain\rice_count_pipeline\methods\autolabel_proxy')
sys.path.insert(0,str(OLD_CODE))
from autolabel import label_image
path=Path(sys.argv[1])
image=cv2.imread(str(path))
if image is None: raise ValueError(f'Cannot read {path}')
h,w=image.shape[:2]; scale=min(1.0,1536/max(h,w))
vis=cv2.resize(image,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA) if scale<1 else image.copy()
counts={'rice':0,'foreign':0}
def draw_instance(mask,cls):
    counts[cls]+=1
    color=(255,0,255) if cls=='rice' else (0,0,255)
    contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis,contours,-1,color,2,cv2.LINE_AA)
old_out=OUT/'old_proxy_rerun'; tmp_root=OUT/'old_proxy_temp'; tmp_root.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory(prefix='img_',dir=tmp_root) as td:
    tmp=Path(td)/path.name
    if not cv2.imwrite(str(tmp),vis): raise OSError(f'Could not write temp {tmp}')
    with threadpool_limits(limits=1):
        _,_,ref=label_image(tmp,min_area=200,min_foreign_area=700,instance_callback=draw_instance,max_split_clusters=50)
m=re.search(r'(\d+)\s*whole.*?(\d+)\s*broken.*?(\d+)\s*stones',path.name,re.I)
if not m: raise ValueError(f'Cannot parse truth in {path.name}')
whole,broken,stones=map(int,m.groups()); ar=whole+broken; pr,pf=counts['rice'],counts['foreign']
cv2.putText(vis,f'OLD PROXY rice={pr}/{ar}; foreign={pf}/{stones}',(16,38),cv2.FONT_HERSHEY_SIMPLEX,.75,(0,0,0),2,cv2.LINE_AA)
overlays=old_out/'overlays'; overlays.mkdir(parents=True,exist_ok=True)
op=overlays/(path.stem+'_overlay.jpg')
if not cv2.imwrite(str(op),vis): raise OSError(f'Could not save {op}')
row={'image':path.name,'actual_whole_rice':whole,'actual_broken_rice':broken,'actual_rice_total':ar,'predicted_rice':pr,'actual_foreign':stones,'predicted_foreign':pf,'rice_abs_error':abs(pr-ar),'foreign_abs_error':abs(pf-stones),'rice_exact':pr==ar,'foreign_exact':pf==stones,'both_classes_exact':pr==ar and pf==stones,'single_grain_reference_area_px':float(ref),'overlay':str(op.relative_to(OUT))}
per=old_out/'per_image';per.mkdir(parents=True,exist_ok=True)
(per/(path.stem+'.json')).write_text(json.dumps(row),encoding='utf-8')
print(f'{path.name}: old rice={pr}/{ar}, foreign={pf}/{stones}',flush=True)
