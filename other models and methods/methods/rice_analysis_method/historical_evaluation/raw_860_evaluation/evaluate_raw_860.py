from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse, csv, os, re, sys, time

ROOT=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent
SOURCE=Path(r'A:\grain\data\rice_count\raw')
sys.path.insert(0,str(ROOT))
import rice_labeler

FIELDS=['image','actual_rice','predicted_rice','rice_signed_error','rice_abs_error','rice_exact',
        'rice_within_1','actual_foreign','predicted_foreign','foreign_signed_error','foreign_abs_error',
        'foreign_exact','foreign_within_1','both_exact','seconds','reference_grain_area_px']

def natural_key(p):
    try:return (0,int(p.stem))
    except ValueError:return (1,p.name.lower())

def evaluate_one(path_string):
    import cv2
    import rice_labeler as labeler
    from time import perf_counter
    cv2.setNumThreads(1)
    path=Path(path_string)
    counts={'rice':0,'foreign':0}
    def count_only(mask,cls): counts[cls]+=1
    start=perf_counter()
    _,_,ref=labeler.label_image(path,instance_callback=count_only,seed=0)
    duration=perf_counter()-start
    actual_rice,actual_foreign=100,10
    pr,pf=counts['rice'],counts['foreign']
    return {'image':path.name,'actual_rice':actual_rice,'predicted_rice':pr,
        'rice_signed_error':pr-actual_rice,'rice_abs_error':abs(pr-actual_rice),
        'rice_exact':pr==actual_rice,'rice_within_1':abs(pr-actual_rice)<=1,
        'actual_foreign':actual_foreign,'predicted_foreign':pf,
        'foreign_signed_error':pf-actual_foreign,'foreign_abs_error':abs(pf-actual_foreign),
        'foreign_exact':pf==actual_foreign,'foreign_within_1':abs(pf-actual_foreign)<=1,
        'both_exact':pr==actual_rice and pf==actual_foreign,'seconds':round(duration,3),
        'reference_grain_area_px':round(float(ref),2)}

def main():
    ap=argparse.ArgumentParser(description='Count-evaluate the fresh proxy on raw photos using nominal 100 rice + 10 foreign references.')
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--max-tasks-per-worker',type=int,default=25)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    paths=sorted((p for p in SOURCE.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'}),key=natural_key)
    if not paths: raise SystemExit(f'No images in {SOURCE}')
    counts_path=OUT/'counts.csv'; errors_path=OUT/'failed_images.csv'
    existing={}
    if args.resume and counts_path.exists():
        with counts_path.open(newline='',encoding='utf-8-sig') as f:
            existing={r['image']:r for r in csv.DictReader(f) if r.get('image')}
    todo=[p for p in paths if p.name not in existing]
    started=time.perf_counter(); done=len(existing); failures=[]
    mode='a' if existing and counts_path.exists() else 'w'
    with counts_path.open(mode,newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        if mode=='w':writer.writeheader();f.flush()
        with ProcessPoolExecutor(max_workers=max(1,args.workers),max_tasks_per_child=max(1,args.max_tasks_per_worker)) as pool:
            future_paths={pool.submit(evaluate_one,str(p)):p for p in todo}
            for fut in as_completed(future_paths):
                path=future_paths[fut]
                try:
                    row=fut.result();writer.writerow(row);done+=1
                except Exception as e:
                    failures.append({'image':path.name,'error':repr(e)})
                f.flush()
                if done%10==0 or done==len(paths):
                    elapsed=time.perf_counter()-started
                    rate=(done-len(existing))/max(elapsed,1e-9)
                    remain=(len(paths)-done)/max(rate,1e-9)
                    print(f'{done}/{len(paths)} images complete; {len(failures)} failed in this run; elapsed {elapsed/60:.1f} min; ETA {remain/60:.1f} min',flush=True)
    with errors_path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['image','error']);writer.writeheader();writer.writerows(failures)
    print(f'Finished: raw files={len(paths)}, recorded successes={done}, failures={len(failures)}; CSV={counts_path}',flush=True)
    if len(existing)+len(todo)!=len(paths): raise RuntimeError('Internal image accounting mismatch')

if __name__=='__main__':
    main()


