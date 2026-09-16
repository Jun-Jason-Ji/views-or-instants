"""Development screen: dense single-camera bearings + sparse 3D anchors on MCalib record19-23.

Development only (records already used for path experiments). All predictions
are computed before any reference is read for scoring in this process.
"""
import json,argparse,hashlib
from pathlib import Path
from time import perf_counter
import numpy as np
from audit_mcalib_sample import read
from prepare_mcalib_cache import load_record
from limited_arc_cpu import limited_arc
from continuous_motion_cpu import estimate as state_estimate
from mcalib_cache_observer import triangulate
from bearing_fusion_cpu import ray_depth_reconstruct,bearing_smoother,depth_of,polygon,ray_world

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data_external/mcalib_2026-09-13/verified/mocapAssistedRecord'
CAL=ROOT/'experiments/mcalib_3d_cache_2026-09-13_verified/public/calibration.json'
CACHE={'dev':ROOT/'experiments/three_view_pipeline_2026-09-13/new_predictions.jsonl','confirm':ROOT/'experiments/robust_triangulation_2026-09-13/new_predictions.jsonl'}


def load_points(path,records):
    out={}
    for s in path.read_text().splitlines():
        r=json.loads(s)
        if r['record'] in records:out[r['record'],r['frame']]=r['robust']['point']
    return out


def preview_camera(cams,uv,ids,anchor_pts,anchor_ids):
    """Rule fixed before data: full detections on the grid, then smallest median anchor depth."""
    best=None
    for cid in sorted(cams):
        if any(uv[cid][j] is None for j in ids):continue
        z=np.median([depth_of(p,cams[cid]) for p in anchor_pts])
        if best is None or z<best[0]:best=(z,cid)
    return best


def window_predictions(record,start,cams,uv,t,points,budgets,grid_n,noise_px,rng):
    end=start+199;ids33=start+np.rint(np.linspace(0,199,grid_n)).astype(int);rows=[]
    for b in budgets:
        aid=start+np.rint(np.linspace(0,199,b)).astype(int);apts=[points.get((record,int(i))) for i in aid]
        if any(p is None for p in apts):rows.append(dict(record=record,start=start,budget=b,method='anchor_failure',value_m=None));continue
        apts=np.array(apts);at=t[aid]
        base=dict(record=record,start=start,budget=b,anchor_frames=7*b)
        rows.append(dict(base,method='anchors_polygon',value_m=polygon(apts),frames=7*b))
        rows.append(dict(base,method='anchors_limited',value_m=limited_arc(apts,at),frames=7*b))
        rows.append(dict(base,method='anchors_state_mean',value_m=state_estimate(apts,at)['mean_path_m'],frames=7*b))
        choice=preview_camera(cams,uv,ids33,apts,aid)
        if choice is None:rows.append(dict(base,method='no_full_preview_camera',value_m=None));continue
        zmed,cid=choice;dense_ids=np.unique(np.r_[ids33,aid]);dt=t[dense_ids]
        cost=7*b+len(dense_ids)-b  # single-camera preview frames not already in anchors
        for level in noise_px:
            buv=[np.asarray(uv[cid][j],float)+(rng.normal(0,level,2) if level>0 else 0) for j in dense_ids]
            tag=f'_px{level:g}' if level>0 else ''
            for method,space in [('linear','z'),('pchip','z'),('pchip','log'),('cubic','z')]:
                if level>0 and (method,space)!=('pchip','z'):continue
                try:
                    pts,_=ray_depth_reconstruct(dt,buv,at,apts,cams[cid],method,space)
                    rows.append(dict(base,method=f'rd_{method}_{space}_polygon{tag}',value_m=polygon(pts),frames=cost,camera=cid))
                    rows.append(dict(base,method=f'rd_{method}_{space}_limited{tag}',value_m=limited_arc(pts,dt),frames=cost,camera=cid))
                except ValueError as e:
                    rows.append(dict(base,method=f'rd_{method}_{space}_limited{tag}',value_m=None,failure=str(e),frames=cost,camera=cid))
            try:
                tick=perf_counter();ks=bearing_smoother(at,apts,dt,buv,cams[cid],sigma_px=max(level,1.));ms=(perf_counter()-tick)*1000
                rows.append(dict(base,method=f'ks_mean{tag}',value_m=ks['mean_path_m'],frames=cost,camera=cid,q=ks['q'],ms=ms))
                rows.append(dict(base,method=f'ks_polygon{tag}',value_m=ks['polygon_m'],frames=cost,camera=cid,q=ks['q']))
                rows.append(dict(base,method=f'ks_limited{tag}',value_m=ks['limited_m'],frames=cost,camera=cid,q=ks['q']))
            except (ValueError,np.linalg.LinAlgError) as e:
                rows.append(dict(base,method=f'ks_mean{tag}',value_m=None,failure=str(e),frames=cost,camera=cid))
        # per-camera sensitivity for the main variant (no noise)
        for cid2 in sorted(cams):
            if any(uv[cid2][j] is None for j in dense_ids):continue
            buv=[np.asarray(uv[cid2][j],float) for j in dense_ids]
            try:
                pts,_=ray_depth_reconstruct(dt,buv,at,apts,cams[cid2],'pchip','z')
                rows.append(dict(base,method='rd_pchip_z_limited_cam',value_m=limited_arc(pts,dt),frames=cost,camera=cid2,median_depth=float(np.median([depth_of(p,cams[cid2]) for p in apts]))))
            except ValueError:pass
    # controls independent of budget
    base=dict(record=record,start=start,budget=None)
    full=[points.get((record,int(i))) for i in ids33]
    if all(p is not None for p in full):
        full=np.array(full);rows.append(dict(base,method='full33_polygon',value_m=polygon(full),frames=7*grid_n));rows.append(dict(base,method='full33_limited',value_m=limited_arc(full,t[ids33]),frames=7*grid_n))
    for b in (12,13):
        aid=start+np.rint(np.linspace(0,199,b)).astype(int);apts=[points.get((record,int(i))) for i in aid]
        if all(p is not None for p in apts):
            apts=np.array(apts);rows.append(dict(base,budget=b,method='anchors_limited',value_m=limited_arc(apts,t[aid]),frames=7*b));rows.append(dict(base,budget=b,method='anchors_state_mean',value_m=state_estimate(apts,t[aid])['mean_path_m'],frames=7*b))
    # two-camera dense DLT control at the 33 grid: the two closest cameras by median anchor depth (budget-8 anchors)
    aid=start+np.rint(np.linspace(0,199,8)).astype(int);apts=[points.get((record,int(i))) for i in aid]
    if all(p is not None for p in apts):
        order=[]
        for cid in sorted(cams):
            if any(uv[cid][j] is None for j in ids33):continue
            order.append((np.median([depth_of(p,cams[cid]) for p in apts]),cid))
        order=sorted(order)
        if len(order)>=2:
            pair=[order[0][1],order[1][1]]
            try:
                pts=np.array([triangulate({c:uv[c][j] for c in pair},cams) for j in ids33])
                rows.append(dict(base,method='two_cam33_polygon',value_m=polygon(pts),frames=2*grid_n,camera='+'.join(pair)));rows.append(dict(base,method='two_cam33_limited',value_m=limited_arc(pts,t[ids33]),frames=2*grid_n,camera='+'.join(pair)))
            except ValueError as e:rows.append(dict(base,method='two_cam33_limited',value_m=None,failure=str(e),frames=2*grid_n))
    return rows


def main(split,out,budgets,noise_px,seed):
    cams=json.loads(CAL.read_text())['cameras'];records=[f'record{i}' for i in (range(19,24) if split=='dev' else range(15,19))]
    points=load_points(CACHE[split],set(records));rng=np.random.default_rng(seed);rows=[];tick=perf_counter()
    raw={}
    for rec in records:
        uv={c:read(DATA/rec/f'centroidsUV{c}.pkl') for c in cams};n=len(uv[list(cams)[0]]);t=np.arange(n)*.02+.00006969+.000996/2
        raw[rec]=(t,uv)
        for start in range(0,3000,200):rows.extend(window_predictions(rec,start,cams,uv,t,points,budgets,33,noise_px,rng))
        print(rec,'done',flush=True)
    predict_s=perf_counter()-tick;out.mkdir(parents=True,exist_ok=False)
    (out/'predictions.json').write_text(json.dumps(rows))
    seal=hashlib.sha256((out/'predictions.json').read_bytes()).hexdigest()
    # ---- reference is read only now ----
    truth={}
    for rec in records:
        t,xyz,_,dense=load_record(rec,list(cams))
        for start in range(0,3000,200):
            end=start+199;middle=dense[(dense[:,1]>t[start])&(dense[:,1]<t[end]),2:5];ref=np.vstack([xyz[start],middle,xyz[end]])
            truth[rec,start]=float(np.linalg.norm(np.diff(ref,axis=0),axis=1).sum())
    for r in rows:
        r['truth_m']=truth[r['record'],r['start']];r['error_m']=None if r['value_m'] is None else r['value_m']-r['truth_m']
    (out/'scored.json').write_text(json.dumps(rows))
    summary={}
    for key in sorted({(r['method'],r['budget']) for r in rows}):
        rr=[r for r in rows if (r['method'],r['budget'])==key]
        ok=[r for r in rr if r['error_m'] is not None]
        per_rec={rec:float(np.mean([abs(r['error_m']) for r in ok if r['record']==rec])) for rec in records if any(r['record']==rec for r in ok)}
        summary[f'{key[0]}|B{key[1]}']=dict(n=len(rr),failed=len(rr)-len(ok),mae_m=float(np.mean([abs(r['error_m']) for r in ok])) if ok else None,
            bias_m=float(np.mean([r['error_m'] for r in ok])) if ok else None,p95_m=float(np.quantile([abs(r['error_m']) for r in ok],.95)) if ok else None,
            frames=sorted({r.get('frames') for r in rr}),per_record_mae=per_rec)
    (out/'summary.json').write_text(json.dumps(dict(split=split,records=records,predict_wall_s=predict_s,prediction_seal=seal,summary=summary),indent=2))
    for k,v in summary.items():
        if v['mae_m'] is not None:print(f"{k:42s} n={v['n']:3d} fail={v['failed']} frames={v['frames']} MAE={v['mae_m']:.6f} bias={v['bias_m']:+.6f} p95={v['p95_m']:.6f}")


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='dev');p.add_argument('--out',type=Path,required=True);p.add_argument('--budgets',default='4,8,16');p.add_argument('--noise',default='0,2,5');p.add_argument('--seed',type=int,default=0)
    a=p.parse_args();main(a.split,a.out,[int(x) for x in a.budgets.split(',')],[float(x) for x in a.noise.split(',')],a.seed)
