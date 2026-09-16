"""Observation allocation sweep: views-per-moment (k) versus moments (m) at cost = k*m camera frames.

Uniform moments over each 3.98 s window; camera subsets from calibration; plain DLT
triangulation on the subset. Path via polygon, limited_arc and Wiener-velocity state mean.
Predictions for a split are written and sealed before the reference is read.
"""
import json,argparse,hashlib
from itertools import combinations
from pathlib import Path
from time import perf_counter
import numpy as np
from audit_mcalib_sample import read
from prepare_mcalib_cache import load_record
from limited_arc_cpu import limited_arc
from continuous_motion_cpu import estimate as state_estimate
from mcalib_cache_observer import triangulate
from robust_triangulation_cpu import reprojection_errors

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data_external/mcalib_2026-09-13/verified/mocapAssistedRecord'
CAL=ROOT/'experiments/mcalib_3d_cache_2026-09-13_verified/public/calibration.json'
SPLITS={'dev':[f'record{i}' for i in range(19,24)],'confirm':[f'record{i}' for i in range(15,19)]}


def polygon(p):return float(np.linalg.norm(np.diff(np.asarray(p),axis=0),axis=1).sum())


def subset_points(uv,cams,subset,n):
    pts=np.full((n,3),np.nan);maxerr=np.full(n,np.nan)
    for i in range(n):
        obs={c:uv[c][i] for c in subset if uv[c][i] is not None}
        if len(obs)<2:continue
        try:p=triangulate(obs,cams)
        except ValueError:continue
        pts[i]=p
        if len(obs)>=3:maxerr[i]=max(reprojection_errors(p,obs,cams).values())
    return pts,maxerr


def main(split,out,subsets_spec,moments,sizes):
    cams=json.loads(CAL.read_text())['cameras'];ids=sorted(cams);records=SPLITS[split]
    subsets=[]
    for k in sizes:subsets.extend(combinations(ids,k))
    if subsets_spec:subsets=[tuple(s.split('+')) for s in subsets_spec.split(',')]
    rows=[];tick=perf_counter();tri_count=0
    for rec in records:
        uv={c:read(DATA/rec/f'centroidsUV{c}.pkl') for c in ids};n=len(uv[ids[0]]);t=np.arange(n)*.02+.00006969+.000996/2
        for sub in subsets:
            pts,maxerr=subset_points(uv,cams,sub,n);tri_count+=n
            for start in range(0,3000,200):
                for m in moments:
                    sel=start+np.rint(np.linspace(0,199,m)).astype(int);p=pts[sel];ok=np.all(np.isfinite(p),axis=1)
                    base=dict(record=rec,start=start,views=len(sub),cameras='+'.join(sub),moments=m,frames=len(sub)*m,missing=int((~ok).sum()),
                              max_reproj_px=float(np.nanmax(maxerr[sel])) if len(sub)>=3 and np.any(np.isfinite(maxerr[sel])) else None)
                    if ok.sum()<2 or not ok[0] or not ok[-1]:
                        rows.append(dict(base,method='polygon',value_m=None,failure='endpoint_or_too_few'));continue
                    q=p[ok];qt=t[sel][ok]
                    rows.append(dict(base,method='polygon',value_m=polygon(q)))
                    rows.append(dict(base,method='limited',value_m=limited_arc(q,qt)))
                    if m<=50:rows.append(dict(base,method='state_mean',value_m=state_estimate(q,qt)['mean_path_m']))
        print(rec,'done',f'{perf_counter()-tick:.1f}s',flush=True)
    predict_s=perf_counter()-tick;out.mkdir(parents=True,exist_ok=False)
    (out/'predictions.json').write_text(json.dumps(rows));seal=hashlib.sha256((out/'predictions.json').read_bytes()).hexdigest()
    (out/'prediction_seal.json').write_text(json.dumps(dict(sha256=seal,predict_wall_s=predict_s,triangulations=tri_count,records=records,subsets=['+'.join(s) for s in subsets],moments=moments)))
    # ---- reference read only after sealing ----
    truth={}
    for rec in records:
        t,xyz,_,dense=load_record(rec,ids)
        for start in range(0,3000,200):
            end=start+199;middle=dense[(dense[:,1]>t[start])&(dense[:,1]<t[end]),2:5];ref=np.vstack([xyz[start],middle,xyz[end]])
            truth[rec,start]=float(np.linalg.norm(np.diff(ref,axis=0),axis=1).sum())
    for r in rows:
        r['truth_m']=truth[r['record'],r['start']];r['error_m']=None if r['value_m'] is None else r['value_m']-r['truth_m']
    (out/'scored.json').write_text(json.dumps(rows))
    summary=[]
    for key in sorted({(r['cameras'],r['moments'],r['method']) for r in rows}):
        rr=[r for r in rows if (r['cameras'],r['moments'],r['method'])==key];ok=[r for r in rr if r['error_m'] is not None]
        summary.append(dict(cameras=key[0],views=rr[0]['views'],moments=key[1],frames=rr[0]['frames'],method=key[2],n=len(rr),failed=len(rr)-len(ok),
            missing_moments=int(sum(r['missing'] for r in rr)),mae_m=float(np.mean([abs(r['error_m']) for r in ok])) if ok else None,
            bias_m=float(np.mean([r['error_m'] for r in ok])) if ok else None,p95_m=float(np.quantile([abs(r['error_m']) for r in ok],.95)) if ok else None,
            per_record_mae={rec:float(np.mean([abs(r['error_m']) for r in ok if r['record']==rec])) for rec in records if any(r['record']==rec for r in ok)}))
    (out/'summary.json').write_text(json.dumps(dict(split=split,seal=seal,predict_wall_s=predict_s,summary=summary),indent=1))
    print('rows',len(rows),'wall',round(predict_s,1))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='dev');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--subsets',default='');p.add_argument('--moments',default='8,12,16,25,33,50,100,200');p.add_argument('--sizes',default='2,3,7')
    a=p.parse_args();main(a.split,a.out,a.subsets,[int(x) for x in a.moments.split(',')],[int(x) for x in a.sizes.split(',')])
