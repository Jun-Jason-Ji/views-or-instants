"""Dense few-view allocation with view-consensus checks where possible and a temporal gate elsewhere.

Triangulation per moment: k=3 with 3 visible -> three-view checks (reject inconsistent);
k=3 with 2 visible -> plain DLT (kept); k=2 -> plain DLT; k=7 -> frozen robust rule.
Then, per query, the innovation gate (gated_state_cpu.gate) drops temporally inconsistent
moments before polygon / limited / state-mean estimation. If the last moment is gated
the query fails ('endpoint_gated'); the first two moments are never gated.
Designed on record19-23; confirmation records 15-18 are already exposed (follow-up, not fresh).
"""
import json,argparse,hashlib
from itertools import combinations
from pathlib import Path
from time import perf_counter
from datetime import datetime,timezone
import numpy as np
from audit_mcalib_sample import read
from prepare_mcalib_cache import load_record
from limited_arc_cpu import limited_arc
from continuous_motion_cpu import estimate as state_estimate
from three_view_cpu import three_view_triangulate
from robust_triangulation_cpu import robust_triangulate
from mcalib_cache_observer import triangulate
from gated_state_cpu import gate
from run_allocation_sweep import polygon,SPLITS
from score_allocation_v2 import aggregate

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data_external/mcalib_2026-09-13/verified/mocapAssistedRecord'
CAL=ROOT/'experiments/mcalib_3d_cache_2026-09-13_verified/public/calibration.json'
MOMENTS=[8,12,16,19,25,28,33,50,100,200]


def tri(obs,cams,k):
    valid={c:v for c,v in obs.items() if v is not None}
    if k==7:return robust_triangulate(obs,cams)[0]
    if k==3 and len(valid)==3:return three_view_triangulate(obs,cams)[0]
    if len(valid)<2:raise ValueError('insufficient_visible_views')
    return triangulate(valid,cams)


def subset_points(uv,cams,subset,n):
    pts=np.full((n,3),np.nan);reasons={}
    for i in range(n):
        try:pts[i]=tri({c:uv[c][i] for c in subset},cams,len(subset))
        except ValueError as e:reasons[str(e)]=reasons.get(str(e),0)+1
    return pts,reasons


def main(split,out,sizes,nis):
    cams=json.loads(CAL.read_text())['cameras'];ids=sorted(cams);records=SPLITS[split];out.mkdir(parents=True,exist_ok=False)
    subsets=[]
    for k in sizes:subsets.extend([tuple(ids)] if k==7 else list(combinations(ids,k)))
    rows=[];reject={};tick=perf_counter();gated_total=0
    for rec in records:
        uv={c:read(DATA/rec/f'centroidsUV{c}.pkl') for c in ids};n=len(uv[ids[0]]);t=np.arange(n)*.02+.00006969+.000996/2
        for sub in subsets:
            pts,reasons=subset_points(uv,cams,sub,n);reject[f"{rec}|{'+'.join(sub)}"]=reasons
            for start in range(0,3000,200):
                for m in MOMENTS:
                    sel=start+np.rint(np.linspace(0,199,m)).astype(int);p=pts[sel];ok=np.all(np.isfinite(p),axis=1)
                    base=dict(record=rec,start=start,views=len(sub),cameras='+'.join(sub),moments=m,frames=len(sub)*m,missing=int((~ok).sum()),gate_nis=nis)
                    if ok.sum()<2 or not ok[0] or not ok[-1]:rows.append(dict(base,method='polygon',value_m=None,failure='endpoint_or_too_few'));continue
                    q=p[ok];qt=t[sel][ok]
                    if len(q)>=3:
                        acc,log=gate(q,qt);gated_total+=len(log)
                        if acc[-1]!=len(q)-1:rows.append(dict(base,method='polygon',value_m=None,failure='endpoint_gated',gated=len(log)));continue
                        q=q[acc];qt=qt[acc];base['gated']=len(log)
                    rows.append(dict(base,method='polygon',value_m=polygon(q)));rows.append(dict(base,method='limited',value_m=limited_arc(q,qt)))
                    if m<=50:rows.append(dict(base,method='state_mean',value_m=state_estimate(q,qt)['mean_path_m']))
        print(rec,'predicted',f'{perf_counter()-tick:.0f}s',flush=True)
    (out/'predictions.json').write_text(json.dumps(rows));seal=hashlib.sha256((out/'predictions.json').read_bytes()).hexdigest()
    (out/'prediction_seal.json').write_text(json.dumps(dict(sha256=seal,rows=len(rows),predict_wall_s=perf_counter()-tick,utc=datetime.now(timezone.utc).isoformat(),gated_observations=gated_total,rejections=reject),indent=1))
    truth={}
    for rec in records:
        t,xyz,_,dense=load_record(rec,ids)
        for start in range(0,3000,200):
            end=start+199;middle=dense[(dense[:,1]>t[start])&(dense[:,1]<t[end]),2:5];ref=np.vstack([xyz[start],middle,xyz[end]]);truth[rec,start]=float(np.linalg.norm(np.diff(ref,axis=0),axis=1).sum())
    for r in rows:r['truth_m']=truth[r['record'],r['start']];r['error_m']=None if r['value_m'] is None else r['value_m']-r['truth_m']
    (out/'scored.json').write_text(json.dumps(rows));summary=aggregate(rows,records)
    (out/'summary_v2.json').write_text(json.dumps(dict(split=split,gate=True,summary=summary),indent=1))
    off_path=ROOT/('experiments/allocation_sweep_dev_v1_2026-09-15' if split=='dev' else 'experiments/allocation_confirm_v1_2026-09-15')/'summary_v2.json'
    off={(s['cameras'],s['moments'],s['method']):s for s in json.load(open(off_path))['summary']};cmp=[]
    for s in summary:
        o=off.get((s['cameras'],s['moments'],s['method']))
        if o and o['mae_m'] is not None and s['mae_m'] is not None:
            cmp.append(dict(cameras=s['cameras'],views=s['views'],moments=s['moments'],method=s['method'],frames=s['frames'],mae_off=o['mae_m'],mae_on=s['mae_m'],failed_off=o['failed'],failed_on=s['failed'],p95_off=o['p95_m'],p95_on=s['p95_m']))
    (out/'gate_on_off_comparison.json').write_text(json.dumps(cmp,indent=1))
    print('rows',len(rows),'gated observations',gated_total,'wall',round(perf_counter()-tick,1))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='dev');p.add_argument('--out',type=Path,required=True);p.add_argument('--sizes',default='2,3,7');p.add_argument('--nis',type=float,default=16.27)
    a=p.parse_args();main(a.split,a.out,[int(x) for x in a.sizes.split(',')],a.nis)
