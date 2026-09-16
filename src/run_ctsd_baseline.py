"""Does a calibrated continuous-time speed-and-distance estimator remove the allocation effect?

Noonan et al. 2019 (Movement Ecology 7:35) correct the sampling-density bias in
distance travelled with CTSD: calibrate the location error, fit a continuous-time
movement model, then integrate the conditional expected SPEED rather than the
length of the conditional mean path. By Fubini, E[int ||v|| dt] = int E[||v||] dt,
so the frozen `posterior_path_m` of continuous_motion_cpu IS that estimator under
a Wiener-velocity (IOU) model with isotropic error.

The thing this project never did is CALIBRATE the error, which Noonan et al. insist
on: every earlier run assumed sigma = 3 mm. Here sigma is calibrated per camera
subset from a dense consecutive burst (measurement noise dominates process noise at
the native frame interval), q is then fitted by innovation likelihood, and the whole
(views x moments) grid is re-scored with four estimators:

  polygon      straight-line displacement sum (no correction)
  limited      frozen limited-arc interpolation
  state_mean   length of the conditional mean path (this project's strong baseline)
  ctsd         integral of conditional expected speed, calibrated sigma  <- Noonan et al.
  ctsd_fixed   same but with the old assumed sigma = 3 mm, to separate calibration

If ctsd flattens the error-vs-allocation curve, the allocation finding is an artefact
of an uncorrected estimator and the contribution collapses to "use CTSD".
"""
import json,argparse,hashlib
from pathlib import Path
from time import perf_counter
from datetime import datetime,timezone
import numpy as np
from audit_mcalib_sample import read
from prepare_mcalib_cache import load_record
from run_allocation_sweep import subset_points,polygon,SPLITS
from allocation_selfcheck_cpu import fit_state_params
from continuous_motion_cpu import estimate as state_estimate,filter_states,Q_GRID
from limited_arc_cpu import limited_arc

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data_external/mcalib_2026-09-13/verified/mocapAssistedRecord'
CAL=ROOT/'experiments/mcalib_3d_cache_2026-09-13_verified/public/calibration.json'
MOMENTS=[8,12,16,25,33,50,100,200];RULE={2:'3e0f8f0+44c4b2e',3:'216f21c1+3e0f8f0+44c4b2e'}


def ctsd(points,times,sigma):
    """Integral of conditional expected speed with a given (calibrated) error sigma."""
    return state_estimate(points,times,sigma)


def main(split,out,burst):
    cams=json.loads(CAL.read_text())['cameras'];ids=sorted(cams);records=SPLITS[split]
    subsets={2:tuple(RULE[2].split('+')),3:tuple(RULE[3].split('+')),7:tuple(ids)}
    out.mkdir(parents=True,exist_ok=False);rows=[];probe={};tick=perf_counter()
    for rec in records:
        uv={c:read(DATA/rec/f'centroidsUV{c}.pkl') for c in ids};n=len(uv[ids[0]]);t=np.arange(n)*.02+.00006969+.000996/2
        pts={k:subset_points(uv,cams,s,n)[0] for k,s in subsets.items()};sig={}
        for k in subsets:
            bp=pts[k][:burst];okb=np.all(np.isfinite(bp),axis=1);f=fit_state_params(bp[okb],t[:burst][okb])
            sig[k]=f['sigma'];probe[f'{rec}|k{k}']=dict(sigma_m=f['sigma'],at_boundary=f['sigma_at_boundary'],used=int(okb.sum()))
        print(rec,'calibrated sigma mm',{k:round(sig[k]*1000,3) for k in sig},flush=True)
        for start in range(0,3000,200):
            for k in subsets:
                for m in MOMENTS:
                    sel=start+np.rint(np.linspace(0,199,m)).astype(int);p=pts[k][sel];ok=np.all(np.isfinite(p),axis=1)
                    base=dict(record=rec,start=start,views=k,cameras='+'.join(subsets[k]),moments=m,frames=k*m,missing=int((~ok).sum()),sigma_calibrated=sig[k])
                    if ok.sum()<3 or not ok[0] or not ok[-1]:rows.append(dict(base,method='polygon',value_m=None,failure='endpoint_or_too_few'));continue
                    q=p[ok];qt=t[sel][ok]
                    rows.append(dict(base,method='polygon',value_m=polygon(q)))
                    rows.append(dict(base,method='limited',value_m=limited_arc(q,qt)))
                    cal=ctsd(q,qt,sig[k]);fix=ctsd(q,qt,.003)
                    rows.append(dict(base,method='state_mean',value_m=fix['mean_path_m']))
                    rows.append(dict(base,method='state_mean_calibrated',value_m=cal['mean_path_m'],q=cal['q']))
                    rows.append(dict(base,method='ctsd',value_m=cal['posterior_path_m'],q=cal['q']))
                    rows.append(dict(base,method='ctsd_fixed_sigma',value_m=fix['posterior_path_m'],q=fix['q']))
        print(rec,'done',f'{perf_counter()-tick:.0f}s',flush=True)
    (out/'predictions.json').write_text(json.dumps(dict(probe=probe,rows=rows)))
    (out/'seal.json').write_text(json.dumps(dict(sha256=hashlib.sha256((out/'predictions.json').read_bytes()).hexdigest(),
        wall_s=perf_counter()-tick,burst=burst,utc=datetime.now(timezone.utc).isoformat())))
    # ---- reference read only now ----
    truth={}
    for rec in records:
        tt,xyz,_,dense=load_record(rec,ids)
        for start in range(0,3000,200):
            end=start+199;mid=dense[(dense[:,1]>tt[start])&(dense[:,1]<tt[end]),2:5];ref=np.vstack([xyz[start],mid,xyz[end]])
            truth[rec,start]=float(np.linalg.norm(np.diff(ref,axis=0),axis=1).sum())
    for r in rows:r['truth_m']=truth[r['record'],r['start']];r['error_m']=None if r['value_m'] is None else r['value_m']-r['truth_m']
    (out/'scored.json').write_text(json.dumps(rows))
    methods=['polygon','limited','state_mean','state_mean_calibrated','ctsd','ctsd_fixed_sigma'];summary=[]
    for k in sorted(subsets):
        for m in MOMENTS:
            for meth in methods:
                rr=[r for r in rows if r['views']==k and r['moments']==m and r['method']==meth]
                ok=[r for r in rr if r['error_m'] is not None]
                wins=len({(r['record'],r['start']) for r in rows if r['views']==k and r['moments']==m})
                summary.append(dict(views=k,moments=m,frames=k*m,method=meth,windows=wins,completed=len(ok),
                    mae_m=float(np.mean([abs(r['error_m']) for r in ok])) if ok else None,
                    bias_m=float(np.mean([r['error_m'] for r in ok])) if ok else None,
                    p95_m=float(np.quantile([abs(r['error_m']) for r in ok],.95)) if ok else None))
    (out/'summary.json').write_text(json.dumps(dict(probe=probe,summary=summary),indent=1))
    for k in sorted(subsets):
        print(f'\n== k={k} ==  frames | '+' | '.join(f'{x:>22s}' for x in methods))
        for m in MOMENTS:
            cells=[]
            for meth in methods:
                s=[x for x in summary if x['views']==k and x['moments']==m and x['method']==meth][0]
                cells.append('   n/a' if s['mae_m'] is None else f"{s['mae_m']*1000:8.2f}({s['bias_m']*1000:+7.2f})")
            print(f'  m={m:3d} {k*m:5d} | '+' | '.join(f'{c:>22s}' for c in cells))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='dev');p.add_argument('--out',type=Path,required=True);p.add_argument('--burst',type=int,default=25)
    a=p.parse_args();main(a.split,a.out,a.burst)
