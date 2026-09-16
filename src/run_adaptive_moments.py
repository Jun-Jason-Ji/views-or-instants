"""Does closed-form non-uniform moment allocation beat uniform at equal cost?

Protocol per window and camera subset, for a total budget of m moments:
  uniform  : m uniform moments.
  adaptive : m//2 uniform probe moments, fit the state model, compute the error
             density rho, then place the remaining moments at equal increments of
             the cumulative rho while KEEPING the probe moments. Total distinct
             moments = m, so the camera-frame cost is identical.
  oracle_nonuniform (diagnostic only): same count, breakpoints chosen using the
             reference trajectory; an upper bound on what any non-uniform rule
             could reach here, never a deployable method.

Sigma is calibrated from a consecutive burst, as in run_ctsd_baseline.
Reference is read only for scoring (and for the explicitly flagged oracle).
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
from adaptive_moments_cpu import error_density,allocate,fit_q
from continuous_motion_cpu import estimate as state_estimate
from limited_arc_cpu import limited_arc

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data_external/mcalib_2026-09-13/verified/mocapAssistedRecord'
CAL=ROOT/'experiments/mcalib_3d_cache_2026-09-13_verified/public/calibration.json'
RULE={2:'3e0f8f0+44c4b2e',3:'216f21c1+3e0f8f0+44c4b2e'};BUDGETS=[8,12,16,25,33]


def estimates(p,t,sigma):
    st=state_estimate(p,t,sigma)
    return dict(state_mean=st['mean_path_m'],limited=limited_arc(p,t),polygon=polygon(p))


def main(split,out,burst,oracle):
    cams=json.loads(CAL.read_text())['cameras'];ids=sorted(cams);records=SPLITS[split]
    subsets={2:tuple(RULE[2].split('+')),3:tuple(RULE[3].split('+')),7:tuple(ids)}
    out.mkdir(parents=True,exist_ok=False);rows=[];probe={};tick=perf_counter()
    truth={}
    for rec in records:
        uv={c:read(DATA/rec/f'centroidsUV{c}.pkl') for c in ids};n=len(uv[ids[0]]);t=np.arange(n)*.02+.00006969+.000996/2
        pts={k:subset_points(uv,cams,s,n)[0] for k,s in subsets.items()};sig={}
        for k in subsets:
            bp=pts[k][:burst];okb=np.all(np.isfinite(bp),axis=1);f=fit_state_params(bp[okb],t[:burst][okb])
            sig[k]=f['sigma'];probe[f'{rec}|k{k}']=dict(sigma_m=f['sigma'],burst_used=int(okb.sum()))
        ref_t,ref_x,_,dense=load_record(rec,ids) if oracle else (None,None,None,None)
        for start in range(0,3000,200):
            win=np.arange(start,start+200);wt=t[win]
            for k in subsets:
                wp=pts[k][win];ok=np.all(np.isfinite(wp),axis=1)
                for m in BUDGETS:
                    base=dict(record=rec,start=start,views=k,cameras='+'.join(subsets[k]),moments=m,frames=k*m,sigma=sig[k])
                    ui=np.rint(np.linspace(0,199,m)).astype(int)
                    if not ok[ui].all() or not ok[0] or not ok[199]:rows.append(dict(base,arm='uniform',failure='missing'));continue
                    for nm,v in estimates(wp[ui],wt[ui],sig[k]).items():rows.append(dict(base,arm='uniform',estimator=nm,value_m=v,selected=ui.tolist() if nm=='state_mean' else None))
                    # ---- adaptive ----
                    mp=max(4,m//2);pi=np.rint(np.linspace(0,199,mp)).astype(int)
                    if not ok[pi].all():rows.append(dict(base,arm='adaptive',failure='probe_missing'));continue
                    try:
                        rho,q,_=error_density(wp[pi],wt[pi],sig[k],wt)
                        ai=allocate(rho,wt,m,must_keep=pi)
                        ai=np.array(sorted(set(ai.tolist())|set(pi.tolist())))
                        if len(ai)>m:   # keep probe, drop extras with least density contribution
                            ai=allocate(rho,wt,m,must_keep=pi)
                        ai=np.clip(ai,0,199)
                        if not ok[ai].all():raise ValueError('adaptive_missing')
                        for nm,v in estimates(wp[ai],wt[ai],sig[k]).items():
                            rows.append(dict(base,arm='adaptive',estimator=nm,value_m=v,actual_moments=int(len(ai)),q=q,selected=ai.tolist() if nm=='state_mean' else None))
                    except (ValueError,np.linalg.LinAlgError) as e:rows.append(dict(base,arm='adaptive',failure=str(e)))
                    if oracle:
                        mid=dense[(dense[:,1]>wt[0])&(dense[:,1]<wt[-1])]
                        rx=np.vstack([ref_x[start],mid[:,2:5],ref_x[start+199]])
                        rt=np.r_[wt[0],mid[:,1],wt[-1]]
                        xv=np.array([np.interp(wt,rt,rx[:,j]) for j in range(3)]).T
                        v=np.gradient(xv,wt,axis=0,edge_order=2);a=np.gradient(v,wt,axis=0,edge_order=2)
                        sp=np.linalg.norm(v,axis=1);rr=np.where(sp>1e-9,np.linalg.norm(np.cross(v,a),axis=1)**(2/3)/np.maximum(sp,1e-9),0.)
                        oi=np.clip(allocate(rr+.05*rr.max(),wt,m),0,199)
                        if ok[oi].all():
                            for nm,vv in estimates(wp[oi],wt[oi],sig[k]).items():rows.append(dict(base,arm='oracle_nonuniform',estimator=nm,value_m=vv))
        print(rec,'done',f'{perf_counter()-tick:.0f}s',flush=True)
    (out/'predictions.json').write_text(json.dumps(dict(probe=probe,rows=rows)))
    (out/'seal.json').write_text(json.dumps(dict(sha256=hashlib.sha256((out/'predictions.json').read_bytes()).hexdigest(),
        wall_s=perf_counter()-tick,burst=burst,oracle=oracle,utc=datetime.now(timezone.utc).isoformat())))
    for rec in records:
        tt,xyz,_,dense=load_record(rec,ids)
        for start in range(0,3000,200):
            end=start+199;mid=dense[(dense[:,1]>tt[start])&(dense[:,1]<tt[end]),2:5];ref=np.vstack([xyz[start],mid,xyz[end]])
            truth[rec,start]=float(np.linalg.norm(np.diff(ref,axis=0),axis=1).sum())
    for r in rows:
        r['truth_m']=truth[r['record'],r['start']];r['error_m']=None if r.get('value_m') is None else r['value_m']-r['truth_m']
    (out/'scored.json').write_text(json.dumps(rows))
    arms=['uniform','adaptive']+(['oracle_nonuniform'] if oracle else []);summary=[]
    for k in sorted(subsets):
        for m in BUDGETS:
            for est in ('state_mean','limited','polygon'):
                cell={}
                common=None
                for arm in arms:
                    d={(r['record'],r['start']):abs(r['error_m']) for r in rows if r['views']==k and r['moments']==m and r['arm']==arm and r.get('estimator')==est and r.get('error_m') is not None}
                    cell[arm]=d;common=set(d) if common is None else common&set(d)
                if not common:continue
                row=dict(views=k,moments=m,frames=k*m,estimator=est,windows=len(common))
                for arm in arms:
                    v=np.array([cell[arm][w] for w in sorted(common)])
                    row[f'{arm}_mae']=float(v.mean());row[f'{arm}_median']=float(np.median(v))
                u=np.array([cell['uniform'][w] for w in sorted(common)]);a=np.array([cell['adaptive'][w] for w in sorted(common)])
                row['adaptive_better_windows']=int((a<u).sum());row['gain_pct']=float((u.mean()-a.mean())/u.mean()*100);row['median_gain_pct']=float((np.median(u)-np.median(a))/np.median(u)*100)
                summary.append(row)
    (out/'summary.json').write_text(json.dumps(summary,indent=1))
    for est in ('state_mean','limited'):
        print(f'\n=== {est} ===  (MAE mm / median mm)')
        for k in sorted(subsets):
            for s in [x for x in summary if x['views']==k and x['estimator']==est]:
                o=f" oracle={s['oracle_nonuniform_mae']*1000:8.2f}/{s['oracle_nonuniform_median']*1000:7.2f}" if oracle and 'oracle_nonuniform_mae' in s else ''
                print(f"  k={k} m={s['moments']:3d} n={s['windows']:3d} uniform={s['uniform_mae']*1000:8.2f}/{s['uniform_median']*1000:7.2f}  adaptive={s['adaptive_mae']*1000:8.2f}/{s['adaptive_median']*1000:7.2f}{o}  better={s['adaptive_better_windows']:3d}/{s['windows']:3d} gain={s['gain_pct']:+6.1f}% medgain={s['median_gain_pct']:+6.1f}%")


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='dev');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--burst',type=int,default=25);p.add_argument('--oracle',action='store_true')
    a=p.parse_args();main(a.split,a.out,a.burst,a.oracle)
