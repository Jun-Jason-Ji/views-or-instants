"""Reference-free allocator: predict the best (views, moments) from a probe, then score the choice.

Session probe (once per record): BURST consecutive frames read with all views ->
per-subset observation noise sigma_hat(k). The burst is at the native 50 Hz, where
measurement noise dominates process noise, so sigma is identifiable.

Window probe: SPREAD moments across the window -> q_hat with sigma held fixed.

Prediction: parametric bootstrap of the deployed estimator over candidate m, for
each candidate subset, using (q_hat, sigma_hat(k)). Choose the (k,m) with the
lowest predicted error whose cost k*m fits the budget.

Nothing in the prediction path reads the mocap reference; it is read only to score.
"""
import json,argparse,hashlib
from itertools import combinations
from pathlib import Path
from time import perf_counter
from datetime import datetime,timezone
import numpy as np
from audit_mcalib_sample import read
from run_allocation_sweep import subset_points,SPLITS
from allocation_selfcheck_cpu import fit_state_params,fit_two_part_probe,predict_error_curve,burst_noise_sample
from continuous_motion_cpu import filter_states,Q_GRID

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data_external/mcalib_2026-09-13/verified/mocapAssistedRecord'
CAL=ROOT/'experiments/mcalib_3d_cache_2026-09-13_verified/public/calibration.json'
SWEEP={'dev':ROOT/'experiments/allocation_sweep_dev_v1_2026-09-15','confirm':ROOT/'experiments/allocation_confirm_v1_2026-09-15'}
MOMENTS=[8,12,16,25,33,50,100,200];BUDGETS=[50,60,75,100,150,200];RULE={2:'3e0f8f0+44c4b2e',3:'216f21c1+3e0f8f0+44c4b2e'}


def fit_q(points,times,sigma):
    best=(np.inf,None)
    for qi,q in enumerate(Q_GRID):
        nll=filter_states(np.asarray(points,float),np.asarray(times,float),float(q),sigma)[-1]
        if nll<best[0]:best=(nll,qi)
    return float(Q_GRID[best[1]]),best[1] in (0,len(Q_GRID)-1)


def main(split,out,burst,spread,realizations,estimator):
    cams=json.loads(CAL.read_text())['cameras'];ids=sorted(cams);records=SPLITS[split]
    subsets={2:tuple(RULE[2].split('+')),3:tuple(RULE[3].split('+')),7:tuple(ids)}
    out.mkdir(parents=True,exist_ok=False);rows=[];probes={};tick=perf_counter()
    for rec in records:
        uv={c:read(DATA/rec/f'centroidsUV{c}.pkl') for c in ids};n=len(uv[ids[0]]);t=np.arange(n)*.02+.00006969+.000996/2
        pts={k:subset_points(uv,cams,s,n)[0] for k,s in subsets.items()}
        # ---- session probe: BURST consecutive frames, all views read once ----
        sig={}
        for k in subsets:
            bp=pts[k][:burst];okb=np.all(np.isfinite(bp),axis=1)
            f=fit_state_params(bp[okb],t[:burst][okb]);pool=burst_noise_sample(bp[okb])
            miss=float(1-okb.mean())
            sig[k]=dict(sigma=f['sigma'],at_boundary=f['sigma_at_boundary'],burst_q=f['q'],used=int(okb.sum()),miss_rate=miss,
                        pool=None if pool is None else pool.tolist(),pool_p95_mm=None if pool is None else float(np.quantile(np.linalg.norm(pool,axis=1),.95))*1000)
        probes[rec]=dict(burst_frames=burst,camera_frames=burst*len(ids),
            sigma_hat={str(k):{kk:vv for kk,vv in sig[k].items() if kk!='pool'} for k in sig})
        print(rec,'probe sigma',{k:round(sig[k]['sigma']*1000,2) for k in sig},'mm',flush=True)
        for start in range(0,3000,200):
            duration=float(t[start+199]-t[start])
            for k in subsets:
                sel=start+np.rint(np.linspace(0,199,spread)).astype(int);p=pts[k][sel];ok=np.all(np.isfinite(p),axis=1)
                row=dict(record=rec,start=start,views=k,cameras='+'.join(subsets[k]),sigma_hat=sig[k]['sigma'],miss_rate=sig[k]['miss_rate'],spread=spread,spread_missing=int((~ok).sum()))
                if ok.sum()<3 or not ok[0] or not ok[-1]:rows.append(dict(row,failure='spread_incomplete'));continue
                q,qb=fit_q(p[ok],t[sel][ok],sig[k]['sigma']);row.update(q_hat=q,q_boundary=qb)
                mean,se=predict_error_curve(q,sig[k]['sigma'],duration,MOMENTS,realizations=realizations,seed=abs(hash((rec,start,k)))%2**31,
                    estimator=estimator,noise_pool=sig[k]['pool'],miss_rate=sig[k]['miss_rate'])
                row['predicted']={str(m):mean[m] for m in mean};row['predicted_argmin']=int(min(mean,key=mean.get));rows.append(row)
        print(rec,'done',f'{perf_counter()-tick:.0f}s',flush=True)
    (out/'predictions.json').write_text(json.dumps(dict(probes=probes,rows=rows)))
    (out/'seal.json').write_text(json.dumps(dict(sha256=hashlib.sha256((out/'predictions.json').read_bytes()).hexdigest(),wall_s=perf_counter()-tick,
        burst=burst,spread=spread,realizations=realizations,estimator=estimator,utc=datetime.now(timezone.utc).isoformat())))
    score(split,out,estimator)


def score(split,out,estimator):
    d=json.loads((out/'predictions.json').read_text());rows=d['rows']
    meas=json.load(open(SWEEP[split]/'scored.json'));M={}
    for r in meas:
        if r['method']==estimator and r['error_m'] is not None:M[r['cameras'],r['record'],r['start'],r['moments']]=abs(r['error_m'])
    from scipy.stats import spearmanr
    per=[]
    for row in rows:
        if row.get('failure'):continue
        key=(row['cameras'],row['record'],row['start']);mc={m:M[key+(m,)] for m in MOMENTS if key+(m,) in M}
        if len(mc)<len(MOMENTS)-2:continue
        mm=np.array(sorted(mc));mv=np.array([mc[m] for m in mm]);pv=np.array([row['predicted'][str(m)] for m in mm])
        per.append(dict(record=row['record'],start=row['start'],views=row['views'],q=row['q_hat'],sigma=row['sigma_hat'],
            predicted_argmin=int(mm[np.argmin(pv)]),measured_argmin=int(mm[np.argmin(mv)]),spearman=float(spearmanr(pv,mv)[0]),
            measured_at_predicted=float(mv[np.argmin(pv)]),measured_best=float(mv.min()),regret=float(mv[np.argmin(pv)]/mv.min()-1),
            pred_over_meas=float(np.median(pv/mv))))
    agg={}
    for k in sorted({p['views'] for p in per}):
        pk=[p for p in per if p['views']==k]
        agg[f'k{k}']=dict(windows=len(pk),median_spearman=float(np.median([p['spearman'] for p in pk])),
            exact_argmin=sum(p['predicted_argmin']==p['measured_argmin'] for p in pk),median_regret=float(np.median([p['regret'] for p in pk])),
            mean_regret=float(np.mean([p['regret'] for p in pk])),regret_le_25pct=sum(p['regret']<=.25 for p in pk),
            mae_at_predicted=float(np.mean([p['measured_at_predicted'] for p in pk])),mae_at_oracle=float(np.mean([p['measured_best'] for p in pk])),
            predicted_argmin_hist={str(v):int(c) for v,c in zip(*np.unique([p['predicted_argmin'] for p in pk],return_counts=True))},
            measured_argmin_hist={str(v):int(c) for v,c in zip(*np.unique([p['measured_argmin'] for p in pk],return_counts=True))},
            median_sigma_mm=float(np.median([p['sigma'] for p in pk]))*1000,median_q=float(np.median([p['q'] for p in pk])))
    # ---- budget-constrained joint (k,m) allocation, aggregated over windows ----
    alloc=[]
    for budget in BUDGETS:
        pick={}
        for row in rows:
            if row.get('failure'):continue
            key=(row['record'],row['start']);cand=[(row['views'],m,row['predicted'][str(m)]) for m in MOMENTS if row['views']*m<=budget]
            if not cand:continue
            best=min(cand,key=lambda c:c[2])
            if key not in pick or best[2]<pick[key][2]:pick[key]=best+( '+'.join(sorted({row['cameras']})),)
        realized=[];oracle=[];prior=[];chosen={}
        for key,(k,m,_,cam) in pick.items():
            kk=(cam,key[0],key[1],m)
            if kk not in M:continue
            realized.append(M[kk]);chosen[f'{k}x{m}']=chosen.get(f'{k}x{m}',0)+1
            best=min([M[c,key[0],key[1],mm] for c in {r['cameras'] for r in rows} for mm in MOMENTS if (c,key[0],key[1],mm) in M and len(c.split('+'))*mm<=budget],default=None)
            if best is not None:oracle.append(best)
            pk=('+'.join(sorted({r['cameras'] for r in rows if r['views']==7})),key[0],key[1],8)
            if pk in M and 7*8<=budget:prior.append(M[pk])
        alloc.append(dict(budget=budget,windows=len(realized),chosen=chosen,mae_realized=float(np.mean(realized)) if realized else None,
                          mae_oracle=float(np.mean(oracle)) if oracle else None,mae_prior_7x8=float(np.mean(prior)) if prior else None,
                          regret_vs_oracle=float(np.mean(realized)/np.mean(oracle)-1) if realized and oracle else None))
    (out/'scored.json').write_text(json.dumps(dict(per_window=per,aggregate=agg,allocation=alloc),indent=1))
    print(json.dumps(agg,indent=1));print(json.dumps(alloc,indent=1))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='dev');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--burst',type=int,default=25);p.add_argument('--spread',type=int,default=8);p.add_argument('--realizations',type=int,default=24)
    p.add_argument('--estimator',default='limited');p.add_argument('--score-only',action='store_true')
    a=p.parse_args()
    if a.score_only:score(a.split,a.out,a.estimator)
    else:main(a.split,a.out,a.burst,a.spread,a.realizations,a.estimator)
