"""Does a reference-free probe predict the measured error-vs-allocation curve?

For each window: take a probe of m0 moments with a camera subset, fit (q, sigma)
by innovation likelihood, bootstrap the predicted error curve over m (and over k
by rescaling sigma from the probe's per-view geometry), then compare with the
error actually measured against the mocap reference in the sealed sweep.

No reference value enters the prediction. The reference is read only to score it.
"""
import json,argparse,hashlib
from itertools import combinations
from pathlib import Path
from time import perf_counter
from datetime import datetime,timezone
import numpy as np
from audit_mcalib_sample import read
from run_allocation_sweep import subset_points,polygon,SPLITS
from allocation_selfcheck_cpu import fit_state_params,predict_error_curve
from limited_arc_cpu import limited_arc
from continuous_motion_cpu import estimate as state_estimate

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data_external/mcalib_2026-09-13/verified/mocapAssistedRecord'
CAL=ROOT/'experiments/mcalib_3d_cache_2026-09-13_verified/public/calibration.json'
SWEEP={'dev':ROOT/'experiments/allocation_sweep_dev_v1_2026-09-15','confirm':ROOT/'experiments/allocation_confirm_v1_2026-09-15'}
MOMENTS=[8,12,16,19,25,28,33,50,100,200]
RULE={2:'3e0f8f0+44c4b2e',3:'216f21c1+3e0f8f0+44c4b2e'}


def main(split,out,probe,realizations,estimator):
    cams=json.loads(CAL.read_text())['cameras'];ids=sorted(cams);records=SPLITS[split]
    subsets={2:tuple(RULE[2].split('+')),3:tuple(RULE[3].split('+')),7:tuple(ids)}
    out.mkdir(parents=True,exist_ok=False);rows=[];tick=perf_counter()
    for rec in records:
        uv={c:read(DATA/rec/f'centroidsUV{c}.pkl') for c in ids};n=len(uv[ids[0]]);t=np.arange(n)*.02+.00006969+.000996/2
        pts={k:subset_points(uv,cams,s,n)[0] for k,s in subsets.items()}
        for start in range(0,3000,200):
            duration=float(t[start+199]-t[start])
            for k in subsets:
                sel=start+np.rint(np.linspace(0,199,probe)).astype(int);p=pts[k][sel];ok=np.all(np.isfinite(p),axis=1)
                row=dict(record=rec,start=start,views=k,cameras='+'.join(subsets[k]),probe=probe,probe_missing=int((~ok).sum()))
                if ok.sum()<probe-1 or not ok[0] or not ok[-1]:rows.append(dict(row,failure='probe_incomplete'));continue
                f=fit_state_params(p[ok],t[sel][ok]);row.update(q_hat=f['q'],sigma_hat=f['sigma'],q_boundary=f['q_at_boundary'],sigma_boundary=f['sigma_at_boundary'])
                mean,se=predict_error_curve(f['q'],f['sigma'],duration,MOMENTS,realizations=realizations,seed=abs(hash((rec,start,k)))%2**31,estimator=estimator)
                row['predicted']={str(m):mean[m] for m in mean};row['predicted_se']={str(m):se[m] for m in se}
                row['predicted_argmin']=int(min(mean,key=mean.get));rows.append(row)
        print(rec,'done',f'{perf_counter()-tick:.0f}s',flush=True)
    (out/'predictions.json').write_text(json.dumps(rows))
    (out/'seal.json').write_text(json.dumps(dict(sha256=hashlib.sha256((out/'predictions.json').read_bytes()).hexdigest(),wall_s=perf_counter()-tick,
        probe=probe,realizations=realizations,estimator=estimator,utc=datetime.now(timezone.utc).isoformat())))
    # ---- scoring against the sealed measured sweep (reference read only here) ----
    meas=json.load(open(SWEEP[split]/'scored.json'));M={}
    for r in meas:
        if r['method']==estimator and r['error_m'] is not None:M[r['cameras'],r['record'],r['start'],r['moments']]=abs(r['error_m'])
    per=[];agg={}
    for row in rows:
        if row.get('failure'):continue
        key=(row['cameras'],row['record'],row['start'])
        meas_curve={m:M.get(key+(m,)) for m in MOMENTS if M.get(key+(m,)) is not None}
        if len(meas_curve)<len(MOMENTS)-2:continue
        pred={m:row['predicted'][str(m)] for m in meas_curve}
        mm=np.array(sorted(meas_curve));mv=np.array([meas_curve[m] for m in mm]);pv=np.array([pred[m] for m in mm])
        from scipy.stats import spearmanr
        per.append(dict(record=row['record'],start=row['start'],views=row['views'],q=row['q_hat'],sigma=row['sigma_hat'],
            predicted_argmin=int(mm[np.argmin(pv)]),measured_argmin=int(mm[np.argmin(mv)]),
            spearman=float(spearmanr(pv,mv)[0]),measured_at_predicted=float(mv[np.argmin(pv)]),measured_best=float(mv.min()),
            regret=float(mv[np.argmin(pv)]/mv.min()-1),ratio_median=float(np.median(pv/mv))))
    for k in sorted({p['views'] for p in per}):
        pk=[p for p in per if p['views']==k]
        agg[f'k{k}']=dict(windows=len(pk),median_spearman=float(np.median([p['spearman'] for p in pk])),
            exact_argmin=sum(p['predicted_argmin']==p['measured_argmin'] for p in pk),
            median_regret=float(np.median([p['regret'] for p in pk])),mean_regret=float(np.mean([p['regret'] for p in pk])),
            regret_below_10pct=sum(p['regret']<=.1 for p in pk),median_pred_over_meas=float(np.median([p['ratio_median'] for p in pk])),
            predicted_argmin_hist={str(v):int(c) for v,c in zip(*np.unique([p['predicted_argmin'] for p in pk],return_counts=True))},
            measured_argmin_hist={str(v):int(c) for v,c in zip(*np.unique([p['measured_argmin'] for p in pk],return_counts=True))},
            median_sigma_hat=float(np.median([p['sigma'] for p in pk])),median_q_hat=float(np.median([p['q'] for p in pk])))
    (out/'scored.json').write_text(json.dumps(dict(per_window=per,aggregate=agg),indent=1))
    print(json.dumps(agg,indent=1))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='dev');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--probe',type=int,default=16);p.add_argument('--realizations',type=int,default=24);p.add_argument('--estimator',default='state_mean')
    a=p.parse_args();main(a.split,a.out,a.probe,a.realizations,a.estimator)
