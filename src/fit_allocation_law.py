"""Explicit allocation law MAE(m,k) = a*m^-p + b*m^q*k^-r, fitted on dev, tested on confirmation.

Two terms: motion missed between the m uniform moments (falls with m) and per-point
noise accumulated along the polyline (grows with m, shrinks with more views k).
Fitted in log space on per-(k,m) medians over camera subsets. The confirmation
records are used only to test predictions; nothing is refitted on them.
"""
import json,argparse
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[1]
DEV=ROOT/'experiments/allocation_sweep_dev_v1_2026-09-15/summary_v2.json';CONF=ROOT/'experiments/allocation_confirm_v1_2026-09-15/summary_v2.json'


def table(path,method):
    S=json.load(open(path))['summary'];per={}
    for s in S:
        if s['method']==method and s['mae_m'] is not None:per.setdefault((s['views'],s['moments']),[]).append(s['mae_m'])
    return {key:float(np.median(v)) for key,v in per.items()}


def model(theta,m,k):
    la,p,lb,q,r=theta;return np.exp(la)*m**(-p)+np.exp(lb)*m**q*k**(-r)


def fit(tab):
    keys=sorted(tab);m=np.array([x[1] for x in keys],float);k=np.array([x[0] for x in keys],float);y=np.array([tab[x] for x in keys])
    res=least_squares(lambda th:np.log(model(th,m,k))-np.log(y),x0=[np.log(10),2.,np.log(1e-4),1.,.5],bounds=([-20,0,-30,0,0],[20,6,10,4,4]))
    return res.x,float(np.sqrt(np.mean(res.fun**2)))


def evaluate(theta,tab):
    keys=sorted(tab);m=np.array([x[1] for x in keys],float);k=np.array([x[0] for x in keys],float);y=np.array([tab[x] for x in keys]);pred=model(theta,m,k)
    return keys,y,pred,float(np.sqrt(np.mean((np.log(pred)-np.log(y))**2))),float(np.median(np.abs(pred/y-1)))


def main():
    report={}
    for method in ('limited','state_mean'):
        dev=table(DEV,method);conf=table(CONF,method);theta,rms_dev=fit(dev);keys,y,pred,rms_conf,medrel=evaluate(theta,conf)
        la,p,lb,q,r=theta
        print(f'\n== {method}: MAE = {np.exp(la):.4g} * m^-{p:.3f} + {np.exp(lb):.3g} * m^{q:.3f} * k^-{r:.3f}   (dev log-RMS {rms_dev:.3f}, confirm log-RMS {rms_conf:.3f}, confirm median |rel err| {medrel:.1%})')
        rows=[]
        for (k,m),yy,pp in zip(keys,y,pred):rows.append(dict(views=k,moments=m,frames=k*m,measured=yy,predicted=float(pp),rel_err=float(pp/yy-1)))
        for row in rows:print(f"  k={row['views']} m={row['moments']:3d} frames={row['frames']:4d} measured={row['measured']:.4f} predicted={row['predicted']:.4f} rel={row['rel_err']:+.0%}")
        # optimal allocation per cost budget: predicted (from dev fit) vs measured on confirmation, over the confirmation grid
        grid=sorted(conf);checks=[]
        for budget in (50,60,75,100,150,200):
            feas=[g for g in grid if g[0]*g[1]<=budget]
            if not feas:continue
            pred_best=min(feas,key=lambda g:model(theta,g[1],g[0]));meas_best=min(feas,key=lambda g:conf[g])
            checks.append(dict(budget=budget,predicted_best=pred_best,measured_best=meas_best,measured_mae_at_predicted=conf[pred_best],measured_mae_best=conf[meas_best],
                               prior_7x8=conf.get((7,8)),regret=conf[pred_best]/conf[meas_best]-1))
            print(f"  budget<={budget:3d}: predicted best (k,m)={pred_best} -> measured {conf[pred_best]:.4f}; measured best {meas_best} {conf[meas_best]:.4f}; regret {conf[pred_best]/conf[meas_best]-1:+.0%}")
        # predicted optimal m per k on a continuous grid
        opt={}
        for k in (2,3,7):
            ms=np.arange(8,201);v=model(theta,ms,k);opt[k]=dict(m_opt=int(ms[np.argmin(v)]),mae_opt=float(v.min()))
        print('  predicted optimum per k:',opt)
        report[method]=dict(theta=dict(a=float(np.exp(la)),p=float(p),b=float(np.exp(lb)),q=float(q),r=float(r)),dev_log_rms=rms_dev,confirm_log_rms=rms_conf,confirm_median_rel_err=medrel,rows=rows,budget_checks=checks,optimum_per_k=opt)
    out=ROOT/'experiments/allocation_confirm_v1_2026-09-15/allocation_law.json';out.write_text(json.dumps(report,indent=1));print('\nwritten',out)


if __name__=='__main__':main()
