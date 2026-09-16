"""Error-cost frontier plot for the observation allocation sweep (dev and, if present, confirmation)."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def best_per(S,views,moments):
    """Per subset the best estimator MAE; return median, min, max over subsets."""
    per={}
    for r in S:
        if r['views']==views and r['moments']==moments and r['mae_m'] is not None:per.setdefault(r['cameras'],[]).append(r['mae_m'])
    v=np.array([min(x) for x in per.values()]);return (np.median(v),v.min(),v.max()) if len(v) else (np.nan,)*3


def panel(ax,S,title):
    colors={2:'#1f77b4',3:'#2ca02c',7:'#d62728'}
    for k in (2,3,7):
        ms=sorted({r['moments'] for r in S if r['views']==k});x=[];med=[];lo=[];hi=[]
        for m in ms:
            a,b,c=best_per(S,k,m);x.append(k*m);med.append(a);lo.append(b);hi.append(c)
        ax.plot(x,med,'o-',color=colors[k],label=f'{k} views (median over camera subsets)')
        if k!=7:ax.fill_between(x,lo,hi,color=colors[k],alpha=.15,label=f'{k} views: best-worst subset')
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('camera frame requests per 3.98 s window');ax.set_ylabel('path-length MAE (m)')
    ax.axvline(56,color='gray',ls='--',lw=.8);ax.text(58,ax.get_ylim()[0]*1.5 if ax.get_ylim()[0]>0 else 1e-3,'56 frames\n(prior 7x8)',fontsize=8,color='gray')
    ax.set_title(title);ax.grid(True,which='both',alpha=.3);ax.legend(fontsize=7)


def main():
    dev=json.load(open(ROOT/'experiments/allocation_sweep_dev_v1_2026-09-15/summary_v2.json'))['summary']
    conf_path=ROOT/'experiments/allocation_confirm_v1_2026-09-15/summary_v2.json'
    n=2 if conf_path.exists() else 1;fig,axes=plt.subplots(1,n,figsize=(6.5*n,4.6))
    axes=np.atleast_1d(axes);panel(axes[0],dev,'Development: record19-23, 75 windows')
    if n==2:
        conf=json.load(open(conf_path))['summary'];panel(axes[1],conf,'Pre-registered confirmation: record15-18, 60 windows')
    fig.tight_layout();out=ROOT/'experiments/allocation_confirm_v1_2026-09-15/allocation_frontier.png' if n==2 else ROOT/'experiments/allocation_sweep_dev_v1_2026-09-15/allocation_frontier.png'
    fig.savefig(out,dpi=150);print(out)


if __name__=='__main__':main()
