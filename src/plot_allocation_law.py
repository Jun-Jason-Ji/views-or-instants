"""Measured vs law-predicted MAE on the confirmation records (fit from dev only)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments/allocation_confirm_v1_2026-09-15'
law=json.load(open(EXP/'allocation_law.json'));colors={2:'#1f77b4',3:'#2ca02c',7:'#d62728'}
fig,axes=plt.subplots(1,2,figsize=(12,4.6))
for ax,(method,L) in zip(axes,law.items()):
    th=L['theta']
    for k in (2,3,7):
        rows=[r for r in L['rows'] if r['views']==k];ms=np.array([r['moments'] for r in rows]);frames=k*ms
        ax.plot(frames,[r['measured'] for r in rows],'o',color=colors[k],label=f'k={k} measured (record15-18)')
        mm=np.arange(8,201);ax.plot(k*mm,th['a']*mm**-th['p']+th['b']*mm**th['q']*k**-th['r'],'-',color=colors[k],alpha=.7,label=f'k={k} law fitted on record19-23')
    ax.set_xscale('log');ax.set_yscale('log');ax.grid(True,which='both',alpha=.3);ax.set_xlabel('camera frame requests per window');ax.set_ylabel('path-length MAE (m)')
    ax.set_title(f"{method}: MAE = {th['a']:.3g} m^-{th['p']:.2f} + {th['b']:.2g} m^{th['q']:.2f} k^-{th['r']:.2f}",fontsize=9);ax.legend(fontsize=7)
fig.tight_layout();out=EXP/'allocation_law.png';fig.savefig(out,dpi=150);print(out)
