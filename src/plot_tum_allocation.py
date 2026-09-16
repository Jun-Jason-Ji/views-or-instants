"""TUM fr3: path MAE vs frames per 4 s window for cheap CPU front ends (dev windows and unseen windows)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
dev=json.load(open(ROOT/'experiments/tum_dense_cheap_v1_2026-09-15/summary.json'))
conf=json.load(open(ROOT/'experiments/tum_dense_cheap_confirm_v1_2026-09-15/decision.json'))['decision']
fig,axes=plt.subplots(1,2,figsize=(12,4.6))
ax=axes[0]
for fe,c in (('orb','#1f77b4'),('xfeat','#2ca02c')):
    rows=[s for s in dev if s['label'].startswith(fe+'_')];x=[s['frames'] for s in rows];y=[s['path_mae_state_mean_seq_equal']*1000 for s in rows]
    ax.plot(x,y,'o-',color=c,label=f'{fe} CPU (complete {min(s["completed"] for s in rows)}-{max(s["completed"] for s in rows)}/15)')
for lab,mk,c in (('lg_baseline','s','#d62728'),('depth_weighted','^','#ff7f0e')):
    rows=[s for s in dev if s['label'].startswith(lab)];ax.plot([s['frames'] for s in rows],[s['path_mae_state_mean_seq_equal']*1000 for s in rows],mk,color=c,ms=9,label=f'{lab} GPU LighterGlue (frozen 2026-09-13)')
ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('frames per 4 s window');ax.set_ylabel('path-length MAE (mm), state_mean');ax.set_title('fr3 development windows (15, offsets 1-17 s)');ax.grid(True,which='both',alpha=.3);ax.legend(fontsize=7)
ax=axes[1]
for fe,c in (('orb','#1f77b4'),('xfeat','#2ca02c')):
    pb=conf[fe]['per_budget'];x=[int(b) for b in pb];y=[pb[b]['mae_seq_equal']*1000 for b in pb]
    ax.plot(x,y,'o-',color=c,label=f'{fe} CPU (complete {min(v["completed"] for v in pb.values())}/{list(pb.values())[0]["windows"]})')
    for b in pb:ax.annotate(f"{pb[b]['bias']*1000:+.0f}",(int(b),pb[b]['mae_seq_equal']*1000),textcoords='offset points',xytext=(4,4),fontsize=7,color=c)
ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('frames per 4 s window');ax.set_ylabel('path-length MAE (mm), state_mean');ax.set_title('fr3 unseen windows (14, offsets >= 21 s), pre-registered; labels = signed bias mm');ax.grid(True,which='both',alpha=.3);ax.legend(fontsize=7)
fig.tight_layout();out=ROOT/'experiments/tum_dense_cheap_confirm_v1_2026-09-15/tum_allocation.png';fig.savefig(out,dpi=150);print(out)
