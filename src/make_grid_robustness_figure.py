"""Paired grid sensitivity figure for v1.0.4; no independence implied."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from allocation_revision_v104 import grids
ROOT=Path(__file__).resolve().parents[1]

def make():
    data=json.loads((ROOT/'experiments/allocation_revision_v104_2026-09-16/summary.json').read_text())
    names=list(grids(28)); labels=['U','-0.4','-0.2','+0.2','+0.4','J1','J2','J3','J4','J5','J6']
    plt.rcParams.update({'font.size':9,'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42})
    fig,axs=plt.subplots(1,2,figsize=(7.1,3.25),sharey=True)
    for ax,b in zip(axs,(84,168)):
        for arm,color,marker,label in [('dlt','#D55E00','o','DLT'),('spatial','#0072B2','s','Spatial checks'),('spatial_temporal','#009E73','^','Spatial + temporal')]:
            values={r['grid']:r['difference_mm'] for r in data['paired'] if r['split']=='evaluation' and r['budget']==b and r['arm']==arm}
            ax.plot(range(len(names)),[values[n] for n in names],color=color,marker=marker,ms=4,lw=1.1,label=label)
        ax.axhline(0,color='#555555',lw=.8,ls='--');ax.set_xticks(range(len(names)),labels,rotation=45)
        ax.set_xlabel('Uniform (U), offsets, seeded jitters (J)')
        ax.set_title(f'{"(a)" if b==84 else "(b)"} {b} frames per window')
        ax.grid(axis='y',alpha=.2);ax.set_ylim(-95,165);ax.set_yticks([-75,-50,-25,0,25,50,75,100,125,150])
    axs[0].set_ylabel('Paired MAE difference (mm)\n3 views minus 7 views')
    axs[0].legend(loc='upper left',frameon=False)
    fig.tight_layout(pad=.8)
    for ext in ('pdf','png'): fig.savefig(ROOT/f'paper/mst/fig3.{ext}',dpi=220)
    plt.close(fig)

if __name__=='__main__':make()
