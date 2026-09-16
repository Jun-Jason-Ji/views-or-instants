"""Budget and tail-error figure for the exploratory v1.0.3 analysis."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
ROOT=Path(__file__).resolve().parents[1]

def make():
    data=json.loads((ROOT/'experiments/allocation_revision_v103_2026-09-16/summary.json').read_text())
    rows=[r for r in data['summary'] if r['sigma_mm']==3]
    plt.rcParams.update({'font.size':9,'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42})
    fig,axs=plt.subplots(1,2,figsize=(7.1,3.1))
    for k,marker in zip(range(2,8),['o','s','^','D','v','P']):
        a=sorted([r for r in rows if r['views']==k],key=lambda r:r['budget'])
        axs[0].plot([r['budget'] for r in a],[r['mae_mm'] for r in a],marker=marker,lw=1.2,label=f'{k} views'+(' (48/60)' if k==2 else ''))
    for k,color,marker in [(3,'#0072B2','s'),(7,'#D55E00','^')]:
        a=sorted([r for r in rows if r['views']==k],key=lambda r:r['budget'])
        for field,style,label in [('p95_mm','--','95th percentile'),('maximum_mm','-','maximum')]:
            axs[1].plot([r['budget'] for r in a],[r[field] for r in a],color=color,marker=marker,ls=style,lw=1.2,label=f'{k} views: {label}')
    for ax in axs:
        ax.set_xticks(data['budgets']);ax.set_xlabel('Frame budget cap, B');ax.set_yscale('log')
        ax.yaxis.set_major_locator(FixedLocator([10,20,50,100,200,500,1000,2000,5000]));ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'));ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(alpha=.18);ax.legend(frameon=False)
    axs[0].set_ylabel('Completed-window MAE (mm)');axs[0].set_title('(a) Fixed geometric subset per view count')
    axs[1].set_ylabel('Absolute path error (mm)');axs[1].set_title('(b) Tail errors, all 60 windows complete')
    axs[0].set_ylim(6,900);axs[1].set_ylim(12,16000)
    fig.tight_layout(pad=.8)
    for ext in ('pdf','png'):fig.savefig(ROOT/f'paper/mst/fig1.{ext}',dpi=220)
    plt.close(fig)

if __name__=='__main__':make()
