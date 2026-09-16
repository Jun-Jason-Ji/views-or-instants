"""Regenerate the figures at IEEE Access column widths (double column span 7.16 in, single 3.45 in)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'paper/ieee';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':7,'axes.labelsize':7,'axes.titlesize':7,'legend.fontsize':6,
    'xtick.labelsize':6.5,'ytick.labelsize':6.5,'axes.grid':True,'grid.alpha':.25,'grid.linewidth':.4,
    'lines.linewidth':1.0,'lines.markersize':3,'axes.linewidth':.6,'font.family':'serif',
    'font.serif':['Times New Roman','DejaVu Serif'],'mathtext.fontset':'stix','savefig.dpi':600,'savefig.bbox':'tight'})
W2,W1=7.16,3.45;C={2:'#1f77b4',3:'#2ca02c',7:'#d62728'}


def stat(S,k,m,method='state_mean'):
    v=[s['mae_m'] for s in S if s['views']==k and s['moments']==m and s['method']==method and s['mae_m'] is not None]
    return (np.median(v)*1000,min(v)*1000,max(v)*1000) if v else (None,)*3


def main():
    dev=json.load(open(ROOT/'experiments/allocation_sweep_dev_v1_2026-09-15/summary_v2.json'))['summary']
    conf=json.load(open(ROOT/'experiments/allocation_confirm_v1_2026-09-15/summary_v2.json'))['summary']

    fig,ax=plt.subplots(1,2,figsize=(W2,2.5))
    for a,S,t,ms in ((ax[0],dev,'(a) development: 5 records, 75 windows',[8,12,16,25,33,50]),
                     (ax[1],conf,'(b) pre-registered confirmation: 4 records, 60 windows',[8,12,16,19,25,28,33,50])):
        for k in (2,3,7):
            x=[];y=[];lo=[];hi=[]
            for m in ms:
                med,mn,mx=stat(S,k,m)
                if med is None:continue
                x.append(k*m);y.append(med);lo.append(mn);hi.append(mx)
            a.plot(x,y,'o-',color=C[k],label=f'$k$ = {k} views')
            if k!=7:a.fill_between(x,lo,hi,color=C[k],alpha=.13,lw=0)
        a.set_xscale('log');a.set_yscale('log');a.set_xlabel('camera frames per 4 s window, $B=km$')
        a.set_ylabel('path-length MAE (mm)');a.set_title(t);a.legend(frameon=False)
    fig.tight_layout(pad=.3);fig.savefig(OUT/'fig1.pdf');fig.savefig(OUT/'fig1.png');plt.close(fig)

    fig,ax=plt.subplots(figsize=(W1,2.4))
    for m in (16,19,25,28,33):
        y=[];e=[]
        for k in (2,3,7):
            med,mn,mx=stat(conf,k,m);y.append(med);e.append([med-mn,mx-med])
        ax.errorbar([2,3,7],y,yerr=np.array(e).T,marker='o',capsize=1.5,lw=.9,label=f'$m$ = {m}')
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xticks([2,3,7]);ax.set_xticklabels(['2','3','7']);ax.minorticks_off()
    ax.set_xlabel('views per instant, $k$');ax.set_ylabel('path-length MAE (mm)')
    ax.legend(frameon=False,ncol=2,columnspacing=.8)
    fig.tight_layout(pad=.3);fig.savefig(OUT/'fig2.pdf');fig.savefig(OUT/'fig2.png');plt.close(fig)

    ad=json.load(open(ROOT/'experiments/adaptive_moments_dev_v1_2026-09-15/summary.json'))
    fig,ax=plt.subplots(figsize=(W1,2.4))
    rows=sorted([s for s in ad if s['views']==3 and s['estimator']=='state_mean'],key=lambda s:s['moments'])
    x=[s['moments'] for s in rows]
    ax.plot(x,[s['uniform_mae']*1000 for s in rows],'o-',color='#333',label='uniform')
    ax.plot(x,[s['adaptive_mae']*1000 for s in rows],'s--',color='#d62728',label='curvature density')
    ax.plot(x,[s['oracle_nonuniform_mae']*1000 for s in rows],'^:',color='#1f77b4',label='oracle placement')
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xticks(x);ax.set_xticklabels([str(v) for v in x]);ax.minorticks_off()
    ax.set_xlabel('time instants, $m$ ($k=3$)');ax.set_ylabel('path-length MAE (mm)');ax.legend(frameon=False)
    fig.tight_layout(pad=.3);fig.savefig(OUT/'fig3.pdf');fig.savefig(OUT/'fig3.png');plt.close(fig)

    mc=json.load(open(ROOT/'experiments/ctsd_baseline_dev_v1_2026-09-15/scored.json'))
    tu=json.load(open(ROOT/'experiments/tum_dense_cheap_confirm_v1_2026-09-15/decision.json'))['decision']
    fig,ax=plt.subplots(1,2,figsize=(W2,2.5))
    M=[8,12,16,25,33,50,100,200]
    for lab,fn,st in (('mean',np.mean,'o-'),('median',np.median,'s--')):
        y=[fn(np.array([abs(r['error_m']) for r in mc if r['views']==3 and r['moments']==m
            and r['method']=='state_mean_calibrated' and r['error_m'] is not None])*1000) for m in M]
        ax[0].plot([3*m for m in M],y,st,label=f'{lab} over 75 windows')
    ax[0].set_xscale('log');ax[0].set_yscale('log');ax[0].set_xticks([24,48,99,150,300,600])
    ax[0].set_xticklabels(['24','48','99','150','300','600']);ax[0].minorticks_off()
    ax[0].set_xlabel('camera frames per window');ax[0].set_ylabel('path-length error (mm)')
    ax[0].set_title('(a) the dense-regime rise appears only in the mean');ax[0].legend(frameon=False)
    for fe,c in (('orb','#1f77b4'),('xfeat','#2ca02c')):
        pb=tu[fe]['per_budget'];x=sorted(int(b) for b in pb)
        ax[1].plot(x,[pb[str(b)]['mae_seq_equal']*1000 for b in x],'o-',color=c,label=f'{fe.upper()} front end')
        ax[1].plot(x,[abs(pb[str(b)]['bias'])*1000 for b in x],'--',color=c,alpha=.45,lw=.7,label=f'{fe.upper()} |bias|')
    ax[1].set_xscale('log');ax[1].set_yscale('log');ax[1].set_xticks([8,16,32,64])
    ax[1].set_xticklabels(['8','16','32','64']);ax[1].minorticks_off()
    ax[1].set_xlabel('frames per 4 s window');ax[1].set_ylabel('path-length MAE (mm)')
    ax[1].set_title('(b) single-camera chain, 14 unseen windows');ax[1].legend(frameon=False)
    fig.tight_layout(pad=.3);fig.savefig(OUT/'fig4.pdf');fig.savefig(OUT/'fig4.png');plt.close(fig)
    for p in sorted(OUT.glob('*')):print(f'  {p.stat().st_size/1024:8.1f} KB  {p.name}')


if __name__=='__main__':main()
