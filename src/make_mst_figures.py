"""Figures for the Measurement Science and Technology submission.

IOP requirements applied here (iopjournal-guidelines.pdf and the MST author pages):
  - vector PDF
  - text 8-12 pt at final size, so each figure is generated at its final width
    and included at natural size
  - colour must not be the only carrier of information: every series also has a
    distinct line style and marker, so the figures survive greyscale printing
  - file names use only a-z, A-Z, 0-9 and underscore
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/mst'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.grid': True, 'grid.alpha': .25,
    'grid.linewidth': .4, 'lines.linewidth': 1.1, 'lines.markersize': 4,
    'axes.linewidth': .7, 'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'savefig.bbox': 'tight', 'savefig.pad_inches': .02, 'pdf.fonttype': 42})

W2, W1 = 6.0, 4.3
# colour plus a distinct dash pattern and marker for every series
STYLE = {2: ('#1f77b4', 'o', '-'), 3: ('#2ca02c', 's', '--'), 7: ('#d62728', '^', ':')}


def plain_log_ticks(axis, ticks):
    """Label a log axis with plain numbers at the given positions.

    Left to itself matplotlib writes 10^1, 10^2 at the decades and labels the
    minor ticks as 2x10^1 etc; the latter overlap at column width and the
    former are harder to read than 10, 20, 50.  Every log axis in this file
    goes through here or through an explicit set_xticks, so the panels agree.
    """
    axis.set_ticks(ticks)
    axis.set_ticklabels([str(t) for t in ticks])


def stat(S, k, m, method='state_mean'):
    v = [s['mae_m'] for s in S if s['views'] == k and s['moments'] == m
         and s['method'] == method and s['mae_m'] is not None]
    return (np.median(v) * 1000, min(v) * 1000, max(v) * 1000) if v else (None,) * 3


def main():
    dev = json.load(open(ROOT / 'experiments/allocation_sweep_dev_v1_2026-09-15/summary_v2.json'))['summary']
    conf = json.load(open(ROOT / 'experiments/allocation_confirm_v1_2026-09-15/summary_v2.json'))['summary']

    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    for a, S, t, ms in ((ax[0], dev, '(a) development: 5 records, 75 windows', [8, 12, 16, 25, 33, 50]),
                        (ax[1], conf, '(b) confirmation: 4 records, 60 windows', [8, 12, 16, 19, 25, 28, 33, 50])):
        for k in (2, 3, 7):
            c, mk, ls = STYLE[k]
            x, y, lo, hi = [], [], [], []
            for m in ms:
                med, mn, mx = stat(S, k, m)
                if med is None:
                    continue
                x.append(k * m); y.append(med); lo.append(mn); hi.append(mx)
            a.plot(x, y, marker=mk, linestyle=ls, color=c, label=f'$k$ = {k} views')
            if k != 7:
                a.fill_between(x, lo, hi, color=c, alpha=.12, lw=0)
        a.set_xscale('log'); a.set_yscale('log')
        # Explicit ticks with the minor ones off. On a log axis matplotlib
        # otherwise labels the minor ticks too, and at this figure width those
        # labels collide with each other.
        a.set_xticks([20, 50, 100, 200, 400])
        a.set_xticklabels(['20', '50', '100', '200', '400'])
        a.set_yticks([10, 20, 50, 100, 200, 500])
        a.set_yticklabels(['10', '20', '50', '100', '200', '500'])
        a.minorticks_off()
        a.set_xlabel(r'camera frames per window, $B = k \times m$')
        a.set_ylabel('path-length MAE (mm)')
        a.set_title(t); a.legend(frameon=False)
    fig.tight_layout(pad=.3); fig.savefig(OUT / 'fig1.pdf'); fig.savefig(OUT / 'fig1.png', dpi=400); plt.close(fig)

    fig, ax = plt.subplots(figsize=(W1, 2.7))
    marks = ['o', 's', '^', 'D', 'v']; lss = ['-', '--', ':', '-.', (0, (3, 1, 1, 1))]
    for i, m in enumerate((16, 19, 25, 28, 33)):
        y, e = [], []
        for k in (2, 3, 7):
            med, mn, mx = stat(conf, k, m)
            y.append(med); e.append([med - mn, mx - med])
        ax.errorbar([2, 3, 7], y, yerr=np.array(e).T, marker=marks[i], linestyle=lss[i],
                    capsize=2, lw=1.0, label=f'$m$ = {m}')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks([2, 3, 7]); ax.set_xticklabels(['2', '3', '7'])
    plain_log_ticks(ax.yaxis, [10, 20, 50, 100]); ax.minorticks_off()
    ax.set_xlabel('views per instant, $k$'); ax.set_ylabel('path-length MAE (mm)')
    ax.legend(frameon=False, ncol=2, columnspacing=.8)
    fig.tight_layout(pad=.3); fig.savefig(OUT / 'fig2.pdf'); fig.savefig(OUT / 'fig2.png', dpi=400); plt.close(fig)

    ad = json.load(open(ROOT / 'experiments/adaptive_moments_dev_v1_2026-09-15/summary.json'))
    fig, ax = plt.subplots(figsize=(W1, 2.7))
    rows = sorted([s for s in ad if s['views'] == 3 and s['estimator'] == 'state_mean'],
                  key=lambda s: s['moments'])
    x = [s['moments'] for s in rows]
    ax.plot(x, [s['uniform_mae'] * 1000 for s in rows], marker='o', linestyle='-', color='#333333', label='uniform')
    ax.plot(x, [s['adaptive_mae'] * 1000 for s in rows], marker='s', linestyle='--', color='#d62728', label='curvature density')
    ax.plot(x, [s['oracle_nonuniform_mae'] * 1000 for s in rows], marker='^', linestyle=':', color='#1f77b4', label='oracle placement')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks(x); ax.set_xticklabels([str(v) for v in x])
    plain_log_ticks(ax.yaxis, [10, 20, 50, 100, 200, 500]); ax.minorticks_off()
    ax.set_xlabel('time instants, $m$ (three views)')
    ax.set_ylabel('path-length MAE (mm)'); ax.legend(frameon=False)
    fig.tight_layout(pad=.3); fig.savefig(OUT / 'fig3.pdf'); fig.savefig(OUT / 'fig3.png', dpi=400); plt.close(fig)

    mc = json.load(open(ROOT / 'experiments/ctsd_baseline_dev_v1_2026-09-15/scored.json'))
    tu = json.load(open(ROOT / 'experiments/tum_dense_cheap_confirm_v1_2026-09-15/decision.json'))['decision']
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    M = [8, 12, 16, 25, 33, 50, 100, 200]
    for lab, fn, mk, ls in (('mean over 75 windows', np.mean, 'o', '-'),
                            ('median over 75 windows', np.median, 's', '--')):
        y = [fn(np.array([abs(r['error_m']) for r in mc if r['views'] == 3 and r['moments'] == m
                          and r['method'] == 'state_mean_calibrated' and r['error_m'] is not None]) * 1000)
             for m in M]
        ax[0].plot([3 * m for m in M], y, marker=mk, linestyle=ls, color='#333333' if mk == 'o' else '#d62728', label=lab)
    ax[0].set_xscale('log'); ax[0].set_yscale('log')
    ax[0].set_xticks([24, 48, 99, 150, 300, 600])
    ax[0].set_xticklabels(['24', '48', '99', '150', '300', '600'])
    plain_log_ticks(ax[0].yaxis, [5, 10, 20, 50, 100, 200]); ax[0].minorticks_off()
    ax[0].set_xlabel('camera frames per window'); ax[0].set_ylabel('path-length error (mm)')
    ax[0].set_title('(a) rise appears only in the mean'); ax[0].legend(frameon=False)
    for fe, c, mk, ls in (('orb', '#1f77b4', 'o', '-'), ('xfeat', '#2ca02c', 's', '--')):
        pb = tu[fe]['per_budget']; x = sorted(int(b) for b in pb)
        name = 'binary descriptor' if fe == 'orb' else 'learned descriptor'
        ax[1].plot(x, [pb[str(b)]['mae_seq_equal'] * 1000 for b in x], marker=mk, linestyle=ls, color=c, label=name)
        ax[1].plot(x, [abs(pb[str(b)]['bias']) * 1000 for b in x], linestyle=(0, (1, 1)), color=c,
                   alpha=.55, lw=.8, label=f'{name}, |bias|')
    ax[1].set_xscale('log'); ax[1].set_yscale('log')
    ax[1].set_xticks([8, 16, 32, 64]); ax[1].set_xticklabels(['8', '16', '32', '64'])
    # The curves fill the panel from lower left to upper right, so there is no
    # empty corner for a four-entry legend. Extend the axis upwards and put
    # the legend in the space that creates; nothing is drawn above 200 mm.
    ax[1].set_ylim(1.3, 1500)
    plain_log_ticks(ax[1].yaxis, [2, 5, 10, 20, 50, 100, 200, 500]); ax[1].minorticks_off()
    ax[1].set_xlabel('frames per window'); ax[1].set_ylabel('path-length MAE (mm)')
    ax[1].set_title('(b) single-camera chain, 14 unseen windows')
    ax[1].legend(frameon=False, loc='upper left')
    fig.tight_layout(pad=.3); fig.savefig(OUT / 'fig4.pdf'); fig.savefig(OUT / 'fig4.png', dpi=400); plt.close(fig)

    for p in sorted(OUT.glob('*.pdf')):
        print(f'  {p.stat().st_size/1024:7.1f} KB  {p.name}')


if __name__ == '__main__':
    main()
