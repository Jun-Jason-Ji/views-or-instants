# -*- coding: utf-8 -*-
"""Test the closed-form prediction m* = T * sqrt(kappa * v^2 / (7 sigma_a)).

Derivation: per-window corner-cutting deficit  D(m) = T^3 kappa^2 v^3 / (24 m^2);
noise inflation of a polyline with per-axis noise sigma_a, I(m) = 2 sigma_a^2 m^2 / L
with L = v T.  Setting D = I gives m*^4 = T^4 kappa^2 v^4 / (48 sigma_a^2).

kappa is taken from the REFERENCE trajectory (median over the window of
|r' x r''| / |r'|^3, on a lightly smoothed trace), v from the reference path
length, sigma_a from the per-instant 3-D error already measured for each rig.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dpjait_record import Record          # noqa: E402
from dpjait_sim_record import SimRecord   # noqa: E402

W = 100  # video frames per window


def smooth(x, k=5):
    if len(x) < k:
        return x
    ker = np.ones(k) / k
    return np.stack([np.convolve(x[:, i], ker, mode='same') for i in range(x.shape[1])], axis=1)


def curvature_median(P, dt):
    """median |kappa| over a trajectory segment, 1/m."""
    P = smooth(np.asarray(P, float))
    if len(P) < 5:
        return np.nan
    d1 = np.gradient(P, dt, axis=0)
    d2 = np.gradient(d1, dt, axis=0)
    num = np.linalg.norm(np.cross(d1, d2), axis=1)
    den = np.linalg.norm(d1, axis=1) ** 3
    ok = den > 1e-9
    if ok.sum() < 5:
        return np.nan
    k = num[ok] / den[ok]
    return float(np.median(k[2:-2])) if len(k) > 4 else float(np.median(k))


def rig_rows(name, rec, ref_hz, windows):
    """per window: path length (m), median curvature (1/m)."""
    out = []
    for w in windows:
        f0, f1 = w * W, (w + 1) * W
        for dr in sorted(rec.reference):
            try:
                L = rec.reference_path_length(dr, f0, f1)
            except Exception:
                continue
            if not np.isfinite(L) or L <= 0:
                continue
            per = int(round(ref_hz / 25.0))     # reference samples per video frame
            seg = rec.reference[dr][f0 * per: f1 * per + 1]
            seg = seg[~np.isnan(seg).any(axis=1)]
            k = curvature_median(seg, 1.0 / ref_hz)
            if np.isfinite(k):
                out.append((L, k))
    return out


def predict(T, L, kappa, sigma_a):
    v = L / T
    return T * np.sqrt(kappa * v * v / (7.0 * sigma_a))


def main():
    T = 4.0
    print('%-10s %7s %7s %8s %8s %6s %6s' % ('record', 'L(m)', 'v(m/s)', 'kappa', 'sig_a(mm)', 'pred', 'obs'))

    real_dir = ROOT / 'data_external/dpjait_2026-09-19/Real_Data'
    sim_dir = ROOT / 'data_external/dpjait_2026-09-19/Simulated_Data'
    es_dir = ROOT / 'experiments/dpjait_prereg_v0.1/error_structure'
    obs = {'R05_D1': 6, 'R06_D1': 4, 'R07_D1': 4, 'R08_D1': 6, 'R09_D1': 10,
           'R10_D1': 10, 'R11_D1': 8, 'R16_D1_A': 6,
           'S08_D8': 25, 'S09_D6': 17, 'S10_D6': 6}
    # per-axis noise: real rigs from the error-structure "spread" (3-D rms), sim from
    # the post-hoc diagnostic (median 3-D error); both divided by sqrt(3).
    sim_noise = {r['record']: r['err3d_mm_median']
                 for r in json.loads((ROOT / 'experiments/dpjait_prereg_v0.1/sim_obs_noise.json')
                                     .read_text())['records']}

    rows = []
    for name in obs:
        is_sim = name.startswith('S')
        folder = (sim_dir if is_sim else real_dir) / name
        if not folder.exists():
            cand = [p for p in (sim_dir if is_sim else real_dir).iterdir() if p.name == name]
            if not cand:
                print('%-10s  (folder not found)' % name)
                continue
            folder = cand[0]
        rec = SimRecord(folder) if is_sim else Record(folder)
        ref_hz = 25.0 if is_sim else 100.0
        nwin = rec.n_video_frames // W
        wins = list(range(min(nwin, 40)))
        data = rig_rows(name, rec, ref_hz, wins)
        if not data:
            print('%-10s  (no reference windows)' % name)
            continue
        L = float(np.median([d[0] for d in data]))
        kap = float(np.median([d[1] for d in data]))
        if is_sim:
            sig = sim_noise[name] / np.sqrt(3.0) / 1000.0
        else:
            es = json.loads((es_dir / (name + '.json')).read_text())
            sig = float(es.get('spread_mm', es.get('scatter_mm', np.nan))) / np.sqrt(3.0) / 1000.0
        p = predict(T, L, kap, sig)
        rows.append((name, L, L / T, kap, sig * 1000, p, obs[name]))
        print('%-10s %7.2f %7.2f %8.3f %8.1f %6.1f %6d'
              % (name, L, L / T, kap, sig * 1000, p, obs[name]))

    if rows:
        pr = np.array([r[5] for r in rows])
        ob = np.array([r[6] for r in rows], float)
        ratio = pr / ob
        print('\npredicted/observed: median %.2f  range %.2f-%.2f' %
              (np.median(ratio), ratio.min(), ratio.max()))


if __name__ == '__main__':
    main()
