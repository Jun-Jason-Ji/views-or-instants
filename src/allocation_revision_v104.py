"""Retrospective grid and robustness checks; no fresh holdout or preregistration.

Settings are written before execution. Original archived experiments stay intact.
Run this script, then allocation_revision_v104_report.py.
"""
import concurrent.futures
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/allocation_revision_v104_2026-09-16'
BUDGETS = (84, 168)
ARMS = ('dlt', 'spatial', 'spatial_temporal')
PROTOCOL = {
    'status': 'Exploratory retrospective checks on previously examined development and evaluation records.',
    'development': list(range(19, 24)), 'evaluation': list(range(15, 19)),
    'budgets': BUDGETS, 'views': (3, 7), 'sigma_m': .003,
    'grids': 'Uniform; internal offsets -0.4,-0.2,+0.2,+0.4 of one ideal interval; six independent seeded jitters uniform in +/-0.4 interval. Endpoints fixed. No outcome-based grid selection.',
    'seeds': list(range(10401, 10407)), 'arms': ARMS,
    'spatial': 'Existing three_view_cpu (4 px, 5 degrees, 20 mm pair spread) with unchanged two-visible-view DLT fallback; seven-view robust_triangulate (4 px, >=4 and majority support).',
    'temporal': 'Existing gate, sigma=3 mm, NIS=16.27, q from ungated innovation likelihood; first two valid observations retained; endpoint loss fails window.',
    'subset_rule': 'Historical subset_score is MINIMUM over pairs of MAXIMUM depth/focal-length scale divided by ray sine, not a worst-pair score. Equal scores use lexicographic camera order.',
    'selection': 'Retrospective development-only selection among uniform-grid 3/7-view, DLT/spatial/spatial_temporal schedules with cost<=cap (including cheaper 84-frame choices at 168). Eligible only if all 75 development windows complete and empirical p95<=100 mm; minimize MAE, then cost, views, arm name. No eligible candidate means abstain. Evaluate chosen candidate unchanged on records15-18. No unseen-data claim.',
}


def grids(m):
    x = np.linspace(0, 199, m)
    step = 199 / (m - 1)
    out = {'uniform': np.rint(x).astype(int)}
    for shift in (-.4, -.2, .2, .4):
        z = x.copy(); z[1:-1] += shift * step
        out[f'phase{shift:+.1f}'] = np.rint(z).astype(int)
    for seed in PROTOCOL['seeds']:
        z = x.copy()
        z[1:-1] += np.random.default_rng(seed).uniform(-.4, .4, m-2) * step
        out[f'jitter{seed}'] = np.rint(z).astype(int)
    for z in out.values():
        assert z[0] == 0 and z[-1] == 199 and len(z) == m and np.all(np.diff(z) > 0)
    return out


def record_job(number):
    from analyze_allocation import scene_center, subset_score
    from prepare_mcalib_cache import load_record
    from run_allocation_sweep import subset_points
    from run_allocation_gate import subset_points as spatial_points
    from continuous_motion_cpu import estimate
    from gated_state_cpu import gate
    rec = f'record{number}'
    cams = json.loads((ROOT/'experiments/mcalib_3d_cache_2026-09-13_verified/public/calibration.json').read_text())['cameras']
    ids = sorted(cams); center = scene_center(cams)
    subsets = {k: min(itertools.combinations(ids, k), key=lambda s: subset_score(cams, s, center)) for k in (3, 7)}
    t, xyz, uv, dense = load_record(rec, ids)
    rows = []
    for k, sub in subsets.items():
        points, _ = subset_points(uv, cams, sub, 3000)
        spatial, _ = spatial_points(uv, cams, sub, 3000)
        for budget in BUDGETS:
            m = budget // k
            for grid, indices in grids(m).items():
                for start in range(0, 3000, 200):
                    sel = start + indices
                    middle = dense[(dense[:,1] > t[start]) & (dense[:,1] < t[start+199]), 2:5]
                    reference = float(np.linalg.norm(np.diff(np.vstack([xyz[start], middle, xyz[start+199]]),axis=0),axis=1).sum())
                    missing = int((~np.isfinite(points[sel]).all(axis=1)).sum())
                    for arm in ARMS:
                        p = (points if arm == 'dlt' else spatial)[sel]
                        ok = np.isfinite(p).all(axis=1)
                        spatial_rejected = int((np.isfinite(points[sel]).all(axis=1) & ~ok).sum())
                        temporal_rejected = 0; failure = None; value = None
                        if not (ok[0] and ok[-1] and ok.sum() >= 2):
                            failure = 'spatial_endpoint_or_too_few'
                        else:
                            q, qt = p[ok], t[sel][ok]
                            est = estimate(q, qt, sigma=.003)
                            if arm == 'spatial_temporal':
                                accepted, log = gate(q, qt, sigma=.003, nis_threshold=16.27, q=est['q'])
                                temporal_rejected = len(log)
                                if accepted[-1] != len(q)-1:
                                    failure = 'temporal_endpoint'
                                elif len(log):
                                    est = estimate(q[accepted], qt[accepted], sigma=.003)
                            if failure is None:
                                value = est['mean_path_m']
                        rows.append(dict(record=rec, split='development' if number>=19 else 'evaluation',
                            start=start, views=k, cameras=list(sub), budget=budget, moments=m, frames=k*m,
                            grid=grid, arm=arm, selected_indices=indices.tolist(), missing=missing,
                            spatial_rejected=spatial_rejected, temporal_rejected=temporal_rejected,
                            value_m=value, reference_m=reference,
                            error_mm=None if value is None else 1000*(value-reference), failure=failure))
    (OUT/f'{rec}.json').write_text(json.dumps(rows,separators=(',',':')),encoding='utf-8')
    return rec, len(rows)


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    path = OUT/'protocol.json'
    if not path.exists():
        path.write_text(json.dumps(dict(PROTOCOL, recorded_utc=datetime.now(timezone.utc).isoformat()),indent=2),encoding='utf-8')
    else:
        old=json.loads(path.read_text()); old.pop('recorded_utc')
        assert json.loads(json.dumps(PROTOCOL)) == old, 'Do not modify an executed protocol'
    tracked = ['allocation_revision_v104.py','analyze_allocation.py','run_allocation_sweep.py',
               'run_allocation_gate.py','three_view_cpu.py','robust_triangulation_cpu.py',
               'gated_state_cpu.py','continuous_motion_cpu.py','prepare_mcalib_cache.py']
    if not (OUT/'provenance.json').exists():
        import shutil
        (OUT/'source_snapshot').mkdir(exist_ok=True)
        for name in tracked:
            shutil.copy2(ROOT/'src'/name, OUT/'source_snapshot'/name)
        (OUT/'provenance.json').write_text(json.dumps({'sha256':{name:hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest() for name in tracked}},indent=2))
    todo=[i for i in list(range(19,24))+list(range(15,19)) if not (OUT/f'record{i}.json').exists()]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(record_job, todo):
            print(result, flush=True)


if __name__ == '__main__':
    main()
