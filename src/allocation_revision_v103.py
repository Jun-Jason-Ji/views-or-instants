"""Exploratory robustness checks after inspecting all previous results.

Never alters frozen files. Run from the project with the original public data
installed. Budget grid and sensitivity values are fixed here, not optimized
against confirmation errors. Cached original estimates are checked explicitly.
"""
import concurrent.futures
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/allocation_revision_v103_2026-09-16'
BUDGETS = (56, 84, 112, 168)
SIGMAS = (.0004, .001, .003)


def read(rel):
    import gzip
    p = ROOT / rel
    if p.exists():
        return json.loads(p.read_text(encoding='utf-8'))
    with gzip.open(str(p)+'.gz', 'rt', encoding='utf-8') as f:
        return json.load(f)


def record_job(rec):
    from analyze_allocation import scene_center, subset_score
    from prepare_mcalib_cache import load_record
    from run_allocation_sweep import subset_points
    from continuous_motion_cpu import estimate
    cams = read('experiments/mcalib_3d_cache_2026-09-13_verified/public/calibration.json')['cameras']
    ids = sorted(cams)
    center = scene_center(cams)
    subsets = {k: min(itertools.combinations(ids, k), key=lambda s: subset_score(cams, s, center))
               for k in range(2, 8)}
    t, xyz, uv, dense = load_record(rec, ids)
    refs = {}
    for start in range(0, 3000, 200):
        middle = dense[(dense[:, 1] > t[start]) & (dense[:, 1] < t[start+199]), 2:5]
        refs[start] = {stride: float(np.linalg.norm(np.diff(np.vstack([xyz[start], middle[::stride], xyz[start+199]]), axis=0), axis=1).sum())
                       for stride in (1, 2, 4)}
    rows = []
    for k, sub in subsets.items():
        pts, _ = subset_points(uv, cams, sub, 3000)
        for budget in BUDGETS:
            m = budget // k
            for sigma in (SIGMAS if budget == 84 and k in (3, 7) else (.003,)):
                for start in range(0, 3000, 200):
                    sel = start + np.rint(np.linspace(0, 199, m)).astype(int)
                    p = pts[sel]
                    ok = np.all(np.isfinite(p), axis=1)
                    failure = not (ok[0] and ok[-1] and ok.sum() >= 2)
                    val = None if failure else estimate(p[ok], t[sel][ok], sigma=sigma)['mean_path_m']
                    rows.append(dict(record=rec, start=start, views=k, cameras='+'.join(sub), budget=budget,
                                     frames=k*m, moments=m, sigma_mm=sigma*1000,
                                     missing=int((~ok).sum()), value_m=val,
                                     reference_m=refs[start],
                                     error_mm=None if val is None else 1000*(val-refs[start][1])))
    return rows


def summarize(rows):
    summaries = []
    for budget, k, sigma in sorted({(r['budget'], r['views'], r['sigma_mm']) for r in rows}):
        rr = [r for r in rows if (r['budget'], r['views'], r['sigma_mm']) == (budget, k, sigma)]
        good = [r for r in rr if r['error_mm'] is not None]
        errors = np.array([abs(r['error_mm']) for r in good])
        summaries.append(dict(budget=budget, views=k, sigma_mm=sigma, moments=rr[0]['moments'], frames=rr[0]['frames'],
                              completed=len(good), attempted=len(rr), mae_mm=float(errors.mean()),
                              p95_mm=float(np.quantile(errors,.95)), maximum_mm=float(errors.max()),
                              over100mm=int((errors>100).sum()),
                              per_record_mae_mm={rec:float(np.mean([abs(r['error_mm']) for r in good if r['record']==rec]))
                                                 for rec in sorted({r['record'] for r in rr})}))
    paired = []
    for budget in BUDGETS:
        for sigma in (SIGMAS if budget == 84 else (.003,)):
            a = {(r['record'], r['start']):r for r in rows if r['budget']==budget and r['views']==3 and r['sigma_mm']==sigma*1000 and r['value_m'] is not None}
            b = {(r['record'], r['start']):r for r in rows if r['budget']==budget and r['views']==7 and r['sigma_mm']==sigma*1000 and r['value_m'] is not None}
            common = sorted(a.keys() & b.keys())
            for stride in ((1,2,4) if budget == 84 else (1,)):
                err = lambda r:abs(1000*(r['value_m']-r['reference_m'][stride]))
                byrec = {rec:float(np.mean([err(a[x])-err(b[x]) for x in common if x[0]==rec])) for rec in sorted({x[0] for x in common})}
                vals = list(byrec.values())
                loo = {rec:float(np.mean([v for other,v in byrec.items() if other!=rec])) for rec in byrec}
                paired.append(dict(budget=budget,sigma_mm=sigma*1000,reference_stride=stride,common=len(common),
                                   mae3_mm=float(np.mean([err(a[x]) for x in common])),mae7_mm=float(np.mean([err(b[x]) for x in common])),
                                   difference_mm=float(np.mean(vals)),per_record_difference_mm=byrec,leave_one_record_out_mm=loo))
    return summaries, paired


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for rec, result in zip(range(15,19), pool.map(record_job, [f'record{i}' for i in range(15,19)])):
            rows.extend(result)
            print('Completed record', rec, len(result), 'window configurations', flush=True)
    summary, paired = summarize(rows)
    old = read('experiments/allocation_confirm_v1_2026-09-15/scored.json')
    index = {(r['record'],r['start'],r['cameras'],r['moments']):r for r in old if r['method']=='state_mean' and r['value_m'] is not None}
    differences = [abs(r['value_m']-index[(r['record'],r['start'],r['cameras'],r['moments'])]['value_m'])
                   for r in rows if r['sigma_mm']==3 and r['value_m'] is not None and (r['record'],r['start'],r['cameras'],r['moments']) in index]
    assert differences and max(differences)<1e-9, 'Original-estimate replication failed'
    adaptive = read('experiments/adaptive_moments_dev_v1_2026-09-15/scored.json')
    adaptive_summary = read('experiments/adaptive_moments_dev_v1_2026-09-15/summary.json')
    completion = []
    for k,m in sorted({(r['views'],r['moments']) for r in adaptive}):
        counts = {arm:len({(r['record'],r['start']) for r in adaptive if r['views']==k and r['moments']==m and r['arm']==arm and r.get('estimator')=='state_mean' and r.get('error_m') is not None}) for arm in ('uniform','adaptive','oracle_nonuniform')}
        common = next(r['windows'] for r in adaptive_summary if r['views']==k and r['moments']==m and r['estimator']=='state_mean')
        completion.append(dict(views=k,moments=m,total=75,recorded_success=counts,common=common,
                               note='Sequential eligibility screens: absent downstream results may be unattempted, not independent method failures.'))
    result = dict(status='Exploratory analyses after previous outcomes; not a new holdout or preregistration.',
                  budgets=list(BUDGETS),sigma_mm=[s*1000 for s in SIGMAS],
                  selection_rule='Calibration-only minimum worst-pair geometric score; extended to k=4,5,6 without reference outcomes.',
                  original_estimates_replicated=len(differences),max_replication_error_m=max(differences),
                  summary=summary,paired=paired,adaptive_completion=completion)
    (OUT/'rows.json').write_text(json.dumps(rows,indent=1),encoding='utf-8')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    tracked=[Path(__file__), ROOT/'src/continuous_motion_cpu.py',ROOT/'src/analyze_allocation.py',
             ROOT/'experiments/mcalib_3d_cache_2026-09-13_verified/public/calibration.json']
    (OUT/'provenance.json').write_text(json.dumps({'status':result['status'],'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}},indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=1),flush=True)


if __name__ == '__main__':
    main()
