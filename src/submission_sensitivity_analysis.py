"""Exploratory sensitivity analyses; never modify the frozen confirmation artefacts.

Run after inspecting the confirmation outcomes. No new preregistration or fresh
holdout is claimed. All aggregation uses explicit units and weights.
"""
import argparse
import hashlib
import gzip
import itertools
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/submission_sensitivity_2026-09-16'
RULE = ('216f21c1', '3e0f8f0', '44c4b2e')


def read(rel):
    path = ROOT / rel
    if path.exists():
        return json.loads(path.read_text())
    with gzip.open(str(path)+'.gz', 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def save(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2), encoding='utf-8')


def input_digest(path):
    if path.exists():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    with gzip.open(str(path)+'.gz', 'rb') as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def summaries():
    rows = read('experiments/allocation_confirm_v1_2026-09-15/scored.json')
    all7 = next(r['cameras'] for r in rows if r['views'] == 7)
    def errors(cam, m):
        return {(r['record'], r['start']): r['error_m'] * 1000 for r in rows
                if r['cameras'] == cam and r['moments'] == m
                and r['method'] == 'state_mean' and r['error_m'] is not None}
    comparisons = []
    for m3, m7 in [(16,16), (19,19), (25,25), (28,28), (33,33), (28,12)]:
        a, b = errors('+'.join(RULE),m3), errors(all7,m7)
        common = sorted(a.keys() & b.keys())
        recs = sorted({k[0] for k in common})
        means = [np.mean([abs(a[k])-abs(b[k]) for k in common if k[0]==rec]) for rec in recs]
        # Exhaustive resampling of the four records, retaining all windows.
        boot = [np.mean([means[i] for i in inds]) for inds in itertools.product(range(len(recs)),repeat=len(recs))]
        comparisons.append(dict(m3=m3,m7=m7,frames3=3*m3,frames7=7*m7,windows=len(common),
            mae3_mm=float(np.mean([abs(a[k]) for k in common])),mae7_mm=float(np.mean([abs(b[k]) for k in common])),
            per_record_difference_mm=dict(zip(recs,map(float,means))),
            difference_mm=float(np.mean(means)),cluster_percentile95_mm=np.quantile(boot,[.025,.975]).tolist()))
    save('paired_records.json',dict(status='exploratory; four record clusters, not an equivalence test',comparisons=comparisons))
    tum = read('experiments/tum_dense_cheap_confirm_v1_2026-09-15/evaluation.json')
    result = {}
    for fe in ('orb','xfeat'):
        result[fe] = {'per_budget':{}}
        for b in ('8','16','32','64'):
            subset = [r for r in tum if r['method']==fe and r['budget_label']==b and r['errors_m']]
            seqs = sorted({r['sequence'] for r in subset})
            byseq = {s:np.array([r['errors_m']['state_mean'] for r in subset if r['sequence']==s]) for s in seqs}
            mae = float(np.mean([np.mean(abs(v)) for v in byseq.values()]))
            bias = float(np.mean([np.mean(v) for v in byseq.values()]))
            assert abs(bias) <= mae + 1e-12
            result[fe]['per_budget'][b] = dict(mae_seq_equal=mae,bias_seq_equal=bias,
                n_by_sequence={s:len(v) for s,v in byseq.items()})
    save('tum_sequence_equal.json',result)
    gate = read('experiments/allocation_gate_dev_v1_2026-09-15/scored.json')
    selected = [r for r in gate if r['views']==3 and r['moments']==200]
    arms = {(r['record'],r['start'],r['cameras']) for r in selected}
    success = [r for r in selected if r['method']=='limited' and r['error_m'] is not None]
    gated = { (r['record'],r['start'],r['cameras']):r.get('gated',0) for r in selected }
    save('gate_completion.json',dict(window_arms=len(arms),completed=len(success),
        failed=len(arms)-len(success),gated_observations=sum(gated.values()),
        window_arms_with_temporal_rejections=sum(v>0 for v in gated.values()),
        note='Combined view-consensus and temporal gate; failures remain in completion denominator.'))
    print(json.dumps(dict(paired=comparisons,tum=result,gate=read('experiments/submission_sensitivity_2026-09-16/gate_completion.json')),indent=1))


def geometry():
    from audit_mcalib_sample import read as read_pickle
    from prepare_mcalib_cache import load_record
    from run_allocation_sweep import subset_points
    from robust_triangulation_cpu import robust_triangulate
    from continuous_motion_cpu import estimate
    from view_geometry_analysis import jacobian
    cams = read('experiments/mcalib_3d_cache_2026-09-13_verified/public/calibration.json')['cameras']
    ids = sorted(cams)
    rest = [c for c in ids if c not in RULE]
    data = ROOT / 'data_external/mcalib_2026-09-13/verified/mocapAssistedRecord'
    dev = read('experiments/allocation_sweep_dev_v1_2026-09-15/scored.json')
    lookup = {(r['record'],r['start'],r['cameras'],r['moments']):r for r in dev if r['method']=='state_mean'}
    rng = np.random.default_rng(20260916)
    records, robust_rows = {}, []
    for rec in [f'record{i}' for i in range(19,24)]:
        t, truth, uv, _ = load_record(rec,ids)
        p3,_ = subset_points(uv,cams,RULE,3000)
        p7,_ = subset_points(uv,cams,ids,3000)
        p4,_ = subset_points(uv,cams,rest,3000)
        def cosine(a,b):
            ok = np.all(np.isfinite(a),axis=1)&np.all(np.isfinite(b),axis=1)
            a,b = a[ok],b[ok]
            return float(np.median(np.sum(a*b,axis=1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-15)))
        e3,e7,e4 = p3-truth[:3000],p7-truth[:3000],p4-truth[:3000]
        valid = np.all(np.isfinite(e3),axis=1)
        pair = valid[1:]&valid[:-1]
        rms = np.sqrt(np.mean(np.sum(e3[valid]**2,axis=1)))
        diff = np.sqrt(np.mean(np.sum((e3[1:][pair]-e3[:-1][pair])**2,axis=1)))
        # i.i.d. unit pixel noise under the same linearized least-squares geometry.
        # Shared cameras reuse the same simulated noise. The reference is exact.
        sim3,sim7,sim4 = [],[],[]
        for x in truth[:3000:40]:
            js = [jacobian(x,cams[c]) for c in ids]
            if any(j is None for j in js): continue
            noise = rng.normal(size=(100,len(ids),2))
            for sub,dest in [(RULE,sim3),(ids,sim7),(rest,sim4)]:
                inds=[ids.index(c) for c in sub];J=np.vstack([js[i] for i in inds])
                err=noise[:,inds,:].reshape(100,-1)@np.linalg.pinv(J).T
                dest.extend(err)
        records[rec] = dict(nested_cosine=cosine(e3,e7),disjoint_cosine=cosine(e3,e4),
            iid_nested_cosine=cosine(np.array(sim3),np.array(sim7)),iid_disjoint_cosine=cosine(np.array(sim3),np.array(sim4)),
            position_rms3_mm=float(rms*1000),increment_rms3_mm=float(diff*1000),
            increment_over_position=float(diff/rms))
        for start in range(0,3000,200):
            sel=start+np.rint(np.linspace(0,199,33)).astype(int)
            points=[];rejected=0
            for idx in sel:
                try: p,_=robust_triangulate({c:uv[c][idx] for c in ids},cams,threshold_px=4.)
                except ValueError: p=np.full(3,np.nan);rejected+=1
                points.append(p)
            p=np.array(points);ok=np.all(np.isfinite(p),axis=1)
            base=lookup[(rec,start,'+'.join(ids),33)]
            failure=not (ok[0] and ok[-1] and ok.sum()>=2)
            val=None if failure else estimate(p[ok],t[sel][ok])['mean_path_m']
            robust_rows.append(dict(record=rec,start=start,rejected_instants=rejected,
                error_m=None if val is None else val-base['truth_m'],
                dlt7_error_m=base['error_m'],dlt3_error_m=lookup[(rec,start,'+'.join(RULE),33)]['error_m']))
        print(rec,records[rec],flush=True)
    ok=[r for r in robust_rows if r['error_m'] is not None and r['dlt7_error_m'] is not None and r['dlt3_error_m'] is not None]
    summary=dict(windows=len(robust_rows),completed=len(ok),rejected_instants=sum(r['rejected_instants'] for r in robust_rows),
        mae_mm={k:float(np.mean([abs(r[k])*1000 for r in ok])) for k in ['error_m','dlt7_error_m','dlt3_error_m']},
        per_record_mae_mm={rec:{k:float(np.mean([abs(r[k])*1000 for r in ok if r['record']==rec])) for k in ['error_m','dlt7_error_m','dlt3_error_m']} for rec in records})
    save('geometry_controls.json',dict(status='exploratory development analysis; fixed existing 4-pixel consensus rule; no temporal gate',
        records=records,robust_m33=summary,robust_rows=robust_rows))
    print(json.dumps(summary,indent=1))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--geometry',action='store_true');a=p.parse_args()
    summaries()
    if a.geometry: geometry()
    save('provenance.json',dict(status='exploratory analysis after confirmation outcomes were examined',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256={str(p.relative_to(ROOT)):input_digest(p) for p in [
            ROOT/'experiments/allocation_confirm_v1_2026-09-15/scored.json',
            ROOT/'experiments/tum_dense_cheap_confirm_v1_2026-09-15/evaluation.json',
            ROOT/'experiments/tum_dense_cheap_confirm_v1_2026-09-15/config.json']}))
