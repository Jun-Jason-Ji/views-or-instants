"""Pre-registered confirmation of dense few-view allocation on MCalib record15-18.

freeze : write config + source hashes before any confirmation record is read.
predict: compute all predictions, seal them (sha256) before reading any reference.
score  : read reference, evaluate pre-declared gates, write decision.

Records 15-18 were previously used only for per-frame triangulation QA
(robust_triangulation_2026-09-13); no path/sampling design used them.
"""
import json,argparse,hashlib,shutil
from datetime import datetime,timezone
from pathlib import Path
from time import perf_counter
import numpy as np
from audit_mcalib_sample import read
from prepare_mcalib_cache import load_record
from limited_arc_cpu import limited_arc
from continuous_motion_cpu import estimate as state_estimate
from run_allocation_sweep import subset_points,polygon
from analyze_allocation import scene_center,subset_score
from itertools import combinations

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data_external/mcalib_2026-09-13/verified/mocapAssistedRecord'
CAL=ROOT/'experiments/mcalib_3d_cache_2026-09-13_verified/public/calibration.json';OUT=ROOT/'experiments/allocation_confirm_v1_2026-09-15'
SOURCES=['src/run_allocation_confirm.py','src/run_allocation_sweep.py','src/analyze_allocation.py','src/limited_arc_cpu.py','src/continuous_motion_cpu.py','src/mcalib_cache_observer.py','src/robust_triangulation_cpu.py','src/audit_mcalib_sample.py','src/prepare_mcalib_cache.py','tests/test_bearing_fusion.py','experiments/mcalib_3d_cache_2026-09-13_verified/public/calibration.json']


def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rule_subsets(cams):
    """Calibration-only: for k=2 and k=3 choose the subset with the smallest predicted error scale."""
    center=scene_center(cams);out={}
    for k in (2,3):
        best=min(combinations(sorted(cams),k),key=lambda s:subset_score(cams,s,center));out[k]='+'.join(best)
    return out,center.tolist()


def freeze():
    OUT.mkdir(exist_ok=False);cams=json.loads(CAL.read_text())['cameras'];subs,center=rule_subsets(cams)
    cfg=dict(records=[f'record{i}' for i in range(15,19)],windows='start 0,200,...,2800; end=start+199 (15 per record, 60 total)',
        cost_unit='camera frame requests = views x moments; missing detections still charged',
        prior_protocol=dict(views=7,moments=8,frames=56,estimators=['state_mean','limited']),
        candidates=[dict(views=2,cameras=subs[2],moments=28,frames=56,estimator='state_mean'),dict(views=2,cameras=subs[2],moments=33,frames=66,estimator='state_mean'),
                    dict(views=3,cameras=subs[3],moments=19,frames=57,estimator='state_mean')],
        secondary=dict(all_pairs_and_triples=True,views=[2,3,7],moments=[8,12,16,19,25,28,33,50,100,200],estimators=['polygon','limited','state_mean(m<=50)'],estimator_choice='state_mean chosen on dev for m<=33; limited for m>33'),
        camera_rule='calibration-only scene center (closest point to all optical axes) then min over subsets of worst-pair z/(f*sin(angle))',
        scene_center_m=center,
        failure_policy='A moment with <2 visible views is dropped from the polyline; if either endpoint is missing the query fails. Failures are counted and reported, never dropped from denominators.',
        gates=dict(primary='Each candidate at equal-or-lower frames (56/57) vs prior 7x8 state_mean: MAE reduction >=50% overall, every record improves, failed queries <= prior + 0',
                   also='66-frame 2x33 candidate reported vs 7x8 (56) and vs 7x12 (84) state_mean; not part of the primary gate'),
        scope='Same rig, same session, publisher 2D caches, mocap reference; NOT end-to-end RGB, NOT an independent laboratory. Per-frame cost model only.',
        utc=datetime.now(timezone.utc).isoformat())
    save(OUT/'config.json',cfg);save(OUT/'freeze.json',dict(hashes={s:sha(ROOT/s) for s in SOURCES},utc=cfg['utc']))
    for s in SOURCES:
        d=OUT/'source_snapshot'/s;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/s,d)
    print(json.dumps(dict(subsets=subs,center=center)))


def verify():
    for s,h in json.loads((OUT/'freeze.json').read_text())['hashes'].items():assert sha(ROOT/s)==h,s


def predict():
    verify();cfg=json.loads((OUT/'config.json').read_text());cams=json.loads(CAL.read_text())['cameras'];ids=sorted(cams)
    subsets=[tuple(ids)]+list(combinations(ids,2))+list(combinations(ids,3))
    inputs={};rows=[];tick=perf_counter()
    for rec in cfg['records']:
        for c in ids:inputs[f'{rec}/centroidsUV{c}.pkl']=sha(DATA/rec/f'centroidsUV{c}.pkl')
        uv={c:read(DATA/rec/f'centroidsUV{c}.pkl') for c in ids};n=len(uv[ids[0]]);t=np.arange(n)*.02+.00006969+.000996/2
        for sub in subsets:
            pts,maxerr=subset_points(uv,cams,sub,n)
            for start in range(0,3000,200):
                for m in cfg['secondary']['moments']:
                    sel=start+np.rint(np.linspace(0,199,m)).astype(int);p=pts[sel];ok=np.all(np.isfinite(p),axis=1)
                    base=dict(record=rec,start=start,views=len(sub),cameras='+'.join(sub),moments=m,frames=len(sub)*m,missing=int((~ok).sum()))
                    if ok.sum()<2 or not ok[0] or not ok[-1]:rows.append(dict(base,method='polygon',value_m=None,failure='endpoint_or_too_few'));continue
                    q=p[ok];qt=t[sel][ok]
                    rows.append(dict(base,method='polygon',value_m=polygon(q)));rows.append(dict(base,method='limited',value_m=limited_arc(q,qt)))
                    if m<=50:rows.append(dict(base,method='state_mean',value_m=state_estimate(q,qt)['mean_path_m']))
        print(rec,'predicted',flush=True)
    save(OUT/'input_hashes.json',inputs);(OUT/'predictions.json').write_text(json.dumps(rows))
    save(OUT/'prediction_seal.json',dict(sha256=sha(OUT/'predictions.json'),rows=len(rows),predict_wall_s=perf_counter()-tick,utc=datetime.now(timezone.utc).isoformat()))


def score():
    verify();cfg=json.loads((OUT/'config.json').read_text());cams=json.loads(CAL.read_text())['cameras'];ids=sorted(cams)
    assert sha(OUT/'predictions.json')==json.loads((OUT/'prediction_seal.json').read_text())['sha256']
    rows=json.loads((OUT/'predictions.json').read_text());truth={};ref_utc=datetime.now(timezone.utc).isoformat()
    for rec in cfg['records']:
        t,xyz,_,dense=load_record(rec,ids)
        for start in range(0,3000,200):
            end=start+199;middle=dense[(dense[:,1]>t[start])&(dense[:,1]<t[end]),2:5];ref=np.vstack([xyz[start],middle,xyz[end]]);truth[rec,start]=float(np.linalg.norm(np.diff(ref,axis=0),axis=1).sum())
    for r in rows:r['truth_m']=truth[r['record'],r['start']];r['error_m']=None if r['value_m'] is None else r['value_m']-r['truth_m']
    (OUT/'scored.json').write_text(json.dumps(rows))
    def agg(views,cameras,m,method):
        rr=[r for r in rows if r['views']==views and r['moments']==m and r['method']==method and (cameras is None or r['cameras']==cameras)]
        ok=[r for r in rr if r['error_m'] is not None]
        return dict(n=len(rr),failed=len(rr)-len(ok),mae_m=float(np.mean([abs(r['error_m']) for r in ok])) if ok else None,
                    p95_m=float(np.quantile([abs(r['error_m']) for r in ok],.95)) if ok else None,bias_m=float(np.mean([r['error_m'] for r in ok])) if ok else None,
                    per_record={rec:float(np.mean([abs(r['error_m']) for r in ok if r['record']==rec])) for rec in cfg['records']},frames=views*m,cameras=cameras or 'all7')
    prior=agg(7,None,8,'state_mean');prior_lim=agg(7,None,8,'limited');prior12=agg(7,None,12,'state_mean');decision=dict(prior_7x8_state_mean=prior,prior_7x8_limited=prior_lim,prior_7x12_state_mean=prior12,candidates=[])
    for c in cfg['candidates']:
        a=agg(c['views'],c['cameras'],c['moments'],c['estimator']);gain=(prior['mae_m']-a['mae_m'])/prior['mae_m']
        every=all(a['per_record'][k]<prior['per_record'][k] for k in cfg['records'])
        passed=(c['frames']<=57) and gain>=.5 and every and a['failed']<=prior['failed']
        decision['candidates'].append(dict(candidate=c,result=a,relative_gain=gain,every_record_improves=every,primary_gate=passed if c['frames']<=57 else 'not_in_primary_gate'))
    summary=[]
    for key in sorted({(r['cameras'],r['moments'],r['method']) for r in rows}):
        rr=[r for r in rows if (r['cameras'],r['moments'],r['method'])==key];a=agg(rr[0]['views'],key[0],key[1],key[2]);a.update(moments=key[1],method=key[2]);summary.append(a)
    save(OUT/'decision.json',dict(reference_first_read_utc=ref_utc,decision=decision));save(OUT/'summary.json',summary)
    print(json.dumps(decision,indent=1)[:6000])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','predict','score']);a=p.parse_args();globals()[a.action]()
