"""Pre-registered TUM fr3 check on never-used windows (offsets >= 21 s): does the allocation optimum sit at LOW frame counts?

Hypothesis (from the allocation law, slow-motion / chained-noise regime): with the cheap CPU XFeat
front end, state_mean path MAE at B8 is lower than at B32 and B64, in every sequence, with no lower
completion. B16 vs B8 is reported as secondary. ORB is a secondary front end.
freeze -> run (sealed predictions) -> evaluate (GT read last).
"""
import json,argparse,hashlib,shutil
from datetime import datetime,timezone
from pathlib import Path
from time import perf_counter
import numpy as np
from run_tum_rgbd_cpu import timestamps
from run_fixed_observer_replication_cpu import associate,windows_for
from calibrated_measurement_core_v1 import validate_calibration
from run_tum_dense_cheap_v1 import query,make_model,K
from feature_chain_core import reference_metrics
from pose_propagation_cpu import interpolate_pose,pose_errors

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/tum_dense_cheap_confirm_v1_2026-09-15'
SEQS=['structure_texture_far','nostructure_texture_near_withloop','sitting_xyz']
SOURCES=['src/run_tum_dense_cheap_confirm.py','src/run_tum_dense_cheap_v1.py','src/calibrated_measurement_core_v1.py','src/rgbd_rigid_route_cpu.py','src/rgbd_odometry_cpu.py','src/continuous_motion_cpu.py','src/limited_arc_cpu.py','src/feature_chain_core.py','src/xfeat_cpu_adapter.py','src/run_fixed_observer_replication_cpu.py','src/run_tum_rgbd_cpu.py']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def plan():
    out={}
    for s in SEQS:
        pub=ROOT/f'data_external/tum_rgbd_fr3_{s}_2026-09-13/public';rgb=timestamps(pub/'rgb.txt');depth=timestamps(pub/'depth.txt')
        ws=[w for w in windows_for(rgb,associate(rgb,depth)) if w['offset']>=21];out[s]=ws
    return out


def freeze():
    OUT.mkdir(exist_ok=False);p=plan()
    cfg=dict(sequences=SEQS,windows={s:[dict(offset=w['offset'],frames=len(w['frames']),public_failure=w['public_failure']) for w in ws] for s,ws in p.items()},
        previously_used_offsets=[1,5,9,13,17],new_offsets='all valid windows with offset>=21 s from the frozen windows_for rule',
        frontends=dict(primary='xfeat',secondary='orb'),budgets=[8,16,32,64],K=K,
        primary_gate='xfeat state_mean sequence-equal MAE: B8 < B32 and B8 < B64; each sequence B8 <= B32; completed(B8) >= completed(B32). Windows failing in either arm are excluded from the paired MAE but counted.',
        secondary='B8 vs B16 (report only); orb same gates (report only)',
        scope='Same three fr3 sequences as before, new time windows; same rig/day; development chain frozen from tum_dense_cheap_v1',utc=datetime.now(timezone.utc).isoformat())
    (OUT/'config.json').write_text(json.dumps(cfg,indent=2));(OUT/'plan.json').write_text(json.dumps(p))
    (OUT/'freeze.json').write_text(json.dumps(dict(hashes={s:sha(ROOT/s) for s in SOURCES},utc=cfg['utc']),indent=1))
    for s in SOURCES:
        d=OUT/'source_snapshot'/s;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/s,d)
    print(json.dumps(cfg['windows']))


def verify():
    for s,h in json.loads((OUT/'freeze.json').read_text())['hashes'].items():assert sha(ROOT/s)==h,s


def run():
    verify();cfg=json.loads((OUT/'config.json').read_text());p=json.loads((OUT/'plan.json').read_text());matrix=validate_calibration(K);rows=[];tick=perf_counter()
    for fe in ('xfeat','orb'):
        model=make_model(fe)
        for s,ws in p.items():
            root=ROOT/f'data_external/tum_rgbd_fr3_{s}_2026-09-13/public'
            for w in ws:
                if w['public_failure']:continue
                for b in cfg['budgets']:
                    r=query(w,s,b,fe,model,root,matrix);r['budget_label']=str(b);rows.append(r);print(fe,s,w['offset'],b,'fail' if r['failure'] else 'ok',round(r['wall_seconds']['total'],1),flush=True)
    (OUT/'predictions.json').write_text(json.dumps(rows))
    (OUT/'prediction_seal.json').write_text(json.dumps(dict(sha256=sha(OUT/'predictions.json'),rows=len(rows),wall_s=perf_counter()-tick,utc=datetime.now(timezone.utc).isoformat())))


def evaluate():
    verify();rows=json.loads((OUT/'predictions.json').read_text());assert sha(OUT/'predictions.json')==json.loads((OUT/'prediction_seal.json').read_text())['sha256']
    ref_utc=datetime.now(timezone.utc).isoformat();gts={s:np.loadtxt(ROOT/f'data_external/tum_rgbd_fr3_{s}_2026-09-13/reference/groundtruth.txt') for s in SEQS}
    for r in rows:
        truth,inv=reference_metrics(gts[r['sequence']],r['times']);r['truth']=truth;r['reference_failure']=inv;r['errors_m']=None;r['pose_errors']=None
        if r['failure'] is None and inv is None:
            r['errors_m']={k:v-truth['displacement_m' if k=='displacement' else 'path_m'] for k,v in r['values_m'].items()}
            gp=np.array([interpolate_pose(gts[r['sequence']],t) for t in r['times']]);r['pose_errors']=pose_errors(np.array(r['poses']),np.linalg.inv(gp[0])@gp)
    (OUT/'evaluation.json').write_text(json.dumps(rows));dec={}
    for fe in ('xfeat','orb'):
        arms={b:{(r['sequence'],r['offset']):r for r in rows if r['method']==fe and r['budget_label']==b} for b in ('8','16','32','64')}
        def stats(b,common=None):
            ok={k:r for k,r in arms[b].items() if r['errors_m'] and (common is None or k in common)}
            per={s:[abs(r['errors_m']['state_mean']) for k,r in ok.items() if k[0]==s] for s in SEQS}
            return dict(windows=len(arms[b]),completed=sum(1 for r in arms[b].values() if r['errors_m']),mae_seq_equal=float(np.mean([np.mean(v) for v in per.values() if v])) if ok else None,
                        per_sequence={s:(float(np.mean(v)) if v else None) for s,v in per.items()},bias=float(np.mean([r['errors_m']['state_mean'] for r in ok.values()])) if ok else None,
                        pos_rmse=float(np.mean([r['pose_errors']['position_rmse_m'] for r in ok.values()])) if ok else None,mean_wall_s=float(np.mean([r['wall_seconds']['total'] for r in arms[b].values()])))
        res={b:stats(b) for b in arms};pairs={}
        for b in ('16','32','64'):
            common={k for k in arms['8'] if arms['8'][k]['errors_m'] and arms[b][k]['errors_m']};a=stats('8',common);c=stats(b,common)
            pairs[f'8_vs_{b}']=dict(common_windows=len(common),mae_8=a['mae_seq_equal'],mae_b=c['mae_seq_equal'],per_sequence_8=a['per_sequence'],per_sequence_b=c['per_sequence'],
                                    b8_better=a['mae_seq_equal']<c['mae_seq_equal'] if a['mae_seq_equal'] and c['mae_seq_equal'] else None,
                                    each_sequence_b8_not_worse=all((a['per_sequence'][s] or np.inf)<=(c['per_sequence'][s] or np.inf) for s in SEQS if a['per_sequence'][s] is not None))
        gate=dict(b8_lt_b32=pairs['8_vs_32']['b8_better'],b8_lt_b64=pairs['8_vs_64']['b8_better'],each_seq_b8_le_b32=pairs['8_vs_32']['each_sequence_b8_not_worse'],completion_b8_ge_b32=res['8']['completed']>=res['32']['completed'])
        dec[fe]=dict(per_budget=res,paired=pairs,gate=gate,gate_pass=all(v for v in gate.values()))
    (OUT/'decision.json').write_text(json.dumps(dict(reference_first_read_utc=ref_utc,decision=dec),indent=1))
    for fe,d in dec.items():
        print(f'== {fe} == gate pass:',d['gate_pass'],d['gate'])
        for b,s in d['per_budget'].items():print(f"  B{b}: complete {s['completed']}/{s['windows']} MAE {s['mae_seq_equal'] and round(s['mae_seq_equal']*1000,2)} mm bias {s['bias'] and round(s['bias']*1000,1)} posRMSE {s['pos_rmse'] and round(s['pos_rmse']*1000,1)} wall {round(s['mean_wall_s'],1)} s per-seq {{ {', '.join(f'{k[:8]}:{v and round(v*1000,1)}' for k,v in s['per_sequence'].items())} }}")
        for k,p in d['paired'].items():print(f"  paired {k}: n={p['common_windows']} MAE8={p['mae_8'] and round(p['mae_8']*1000,2)} MAEb={p['mae_b'] and round(p['mae_b']*1000,2)}")


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','evaluate']);globals()[p.parse_args().action]()
