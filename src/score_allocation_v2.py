"""Corrected aggregation for allocation sweep/confirmation: a failed window is failed for every estimator.

The frozen score() in run_allocation_confirm.py only stored failure rows under method='polygon',
so its per-method 'failed' counts for limited/state_mean were 0 and n was reduced. This scorer
re-aggregates the sealed, already-scored rows without touching predictions or references.
Gates are re-evaluated with failures counted; matched-window comparisons are added.
"""
import json,argparse
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def aggregate(rows,records):
    keys=sorted({(r['cameras'],r['views'],r['moments']) for r in rows});out=[]
    for cam,k,m in keys:
        rr=[r for r in rows if (r['cameras'],r['moments'])==(cam,m)]
        windows=sorted({(r['record'],r['start']) for r in rr});failed_w={(r['record'],r['start']) for r in rr if r.get('failure')}
        for method in ('polygon','limited','state_mean'):
            vals={(r['record'],r['start']):r['error_m'] for r in rr if r['method']==method and r['error_m'] is not None}
            if not vals and method!='polygon':continue
            ok=[w for w in windows if w in vals]
            out.append(dict(cameras=cam,views=k,moments=m,frames=k*m,method=method,windows=len(windows),failed=len(windows)-len(ok),
                mae_m=float(np.mean([abs(vals[w]) for w in ok])) if ok else None,bias_m=float(np.mean([vals[w] for w in ok])) if ok else None,
                p95_m=float(np.quantile([abs(vals[w]) for w in ok],.95)) if ok else None,
                per_record={rec:float(np.mean([abs(vals[w]) for w in ok if w[0]==rec])) for rec in records if any(w[0]==rec for w in ok)},
                ok_windows=[list(w) for w in ok]))
    return out


def main(exp):
    rows=json.load(open(exp/'scored.json'));records=sorted({r['record'] for r in rows});summary=aggregate(rows,records)
    (exp/'summary_v2.json').write_text(json.dumps(dict(note='failures counted for every estimator; MAE over successful windows only',summary=summary),indent=1))
    cfg_path=exp/'config.json'
    if not cfg_path.exists():print('summary_v2 written (dev sweep)');return
    cfg=json.load(open(cfg_path));find=lambda cam,m,meth:next(s for s in summary if s['cameras']==cam and s['moments']==m and s['method']==meth)
    all7='+'.join(sorted({c for r in rows if r['views']==7 for c in r['cameras'].split('+')}))
    prior=find(all7,8,'state_mean');prior12=find(all7,12,'state_mean');decision=dict(prior_7x8_state_mean=prior,prior_7x12_state_mean=prior12,candidates=[])
    err_prior={(r['record'],r['start']):r['error_m'] for r in rows if r['views']==7 and r['moments']==8 and r['method']=='state_mean'}
    for c in cfg['candidates']:
        a=find(c['cameras'],c['moments'],c['estimator']);ok=[tuple(w) for w in a['ok_windows']]
        matched_prior=float(np.mean([abs(err_prior[w]) for w in ok])) if ok else None
        gain=(prior['mae_m']-a['mae_m'])/prior['mae_m'];every=all(a['per_record'].get(k,np.inf)<prior['per_record'][k] for k in records)
        gate=dict(frames_ok=c['frames']<=57,gain_ge_50pct=gain>=.5,every_record_improves=every,no_new_failures=a['failed']<=prior['failed'])
        decision['candidates'].append(dict(candidate=c,result={k:v for k,v in a.items() if k!='ok_windows'},relative_gain_vs_prior_all_windows=gain,
            prior_mae_on_same_successful_windows=matched_prior,gate=gate,primary_gate_pass=all(gate.values()) if c['frames']<=57 else 'not_in_primary_gate'))
    # robustness: every pair / triple at the equal-cost moment counts
    rob={}
    for k,m in ((2,28),(3,19)):
        items=[s for s in summary if s['views']==k and s['moments']==m and s['method']=='state_mean']
        rob[f'{k}x{m}']=dict(subsets=len(items),mae_min=min(s['mae_m'] for s in items),mae_median=float(np.median([s['mae_m'] for s in items])),mae_max=max(s['mae_m'] for s in items),
            subsets_with_failures=sum(s['failed']>0 for s in items),failed_windows_total=sum(s['failed'] for s in items),max_failed_per_subset=max(s['failed'] for s in items),
            all_subsets_beat_prior_on_successful_windows=all(s['mae_m']<prior['mae_m'] for s in items))
    decision['robustness_all_subsets']=rob
    (exp/'decision_v2.json').write_text(json.dumps(decision,indent=1));print(json.dumps({k:v for k,v in decision.items() if k!='prior_7x12_state_mean'},indent=1)[:5000])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('exp',type=Path);main(p.parse_args().exp)
