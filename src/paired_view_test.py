"""Paired test of k=3 against k=7 on the confirmation set (same windows, same instants)."""
import json
from pathlib import Path
import numpy as np
from scipy import stats

ROOT=Path(__file__).resolve().parents[1];RULE='216f21c1+3e0f8f0+44c4b2e'


def main():
    rows=json.load(open(ROOT/'experiments/allocation_confirm_v1_2026-09-15/scored.json'))
    all7='+'.join(sorted({c for r in rows if r['views']==7 for c in r['cameras'].split('+')}))
    err=lambda cam,m:{(r['record'],r['start']):abs(r['error_m'])*1000 for r in rows
                      if r['cameras']==cam and r['moments']==m and r['method']=='state_mean' and r['error_m'] is not None}
    out=[]
    for m in (16,19,25,28,33):
        e7=err(all7,m);diffs=[]
        for cam in sorted({r['cameras'] for r in rows if r['views']==3}):
            e3=err(cam,m);common=sorted(set(e3)&set(e7))
            if len(common)<50:continue
            diffs.append(float(np.mean([e3[w] for w in common])-np.mean([e7[w] for w in common])))
        diffs=np.array(diffs);better=int((diffs<0).sum())
        e3=err(RULE,m);common=sorted(set(e3)&set(e7));d=np.array([e3[w]-e7[w] for w in common])
        rng=np.random.default_rng(0);bs=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(5000)])
        out.append(dict(moments=m,triples=len(diffs),median_triple_minus_k7_mm=float(np.median(diffs)),
            triples_better_than_k7=better,sign_test_p=float(stats.binomtest(better,len(diffs),.5).pvalue),
            rule_triple_mean_diff_mm=float(d.mean()),boot_ci95_mm=[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))],windows=len(d)))
    (ROOT/'experiments/allocation_confirm_v1_2026-09-15/paired_view_test.json').write_text(json.dumps(out,indent=1))
    for o in out:
        print(f"m={o['moments']:3d} triples={o['triples']} median(k3-k7)={o['median_triple_minus_k7_mm']:+6.2f} mm  "
              f"better={o['triples_better_than_k7']}/{o['triples']} sign p={o['sign_test_p']:.3f}  "
              f"rule triple diff={o['rule_triple_mean_diff_mm']:+6.2f} mm CI95 [{o['boot_ci95_mm'][0]:+.2f},{o['boot_ci95_mm'][1]:+.2f}]")


if __name__=='__main__':main()
