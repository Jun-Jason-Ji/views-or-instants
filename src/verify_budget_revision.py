"""Independently check reported budget cells and robustness contrasts."""
import json
import re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'experiments/allocation_revision_v103_2026-09-16'
rows=json.loads((base/'rows.json').read_text())
summary=json.loads((base/'summary.json').read_text())
tex=(ROOT/'paper/mst/mst_manuscript.tex')
if not tex.exists():tex=ROOT/'paper/mst_manuscript.tex'
src=tex.read_text(encoding='utf-8')
checks=0
for cell in summary['summary']:
    rr=[r for r in rows if all(r[k]==cell[k] for k in ('budget','views','sigma_mm'))]
    good=[r for r in rr if r['value_m'] is not None]
    e=np.array([1000*abs(r['value_m']-r['reference_m']['1']) for r in good])
    assert len(good)==cell['completed'] and len(rr)==cell['attempted']==60
    assert all(r['frames']==r['views']*r['moments']<=r['budget'] for r in rr)
    assert np.isclose(e.mean(),cell['mae_mm'],atol=1e-9,rtol=0)
    assert np.isclose(max(e),cell['maximum_mm'],atol=1e-9,rtol=0)
    if cell['sigma_mm']==3:
        row=next(l for l in src.splitlines() if l.startswith(str(cell['budget'])+' &'))
        shown=row.split('&')[cell['views']-1].strip()
        assert shown.startswith(f"{cell['mae_mm']:.2f} ({cell['frames']})"),shown
    checks+=1
for contrast in summary['paired']:
    a={};b={};stride=str(contrast['reference_stride'])
    for r in rows:
        if r['budget']==contrast['budget'] and r['sigma_mm']==contrast['sigma_mm'] and r['value_m'] is not None:
            if r['views'] in (3,7):
                (a if r['views']==3 else b)[r['record'],r['start']]=1000*abs(r['value_m']-r['reference_m'][stride])
    common=a.keys()&b.keys()
    assert len(common)==contrast['common']
    d={rec:np.mean([a[k]-b[k] for k in common if k[0]==rec]) for rec in contrast['per_record_difference_mm']}
    assert np.isclose(np.mean(list(d.values())),contrast['difference_mm'],atol=1e-9,rtol=0)
    for rec in d:
        assert np.isclose(np.mean([v for name,v in d.items() if name!=rec]),contrast['leave_one_record_out_mm'][rec],atol=1e-9,rtol=0)
    checks+=1
assert '85.66' in src and '10.55' in src and '4110.45' in src
assert summary['original_estimates_replicated']==288 and summary['max_replication_error_m']==0
print(f'PASS {checks+2}: budget cells, completion, tails, paired reference/noise contrasts, record deletion, rank reversal and archived estimate replication.')
