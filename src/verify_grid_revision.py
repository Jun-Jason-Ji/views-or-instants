"""Independent arithmetic, pairing, eligibility and provenance verification."""
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/allocation_revision_v104_2026-09-16'
result=json.loads((OUT/'summary.json').read_text())
rows=[]; checks=0
for name,digest in result['input_sha256'].items():
    path=OUT/name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    rows+=json.loads(path.read_text()); checks+=1
for name,digest in json.loads((OUT/'provenance.json').read_text())['sha256'].items():
    p=OUT/'source_snapshot'/name
    assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
    checks+=1
assert len(rows)==17820
for r in rows:
    indices=r['selected_indices']
    assert indices[0]==0 and indices[-1]==199 and np.all(np.diff(indices)>0)
    assert len(indices)==r['moments'] and r['views']*len(indices)==r['frames']==r['budget']
    assert (r['value_m'] is None)==(r['failure'] is not None)
    if r['value_m'] is not None:
        assert np.isclose((r['value_m']-r['reference_m'])*1000,r['error_mm'],atol=1e-10,rtol=0)
checks+=1
groups={}
for r in rows:groups.setdefault((r['split'],r['budget'],r['views'],r['arm'],r['grid']),[]).append(r)
for s in result['summary']:
    rr=groups[tuple(s[k] for k in ('split','budget','views','arm','grid'))]
    e=np.array([abs(r['error_mm']) for r in rr if r['value_m'] is not None])
    assert len(rr)==s['attempted'] and len(e)==s['completed']
    for key,val in [('mae_mm',np.mean(e)),('median_mm',np.median(e)),('p95_mm',np.quantile(e,.95)),('max_mm',max(e))]:
        assert np.isclose(s[key],val,atol=1e-9,rtol=0)
    assert s['over100mm']==sum(e>100)
    for key in ('spatial_rejected','temporal_rejected','missing'):
        assert s[key]==sum(r[key] for r in rr)
    checks+=1
for s in result['paired']:
    base=(s['split'],s['budget'])
    a={(r['record'],r['start']):r for r in groups[*base,3,s['arm'],s['grid']] if r['value_m'] is not None}
    b={(r['record'],r['start']):r for r in groups[*base,7,s['arm'],s['grid']] if r['value_m'] is not None}
    common=a.keys()&b.keys()
    assert len(common)==s['common']
    delta=np.mean([abs(a[k]['error_mm'])-abs(b[k]['error_mm']) for k in common])
    assert np.isclose(delta,s['difference_mm'],rtol=0,atol=1e-9)
    checks+=1
for s in result['selection']:
    dev=[x for x in result['summary'] if x['split']=='development' and x['grid']=='uniform' and x['budget']<=s['cap']]
    eligible=[x for x in dev if x['completed']==75 and x['p95_mm']<=100]
    assert len(dev)==s['candidates'] and len(eligible)==s['eligible']
    want=min(eligible,key=lambda x:(x['mae_mm'],x['budget'],x['views'],x['arm']))
    assert want==s['chosen']
    assert s['evaluation']['split']=='evaluation'
    assert all(want[k]==s['evaluation'][k] for k in ('budget','views','arm','grid'))
    checks+=1
old=json.loads((OUT.parent/'allocation_revision_v103_2026-09-16/rows.json').read_text())
base={(r['record'],r['start'],r['budget'],r['views']):r for r in old if r['sigma_mm']==3}
overlap=[r for r in rows if r['split']=='evaluation' and r['grid']=='uniform' and r['arm']=='dlt']
assert len(overlap)==240 and all(abs(r['value_m']-base[r['record'],r['start'],r['budget'],r['views']]['value_m'])<1e-9 for r in overlap)
checks+=1
tex=ROOT/'paper/mst/mst_manuscript.tex'
if not tex.exists():tex=ROOT/'paper/mst_manuscript.tex'
body=tex.read_text(encoding='utf-8')
for s in result['grid_ranges']:
    lo,hi=s['paired_difference_range_mm']
    assert f'[{lo:.2f},{hi:.2f}]' in body
    assert f"{s['three_view_lower']}/11" in body
    checks+=1
for val in ('7.92','22.59','17820','0.060','9.89','8.37','22.32','19.17'):
    assert val in body
checks+=1
print(f'PASS {checks}: source/input hashes, 17820 schedules, errors, completion/rejections, common-window comparisons, development selection, 240 old predictions and manuscript results.')
