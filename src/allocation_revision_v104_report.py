"""Summarize grid/robustness sensitivity and retrospective development selection."""
import hashlib
import json
from pathlib import Path
import numpy as np
from allocation_revision_v104 import OUT, ARMS, grids


def stats(rows):
    good=[r for r in rows if r['error_mm'] is not None]
    e=np.array([abs(r['error_mm']) for r in good])
    return dict(attempted=len(rows),completed=len(good),mae_mm=float(e.mean()) if len(e) else None,
        median_mm=float(np.median(e)) if len(e) else None,p95_mm=float(np.quantile(e,.95)) if len(e) else None,
        max_mm=float(e.max()) if len(e) else None,over100mm=int((e>100).sum()),
        missing=sum(r['missing'] for r in rows),spatial_rejected=sum(r['spatial_rejected'] for r in rows),
        temporal_rejected=sum(r['temporal_rejected'] for r in rows),queried=sum(r['moments'] for r in rows))


def main():
    rows=[]
    for num in range(15,24):
        rows+=json.loads((OUT/f'record{num}.json').read_text())
    assert len(rows)==17820
    groups={}
    for r in rows:
        key=(r['split'],r['budget'],r['views'],r['arm'],r['grid'])
        groups.setdefault(key,[]).append(r)
    summary=[dict(split=k[0],budget=k[1],views=k[2],arm=k[3],grid=k[4],**stats(v)) for k,v in sorted(groups.items())]
    paired=[]; effects=[]
    for split in ('development','evaluation'):
        for b in (84,168):
            for arm in ARMS:
                for grid in grids(28):
                    a={(r['record'],r['start']):r for r in groups[split,b,3,arm,grid] if r['error_mm'] is not None}
                    c={(r['record'],r['start']):r for r in groups[split,b,7,arm,grid] if r['error_mm'] is not None}
                    common=sorted(a.keys() & c.keys())
                    aa=[a[x] for x in common]; cc=[c[x] for x in common]
                    sa,sc=stats(aa),stats(cc)
                    paired.append(dict(split=split,budget=b,arm=arm,grid=grid,common=len(common),
                        mae3_mm=sa['mae_mm'],mae7_mm=sc['mae_mm'],difference_mm=sa['mae_mm']-sc['mae_mm'],
                        by_record={rec:float(np.mean([abs(a[x]['error_mm'])-abs(c[x]['error_mm']) for x in common if x[0]==rec])) for rec in sorted({x[0] for x in common})}))
            for k in (3,7):
                for arm in ARMS[1:]:
                    for grid in grids(28):
                        base={(r['record'],r['start']):r for r in groups[split,b,k,'dlt',grid] if r['error_mm'] is not None}
                        after={(r['record'],r['start']):r for r in groups[split,b,k,arm,grid] if r['error_mm'] is not None}
                        common=sorted(base.keys() & after.keys())
                        effects.append(dict(split=split,budget=b,views=k,arm=arm,grid=grid,common=len(common),
                            dlt=stats([base[x] for x in common]),robust=stats([after[x] for x in common])))
    ranges=[]
    for b in (84,168):
        for arm in ARMS:
            pp=[p for p in paired if p['split']=='evaluation' and p['budget']==b and p['arm']==arm]
            rr=[r for r in summary if r['split']=='evaluation' and r['budget']==b and r['arm']==arm]
            ranges.append(dict(budget=b,arm=arm,grids=len(pp),three_view_lower=sum(p['difference_mm']<0 for p in pp),
                paired_difference_range_mm=[min(p['difference_mm'] for p in pp),max(p['difference_mm'] for p in pp)],
                common_range=[min(p['common'] for p in pp),max(p['common'] for p in pp)],
                per_view={k:{'mae_range_mm':[min(r['mae_mm'] for r in rr if r['views']==k),max(r['mae_mm'] for r in rr if r['views']==k)],
                             'completion_range':[min(r['completed'] for r in rr if r['views']==k),max(r['completed'] for r in rr if r['views']==k)]} for k in (3,7)}))
    selection=[]
    for cap in (84,168):
        candidates=[s for s in summary if s['split']=='development' and s['grid']=='uniform' and s['budget']<=cap]
        eligible=[s for s in candidates if s['completed']==s['attempted'] and s['p95_mm']<=100]
        chosen=min(eligible,key=lambda s:(s['mae_mm'],s['budget'],s['views'],s['arm'])) if eligible else None
        evaluation=None if chosen is None else next(s for s in summary if s['split']=='evaluation' and s['grid']=='uniform' and all(s[k]==chosen[k] for k in ('budget','views','arm')))
        selection.append(dict(cap=cap,candidates=len(candidates),eligible=len(eligible),chosen=chosen,evaluation=evaluation))
    old=json.loads((OUT.parent/'allocation_revision_v103_2026-09-16/rows.json').read_text())
    index={(r['record'],r['start'],r['budget'],r['views']):r for r in old if r['sigma_mm']==3}
    diff=[abs(r['value_m']-index[r['record'],r['start'],r['budget'],r['views']]['value_m']) for r in rows if r['split']=='evaluation' and r['arm']=='dlt' and r['grid']=='uniform']
    assert len(diff)==240 and max(diff)<1e-9
    result=dict(status='Retrospective exploratory analysis; grids are correlated perturbations, not independent replicates.',
        configurations=len(rows),original_replicated=len(diff),max_replication_error_m=max(diff),
        summary=summary,paired=paired,robust_effects=effects,grid_ranges=ranges,selection=selection,
        input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('record*.json'))})
    (OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'ranges':ranges,'selection':selection,'uniform_evaluation':[s for s in summary if s['split']=='evaluation' and s['grid']=='uniform']},indent=2))


if __name__=='__main__':
    main()
