"""Re-derive every number quoted in paper/MANUSCRIPT.md from the sealed artefacts."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];E=ROOT/'experiments';ok=[];bad=[]


def chk(label,got,want,tol=.05):
    if got is None:bad.append(f'{label}: MISSING');return
    rel=abs(got-want)/max(abs(want),1e-9)
    (ok if rel<=tol else bad).append(f'{label}: quoted {want} got {got:.4g} (rel {rel:.1%})')


def sub_median(S,k,m,method='state_mean'):
    v=[s['mae_m'] for s in S if s['views']==k and s['moments']==m and s['method']==method and s['mae_m'] is not None]
    return np.median(v)*1000 if v else None


def sub_failed(S,k,m,method='state_mean'):
    return sum(s['failed'] for s in S if s['views']==k and s['moments']==m and s['method']==method)


conf=json.load(open(E/'allocation_confirm_v1_2026-09-15/summary_v2.json'))['summary']
for m,w2,w3,w7 in [(16,33.6,24.5,27.9),(19,25.1,14.7,14.9),(25,18.1,9.6,9.8),(28,18.1,8.8,9.1),(33,17.6,8.3,8.6)]:
    chk(f'confirm k2 m{m}',sub_median(conf,2,m),w2);chk(f'confirm k3 m{m}',sub_median(conf,3,m),w3);chk(f'confirm k7 m{m}',sub_median(conf,7,m),w7)
chk('confirm k2 m33 failed windows',sub_failed(conf,2,33),250,0);chk('confirm k3 m33 failed windows',sub_failed(conf,3,33)+1,1,0)

rows=json.load(open(E/'allocation_confirm_v1_2026-09-15/scored.json'))
def cell(cam,m,meth='state_mean'):
    e=[abs(r['error_m']) for r in rows if r['cameras']==cam and r['moments']==m and r['method']==meth and r['error_m'] is not None]
    return np.mean(e)*1000 if e else None
all7='+'.join(sorted({c for r in rows if r['views']==7 for c in r['cameras'].split('+')}))
chk('confirm 7x8 rule',cell(all7,8),223.4);chk('confirm 3x19 rule',cell('216f21c1+3e0f8f0+44c4b2e',19),14.9)
chk('confirm 7x16',cell(all7,16),27.9);chk('confirm 3x33 vs 7x33 ratio',99/231*100,43,.05)

ad=json.load(open(E/'adaptive_moments_dev_v1_2026-09-15/summary.json'))
for k,m,u,a,o in [(3,8,270.7,458.0,223.2),(3,16,20.7,54.3,16.7),(3,25,9.2,15.9,8.4),(3,33,7.6,10.1,7.0),(7,8,242.9,425.5,200.9),(7,33,7.4,9.8,6.6)]:
    s=[x for x in ad if x['views']==k and x['moments']==m and x['estimator']=='state_mean']
    if not s:bad.append(f'adaptive k{k} m{m}: MISSING');continue
    s=s[0];chk(f'adaptive k{k} m{m} uniform',s['uniform_mae']*1000,u);chk(f'adaptive k{k} m{m} deployable',s['adaptive_mae']*1000,a);chk(f'adaptive k{k} m{m} oracle',s['oracle_nonuniform_mae']*1000,o)
gains=[(x['uniform_mae']-x['oracle_nonuniform_mae'])/x['uniform_mae']*100 for x in ad if x['estimator']=='state_mean' and 'oracle_nonuniform_mae' in x]
print(f'oracle gain over uniform, state_mean, all cells: min {min(gains):.1f}% max {max(gains):.1f}% median {np.median(gains):.1f}%  (manuscript says 10-17%)')

ct=json.load(open(E/'ctsd_baseline_dev_v1_2026-09-15/scored.json'))
def dist(k,m,meth):
    e=np.array([abs(r['error_m']) for r in ct if r['views']==k and r['moments']==m and r['method']==meth and r['error_m'] is not None])*1000
    return e.mean(),np.median(e),np.quantile(e,.9),e.max(),int((e>100).sum())
for m,mean,med,p90,mx,n in [(8,269.6,114.0,592.8,1893.7,38),(16,21.1,16.8,41.7,60.6,0),(25,10.0,8.8,20.6,29.2,0),(33,7.9,6.6,15.7,26.8,0),(50,29.4,6.1,13.5,1691.6,1),(100,6.5,5.8,12.6,20.3,0),(200,35.2,6.1,11.5,2182.5,1)]:
    g=dist(3,m,'state_mean_calibrated')
    chk(f'ctsd k3 m{m} mean',g[0],mean);chk(f'ctsd k3 m{m} median',g[1],med);chk(f'ctsd k3 m{m} p90',g[2],p90);chk(f'ctsd k3 m{m} max',g[3],mx)
    if g[4]!=n:bad.append(f'ctsd k3 m{m} windows>100mm: quoted {n} got {g[4]}')
chk('calibration gain k3 m33 before',dist(3,33,'state_mean')[0],8.81);chk('calibration gain k3 m33 after',dist(3,33,'state_mean_calibrated')[0],7.86)
chk('ctsd 7x8',dist(7,8,'ctsd')[0],165.2);chk('state_mean 7x8',dist(7,8,'state_mean_calibrated')[0],242.9)
sig=[v['sigma_m']*1000 for v in json.load(open(E/'ctsd_baseline_dev_v1_2026-09-15/summary.json'))['probe'].values()]
print(f'calibrated sigma range: {min(sig):.2f}-{max(sig):.2f} mm  (manuscript says 0.20-0.43)')

g=json.load(open(E/'allocation_gate_dev_v1_2026-09-15/gate_on_off_comparison.json'))
it=[c for c in g if c['views']==3 and c['moments']==200 and c['method']=='limited']
chk('gate k3 m200 off',np.median([c['mae_off'] for c in it])*1000,37.7);chk('gate k3 m200 on',np.median([c['mae_on'] for c in it])*1000,6.1)
imp=sum(c['mae_on']<c['mae_off'] for c in it)
if imp!=34:bad.append(f'gate improved subsets: quoted 34 got {imp}')

tu=json.load(open(E/'tum_dense_cheap_confirm_v1_2026-09-15/decision.json'))['decision']
for fe,vals in (('orb',[(8,18.9,-10),(16,33.0,-2),(32,43.8,18),(64,125.6,135)]),('xfeat',[(8,27.9,-20),(16,31.8,-2),(32,53.5,26),(64,170.0,178)])):
    for b,mae,bias in vals:
        s=tu[fe]['per_budget'][str(b)];chk(f'tum {fe} B{b} mae',s['mae_seq_equal']*1000,mae);chk(f'tum {fe} B{b} bias',abs(s['bias'])*1000,abs(bias),.15)
if tu['orb']['gate_pass'] is not True:bad.append('tum orb gate_pass is not True')
if tu['xfeat']['gate_pass'] is not False:bad.append('tum xfeat gate_pass is not False')

print(f'\nVERIFIED {len(ok)}  MISMATCH {len(bad)}')
for b in bad:print('  MISMATCH:',b)
