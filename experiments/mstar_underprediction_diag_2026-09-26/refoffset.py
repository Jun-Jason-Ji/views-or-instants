import sys, json, numpy as np
sys.path.insert(0,'src')
import dpjait_noise_runner as v1
from pathlib import Path
exp=Path('experiments/dpjait_noise_prereg_v0.2'); cfg=json.loads((exp/'config.json').read_text())
Tp=cfg['t_primary']; grid=np.array(cfg['grid_primary']); Ts=Tp/25
out={}
for fam,names in {'real':v1.REAL,'S12_D3':['S12_D3'],'S13_D3':['S13_D3']}.items():
    D=[]; offs={}
    for n in names:
        rec,sim=v1.open_record(n); per=v1.ref_per_frame(sim)
        d=json.loads((exp/'scored'/(n+'.json')).read_text())
        for dr,w in sorted({(r['drone'],r['window']) for r in d['rows']}):
            f0,f1=w*Tp,(w+1)*Tp
            seg=np.asarray(rec.reference[dr][f0*per:f1*per+1],float)
            L100=v1.polyline(seg)
            row=[v1.polyline(np.asarray([rec.reference_at(dr,f) for f in v1.instants(f0,Tp,m)]))-L100 for m in grid]
            # length at decimations of the reference itself
            for step in (1,2,4,8,16):
                offs.setdefault(step,[]).append(v1.polyline(seg[::step])-L100)
            D.append(row)
    D=np.array(D).mean(axis=0)*1000
    # fit D = -a/m^2 + b on m>=4
    X=np.column_stack([-1/grid**2,np.ones_like(grid,dtype=float)]); a,b=np.linalg.lstsq(X,D,rcond=None)[0]
    print(fam,'mean D(m) mm:',{int(m):round(v,1) for m,v in zip(grid,D) if m in (4,6,10,17,28,50,100,200,400)})
    print('   fit D = -a/m^2 + b : a=%.0f mm, b=%.1f mm'%(a,b), '| residual max %.1f'%np.max(np.abs(X@[a,b]-D)))
    print('   reference self-decimation offset (mm), 100Hz->:',{ (100//s if not fam.startswith('S') else 25//s if s<=25 else 0):round(np.mean(v)*1000,1) for s,v in offs.items()})
    out[fam]=dict(D=D.tolist(),a=a,b=b)
json.dump(out,open('experiments/mstar_underprediction_diag_2026-09-26/deficit_fit.json','w'),indent=1)
