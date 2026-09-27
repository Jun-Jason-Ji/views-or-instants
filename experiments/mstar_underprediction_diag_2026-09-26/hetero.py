import sys, json, numpy as np
sys.path.insert(0,'src')
import dpjait_noise_runner as v1
from pathlib import Path
Tp=400; exp=Path('experiments/dpjait_noise_prereg_v0.2')
fit=json.load(open('experiments/mstar_underprediction_diag_2026-09-26/deficit_fit.json'))
diag=json.load(open('experiments/mstar_underprediction_diag_2026-09-26/diagnosis.json'))
out={}
for fam,names in {'real':v1.REAL,'S11_D4':['S11_D4'],'S12_D3':['S12_D3'],'S13_D3':['S13_D3']}.items():
    H={m:[] for m in (10,17,24)}; cv=[]; slowfrac=[]
    for n in names:
        rec,sim=v1.open_record(n); per=v1.ref_per_frame(sim)
        d=json.loads((exp/'scored'/(n+'.json')).read_text())
        for dr,w in sorted({(r['drone'],r['window']) for r in d['rows']}):
            fr=np.arange(w*Tp,(w+1)*Tp+1)
            if fr[-1]*per>=len(rec.reference[dr]): continue
            ref=np.asarray([rec.reference_at(dr,f) for f in fr]); L=v1.polyline(ref)
            sp=np.linalg.norm(np.diff(ref,axis=0),axis=1)*25
            cv.append(sp.std()/sp.mean()); slowfrac.append(np.mean(sp<0.25*np.median(sp)))
            for m in H:
                s=np.linalg.norm(np.diff(ref[v1.instants(0,Tp,m)],axis=0),axis=1)
                H[m].append(np.sum(1/np.maximum(s,1e-4))/((m-1)**2/L))
    out[fam]=dict(speed_cv=float(np.median(cv)),slow_frac=float(np.mean(slowfrac)),harmonic={m:float(np.median(v)) for m,v in H.items()})
    # deficit coefficient ratio at the dense end (actual / formula with median kappa, v)
    D=np.array(fit[fam]['D']) if fam in fit else None
    print('%-7s speed CV %.2f | frac time <25%% of median speed %.2f | inflation factor sum(1/s)/(q^2/L), median: %s'%(fam,out[fam]['speed_cv'],out[fam]['slow_frac'],{m:round(v,2) for m,v in out[fam]['harmonic'].items()}),
          '| deficit act/formula @m=24: %.2f'%diag[fam]['deficit_ratio_actual_over_formula']['24'])
json.dump(out,open('experiments/mstar_underprediction_diag_2026-09-26/heterogeneity.json','w'),indent=1)
