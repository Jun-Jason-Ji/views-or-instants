import sys, json, numpy as np
sys.path.insert(0,'src')
import dpjait_noise_runner as v1
from pathlib import Path
Tp=400
res={}
for ver,exp in [('v0.1_blob','experiments/dpjait_noise_prereg_v0.1'),('v0.2_yolo','experiments/dpjait_noise_prereg_v0.2')]:
    exp=Path(exp)
    for fam,names in {'real':v1.REAL,'S12_D3':['S12_D3'],'S13_D3':['S13_D3']}.items():
        for lev in (0,3,5):
            fac=[]; eig=[]
            for n in names:
                rec,sim=v1.open_record(n); per=v1.ref_per_frame(sim)
                d=json.loads((exp/'scored'/(n+'.json')).read_text()); dz=np.load(exp/'predictions'/(n+'_dense.npz'))
                for dr in sorted({r['drone'] for r in d['rows']}):
                    P=dz['A|%d|0|%s'%(lev,dr)].astype(float)
                    for w in sorted({r['window'] for r in d['rows'] if r['drone']==dr}):
                        fr=np.arange(w*Tp,(w+1)*Tp+1)
                        if fr[-1]*per>=len(rec.reference[dr]): continue
                        ref=np.asarray([rec.reference_at(dr,f) for f in fr]); e=P[fr]-ref
                        ok=~np.isnan(e).any(1)
                        if ok.mean()<0.9: continue
                        e=e[ok]-e[ok].mean(0); S=np.cov(e.T)
                        # segment directions at the crossing scale (m~17)
                        ii=v1.instants(0,Tp,17); seg=np.diff(ref[ii],axis=0); u=seg/np.linalg.norm(seg,axis=1,keepdims=True)
                        perp=np.trace(S)-np.einsum('ti,ij,tj->t',u,S,u)
                        fac.append(np.mean(perp)/(2/3*np.trace(S)))
                        ev=np.sort(np.linalg.eigvalsh(S))[::-1]; eig.append(ev/ev.sum())
            res[(ver,fam,lev)]=(np.mean(fac),np.mean(eig,0))
            print('%-9s %-7s level %d  anisotropy factor (perp var / isotropic) = %.2f   eigen-share %s'%(ver,fam,lev,np.mean(fac),np.round(np.mean(eig,0),2)))
