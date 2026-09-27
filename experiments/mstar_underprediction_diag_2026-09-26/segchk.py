import sys, json, numpy as np
sys.path.insert(0,'src')
import dpjait_noise_runner as v1
from pathlib import Path
Tp=400
for ver,exp in [('v0.1_blob','experiments/dpjait_noise_prereg_v0.1'),('v0.2_yolo','experiments/dpjait_noise_prereg_v0.2')]:
  exp=Path(exp)
  for fam,names in {'real':v1.REAL,'S12_D3':['S12_D3']}.items():
    for lev in (0,5):
      for m in (10,17,24):
        Id=[];Iq=[];If=[];sig2=[];Lw=[]
        for n in names:
          rec,sim=v1.open_record(n); per=v1.ref_per_frame(sim)
          d=json.loads((exp/'scored'/(n+'.json')).read_text()); dz=np.load(exp/'predictions'/(n+'_dense.npz'))
          for dr in sorted({r['drone'] for r in d['rows']}):
            for rep in range(4 if lev else 1):
              P=dz['A|%d|%d|%s'%(lev,rep,dr)].astype(float)
              for w in sorted({r['window'] for r in d['rows'] if r['drone']==dr}):
                fr=np.arange(w*Tp,(w+1)*Tp+1)
                if fr[-1]*per>=len(rec.reference[dr]): continue
                ref=np.asarray([rec.reference_at(dr,f) for f in fr]); e=P[fr]-ref
                if np.isnan(e).any(): continue
                ii=v1.instants(0,Tp,m); r_=ref[ii]; x=(ref+e)[ii]
                seg=np.diff(r_,axis=0); s=np.linalg.norm(seg,axis=1); u=seg/s[:,None]
                de=np.diff(e[ii],axis=0); perp=np.sum(de**2,1)-np.sum(de*u,1)**2
                Id.append(v1.polyline(x)-v1.polyline(r_)); Iq.append(np.sum(perp/(2*s)+np.sum(de*u,1)))
                c=e-e.mean(0); sg=np.sqrt((c**2).mean()); sig2.append(sg**2); L=v1.polyline(ref); Lw.append(L)
                If.append(2*sg**2*(m-1)**2/L)
        # formula as used in the paper: mean sigma, mean 1/L
        sbar=np.mean(np.sqrt(sig2)); Ifp=2*sbar**2*(m-1)**2*np.mean(1/np.array(Lw))
        print('%-9s %-6s lev %d m %2d | direct %6.0f  2nd-order exact %6.0f  per-window formula %6.0f  paper formula(mean sigma) %6.0f mm | direct/paper %.2f'%(ver,fam,lev,m,np.mean(Id)*1e3,np.mean(Iq)*1e3,np.mean(If)*1e3,Ifp*1e3,np.mean(Id)/Ifp))
