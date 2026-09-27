import sys, json, numpy as np
sys.path.insert(0,'src')
import dpjait_noise_runner as v1
from pathlib import Path
grid=[4,6,8,10,12,14,17,20,24,28,34,40,50,60,80,100,134,200,400]
def mavg(X,w):
    k=np.ones(w)/w; return np.stack([np.convolve(X[:,i],k,mode='same') for i in range(3)],1)
fit=json.load(open('experiments/mstar_underprediction_diag_2026-09-26/deficit_fit.json'))
for ver,exp in [('v0.1_blob','experiments/dpjait_noise_prereg_v0.1'),('v0.2_yolo','experiments/dpjait_noise_prereg_v0.2')]:
    exp=Path(exp); Tp=400
    res={'excess_smooth25':[], 'excess_smooth_cross':[],'I_by_m':{m:[] for m in grid}}
    for n in v1.REAL:
        rec,_=v1.open_record(n); d=json.loads((exp/'scored'/(n+'.json')).read_text()); dz=np.load(exp/'predictions'/(n+'_dense.npz'))
        P=dz['A|0|0|drone'].astype(float)
        for w in sorted({r['window'] for r in d['rows']}):
            f0,f1=w*Tp,(w+1)*Tp
            ref=np.asarray([rec.reference_at('drone',f) for f in range(f0,min(f1+1,len(rec.reference["drone"])//4))])
            if len(ref)<Tp+1: continue
            e=P[f0:f0+len(ref)]-ref
            if np.isnan(e).any(): 
                ok=~np.isnan(e).any(1)
                if ok.mean()<0.9: continue
                idx=np.arange(len(e)); e=np.stack([np.interp(idx,idx[ok],e[ok,i]) for i in range(3)],1)
            es=mavg(e,25)          # 1 s low-pass of the observation error
            Lr=v1.polyline(ref)
            res['excess_smooth25'].append(v1.polyline(ref+es)-Lr)
            # at 24 instants (crossing scale): excess from the smooth error only vs full error
            ii=v1.instants(0,Tp,24)
            res['excess_smooth_cross'].append((v1.polyline((ref+es)[ii])-v1.polyline(ref[ii]), v1.polyline((ref+e)[ii])-v1.polyline(ref[ii])))
            for m in grid:
                ii=v1.instants(0,Tp,m)
                res['I_by_m'][m].append(v1.polyline((ref+e)[ii])-v1.polyline(ref[ii]))
    ex=np.array(res['excess_smooth_cross'])
    print('==',ver,'native, windows',len(res['excess_smooth25']))
    print('  path excess from 1-s low-passed error at 25 Hz: mean %.0f mm, median %.0f mm'%(np.mean(res['excess_smooth25'])*1e3,np.median(res['excess_smooth25'])*1e3))
    print('  at m=24: inflation from smooth part %.0f mm of total %.0f mm (%.0f%%)'%(ex[:,0].mean()*1e3,ex[:,1].mean()*1e3,100*ex[:,0].mean()/ex[:,1].mean()))
    print('  inflation I(m) with the SAME-instant reference (mm):',{m:round(np.mean(v)*1e3) for m,v in res['I_by_m'].items() if m in (6,10,17,24,34,50,100,200,400)})
