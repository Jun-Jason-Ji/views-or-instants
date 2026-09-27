import sys, json, numpy as np, ezc3d
sys.path.insert(0,'src')
import dpjait_noise_runner as v1
from pathlib import Path
Tp=400
def mavg(X,w):
    k=np.ones(w)/w; return np.stack([np.convolve(X[:,i],k,mode='same') for i in range(3)],1)
def kabsch(A,B):  # R with R@A_i ~ B_i (centred)
    H=A.T@B; U,S,Vt=np.linalg.svd(H); d=np.sign(np.linalg.det(Vt.T@U.T)); D=np.diag([1,1,d]); return Vt.T@D@U.T
out={}
for ver,exp in [('v0.1_blob','experiments/dpjait_noise_prereg_v0.1'),('v0.2_yolo','experiments/dpjait_noise_prereg_v0.2')]:
    exp=Path(exp); tot=[]; 
    for n in v1.REAL:
        rec,_=v1.open_record(n)
        c=ezc3d.c3d(str(rec.c3d_path)); M=np.transpose(c['data']['points'][:3],(2,1,0))*1e-3; M[M==0]=np.nan  # T x 4 x 3
        full=~np.isnan(M).any(axis=(1,2)); tmpl=M[np.argmax(full)]; tmpl=tmpl-tmpl.mean(0)
        d=json.loads((exp/'scored'/(n+'.json')).read_text()); P=np.load(exp/'predictions'/(n+'_dense.npz'))['A|0|0|drone'].astype(float)
        E=[];Rs=[];W=[];REF=[]
        for w in sorted({r['window'] for r in d['rows']}):
            fr=np.arange(w*Tp,(w+1)*Tp+1)
            if fr[-1]*4>=len(M): continue
            mk=M[fr*4]; ok=~np.isnan(mk).any(axis=(1,2))
            ref=np.asarray([rec.reference_at('drone',f) for f in fr]); e=P[fr]-ref
            ok&=~np.isnan(e).any(1)
            if ok.mean()<0.9: continue
            idx=np.arange(len(fr))
            e=np.stack([np.interp(idx,idx[ok],e[ok,i]) for i in range(3)],1)
            R=np.full((len(fr),3,3),np.nan)
            for i in np.nonzero(ok)[0]: R[i]=kabsch(tmpl, mk[i]-mk[i].mean(0))
            for i in np.nonzero(~ok)[0]: R[i]=R[np.nonzero(ok)[0][np.argmin(abs(np.nonzero(ok)[0]-i))]]
            E.append(e);Rs.append(R);W.append(w);REF.append(ref)
        # fit smooth error: e_s(t) = R(t) l + c_w  (one body-frame lever arm per record, one world offset per window)
        Es=[mavg(e,25) for e in E]
        nw=len(E); A=[];b=[]
        for j,(es,R) in enumerate(zip(Es,Rs)):
            for i in range(0,len(es),5):
                row=np.zeros((3,3+3*nw)); row[:,:3]=R[i]; row[:,3+3*j:6+3*j]=np.eye(3); A.append(row); b.append(es[i])
        A=np.vstack(A); b=np.concatenate(b); x=np.linalg.lstsq(A,b,rcond=None)[0]; l=x[:3]
        for j,(e,es,R,ref) in enumerate(zip(E,Es,Rs,REF)):
            lever=np.einsum('tij,j->ti',R,l); cw=x[3+3*j:6+3*j]
            ii=v1.instants(0,Tp,24); L0=v1.polyline(ref[ii])
            tot.append(dict(rec=n,lever_mm=float(np.linalg.norm(l)*1e3),
                ex_total=v1.polyline((ref+e)[ii])-L0, ex_smooth=v1.polyline((ref+es)[ii])-L0,
                ex_lever=v1.polyline((ref+lever)[ii])-L0,
                var_expl=1-np.mean((es-lever-cw)**2)/np.mean((es-es.mean(0))**2)))
    t=tot
    print('==',ver,'windows',len(t))
    for n in v1.REAL:
        q=[x for x in t if x['rec']==n]
        if q: print('  %-9s |l|=%4.0f mm  var explained by R(t)l+c: %.2f  | m=24 excess mm: total %4.0f smooth %4.0f lever-model %4.0f'%(n,q[0]['lever_mm'],np.mean([x['var_expl'] for x in q]),np.mean([x['ex_total'] for x in q])*1e3,np.mean([x['ex_smooth'] for x in q])*1e3,np.mean([x['ex_lever'] for x in q])*1e3))
    print('  ALL: total %.0f  smooth %.0f  lever-model %.0f mm; var explained median %.2f'%(np.mean([x['ex_total'] for x in t])*1e3,np.mean([x['ex_smooth'] for x in t])*1e3,np.mean([x['ex_lever'] for x in t])*1e3,np.median([x['var_expl'] for x in t])))
    out[ver]=t
json.dump(out,open('experiments/mstar_underprediction_diag_2026-09-26/lever_arm.json','w'),indent=1,default=float)
