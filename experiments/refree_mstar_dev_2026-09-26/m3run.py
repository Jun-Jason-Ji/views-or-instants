import sys, json, numpy as np
sys.path.insert(0,'src')
import refree_mstar_dev as R
import dpjait_noise_runner as v1
rows=[]
for ver,exp in R.EXPS.items():
    cfg=json.loads((exp/'config.json').read_text()); dec=json.loads((exp/'decision.json').read_text())['result']['families']
    for fam,names in R.FAMS.items():
        lv=[l for l in dec[fam]['levels'] if l.get('ladder','A')=='A']
        for li,l in enumerate(lv):
            Ls=[];Pw_all=[];sigs=[]
            for n in names:
                sc=json.loads((exp/'scored'/(n+'.json')).read_text()); dz=np.load(exp/'predictions'/(n+'_dense.npz'))
                for dr in sorted({r['drone'] for r in sc['rows']}):
                    for rep in (range(cfg['ladder_a_reps']) if li else [0]):
                        key='A|%d|%d|%s'%(li,rep,dr)
                        if key not in dz.files: continue
                        P=dz[key].astype(float)
                        for w in sorted({r['window'] for r in sc['rows'] if r['drone']==dr}):
                            f0=w*R.TP
                            if f0+R.TP+1>len(P): continue
                            Pw=R.fill_gaps(P[f0:f0+R.TP+1])
                            if Pw is None: continue
                            Pw_all.append(Pw); sigs.append(R.sigma_white(Pw))
            sig=float(np.sqrt(np.mean(np.square(sigs))))
            Ls=np.mean([[v1.polyline(Pw[v1.instants(0,R.TP,int(m))]) for m in R.GRID] for Pw in Pw_all],axis=0)
            Is=np.mean([[R.inflation_exact(Pw,m,sig) for m in R.GRID] for Pw in Pw_all],axis=0)
            res={}
            for lo,hi in ((6,100),(10,100),(10,50),(17,134)):
                res['M3_%d_%d'%(lo,hi)]=R.m3_predict(Ls,Is,lo,hi)[0]
            eq1=l.get('pred_uncorrected',l.get('predicted'))
            rows.append(dict(chain=ver,family=fam,level=li,sigma_ref_mm=l['sigma_a_mm'],sigma_white_mm=sig*1e3,m_star=l['m_star'],crossing=l['crossing'],eq1_refbased=eq1,**res))
            print('%-9s %-7s L%d sig_ref %6.1f sig_white %6.1f | m*=%3d cross=%6.1f eq1=%5.1f | '%(ver,fam,li,l['sigma_a_mm'],sig*1e3,l['m_star'],l['crossing'] or np.nan,eq1)+' '.join('%s=%5.1f'%(k,v) for k,v in res.items()))
json.dump(rows,open('experiments/refree_mstar_dev_2026-09-26/dev_m3.json','w'),indent=1)
print()
for k in ['eq1_refbased']+[k for k in rows[0] if k.startswith('M3')]:
    r=np.array([q[k]/q['m_star'] for q in rows],float); ok=np.isfinite(r)
    for sub,msk in (('all',np.ones(len(rows),bool)),('real',np.array([q['family']=='real' for q in rows])),('native',np.array([q['level']==0 for q in rows]))):
        rr=r[ok&msk]; print('  %-14s %-6s %2d/%2d in x1.5  median %.2f  rms-log %.2f'%(k,sub,np.sum((rr>=2/3)&(rr<=1.5)),msk.sum(),np.median(rr),np.sqrt(np.mean(np.log(rr)**2))))
