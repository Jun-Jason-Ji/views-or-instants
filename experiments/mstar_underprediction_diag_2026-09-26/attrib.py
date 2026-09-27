import sys, json, numpy as np
sys.path.insert(0,'src')
import dpjait_noise_runner as v1
from pathlib import Path
fit=json.load(open('experiments/mstar_underprediction_diag_2026-09-26/deficit_fit.json'))
diag=json.load(open('experiments/mstar_underprediction_diag_2026-09-26/diagnosis.json'))
res={}
for ver in ['v0.1_blob','v0.2_yolo']:
  exp=Path('experiments/dpjait_noise_prereg_'+('v0.1' if ver.startswith('v0.1') else 'v0.2')); cfg=json.loads((exp/'config.json').read_text())
  Tp=cfg['t_primary']; grid=np.array(cfg['grid_primary'],float); Ts=Tp/25
  for fam in ['real','S12_D3','S13_D3']:
    names=v1.REAL if fam=='real' else [fam]
    D=np.array(fit[fam]['D'])/1000
    dg=diag[fam]; kap,spd=dg['kappa_median'],dg['speed_median']
    rows=[]
    for n in names:
      d=json.loads((exp/'scored'/(n+'.json')).read_text())
      rows+=[r for r in d['rows'] if r.get('T',Tp)==Tp and r.get('ladder','A')=='A' and r['err'] is not None]
    # 1/L from windows
    Ls={}
    for r in rows: Ls[(r.get('drone'),r['window'],r.get('record',''))]=r['reference_m']
    invL=np.mean([1/x for x in Ls.values()])
    dec=json.loads((exp/'decision.json').read_text())['result']['families'][fam]
    lv=[l for l in dec['levels'] if l.get('ladder','A')=='A']
    print('==',ver,fam)
    for li,l in enumerate(lv):
      sig=l['sigma_a_mm']/1000
      S=np.array([np.mean([r['err'] for r in rows if r['level']==li and r['m']==m]) for m in grid])
      I=S-D
      mc=l['crossing']
      if not mc: continue
      lg=np.log(grid); f=lambda y: np.interp(np.log(mc),lg,y)
      Da=-f(D); Ia=f(I)
      Df=Ts**3*kap**2*spd**3/(24*mc**2); If=2*sig**2*mc**2*invL
      eq1=Ts*np.sqrt(kap*spd**2/(7*sig))
      # local power-law exponents of the actual terms around crossing
      i=np.searchsorted(grid,mc); a,b=max(i-2,0),min(i+2,len(grid)-1)
      pD=np.polyfit(lg[a:b+1],np.log(-D[a:b+1]),1)[0]; 
      Iseg=I[a:b+1]; pI=np.polyfit(lg[a:b+1],np.log(np.clip(Iseg,1e-9,None)),1)[0]
      print('  sig %6.1f cross %6.1f eq1 %5.1f | Dact/Dform %.2f  Iact/Iform %.2f | eq1/cross %.2f  predicted (Df/Da*Ia/If)^.25 %.2f | local slopes D %.2f I %.2f'%(l['sigma_a_mm'],mc,eq1,Da/Df,Ia/If,eq1/mc,((Df/Da)*(Ia/If))**0.25,pD,pI))
