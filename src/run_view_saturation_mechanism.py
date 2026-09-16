"""Why does adding views stop helping? Three measurements, all reference-scored.

1. The i.i.d.-pixel-noise geometry model predicts how much k=7 should beat k=3.
2. The realised position-error ratio, and the angle between the k=3 and k=7 error
   vectors: a large common-mode component cannot be averaged away by more views.
3. Leave-one-camera-out: whether any single view is actively harmful.
"""
import json
from pathlib import Path
import numpy as np
from audit_mcalib_sample import read
from prepare_mcalib_cache import load_record
from run_allocation_sweep import subset_points
from view_geometry_analysis import cov

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data_external/mcalib_2026-09-13/verified/mocapAssistedRecord'
CAL=ROOT/'experiments/mcalib_3d_cache_2026-09-13_verified/public/calibration.json'
OUT=ROOT/'experiments/view_saturation_mechanism_v1_2026-09-15';RULE=('216f21c1','3e0f8f0','44c4b2e')


def main():
    cams=json.loads(CAL.read_text())['cameras'];ids=sorted(cams);OUT.mkdir(exist_ok=True);res={}
    for rec in ('record19','record20','record21','record22','record23'):
        uv={c:read(DATA/rec/f'centroidsUV{c}.pkl') for c in ids};t,truth,_,_=load_record(rec,ids)
        p3,_=subset_points(uv,cams,RULE,3000);p7,_=subset_points(uv,cams,tuple(ids),3000)
        ok=np.all(np.isfinite(p3),axis=1)&np.all(np.isfinite(p7),axis=1)
        d3=p3[ok]-truth[ok];d7=p7[ok]-truth[ok]
        rms3=float(np.sqrt((np.linalg.norm(d3,axis=1)**2).mean()));rms7=float(np.sqrt((np.linalg.norm(d7,axis=1)**2).mean()))
        cos=np.sum(d3*d7,axis=1)/np.maximum(np.linalg.norm(d3,axis=1)*np.linalg.norm(d7,axis=1),1e-12)
        pred=[]
        for x in truth[ok][::40]:
            c3=cov(x,cams,RULE);c7=cov(x,cams,ids)
            if c3 is not None and c7 is not None and np.trace(c3)>0 and np.trace(c7)>0:
                pred.append(np.sqrt(np.trace(c7)/np.trace(c3)))
        loo={}
        for c in ids:
            sub=tuple(x for x in ids if x!=c);p,_=subset_points(uv,cams,sub,3000);o=np.all(np.isfinite(p),axis=1)&ok
            loo[c]=float(np.sqrt((np.linalg.norm(p[o]-truth[o],axis=1)**2).mean()))
        res[rec]=dict(n=int(ok.sum()),rms3_mm=rms3*1000,rms7_mm=rms7*1000,realised_ratio=rms7/rms3,
                      predicted_ratio_iid=float(np.mean(pred)),median_cos_err3_err7=float(np.median(cos)),
                      leave_one_out_rms_mm={c:v*1000 for c,v in loo.items()},
                      best_leave_one_out=min(loo,key=loo.get),best_loo_rms_mm=min(loo.values())*1000)
        r=res[rec]
        print(f"{rec}: k3 {r['rms3_mm']:6.3f}  k7 {r['rms7_mm']:6.3f} mm  realised ratio {r['realised_ratio']:.3f} "
              f"vs i.i.d. prediction {r['predicted_ratio_iid']:.3f}   median cos(e3,e7) {r['median_cos_err3_err7']:.3f}   "
              f"best 6-view = drop {r['best_leave_one_out']} -> {r['best_loo_rms_mm']:.3f} mm")
    (OUT/'mechanism.json').write_text(json.dumps(res,indent=1))
    rr=[v['realised_ratio'] for v in res.values()];pp=[v['predicted_ratio_iid'] for v in res.values()]
    cc=[v['median_cos_err3_err7'] for v in res.values()]
    print(f"\nacross 5 records: realised ratio {min(rr):.3f}-{max(rr):.3f} (mean {np.mean(rr):.3f}); "
          f"i.i.d. prediction {np.mean(pp):.3f}; median cos {min(cc):.3f}-{max(cc):.3f}")
    harmful=[(k,v['rms7_mm'],v['best_leave_one_out'],v['best_loo_rms_mm']) for k,v in res.items() if v['best_loo_rms_mm']<v['rms7_mm']]
    print(f"records where dropping one view improves the 7-view solution: {len(harmful)}/5")
    for k,a,c,b in harmful:print(f"   {k}: {a:.3f} -> {b:.3f} mm by dropping {c}  ({(1-b/a)*100:.1f} % better)")


if __name__=='__main__':main()
