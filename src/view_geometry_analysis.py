"""Why does accuracy saturate at k=3? Compare the i.i.d.-pixel-noise prediction with the realised ratio.

Under independent isotropic pixel noise the triangulation covariance is
Cov(X) = sigma_px^2 (sum_i J_i^T J_i)^{-1}, with J_i the 2x3 projection Jacobian.
That model predicts how much a 7-view solution should beat a 3-view one. We compare
that prediction with the ratio actually observed in the path-length error.
"""
import json
from pathlib import Path
import numpy as np
from audit_mcalib_sample import read
from run_allocation_sweep import subset_points

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data_external/mcalib_2026-09-13/verified/mocapAssistedRecord'
CAL=ROOT/'experiments/mcalib_3d_cache_2026-09-13_verified/public/calibration.json';RULE=('216f21c1','3e0f8f0','44c4b2e')


def jacobian(X,cam):
    R=np.asarray(cam['R']);t=np.asarray(cam['t_m']);K=np.asarray(cam['K']);Xc=R@X+t
    if Xc[2]<=0:return None
    fx,fy=K[0,0],K[1,1];x,y,z=Xc
    dpdXc=np.array([[fx/z,0,-fx*x/z**2],[0,fy/z,-fy*y/z**2]])
    return dpdXc@R


def cov(X,cams,subset):
    A=np.zeros((3,3))
    for c in subset:
        J=jacobian(X,cams[c])
        if J is None:return None
        A+=J.T@J
    try:return np.linalg.inv(A)
    except np.linalg.LinAlgError:return None


def main():
    cams=json.loads(CAL.read_text())['cameras'];ids=sorted(cams)
    rec='record19';uv={c:read(DATA/rec/f'centroidsUV{c}.pkl') for c in ids}
    pts,_=subset_points(uv,cams,tuple(ids),3000)
    ok=np.all(np.isfinite(pts),axis=1);X=pts[ok][::25]
    r3=[];r7=[];per_cam={}
    for x in X:
        c3=cov(x,cams,RULE);c7=cov(x,cams,ids)
        if c3 is None or c7 is None:continue
        r3.append(np.sqrt(np.trace(c3)));r7.append(np.sqrt(np.trace(c7)))
    r3=np.array(r3);r7=np.array(r7)
    print(f'points analysed: {len(r3)}')
    print(f'predicted position sigma per unit pixel noise:  k=3 {r3.mean()*1000:.3f} mm/px   k=7 {r7.mean()*1000:.3f} mm/px')
    print(f'predicted k=7 / k=3 error ratio under i.i.d. pixel noise: {(r7/r3).mean():.3f}  (i.e. k=7 should be {(1-(r7/r3).mean())*100:.1f} % better)')
    # marginal value of each additional camera, greedily ordered by information gain
    chosen=[];remaining=list(ids)
    while remaining:
        best=None
        for c in remaining:
            s=chosen+[c];v=[cov(x,cams,s) for x in X[:200]]
            v=[np.sqrt(np.trace(m)) for m in v if m is not None]
            if len(v)<100:continue
            sc=float(np.mean(v))
            if best is None or sc<best[0]:best=(sc,c)
        if best is None:break
        chosen.append(best[1]);remaining.remove(best[1])
        print(f'  {len(chosen)} views -> predicted sigma {best[0]*1000:7.3f} mm/px   ({chosen[-1]})')
    # realised ratio from the confirmation set
    rows=json.load(open(ROOT/'experiments/allocation_confirm_v1_2026-09-15/scored.json'))
    all7='+'.join(sorted({c for r in rows if r['views']==7 for c in r['cameras'].split('+')}))
    e=lambda cam,m:np.mean([abs(r['error_m']) for r in rows if r['cameras']==cam and r['moments']==m and r['method']=='state_mean' and r['error_m'] is not None])
    for m in (25,33):
        print(f'realised k=7 / k=3 path-error ratio at m={m}: {e(all7,m)/e("+".join(RULE),m):.3f}')


if __name__=='__main__':main()


def position_level():
    """Direct test at the position level: predicted vs realised k=7 / k=3 error ratio, and cross-view residual correlation."""
    from prepare_mcalib_cache import load_record
    cams=json.loads(CAL.read_text())['cameras'];ids=sorted(cams);out={}
    for rec in ('record19','record20','record21'):
        uv={c:read(DATA/rec/f'centroidsUV{c}.pkl') for c in ids}
        t,truth,_,_=load_record(rec,ids)
        p3,_=subset_points(uv,cams,RULE,3000);p7,_=subset_points(uv,cams,tuple(ids),3000)
        ok=np.all(np.isfinite(p3),axis=1)&np.all(np.isfinite(p7),axis=1)
        e3=np.linalg.norm(p3[ok]-truth[ok],axis=1);e7=np.linalg.norm(p7[ok]-truth[ok],axis=1)
        # common-mode: how much of the 3-view error is shared with the 7-view error?
        d3=p3[ok]-truth[ok];d7=p7[ok]-truth[ok]
        shared=np.sum(d3*d7,axis=1)/np.maximum(np.linalg.norm(d3,axis=1)*np.linalg.norm(d7,axis=1),1e-12)
        out[rec]=dict(n=int(ok.sum()),rms3_mm=float(np.sqrt((e3**2).mean())*1000),rms7_mm=float(np.sqrt((e7**2).mean())*1000),
                      median3_mm=float(np.median(e3)*1000),median7_mm=float(np.median(e7)*1000),
                      realised_ratio=float(np.sqrt((e7**2).mean())/np.sqrt((e3**2).mean())),
                      median_cos_between_error_vectors=float(np.median(shared)))
        r=out[rec];print(f"{rec}: n={r['n']} RMS k3={r['rms3_mm']:.3f} mm  k7={r['rms7_mm']:.3f} mm  realised ratio={r['realised_ratio']:.3f} "
                         f"(i.i.d. model predicts 0.588)   median cos(err3,err7)={r['median_cos_between_error_vectors']:.3f}")
    Path(ROOT/'experiments/view_geometry_v1_2026-09-15').mkdir(exist_ok=True)
    (ROOT/'experiments/view_geometry_v1_2026-09-15/position_level.json').write_text(json.dumps(out,indent=1))


if __name__=='__main__':position_level()
