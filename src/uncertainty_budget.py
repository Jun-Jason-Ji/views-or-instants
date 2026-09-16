"""GUM/VIM-style uncertainty budget for the recommended configuration.

Components, for the pre-registered confirmation set at the recommended
allocation (k=3 views, m=33 instants, conditional-mean-path estimator):

  b_samp   systematic measurement error from finite temporal sampling
           (VIM 2.18 measurement bias): the mean signed error. Known in sign
           and magnitude, hence correctable in principle; reported uncorrected.
  u_A      Type A standard uncertainty: the experimental standard deviation of
           the signed error over the 60 windows.
  u_ref    Type B: realisation of the reference quantity value. Evaluated by
           decimating the 200 Hz reference and taking the resulting spread as
           the half-width of a rectangular distribution.
  u_ref    Type B: realisation of the reference quantity value.

u_c combines the Type A and Type B terms; U = 2 u_c. The reconstruction error
is NOT added as a further term: it is one of the effects that already varies
between windows and is therefore inside u_A. We instead use it as a consistency
check, and that check is informative. Propagating the measured per-point RMS
position error incoherently over the segments over-predicts the observed
dispersion by a factor of three, because a path length is invariant to a common
translation of its points: the common-mode component that dominates the POSITION
error largely cancels in the LENGTH. That is a property of this measurand, and
it is the same reason additional views buy so little.
"""
import json
from pathlib import Path
import numpy as np
from audit_mcalib_sample import read
from prepare_mcalib_cache import load_record

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data_external/mcalib_2026-09-13/verified/mocapAssistedRecord'
CAL=ROOT/'experiments/mcalib_3d_cache_2026-09-13_verified/public/calibration.json'
RULE='216f21c1+3e0f8f0+44c4b2e';OUT=ROOT/'experiments/uncertainty_budget_v1_2026-09-15'


def reference_sensitivity(records,ids):
    """Spread of the reference path length under decimation of the 200 Hz trace."""
    spread=[]
    for rec in records:
        t,xyz,_,dense=load_record(rec,ids)
        for start in range(0,3000,200):
            end=start+199;mid=dense[(dense[:,1]>t[start])&(dense[:,1]<t[end])]
            ref=np.vstack([xyz[start],mid[:,2:5],xyz[end]])
            lengths=[]
            for stride in (1,2,4):
                ii=np.unique(np.r_[np.arange(0,len(ref),stride),len(ref)-1])
                lengths.append(float(np.linalg.norm(np.diff(ref[ii],axis=0),axis=1).sum()))
            spread.append(max(lengths)-min(lengths))
    return float(np.mean(spread)),float(np.max(spread))


def main():
    cams=json.loads(CAL.read_text())['cameras'];ids=sorted(cams)
    records=[f'record{i}' for i in range(15,19)]
    rows=json.load(open(ROOT/'experiments/allocation_confirm_v1_2026-09-15/scored.json'))
    e=np.array([r['error_m'] for r in rows if r['cameras']==RULE and r['moments']==33
                and r['method']=='state_mean' and r['error_m'] is not None])
    n=len(e);bias=float(e.mean());uA=float(e.std(ddof=1));truth=float(np.mean([r['truth_m'] for r in rows
        if r['cameras']==RULE and r['moments']==33 and r['method']=='state_mean' and r['error_m'] is not None]))
    ref_mean,ref_max=reference_sensitivity(records,ids)
    u_ref=ref_mean/np.sqrt(3)                      # rectangular, half-width = mean decimation spread
    mech=json.load(open(ROOT/'experiments/view_saturation_mechanism_v1_2026-09-15/mechanism.json'))
    rms3=float(np.mean([v['rms3_mm'] for v in mech.values()]))/1000
    cosbar=float(np.mean([v['median_cos_err3_err7'] for v in mech.values()]))
    nseg=32;incoherent=rms3*np.sqrt(2*nseg)      # if every point erred independently
    uc=float(np.sqrt(uA**2+u_ref**2));U=2*uc
    budget=dict(configuration=dict(views=3,instants=33,frames=99,estimator='state_mean',
                windows=n,mean_measurand_m=truth,dataset='pre-registered confirmation, records 15-18'),
        components=[
            dict(symbol='b_samp',source='systematic error from finite temporal sampling (measurement bias, VIM 2.18)',
                 type='determined, uncorrected',value_mm=bias*1000,note='mean signed error; sign is negative because sparse sampling cuts corners'),
            dict(symbol='u_A',source='dispersion of the error over windows',type='A',value_mm=uA*1000,
                 dof=n-1,note='experimental standard deviation of the signed error'),
            dict(symbol='u_ref',source='realisation of the reference quantity value',type='B',value_mm=u_ref*1000,
                 note=f'rectangular, half-width {ref_mean*1000:.2f} mm from decimating the 200 Hz reference (max {ref_max*1000:.2f} mm)'),
        ],
        consistency_check=dict(
            per_point_rms_mm=rms3*1000,segments=nseg,
            incoherent_propagation_mm=incoherent*1000,observed_dispersion_mm=uA*1000,
            ratio=float(incoherent/ (uA if uA>0 else 1)),
            median_common_mode_cosine=cosbar,
            reading=('Incoherent propagation of the per-point error over-predicts the observed dispersion by '
                     f'{incoherent/uA:.1f}x. A path length is invariant to a common translation of its points, so the '
                     'common-mode component that dominates the position error largely cancels in the length. This is why '
                     'the length measurand is more robust than position, and why extra views, which can only reduce the '
                     'already-small independent component, change it so little.')),
        combined_standard_uncertainty_mm=uc*1000,expanded_uncertainty_k2_mm=U*1000,
        relative_expanded=U/truth,
        statement=(f'path length = measured value + {-bias*1000:.1f} mm (bias correction, not applied) '
                   f'with U = {U*1000:.1f} mm (k = 2, approximately 95 % coverage), '
                   f'i.e. {U/truth*100:.2f} % of a {truth:.2f} m measurand'),
        caveats=['Coverage is asserted from a normal approximation, not verified by a coverage study.',
                 'u_A is a top-down estimate over 60 windows of one rig and one motion type; it is not a bottom-up model of the instrument.',
                 'b_samp is configuration-specific: it changes with m and dominates every other term at m < 25.',
                 'Windows within a record share a rig, a calibration and a session, so they are not fully independent; the effective degrees of freedom are below 59.'])
    OUT.mkdir(exist_ok=True);(OUT/'budget.json').write_text(json.dumps(budget,indent=1))
    print(f"configuration: k=3, m=33, {n} windows, mean measurand {truth:.3f} m\n")
    print(f"{'component':<58} {'type':<22} {'value (mm)':>11}")
    for c in budget['components']:print(f"  {c['symbol']:<8}{c['source'][:48]:<48} {c['type']:<22} {c['value_mm']:>11.3f}")
    print(f"\n  combined standard uncertainty u_c            {uc*1000:>11.3f} mm")
    print(f"  expanded uncertainty U (k=2)                 {U*1000:>11.3f} mm  = {U/truth*100:.3f} % of the measurand")
    print(f"\n  {budget['statement']}")


if __name__=='__main__':main()
