"""Build disjoint public measurement cache and private reference for 3D diagnostic."""
import json,hashlib,shutil
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from audit_mcalib_sample import read
from causal_denoise_baselines import smooth5
from visual_observer import quantities
from freeze_pilot_protocol import verify

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data_external/mcalib_2026-09-13/verified'


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(data,f,indent=2)


def load_record(name,cameras):
    folder=DATA/'mocapAssistedRecord'/name;meta=read(folder/'metadata.pkl')
    if meta['exposure']!=-10 or meta['qualisysFrequencyDivisor']!=4 or not meta['hardwareTrigger']:raise ValueError('Unexpected time mapping')
    lines=(folder/'mocapData.tsv').read_text().splitlines();header=next(i for i,s in enumerate(lines) if s.startswith('Frame\tTime'))
    a=np.array([[float(v) if v.strip() else np.nan for v in s.split('\t')[:5]] for s in lines[header+1:] if s.strip()])
    if not np.all(np.isfinite(a)) or not np.all(np.diff(a[:,1])>0):raise ValueError('Invalid reference, do not silently fill')
    a[:,2:5]/=1000
    uv={cid:read(folder/f'centroidsUV{cid}.pkl') for cid in cameras}
    n=len(uv[cameras[0]])
    if any(len(v)!=n for v in uv.values()):raise ValueError('Camera length mismatch')
    t=np.arange(n)*.02+.00006969+.000996/2
    if t[-1]>a[-1,1]:raise ValueError('Reference extrapolation')
    xyz=np.column_stack([np.interp(t,a[:,1],a[:,k]) for k in (2,3,4)])
    return t,xyz,uv,a


def main(out,config):
    out=out.resolve();config=config.resolve()
    verify(ROOT);cfg=json.loads(config.read_text());out.mkdir(exist_ok=False,parents=True);start=perf_counter()
    save(out/'config.json',cfg)
    t,xyz,uv,_=load_record(cfg['calibration_record'],cfg['cameras']);cameras={};qa=[]
    for cid in cfg['cameras']:
        intr=read(DATA/'intrinsicInit'/f'{cid}.pkl');observed=np.array([p if p is not None else [np.nan,np.nan] for p in uv[cid]])
        ids=np.flatnonzero(np.all(np.isfinite(observed),axis=1))[::cfg['calibration_stride']]
        cv2.setRNGSeed(cfg['seed'])
        ok,rvec,tvec,inliers=cv2.solvePnPRansac(xyz[ids],observed[ids],intr['K'],intr['D'],iterationsCount=cfg['ransac_iterations'],reprojectionError=cfg['ransac_reprojection_px'],confidence=.999)
        if not ok or inliers is None:raise ValueError('Calibration failure: '+cid)
        fit=ids[inliers[:,0]];ok,rvec,tvec=cv2.solvePnP(xyz[fit],observed[fit],intr['K'],intr['D'],rvec,tvec,True)
        if not ok:raise ValueError('Calibration refine failure')
        R=cv2.Rodrigues(rvec)[0];cameras[cid]={'K':intr['K'].tolist(),'D':intr['D'].tolist(),'R':R.tolist(),'t_m':tvec.reshape(3).tolist()}
        error=np.linalg.norm(cv2.projectPoints(xyz[fit],rvec,tvec,intr['K'],intr['D'])[0].reshape(-1,2)-observed[fit],axis=1)
        qa.append({'camera':cid,'record':cfg['calibration_record'],'sampled':len(ids),'inliers':len(fit),'fit_median_px':float(np.median(error))})
    save(out/'public/calibration.json',{'calibration_record':cfg['calibration_record'],'cameras':cameras})
    save(out/'calibration_fit_qa.json',qa);manifest=[]
    for record in cfg['development_records']:
        t,xyz,uv,a=load_record(record,cfg['cameras'])
        for begin in cfg['start_frames']:
            end=begin+cfg['duration_frames'];indices=begin+np.rint(np.linspace(0,cfg['duration_frames'],cfg['candidate_count'])).astype(int)
            sid=f'scene_{len(manifest):03d}';scene=out/'public/scenes'/sid
            save(scene/'input.json',{'timestamps_s':t[indices].tolist(),'target':'passive marker','coordinate_unit':'m'})
            for i,j in enumerate(indices):
                entry={cid:np.asarray(uv[cid][j]).tolist() if uv[cid][j] is not None else None for cid in cfg['cameras']}
                save(scene/f'{i:03d}.json',entry)
            middle=a[(a[:,1]>t[begin])&(a[:,1]<t[end])];ref=np.vstack([xyz[begin],middle[:,2:5],xyz[end]])
            rt=np.r_[t[begin],middle[:,1],t[end]];duration=t[end]-t[begin];truth=quantities(ref,duration)
            sensitivity={}
            for stride in (1,2,4):
                ii=np.unique(np.r_[np.arange(0,len(ref),stride),len(ref)-1]);sensitivity[f'stride{stride}']=quantities(ref[ii],duration)['path_length']
            sensitivity['smooth5_200Hz']=quantities(smooth5(ref,rt),duration)['path_length']
            save(out/'private'/f'{sid}.json',{'record':record,'source_indices':indices.tolist(),'reference':truth,
                'candidate_reference_xyz_m':xyz[indices].tolist(),'reference_path_sensitivity_m':sensitivity,
                'reference_times_s':rt.tolist(),'reference_xyz_m':ref.tolist()})
            manifest.append({'scene_id':sid,'record':record,'start_frame':begin,'end_frame':end})
    save(out/'private/manifest.json',manifest)
    save(out/'preparation_cost.json',{'adapter_wall_s':perf_counter()-start,'includes_calibration_and_reference_preparation':True,
        'publisher_detection_cache_construction_s':None,'not_end_to_end_video_cost':True})
    sources=[config,ROOT/'src/prepare_mcalib_cache.py',ROOT/'src/mcalib_cache_observer.py',ROOT/'src/causal_denoise_baselines.py',ROOT/'src/visual_observer.py',ROOT/'src/audit_mcalib_sample.py']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    hashes.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*.json')})
    save(out/'pre_run_hashes.json',hashes)
    snap=out/'source_snapshot';snap.mkdir()
    for p in sources:shutil.copyfile(p,snap/p.name)
    print(json.dumps({'scenes':len(manifest),'records':2,'cameras':len(cameras),'calibration_record':cfg['calibration_record']}))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,required=True);a=p.parse_args();main(a.output,a.config)
