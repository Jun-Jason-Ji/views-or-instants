"""Independent fixed-budget two-depth rigid registration route, CPU only."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
import cv2
import numpy as np
import torch
from feature_chain_core import classic_extractor,classic_matcher,depth_correspondences,score,summarize
from rgbd_odometry_cpu import advance
from continuous_motion_cpu import estimate
from limited_arc_cpu import limited_arc
from pose_propagation_cpu import polygon
from run_classical_feature_routes_cpu import ROOT,OLD,stamp,digest,load,save
from xfeat_cpu_adapter import XFeatAdapter,VENDOR,verify_vendor

OUT=ROOT/'experiments/rgbd_rigid_route_2026-09-13'
CLASSIC=ROOT/'experiments/classical_feature_routes_2026-09-13'
XFEAT=ROOT/'experiments/xfeat_direction_cpu_2026-09-13'
ITERATIONS=1000
DISTANCE=.03
MINIMUM=20
FRACTION=.25
SEED=1309


def fit_rigid(source,target):
    """Return target_from_source; reject line-like geometry, allow planes."""
    source,target=np.asarray(source,float),np.asarray(target,float)
    if source.shape!=target.shape or source.ndim!=2 or source.shape[1]!=3 or len(source)<3:
        raise ValueError('invalid_rigid_correspondences')
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('nonfinite_rigid_correspondences')
    centers=[source.mean(axis=0),target.mean(axis=0)]
    a,b=source-centers[0],target-centers[1]
    for points in [a,b]:
        s=np.linalg.svd(points,compute_uv=False)
        if s[1]<=1e-5 or s[1]<=1e-3*s[0]:raise ValueError('degenerate_rigid_geometry')
    u,_,vt=np.linalg.svd(a.T@b)
    correction=np.eye(3);correction[2,2]=1. if np.linalg.det(vt.T@u.T)>=0 else -1.
    rotation=vt.T@correction@u.T
    transform=np.eye(4);transform[:3,:3]=rotation;transform[:3,3]=centers[1]-rotation@centers[0]
    return transform


def residuals(source,target,transform):
    return np.linalg.norm(source@transform[:3,:3].T+transform[:3,3]-target,axis=1)


def support_ok(mask,pairs):
    return (int(mask.sum())>=MINIMUM and mask.mean()>=FRACTION and
            len(np.unique(pairs[mask,0]))>=MINIMUM and len(np.unique(pairs[mask,1]))>=MINIMUM)


def solve_rigid(source,target,pairs):
    """Fixed 1000 hypotheses; one consensus refit and unchanged metric gate."""
    start=perf_counter();source,target=np.asarray(source,float),np.asarray(target,float);pairs=np.asarray(pairs,np.int64)
    arrays=dict(source_xyz=source,target_xyz=target,pairs=pairs)
    metadata=dict(success=False,failure='insufficient_rigid_correspondences',iterations=ITERATIONS,
        hypotheses_attempted=0,hypotheses_fitted=0,degenerate_samples=0,duplicate_id_samples=0,refine_calls=0,
        correspondence_count=len(source),distance_threshold_m=DISTANCE,rng_seed=SEED)
    if source.shape!=target.shape or source.shape!=(len(pairs),3) or pairs.shape!=(len(source),2):
        raise ValueError('invalid_rigid_input_shape')
    if not np.isfinite(source).all() or not np.isfinite(target).all():raise ValueError('nonfinite_rigid_input')
    if len(source)<MINIMUM or len(np.unique(pairs[:,0]))<MINIMUM or len(np.unique(pairs[:,1]))<MINIMUM:
        metadata['wall_s']=perf_counter()-start
        return dict(metadata=metadata,arrays=arrays)
    rng=np.random.default_rng(SEED);best=None;best_key=(-1,-np.inf)
    for _ in range(ITERATIONS):
        metadata['hypotheses_attempted']+=1
        ii=rng.choice(len(source),3,replace=False)
        if len(np.unique(pairs[ii,0]))<3 or len(np.unique(pairs[ii,1]))<3:
            metadata['duplicate_id_samples']+=1;continue
        try:transform=fit_rigid(source[ii],target[ii])
        except (ValueError,np.linalg.LinAlgError):
            metadata['degenerate_samples']+=1;continue
        metadata['hypotheses_fitted']+=1
        error=residuals(source,target,transform);inside=error<=DISTANCE
        key=(int(inside.sum()),-float(error[inside].mean()) if inside.any() else -np.inf)
        if key>best_key:best_key=key;best=(transform,error,inside)
    metadata['failure']='rigid_consensus_failure'
    if best is not None:
        transform,error,inside=best
        arrays.update(raw_transform=transform,raw_residual_m=error,raw_inlier_indices=np.flatnonzero(inside))
        metadata['raw_support']=dict(inliers=int(inside.sum()),fraction=float(inside.mean()),
            unique_source=len(np.unique(pairs[inside,0])),unique_target=len(np.unique(pairs[inside,1])))
        if support_ok(inside,pairs):
            metadata['refine_calls']+=1
            try:
                transform=fit_rigid(source[inside],target[inside]);error=residuals(source,target,transform);inside=error<=DISTANCE
                arrays.update(refined_transform_candidate=transform,refined_residual_m=error,refined_inlier_indices=np.flatnonzero(inside))
                metadata['refined_support']=dict(inliers=int(inside.sum()),fraction=float(inside.mean()),
                    unique_source=len(np.unique(pairs[inside,0])),unique_target=len(np.unique(pairs[inside,1])),
                    median_inlier_residual_m=float(np.median(error[inside])) if inside.any() else None)
                if support_ok(inside,pairs):
                    metadata['success']=True;metadata['failure']=None;arrays['transform']=transform
                else:metadata['failure']='rigid_refined_support_failure'
            except (ValueError,np.linalg.LinAlgError) as exc:
                metadata['failure']='rigid_refine_degenerate';metadata['refine_error']=str(exc)
    metadata['wall_s']=perf_counter()-start
    return dict(metadata=metadata,arrays=arrays)


def paired_depth(source_keys,target_keys,source_depth,target_depth,pairs):
    xyz,_,kept=depth_correspondences(source_keys,target_keys,source_depth,pairs)
    first=np.asarray(kept,np.int64).reshape(-1,2)
    target_xyz,_,target_kept=depth_correspondences(target_keys,source_keys,target_depth,first[:,::-1])
    # Source feature IDs are unique in ORB one-way ratio and XFeat mutual NN.
    source_map={(int(q),int(t)):i for i,(q,t) in enumerate(first)}
    original_pairs=np.asarray([(q,t) for t,q in target_kept],np.int64).reshape(-1,2)
    ii=np.asarray([source_map[int(q),int(t)] for q,t in original_pairs],np.int64)
    return xyz[ii],target_xyz,original_pairs,len(first)


def query(window,sequence,budget,method,extract,match):
    start=perf_counter();read_s=feature_s=match_s=solver_s=0.;cache={};reads=[];events=[];poses=[np.eye(4)]
    selected=np.rint(np.linspace(0,len(window['frames'])-1,budget)).astype(int).tolist()
    times=[window['frames'][i]['t'] for i in selected];failure=None;values={};q=None
    data=ROOT/f'data_external/tum_rgbd_{sequence}_2026-09-13/public'
    try:
        for index in selected:
            frame=window['frames'][index];reads.append(index);tick=perf_counter()
            gray=cv2.imread(str(data/frame['rgb']),cv2.IMREAD_GRAYSCALE)
            depth=cv2.imread(str(data/frame['depth']),cv2.IMREAD_UNCHANGED);read_s+=perf_counter()-tick
            if gray is None or depth is None or gray.shape!=(480,640) or depth.shape!=(480,640) or depth.dtype!=np.uint16:
                raise ValueError('invalid_rgbd')
            tick=perf_counter()
            try:key,desc=extract(gray)
            finally:feature_s+=perf_counter()-tick
            cache[index]=(key,desc,depth)
        for a,b in zip(selected[:-1],selected[1:]):
            event=dict(a=a,b=b,failure=None);events.append(event);tick=perf_counter()
            try:
                matches=match(cache[a],cache[b])
                xyz,target_xyz,pairs,source_count=paired_depth(cache[a][0],cache[b][0],cache[a][2],cache[b][2],matches)
            finally:match_s+=perf_counter()-tick
            event.update(pairs_before_depth=len(matches),source_depth_survivors=source_count,paired_depth_survivors=len(pairs))
            result=solve_rigid(xyz,target_xyz,pairs);solver_s+=result['metadata']['wall_s']
            name=f'arrays/{sequence}_{window["offset"]}_{budget}_{method}_{a}_{b}.npz'
            with (OUT/name).open('xb') as f:np.savez_compressed(f,**result['arrays'])
            event.update(metadata=result['metadata'],arrays_file=name,failure=result['metadata']['failure'])
            if not result['metadata']['success']:raise ValueError(event['failure'])
            poses.append(advance(poses[-1],result['arrays']['transform']))
        pp=np.asarray(poses);tt=np.asarray(times);state=estimate(pp[:,:3,3],tt,.003);q=state['q']
        values=dict(state_mean=state['mean_path_m'],limited=limited_arc(pp[:,:3,3],tt),polygon=polygon(pp),displacement=float(np.linalg.norm(pp[-1,:3,3])))
    except (ValueError,cv2.error,np.linalg.LinAlgError) as exc:failure=str(exc)
    return dict(sequence=sequence,offset=window['offset'],budget=budget,method=method,selected=selected,times=times,
        failure=failure,poses=[p.tolist() for p in poses],values_m=values,q=q,events=events,read_indices=reads,
        rgbd_pairs_decoded=len(cache),feature_counts={str(i):len(row[0]) for i,row in cache.items()},
        wall_seconds=dict(read=read_s,features=feature_s,matching_and_depth=match_s,solver=solver_s,total=perf_counter()-start))


def verify():
    for p,h in load(OUT/'freeze.json')['hashes'].items():assert digest(ROOT/p)==h,p
    for p,h in load(OLD/'input_bindings.json')['hashes'].items():assert digest(ROOT/p)==h,p
    verify_vendor()


def freeze():
    verify_vendor();OUT.mkdir(exist_ok=False)
    plan={seq:load(OLD/seq/'public_windows.json') for seq in ['desk2','plant']}
    save(OUT/'plan.json',plan)
    save(OUT/'config.json',dict(utc=stamp(),scope='56 CPU queries:all14 existing development windows,B16/B12,ORB and pretrained sparseXFeat. Independent3D3D solver route,not the old fixed-pair depth intervention and not a relaxed PnP gate.',
        methods=['orb_rigid','xfeat_rigid'],frontends='ORB2000ratio.75Hamming versus frozenXFeat2000threshold.05MNN mincos=-1. Same grayscale640x480,frame indices and per-query feature cache.',
        depth='Both source and target use original rounded3x3median,>=5positive,spread<=.05m,z.3-4m,uint16/5000,originalK. Retain original rays and pair identities,only pairs with both depths valid. No LK,GT weighting,threshold tuning or added observations.',
        solver=dict(type='Three-point rigidSVD, target_from_source',rng_seed=SEED,hypotheses=ITERATIONS,distance_m=DISTANCE,
            sample='Three distinctsource andtargetIDs. Both centeredpointsets singularvalue1>1e-5m and>1e-3*singularvalue0; planes allowed,lines rejected.',
            score='Most3cmEuclidean3D inliers;tie smallest mean3Dinlier residual. No earlytermination.',
            gate='At least20correspondence inliers,20unique sourceIDs,20unique targetIDs,and>=.25inlierfraction. OneSVD refit then recompute same3cm support andsamegate;no secondrefit.',
            comparison='3cm is a new metric residual with independent acceptance semantics,not equivalent to prior2pxPnP tolerance. Both depths enable metric alignment but depthnoise and occlusion can bias3D residuals.'),
        scoring='Predictions/edgearrays sealed before GT parsing.Same forwardposecomposition,state_mean path estimator,andreferencegap .1s. Paired common-success pathabsoluteerror+positionRMSE;new/lost separately,bysequence/budget. Allfailures retained.',
        decision='No automaticGPUgate. Compareeachfront-end oldPnP andnewrigid,alsoORBvsXFeat withinrigid. Stable gains specificallyfromlearnedfeatures wouldmotivatepredeclared boundedGPUreplication;classicalrigid gainsalone do not. Do not rewrite the alreadyfailedXFeat directiongate.',
        costs='AllB RGBanddepth frames read/chargedforeachquery,evenearlygeometricfailure;all1000hypothesisattempts chargedfor each eligibleedge.OneCPUthread,two methods independentlyextractandmatch,noGPU.',
        runtime=dict(torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__)))
    files={Path(__file__).resolve(),ROOT/'tests/test_rgbd_rigid_route.py'}
    files.update(Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None)
        and Path(m.__file__).suffix=='.py' and Path(m.__file__).resolve().is_relative_to(ROOT/'src'))
    files.update([OUT/'plan.json',OUT/'config.json',OLD/'input_bindings.json'])
    for prior in [CLASSIC,XFEAT]:files.update(prior/name for name in ['predictions.json','prediction_seal.json','reference_seal.json','freeze.json'])
    files.update(p for p in VENDOR.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    save(OUT/'freeze.json',dict(utc=stamp(),hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(files)}))
    for p in files:
        if p.suffix=='.py':
            dest=OUT/'source_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    verify();print('Frozen56queries,two-depth3D3Droute.',flush=True)


def run():
    verify();torch.set_num_threads(1);cv2.setNumThreads(1);(OUT/'arrays').mkdir(exist_ok=False)
    started=stamp();tick=perf_counter();model=XFeatAdapter(device='cpu');initialization=perf_counter()-tick
    def extract(gray):
        out=model.extract(gray);return out['keypoints'],out['descriptors']
    def match(a,b):return model.match_features({'descriptors':a[1]},{'descriptors':b[1]})
    rows=[]
    for seq,windows in load(OUT/'plan.json').items():
        for wi,window in enumerate(windows):
            for bi,budget in enumerate([16,12]):
                methods=['orb_rigid','xfeat_rigid'] if (wi+bi)%2==0 else ['xfeat_rigid','orb_rigid']
                for method in methods:
                    rows.append(query(window,seq,budget,method,classic_extractor('orb') if method=='orb_rigid' else extract,
                        classic_matcher('orb') if method=='orb_rigid' else match))
            print(seq,window['offset'],'4queries complete',flush=True)
    assert len(rows)==56 and all(len(r['selected'])==r['budget'] and r['read_indices']==r['selected'] for r in rows)
    save(OUT/'predictions.json',rows)
    events=[e for r in rows for e in r['events']];metas=[e['metadata'] for e in events if 'metadata' in e]
    save(OUT/'run_accounting.json',dict(started_utc=started,completed_utc=stamp(),elapsed_s=perf_counter()-tick,
        model_initialization_s=initialization,queries=len(rows),rgbd_pairs_decoded=sum(r['rgbd_pairs_decoded'] for r in rows),
        geometry_attempts=len(events),hypotheses_attempted=sum(m['hypotheses_attempted'] for m in metas),
        hypotheses_fitted=sum(m['hypotheses_fitted'] for m in metas),refine_calls=sum(m['refine_calls'] for m in metas),
        solver_wall_s=sum(r['wall_seconds']['solver'] for r in rows),query_wall_s=sum(r['wall_seconds']['total'] for r in rows),
        gpu_initialized=torch.cuda.is_initialized(),new_frame_times=0))
    assert not torch.cuda.is_initialized();verify()
    paths=[OUT/'predictions.json',OUT/'run_accounting.json']+sorted((OUT/'arrays').glob('*.npz'))
    save(OUT/'prediction_seal.json',dict(completed_utc=stamp(),hashes={str(p.relative_to(OUT)):digest(p) for p in paths}))
    print('Sealed56queries beforeGT.',flush=True)


def evaluate():
    verify();seal=load(OUT/'prediction_seal.json')
    for p,h in seal['hashes'].items():assert digest(OUT/p)==h,p
    started=stamp();assert seal['completed_utc']<started
    rows=load(OUT/'predictions.json')+[r for r in load(CLASSIC/'predictions.json') if r['method']=='orb']+load(XFEAT/'predictions.json')
    assert len(rows)==112
    refs={s:np.loadtxt(ROOT/f'data_external/tum_rgbd_{s}_2026-09-13/reference/groundtruth.txt') for s in ['desk2','plant']}
    evaluated=score(rows,refs);summary=summarize(evaluated)
    save(OUT/'evaluation.json',evaluated);save(OUT/'summary.json',summary)
    save(OUT/'validation.json',dict(scoring_started_utc=started,completed_utc=stamp(),predictions_sealed_before_scoring=True,
        prediction_seal_sha256=digest(OUT/'prediction_seal.json'),queries=56,total_compared=112,
        no_training_or_gpu_gate=True,new_frame_times=0,gpu_initialized=False))
    save(OUT/'reference_seal.json',dict(completed_utc=stamp(),hashes={n:digest(OUT/n) for n in ['evaluation.json','summary.json','validation.json']}))
    print(json.dumps({b:r['pooled']['methods'] for b,r in summary.items()},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','verify','run','evaluate'])
    globals()[parser.parse_args().action]()
