"""Frozen raw RGB-D camera-motion smoke test, including charged flow preview."""
import argparse,json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone
from time import perf_counter
import cv2,numpy as np
from rgbd_odometry_cpu import features,relative,advance,flow_indices,K
from limited_arc_cpu import limited_arc
from continuous_motion_cpu import estimate

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data_external/tum_rgbd_xyz_2026-09-13'
OUT=ROOT/'experiments/tum_rgbd_cpu_2026-09-13'


def save(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)


def timestamps(path):
    return [(float(s.split()[0]),s.split()[1]) for s in path.read_text().splitlines() if s.strip() and not s.startswith('#')]


def prepare():
    OUT.mkdir(exist_ok=False)
    manifest=json.loads((DATA/'manifest.json').read_text())
    for f in manifest['saved']:
        if hashlib.sha256((DATA/f['path']).read_bytes()).hexdigest()!=f['sha256']:raise ValueError('data hash mismatch')
    rgb=timestamps(DATA/'public/rgb.txt');depth=timestamps(DATA/'public/depth.txt');dt=np.array([t for t,_ in depth]);matches=[]
    for i,(t,p) in enumerate(rgb):
        j=int(np.argmin(abs(dt-t)))
        if abs(dt[j]-t)<=.02:matches.append((abs(dt[j]-t),i,j))
    used=set();paired=[]
    for delta,i,j in sorted(matches):
        if j in used:continue
        used.add(j);paired.append(dict(t=rgb[i][0],rgb=rgb[i][1],depth=depth[j][1],sync_delta_s=float(delta)))
    paired.sort(key=lambda x:x['t']);base=rgb[0][0];windows=[]
    for offset in [1.,9.,17.]:
        begin=base+offset;end=begin+4.;rr=[r for r in paired if begin<=r['t']<=end]
        if len(rr)<33 or max(np.diff([r['t'] for r in rr]))>.1:raise ValueError('insufficient paired interval '+str(offset))
        windows.append(dict(offset=offset,frames=rr))
    save(OUT/'public_windows.json',windows)
    save(OUT/'config.json',dict(sequence='freiburg1_xyz',scope='Independent-source raw RGB-D CAMERA egomotion development; reference-only earlier contact. NOT fixed-camera target-motion validation.',
        offsets_s=[1.,9.,17.],duration_s=4.,association_max_s=.02,association='Each RGB nearest depth; retain nearest RGB per depth; unmatched frames counted.',
        rgb_count=len(rgb),depth_count=len(depth),paired_count=len(paired),rgb_unmatched=len(rgb)-len(paired),
        K=K.tolist(),depth_divisor=5000,distortion=None,calibration='Official recommended ROS default for registered depth; do not apply 1.035 scaling again.',
        methods=['uniform8','uniform16','uniform33','flow8','flow16','dense'],candidate_count=33,
        feature='ORB2000; ratio .75; previous depth 3x3 median, >=5 valid pixels, spread<=.05m, .3<=depth<=4m',
        geometry='PnP RANSAC200, 2px, .999; >=20 and25% inliers, iterative refine, positive depth, median residual<=1.5px',
        flow='33 RGB candidates decoded at full size and resized to160x120; Farneback; equal cumulative 75th percentile flow norm, endpoints required; all preview costs charged.',
        estimators=['polygon','limited','state_mean'],state_sigma_m=.003,failures='Any geometry failure fails whole query. No identity fill, no dropping failed queries.',
        evaluation='Reference read only after predictions saved. Native metric path/displacement; no fitted scale or trajectory alignment. Ref gap>0.1s invalidates scoring.',
        seed=1309,opencv_version=cv2.__version__,numpy_version=np.__version__,hardware='CPU single OpenCV thread; OS file cache uncontrolled, no end-to-end speedup claim from one run.'))
    paths=list((ROOT/'src').glob('*.py'))+[ROOT/'tests/test_rgbd_odometry.py',OUT/'config.json',OUT/'public_windows.json',DATA/'manifest.json']
    save(OUT/'freeze.json',dict(utc=datetime.now(timezone.utc).isoformat(),hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    for p in paths:
        if p.suffix=='.py':
            dest=OUT/'source_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)


def verify():
    for p,h in json.loads((OUT/'freeze.json').read_text())['hashes'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p


def run():
    verify();cv2.setNumThreads(1);cfg=json.loads((OUT/'config.json').read_text());rows=[]
    for window in json.loads((OUT/'public_windows.json').read_text()):
        full=window['frames'];candidate=np.rint(np.linspace(0,len(full)-1,33)).astype(int)
        for method in cfg['methods']:
            tick=perf_counter();graycache={};depthloads=0;rgbloads=0;poses=[np.eye(4)];stats=[];failure=None;values={};selected=[];preview_s=0
            def gray(i):
                nonlocal rgbloads
                if i not in graycache:
                    image=cv2.imread(str(DATA/'public'/full[i]['rgb']),cv2.IMREAD_GRAYSCALE)
                    if image is None:raise ValueError('decode_rgb_failure')
                    graycache[i]=image;rgbloads+=1
                return graycache[i]
            try:
                if method=='dense':selected=list(range(len(full)))
                elif method.startswith('uniform'):
                    b=int(method[7:]);selected=candidate[np.rint(np.linspace(0,32,b)).astype(int)].tolist()
                else:
                    pt=perf_counter();ii,_=flow_indices([gray(int(i)) for i in candidate],int(method[4:]));selected=candidate[ii].tolist();preview_s=perf_counter()-pt
                previous=None
                for i in selected:
                    g=gray(i);d=cv2.imread(str(DATA/'public'/full[i]['depth']),cv2.IMREAD_UNCHANGED);depthloads+=1
                    current=features(g,d)
                    if previous is not None:
                        T,info=relative(previous,current);poses.append(advance(poses[-1],T));stats.append(info)
                    previous=current
                xyz=np.array([T[:3,3] for T in poses]);times=np.array([full[i]['t'] for i in selected]);elapsed=times[-1]-times[0]
                values=dict(polygon=float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum()),limited=limited_arc(xyz,times),state_mean=estimate(xyz,times,.003)['mean_path_m'],displacement=float(np.linalg.norm(xyz[-1])))
            except (ValueError,cv2.error,np.linalg.LinAlgError) as e:failure=str(e)
            rows.append(dict(offset=window['offset'],method=method,failure=failure,selected=selected,times=[full[i]['t'] for i in selected],
                positions_m=[T[:3,3].tolist() for T in poses],values_m=values,
                average_speed_mps={} if failure else {k:v/elapsed for k,v in values.items() if k!='displacement'},
                geometry=stats,rgb_decoded=rgbloads,depth_decoded=depthloads,
                rgb_access_indices=list(graycache),query_wall_s=perf_counter()-tick,preview_wall_s=preview_s))
        print(window['offset'],'RGB-D predictions complete',flush=True)
    save(OUT/'predictions.json',rows);verify()


def evaluate():
    verify();rows=json.loads((OUT/'predictions.json').read_text());gt=np.loadtxt(DATA/'reference/groundtruth.txt');result=[]
    for r in rows:
        if not r['times']:invalid='query_no_times'
        else:
            begin,end=r['times'][0],r['times'][-1]
            if begin<gt[0,0] or end>gt[-1,0]:invalid='outside_reference'
            else:
                lo=max(0,np.searchsorted(gt[:,0],begin)-1);hi=min(len(gt),np.searchsorted(gt[:,0],end)+1)
                invalid='reference_gap' if np.max(np.diff(gt[lo:hi,0]))>.1 else None
        truth=None
        if invalid is None:
            endpoints=np.array([[np.interp(t,gt[:,0],gt[:,k]) for k in (1,2,3)] for t in (begin,end)])
            ref=np.vstack([endpoints[0],gt[(gt[:,0]>begin)&(gt[:,0]<end),1:4],endpoints[1]])
            truth=dict(path_m=float(np.linalg.norm(np.diff(ref,axis=0),axis=1).sum()),displacement_m=float(np.linalg.norm(endpoints[1]-endpoints[0])),duration_s=end-begin)
            ii=np.unique(np.r_[np.arange(0,len(ref),2),len(ref)-1]);truth['stride2_path_delta_m']=float(np.linalg.norm(np.diff(ref[ii],axis=0),axis=1).sum()-truth['path_m'])
        result.append(dict(offset=r['offset'],method=r['method'],failure=r['failure'],reference_failure=invalid,truth=truth,values_m=r['values_m'],
            errors_m=None if r['failure'] or invalid else {k:v-(truth['displacement_m'] if k=='displacement' else truth['path_m']) for k,v in r['values_m'].items()},
            average_speed_mps=r['average_speed_mps'],
            speed_errors_mps=None if r['failure'] or invalid else {k:v-truth['path_m']/truth['duration_s'] for k,v in r['average_speed_mps'].items()},
            rgb_decoded=r['rgb_decoded'],depth_decoded=r['depth_decoded'],query_wall_s=r['query_wall_s'],preview_wall_s=r['preview_wall_s']))
    save(OUT/'evaluation.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','evaluate']);a=p.parse_args();globals()[a.action]()
