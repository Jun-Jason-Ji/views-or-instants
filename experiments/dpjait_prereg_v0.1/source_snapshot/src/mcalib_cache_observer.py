"""3D inference from selected multi-camera caches only; no mocap access."""
import json,random
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from causal_denoise_baselines import next_index,smooth5
from visual_observer import choose_frames,quantities


def triangulate(observations,cameras):
    rows=[];used=[]
    for cid,uv in observations.items():
        if uv is None:continue
        c=cameras[cid];K=np.asarray(c['K']);D=np.asarray(c['D'])
        x,y=cv2.undistortPoints(np.asarray(uv,dtype=float).reshape(1,1,2),K,D).reshape(2)
        P=np.column_stack([np.asarray(c['R']),np.asarray(c['t_m']).reshape(3)])
        rows.extend([x*P[2]-P[0],y*P[2]-P[1]]);used.append(P)
    if len(used)<2:raise ValueError('fewer_than_two_visible_cameras')
    _,s,v=np.linalg.svd(np.array(rows));h=v[-1]
    if abs(h[3])<1e-12 or s[-2]<1e-12:raise ValueError('degenerate_triangulation')
    p=h[:3]/h[3]
    if not np.all(np.isfinite(p)) or any((P@np.r_[p,1])[2]<=0 for P in used):raise ValueError('invalid_depth')
    return p


class CacheObserver:
    def __init__(self,scene,budget,cameras):
        self.root=Path(scene);self.budget=budget;self.cameras=cameras
        self.meta=json.loads((self.root/'input.json').read_text());self.cache={};self.log=[];self.load_s=0.;self.triangulate_s=0.
    def observe(self,i):
        if type(i) is not int or not 0<=i<len(self.meta['timestamps_s']):raise ValueError('invalid_index')
        if i not in self.cache and len(self.cache)>=self.budget:raise ValueError('budget_exceeded')
        self.log.append(i)
        if i not in self.cache:
            tick=perf_counter();data=json.loads((self.root/f'{i:03d}.json').read_text());self.load_s+=perf_counter()-tick
            self.cache[i]=data
        tick=perf_counter()
        try:return triangulate(self.cache[i],self.cameras)
        finally:self.triangulate_s+=perf_counter()-tick


def query(scene,budget,method,seed,cameras):
    start=perf_counter();obs=CacheObserver(scene,budget,cameras);points={};policy_s=0.;estimate_s=0.;values=None;failure=None
    try:
        if method.startswith('sequential'):
            for i in (0,len(obs.meta['timestamps_s'])-1):points[i]=obs.observe(i)
            while len(points)<budget:
                tick=perf_counter();i=next_index(points,obs.meta['timestamps_s']);policy_s+=perf_counter()-tick
                points[i]=obs.observe(i)
        else:
            tick=perf_counter();ids=choose_frames(len(obs.meta['timestamps_s']),budget,'random' if method.startswith('random') else 'uniform',seed);policy_s+=perf_counter()-tick
            for i in ids:points[i]=obs.observe(i)
        tick=perf_counter();ids=sorted(points);p=np.array([points[i] for i in ids])
        if method.endswith('_smooth5'):p=smooth5(p,[obs.meta['timestamps_s'][i] for i in ids])
        values=quantities(p,obs.meta['timestamps_s'][-1]-obs.meta['timestamps_s'][0]);estimate_s=perf_counter()-tick
    except ValueError as e:failure=str(e)
    return {'values':values,'indices':sorted(points),'ledger':{'protocol':'B_cached_2D_observations','e2e_s':perf_counter()-start,
        'load_s':obs.load_s,'triangulate_s':obs.triangulate_s,'policy_s':policy_s,'estimate_s':estimate_s,
        'timestamp_requests':len(obs.cache),'camera_frame_requests':len(obs.cache)*len(cameras),
        'missing_camera_observations':sum(v is None for data in obs.cache.values() for v in data.values()),
        'access_log':obs.log,'preview_frames':0,'failure':failure,'hardware':'CPU','precision':'float64','batch_size':1,
        'visual_tokens':None,'peak_gpu_memory':None,'publisher_cache_build_cost':None,'raw_rgb_decoded_in_query':0}}


def run(public,config,output):
    cfg=json.loads(config.read_text());cameras=json.loads((public/'calibration.json').read_text())['cameras'];jobs=[]
    for scene in sorted((public/'scenes').iterdir()):
        for b in cfg['budgets']:
            for method in cfg['methods']:
                for seed in range(cfg['random_seeds'] if method.startswith('random') else 1):
                    for repeat in range(1 if method.startswith('random') else cfg['deterministic_repeats']):jobs.append((scene,b,method,seed,repeat))
        for method in ('full33','full33_smooth5'):
            for repeat in range(cfg['deterministic_repeats']):jobs.append((scene,33,method,0,repeat))
    random.Random(cfg['seed']).shuffle(jobs)
    with output.open('x') as f:
        for order,(scene,b,method,seed,repeat) in enumerate(jobs):
            r=query(scene,b,method,seed,cameras);r.update({'scene_id':scene.name,'budget':b,'method':method,'seed':seed,'repeat':repeat,'order':order})
            f.write(json.dumps(r)+'\n')
            if order%1000==0:print(f'{order}/{len(jobs)} queries',flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--public',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.public,a.config,a.output)
