"""TUM fr3: cheap CPU front ends on more frames vs the frozen LighterGlue chain on 16 frames.

Development screen on the 15 already-seen fr3 windows (offsets 1/5/9/13/17 s, 4 s).
Frozen chain otherwise identical to calibrated_measurement_core_v1.query: uniform
frame selection, paired depth filters, solve_rigid, advance, state_mean/limited/polygon.
First failed edge fails the query (no bridging). Predictions sealed before GT is read.
"""
import json,argparse,hashlib
from datetime import datetime,timezone
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from calibrated_measurement_core_v1 import validate_calibration,paired_depth
from rgbd_rigid_route_cpu import solve_rigid
from rgbd_odometry_cpu import advance
from continuous_motion_cpu import estimate
from limited_arc_cpu import limited_arc
from pose_propagation_cpu import polygon,interpolate_pose,pose_errors
from feature_chain_core import classic_extractor,classic_matcher,reference_metrics

ROOT=Path(__file__).resolve().parents[1];FR3=ROOT/'experiments/fr3_weighted_confirmation_v1_2026-09-13';OUT=ROOT/'experiments/tum_dense_cheap_v1_2026-09-15'
K=[[535.4,0,320.1],[0,539.2,247.6],[0,0,1]]


class OrbModel:
    dev='cpu'
    def __init__(self):self.ex=classic_extractor('orb');self.ma=classic_matcher('orb')
    def extract(self,gray):k,d=self.ex(gray);return dict(keypoints=k,descriptors=d)
    def match_features(self,a,b):return self.ma((None,a['descriptors']),(None,b['descriptors']))


def make_model(name):
    if name=='orb':return OrbModel()
    if name=='xfeat':
        from xfeat_cpu_adapter import XFeatAdapter;return XFeatAdapter(device='cpu',top_k=2000)
    raise ValueError(name)


def query(window,sequence,budget,method,model,data_root,matrix):
    started=perf_counter();n=len(window['frames']);budget=min(budget,n)
    selected=np.rint(np.linspace(0,n-1,budget)).astype(int).tolist();times=[window['frames'][i]['t'] for i in selected]
    common=dict(sequence=sequence,offset=window['offset'],budget=int(budget),requested_budget=budget,frames_in_window=n,method=method,selected=selected,times=times)
    images={};read_s=0.
    for i in selected:
        f=window['frames'][i];tick=perf_counter();g=cv2.imread(str(data_root/f['rgb']),0);d=cv2.imread(str(data_root/f['depth']),-1);read_s+=perf_counter()-tick
        if g is None or d is None or g.shape!=(480,640) or d.shape!=(480,640) or d.dtype!=np.uint16:return dict(**common,failure='invalid_rgbd',poses=[],values_m={},events=[],wall_seconds=dict(read=read_s,total=perf_counter()-started))
        images[i]=(g,d)
    feats={};events=[];poses=[np.eye(4)];failure=None;values={};q=None;fs=ms=ss=0.
    try:
        for i in selected:
            tick=perf_counter();feats[i]=model.extract(images[i][0]);fs+=perf_counter()-tick
        for a,b in zip(selected[:-1],selected[1:]):
            ev=dict(a=a,b=b,failure=None);events.append(ev);tick=perf_counter()
            try:
                pairs=model.match_features(feats[a],feats[b]);ev['pairs']=int(len(pairs))
                src,tgt,kept,_=paired_depth(feats[a]['keypoints'],feats[b]['keypoints'],images[a][1],images[b][1],pairs,K=matrix);ev['depth_pairs']=len(kept)
            except (ValueError,cv2.error,np.linalg.LinAlgError,RuntimeError) as e:ev['failure']=str(e);raise
            finally:ms+=perf_counter()-tick
            tick=perf_counter()
            try:
                r=solve_rigid(src,tgt,kept);ev['inliers']=r['metadata'].get('refined_support',{}).get('inliers')
                if not r['metadata']['success']:raise ValueError(r['metadata']['failure'])
                T=r['arrays']['transform']
            except (ValueError,np.linalg.LinAlgError) as e:ev['failure']=str(e);raise
            finally:ss+=perf_counter()-tick
            poses.append(advance(poses[-1],T))
        pp,tt=np.array(poses),np.array(times);st=estimate(pp[:,:3,3],tt,.003);q=st['q']
        values=dict(state_mean=st['mean_path_m'],limited=limited_arc(pp[:,:3,3],tt),polygon=polygon(pp),displacement=float(np.linalg.norm(pp[-1,:3,3])))
    except (ValueError,cv2.error,np.linalg.LinAlgError,RuntimeError) as e:failure=str(e)
    return dict(**common,failure=failure,poses=[p.tolist() for p in poses],values_m=values,q=q,events=events,edges_done=len(poses)-1,
                feature_counts={str(i):int(len(f['keypoints'])) for i,f in feats.items()},
                wall_seconds=dict(read=read_s,features=fs,matching_and_depth=ms,solver=ss,total=perf_counter()-started))


def run(frontends,budgets):
    OUT.mkdir(exist_ok=False);plan=json.loads((FR3/'plan.json').read_text());matrix=validate_calibration(K)
    cfg=dict(frontends=frontends,budgets=budgets,windows='fr3 confirmation plan.json (15 windows, offsets 1/5/9/13/17 s, 4 s)',K=K,
             chain='uniform selection -> extract -> match -> paired_depth -> solve_rigid -> advance; first failed edge fails query; state_mean(sigma=3mm)/limited/polygon',
             comparison='existing LighterGlue GPU lg_baseline/depth_weighted B16/B32 rows from fr3_weighted_confirmation_v1 evaluation.json',
             scope='development screen on already-seen fr3 windows; not an unseen confirmation',utc=datetime.now(timezone.utc).isoformat())
    (OUT/'config.json').write_text(json.dumps(cfg,indent=2));rows=[];tick=perf_counter()
    for fe in frontends:
        model=make_model(fe)
        for seq,windows in plan.items():
            root=ROOT/f'data_external/tum_rgbd_fr3_{seq}_2026-09-13/public'
            for w in windows:
                for b in budgets:
                    bb=len(w['frames']) if b=='all' else int(b)
                    r=query(w,seq,bb,fe,model,root,matrix);r['budget_label']=str(b);rows.append(r)
                    print(fe,seq,w['offset'],b,'fail' if r['failure'] else 'ok',round(r['wall_seconds']['total'],1),'s',flush=True)
    (OUT/'predictions.json').write_text(json.dumps(rows));seal=hashlib.sha256((OUT/'predictions.json').read_bytes()).hexdigest()
    (OUT/'prediction_seal.json').write_text(json.dumps(dict(sha256=seal,rows=len(rows),wall_s=perf_counter()-tick,utc=datetime.now(timezone.utc).isoformat())))


def evaluate():
    rows=json.loads((OUT/'predictions.json').read_text());assert hashlib.sha256((OUT/'predictions.json').read_bytes()).hexdigest()==json.loads((OUT/'prediction_seal.json').read_text())['sha256']
    gts={s:np.loadtxt(ROOT/f'data_external/tum_rgbd_fr3_{s}_2026-09-13/reference/groundtruth.txt') for s in {r['sequence'] for r in rows}}
    for r in rows:
        truth,inv=reference_metrics(gts[r['sequence']],r['times']);r['truth']=truth;r['reference_failure']=inv;r['errors_m']=None;r['pose_errors']=None
        if r['failure'] is None and inv is None:
            r['errors_m']={k:v-truth['displacement_m' if k=='displacement' else 'path_m'] for k,v in r['values_m'].items()}
            gp=np.array([interpolate_pose(gts[r['sequence']],t) for t in r['times']]);r['pose_errors']=pose_errors(np.array(r['poses']),np.linalg.inv(gp[0])@gp)
    (OUT/'evaluation.json').write_text(json.dumps(rows))
    # existing LG rows
    lg=json.loads((FR3/'evaluation.json').read_text());ref={(r['sequence'],r['offset'],r['budget'],r['method']):r for r in lg}
    seqs=['structure_texture_far','nostructure_texture_near_withloop','sitting_xyz'];summary=[]
    def agg(sel,label,frames,wall):
        ok=[r for r in sel if r['errors_m'] is not None]
        per={s:[abs(r['errors_m']['state_mean']) for r in ok if r['sequence']==s] for s in seqs}
        return dict(label=label,frames=frames,windows=len(sel),completed=len(ok),per_sequence_complete={s:len(per[s]) for s in seqs},
            path_mae_state_mean_seq_equal=float(np.mean([np.mean(v) for v in per.values() if v])) if ok else None,
            path_mae_limited=float(np.mean([abs(r['errors_m']['limited']) for r in ok])) if ok else None,
            path_mae_polygon=float(np.mean([abs(r['errors_m']['polygon']) for r in ok])) if ok else None,
            pos_rmse_mean=float(np.mean([r['pose_errors']['position_rmse_m'] for r in ok])) if ok else None,
            per_sequence_mae={s:(float(np.mean(v)) if v else None) for s,v in per.items()},mean_wall_s=wall,ok_keys=[(r['sequence'],r['offset']) for r in ok])
    for fe in sorted({r['method'] for r in rows}):
        for b in sorted({r['budget_label'] for r in rows},key=lambda x:(x=='all',int(x) if x!='all' else 0)):
            sel=[r for r in rows if r['method']==fe and r['budget_label']==b]
            summary.append(agg(sel,f'{fe}_B{b}',float(np.mean([r['budget'] for r in sel])),float(np.mean([r['wall_seconds']['total'] for r in sel]))))
    for m,b in [('lg_baseline',16),('depth_weighted',16),('lg_baseline',32),('depth_weighted',32)]:
        sel=[ref[s,o,b,m] for s in seqs for o in (1,5,9,13,17)];summary.append(agg(sel,f'{m}_B{b}_GPU',b,None))
    # matched-window comparison of each cheap arm vs LG B16 on the arm's completed windows
    lg16={k[:2]:v for k,v in ref.items() if k[2]==16 and k[3]=='lg_baseline'}
    for s in summary:
        if s['ok_keys'] and not s['label'].endswith('GPU'):
            s['lg_B16_mae_on_same_windows']=float(np.mean([abs(lg16[tuple(k)]['errors_m']['state_mean']) for k in s['ok_keys']]))
        s.pop('ok_keys')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=1))
    for s in summary:print(f"{s['label']:26s} frames={s['frames']:6.1f} complete={s['completed']:2d}/{s['windows']} MAE(state_mean)={s['path_mae_state_mean_seq_equal'] if s['path_mae_state_mean_seq_equal'] is None else round(s['path_mae_state_mean_seq_equal']*1000,2)} mm  limited={None if s['path_mae_limited'] is None else round(s['path_mae_limited']*1000,2)} posRMSE={None if s['pos_rmse_mean'] is None else round(s['pos_rmse_mean']*1000,2)} wall={s['mean_wall_s'] and round(s['mean_wall_s'],1)} lgB16_same={s.get('lg_B16_mae_on_same_windows') and round(s['lg_B16_mae_on_same_windows']*1000,2)}")


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','evaluate']);p.add_argument('--frontends',default='orb,xfeat');p.add_argument('--budgets',default='8,16,32,64,all');a=p.parse_args()
    if a.action=='run':run(a.frontends.split(','),[x if x=='all' else int(x) for x in a.budgets.split(',')])
    else:evaluate()
