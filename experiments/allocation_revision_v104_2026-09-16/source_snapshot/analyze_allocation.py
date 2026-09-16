"""Analyze allocation sweep: error-cost frontier and calibration-only camera-subset rule."""
import json,argparse
from itertools import combinations
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];CAL=ROOT/'experiments/mcalib_3d_cache_2026-09-13_verified/public/calibration.json'


def scene_center(cams):
    """Least-squares point closest to all optical axes; calibration only."""
    A=np.zeros((3,3));b=np.zeros(3)
    for c in cams.values():
        R=np.asarray(c['R']);t=np.asarray(c['t_m']);o=-R.T@t;a=R.T@np.array([0,0,1.]);M=np.eye(3)-np.outer(a,a);A+=M;b+=M@o
    return np.linalg.solve(A,b)


def subset_score(cams,subset,center):
    """Predicted triangulation error scale (m per px) from calibration only: worst-pair z/(f sin angle)."""
    info={}
    for cid in subset:
        c=cams[cid];R=np.asarray(c['R']);t=np.asarray(c['t_m']);o=-R.T@t;K=np.asarray(c['K']);f=(K[0,0]+K[1,1])/2
        v=center-o;info[cid]=(np.linalg.norm(v),v/np.linalg.norm(v),f)
    best=np.inf
    for a,b in combinations(subset,2):
        za,ra,fa=info[a];zb,rb,fb=info[b];s=np.linalg.norm(np.cross(ra,rb))
        best=min(best,max(za/fa,zb/fb)/max(s,1e-6))
    return float(best)


def main(exp):
    cams=json.loads(CAL.read_text())['cameras'];S=json.load(open(exp/'summary.json'))['summary'];center=scene_center(cams)
    print('scene center (calibration only):',np.round(center,3))
    # 1. Frontier: best MAE per (views,moments) across methods, all 7-view and best/median/worst subset for k=2,3
    print('\n== MAE by views x moments (method giving lowest MAE; subsets: k=7 single, k=2/3 median over subsets, best subset in brackets) ==')
    print(f"{'k':>2} {'m':>4} {'frames':>6} | {'best-method':<11} {'median MAE':>11} {'best subset MAE':>15} {'worst':>9} {'n_sub':>5}")
    for k in sorted({r['views'] for r in S}):
        for m in sorted({r['moments'] for r in S}):
            rr=[r for r in S if r['views']==k and r['moments']==m and r['mae_m'] is not None]
            if not rr:continue
            per_sub={}
            for r in rr:per_sub.setdefault(r['cameras'],[]).append((r['mae_m'],r['method']))
            bests={c:min(v) for c,v in per_sub.items()};vals=np.array([v[0] for v in bests.values()])
            bm=max([b[1] for b in bests.values()],key=[b[1] for b in bests.values()].count)
            print(f"{k:>2} {m:>4} {k*m:>6} | {bm:<11} {np.median(vals):>11.4f} {vals.min():>15.4f} {vals.max():>9.4f} {len(vals):>5}")
    # 2. Calibration-only subset rule vs realized MAE at k=2, m=33
    print('\n== k=2, m=33: calibration-only predicted error scale vs realized MAE (limited) ==')
    rows=[]
    for r in S:
        if r['views']==2 and r['moments']==33 and r['method']=='limited':
            rows.append((subset_score(cams,r['cameras'].split('+'),center),r['mae_m'],r['failed'],r['missing_moments'],r['cameras']))
    rows.sort()
    for sc,mae,fail,miss,c in rows:print(f"{c:<20} pred_scale={sc*1e3:7.3f} mm/px  MAE={mae:.4f}  failed={fail} missing_moments={miss}")
    sc=np.array([r[0] for r in rows]);mae=np.array([r[1] for r in rows])
    from scipy.stats import spearmanr
    print('Spearman(pred scale, MAE) =',round(float(spearmanr(sc,mae)[0]),3))
    print('rule pick (min predicted scale):',rows[0][4],'MAE',round(rows[0][1],4),'| rank of rule pick among pairs by MAE:',int(np.argsort(np.argsort(mae))[0])+1)
    print('\n== k=3, m=33 ==')
    rows=[]
    for r in S:
        if r['views']==3 and r['moments']==33 and r['method']=='limited':
            rows.append((subset_score(cams,r['cameras'].split('+'),center),r['mae_m'],r['failed'],r['missing_moments'],r['cameras']))
    rows.sort()
    for sc,mae,fail,miss,c in rows[:8]:print(f"{c:<30} pred_scale={sc*1e3:7.3f} mm/px  MAE={mae:.4f}  failed={fail} missing={miss}")
    sc=np.array([r[0] for r in rows]);mae=np.array([r[1] for r in rows]);print('Spearman =',round(float(spearmanr(sc,mae)[0]),3),'| rule pick',rows[0][4],'MAE',round(rows[0][1],4),'rank',int(np.argsort(np.argsort(mae))[0])+1)
    # 3. Equal-cost comparison table
    print('\n== Equal-cost comparisons (frames): prior protocol 7 views x m vs dense few-view ==')
    def get(k,m,method,cam=None):
        rr=[r for r in S if r['views']==k and r['moments']==m and r['method']==method and (cam is None or r['cameras']==cam)]
        return rr
    for k,m,meth in [(7,8,'state_mean'),(7,8,'limited'),(7,12,'state_mean'),(7,16,'state_mean'),(7,16,'limited'),(7,33,'limited')]:
        rr=get(k,m,meth);print(f"7-view m={m:<3} {meth:<10} frames={7*m:<4} MAE={rr[0]['mae_m']:.4f}")


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('exp',type=Path);main(p.parse_args().exp)
