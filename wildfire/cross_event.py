"""Leave-one-fire-out evaluation of the thesis model across events built by data/build_event.py.

Every design point (reward weights + C_f) is trained and rolled out on every event once, with no
Jaccard training reward, so a policy never sees its own event's burn map. For each held-out event,
the point and spread knob are chosen by the mean IoU over the other events only, then the held-out
event is scored on unseen seeds. The design is a fixed simplex LHS (no Bayesian rounds), because
an adaptive search guided by all events would let the held-out event steer which points exist.
Baselines on each event: every fuel cell burned, and a random policy with the same knob.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from . import calibrate as cal
from .model import Config, Dataset, iou, reward_table
from .rollout import q_table

EVENTS={}

def init_worker(paths,spread,episodes):
    torch.set_num_threads(1)
    for p in paths:
        data=Dataset.load(Path(p)/'dataset.npz');horizon,size,_=data.fields['ndvi'].shape
        EVENTS[Path(p).name]={'data':data,'target':np.load(str(Path(p)/'observed.npy')).astype(bool),'start':tuple(data.metadata['ignition_cell']),
                              'cfg':Config(size=size,horizon=horizon,ndvi_mode='raw',consumed_fuel=True)}
    cal.STATE.update(spread=spread,episodes=episodes)

def run_job(job):
    x,event,seed,knobs,rollouts=job;e=EVENTS[event];cal.START=e['start']
    cfg=replace(e['cfg'],weights=tuple(float(v) for v in x[:-1]),diffusion_scale=float(x[-1]))
    _,_,network=cal.train_visits(e['data'],cfg,cal.STATE['episodes'],seed,e['start'])
    q=q_table(e['data'],network);rewards=reward_table(e['data'],cfg)
    stats=cal.reward_stats(rewards,cal.STATE['spread']);out={}
    for k in knobs:
        maps=[cal.run_spread(e['data'],cfg,q,rewards,s,k,None,stats) for s in range(rollouts)]
        rand=[cal.run_spread(e['data'],cfg,q,rewards,100+s,dict(k,temperature=1e9),None,stats) for s in range(rollouts)]
        out[json.dumps(k)]={'iou':float(np.mean([iou(m,e['target']) for m in maps])),'random':float(np.mean([iou(m,e['target']) for m in rand])),
                            'cells':float(np.mean([m.sum() for m in maps]))}
    return out

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('events',type=Path,nargs='+');parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--points',type=int,default=48);parser.add_argument('--episodes',type=int,default=300)
    parser.add_argument('--seeds',type=int,nargs='+',default=[7,19]);parser.add_argument('--confirm-seeds',type=int,nargs='+',default=[43,101,202])
    parser.add_argument('--workers',type=int,default=20);parser.add_argument('--cf-range',type=float,nargs=2,default=[.005,2.])
    parser.add_argument('--temperatures',type=float,nargs='+',default=[.5,1,2,4,8]);parser.add_argument('--thetas',type=float,nargs='+',default=[0,.2,.4,.6])
    parser.add_argument('--residences',type=int,nargs='+',default=[2,4,8,16]);parser.add_argument('--rollouts',type=int,default=3)
    args=parser.parse_args();started=time.perf_counter()
    cal.CF_RANGE=tuple(args.cf_range);cal.K=4
    spread={'temperatures':args.temperatures,'thetas':args.thetas,'tree_mins':[None],'sharpness':[None],'residences':args.residences,'spots':[None],'spot_m':500.}
    knobs=cal.knobs(spread);names=[p.name for p in args.events]
    points=list(cal.simplex_lhs(args.points,np.random.default_rng(0)))
    setup=([str(p) for p in args.events],spread,args.episodes)
    with ProcessPoolExecutor(args.workers,initializer=init_worker,initargs=setup) as pool:
        jobs=[(x,ev,s,knobs,args.rollouts) for x in points for ev in names for s in args.seeds]
        results=list(pool.map(run_job,jobs))
        keys=[json.dumps(k) for k in knobs]
        # score[point, knob, event] averaged over the search seeds
        get=lambda field:np.array([[[np.mean([results[(i*len(names)+j)*len(args.seeds)+s][k][field] for s in range(len(args.seeds))])
                                     for j in range(len(names))] for k in keys] for i in range(len(points))])
        score,rand,cells=get('iou'),get('random'),get('cells')
        print(json.dumps({'design_done_min':round((time.perf_counter()-started)/60,1)}),flush=True)
        folds={}
        for j,ev in enumerate(names):
            train=[i for i in range(len(names)) if i!=j]
            p,k=np.unravel_index(score[:,:,train].mean(2).argmax(),score.shape[:2])
            confirm=list(pool.map(run_job,[(points[p],ev,s,[knobs[k]],args.rollouts) for s in args.confirm_seeds]))
            target=load_target(args.events[j])
            folds[ev]={'weights':[float(v) for v in points[p][:4]],'diffusion_scale':float(points[p][4]),'knob':knobs[k],
                       'train_mean_iou':float(score[p,k,train].mean()),
                       'heldout_iou':float(np.mean([c[keys[k]]['iou'] for c in confirm])),'heldout_per_seed':[c[keys[k]]['iou'] for c in confirm],
                       'random_same_knob':float(np.mean([c[keys[k]]['random'] for c in confirm])),
                       'all_burned':float(target.mean()),'observed_cells':int(target.sum()),
                       'predicted_cells':float(np.mean([c[keys[k]]['cells'] for c in confirm])),
                       'in_sample_best':float(score[:,:,j].max())}
            print(json.dumps({ev:{a:(round(b,3) if isinstance(b,float) else b) for a,b in folds[ev].items()}}),flush=True)
    h=np.array([f['heldout_iou'] for f in folds.values()])
    summary={'kind':'leave_one_fire_out','events':names,'heldout_mean':float(h.mean()),'heldout_sd':float(h.std()),'heldout_min':float(h.min()),
             'random_mean':float(np.mean([f['random_same_knob'] for f in folds.values()])),'all_burned_mean':float(np.mean([f['all_burned'] for f in folds.values()])),
             'folds':folds,'design':{'points':args.points,'seeds':args.seeds,'confirm_seeds':args.confirm_seeds,'episodes':args.episodes,'cf_range':args.cf_range,
             'knobs':spread,'ji_reward':0,'consumed_fuel':True},'config':asdict(Config()),'minutes':(time.perf_counter()-started)/60}
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(str(args.output/'scores.npz'),iou=score,random=rand,cells=cells,points=np.array(points))
    (args.output/'cross_event.json').write_text(json.dumps(summary,indent=1),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('heldout_mean','heldout_sd','heldout_min','random_mean','all_burned_mean')}),flush=True)

def load_target(path):return np.load(str(Path(path)/'observed.npy')).astype(bool)

if __name__=='__main__':main()
