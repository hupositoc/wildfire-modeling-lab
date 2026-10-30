"""Stochastic cellular automaton calibrated across fires (beyond the thesis, comparison model).

Spread probability from a burning cell i to a neighbour j follows Alexandridis et al. (2008,
Appl. Math. Comput. 204:191) with fuel and moisture terms from the thesis inputs:
  p = p_h * exp(b_veg*(NDVI_j-0.5)) * exp(-b_m*(EMC_j-10)/10) * exp(c1*V) * exp(c2*V*(cos(phi)-1)) * exp(a*theta_s)
phi is the angle between the spread direction and the wind, V the wind speed at i, theta_s the
slope angle from i to j in degrees (elevation difference over the cell distance). Cells with
NDVI <= 0 never burn. A step of 30 minutes has max(1, round(rate*V_mean)) substeps, and a cell
spreads fire for `residence` steps after it ignites. The footprint is the set of cells burned in at
least half of the runs. The eight parameters are fitted by differential evolution to the mean IoU
of the training fires only (leave-one-fire-out); the held-out fire is scored once.
"""
import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path
import time
import numpy as np
from scipy.optimize import differential_evolution
from .model import ACTIONS, Dataset, fuel_moisture, iou

NAMES=['p_h','b_veg','b_m','c1','c2','a','rate','residence']
BOUNDS=[(.02,1.),(0.,6.),(0.,3.),(0.,.3),(0.,.6),(0.,.15),(.02,1.),(1.,16.)]
DIRS=ACTIONS[:8]
EVENTS={}

def load_event(path):
    path=Path(path);data=Dataset.load(path/'dataset.npz');f=data.fields;steps,n,_=f['ndvi'].shape
    z=np.full((n,n),np.nan)
    for row in csv.DictReader(open(path/'raw'/'extras.csv',encoding='utf-8')):z[int(row['y']),int(row['x'])]=float(row['elevation_m'])
    if not np.isfinite(z).all():raise ValueError('Elevation gaps in '+str(path))
    return arrays(path.name,data,z,np.load(str(path/'observed.npy')).astype(bool),tuple(data.metadata['ignition_cell']))

def arrays(name,data,z,target,start):
    f=data.fields;steps,n,_=f['ndvi'].shape;emc=fuel_moisture(f)
    dist=100.*np.linalg.norm(DIRS,axis=1)
    # Neighbour arrays: for direction d, value at j of the source i = j - d.
    src=lambda a,d:np.pad(a,((0,0),(1,1),(1,1)) if a.ndim==3 else 1,mode='edge')[...,1-d[1]:1-d[1]+n,1-d[0]:1-d[0]+n]
    slope=np.stack([np.degrees(np.arctan((z-src(z,d))/dist[k])) for k,d in enumerate(DIRS)])
    unit=DIRS/np.linalg.norm(DIRS,axis=1)[:,None]
    # cos(phi) uses the wind at the source cell; wind_to_deg is counterclockwise from east.
    cosphi=np.stack([np.cos(np.deg2rad(src(f['wind_to_deg'],d)))*unit[k,0]+np.sin(np.deg2rad(src(f['wind_to_deg'],d)))*unit[k,1] for k,d in enumerate(DIRS)],1)
    speed=np.stack([src(f['wind_speed_ms'],d) for d in DIRS],1)
    return {'name':name,'target':target,'start':start,
            'fuel':f['ndvi']>0,'ndvi':f['ndvi'],'emc':emc,'slope':slope,'cosphi':cosphi,'speed':speed,'vmean':f['wind_speed_ms'].mean((1,2)),'steps':steps,'n':n}

def probabilities(ev,p):
    """[steps, 8, n, n] spread probability into each cell from its neighbour in each direction."""
    p_h,b_veg,b_m,c1,c2,a=p[:6]
    base=p_h*np.exp(b_veg*(ev['ndvi']-.5))*np.exp(-b_m*(ev['emc']-10)/10)*ev['fuel']
    wind=np.exp(c1*ev['speed'])*np.exp(c2*ev['speed']*(ev['cosphi']-1))
    return np.clip(base[:,None]*wind*np.exp(a*ev['slope'])[None],0,1)

def simulate(ev,p,runs=8,seed=0):
    rng=np.random.default_rng(seed);n=ev['n'];prob=probabilities(ev,p)
    burned=np.zeros((runs,n,n),bool);ignited=np.full((runs,n,n),-1)
    x,y=ev['start'];burned[:,y,x]=True;ignited[:,y,x]=0
    residence=int(round(p[7]))
    for t in range(ev['steps']):
        for _ in range(max(1,int(round(p[6]*ev['vmean'][t])))):
            active=burned&(ignited>t-residence)
            if not active.any():return burned.mean(0)
            pad=np.pad(active,((0,0),(1,1),(1,1)))
            catch=np.zeros_like(burned)
            for k,d in enumerate(DIRS):
                src=pad[:,1-d[1]:1-d[1]+n,1-d[0]:1-d[0]+n]
                catch|=src&(rng.random((runs,n,n))<prob[t,k])
            new=catch&~burned;burned|=new;ignited[new]=t
    return burned.mean(0)

def score(p,names,runs=8):
    return float(np.mean([iou(simulate(EVENTS[e],p,runs)>=.5,EVENTS[e]['target']) for e in names]))

def init(paths):
    for p in paths:EVENTS[Path(p).name]=load_event(p)

def objective(p,names,runs):
    return -score(p,names,runs)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('events',type=Path,nargs='+');parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runs',type=int,default=8);parser.add_argument('--workers',type=int,default=20)
    parser.add_argument('--maxiter',type=int,default=40);parser.add_argument('--popsize',type=int,default=12)
    parser.add_argument('--confirm-seeds',type=int,nargs='+',default=[43,101,202])
    args=parser.parse_args();started=time.perf_counter()
    args.output.mkdir(parents=True,exist_ok=True)
    paths=[str(p) for p in args.events];init(paths);names=list(EVENTS)
    pool=Pool(args.workers,initializer=init,initargs=(paths,));folds={}
    for held in names+['all']:
        train=[e for e in names if e!=held]
        r=differential_evolution(objective,BOUNDS,args=(train,args.runs),maxiter=args.maxiter,popsize=args.popsize,seed=0,workers=pool.map,
                                 updating='deferred',polish=False,tol=1e-4)
        p=r.x;row={'params':dict(zip(NAMES,map(float,p))),'train_mean_iou':-float(r.fun),'evaluations':int(r.nfev)}
        if held!='all':
            ev=EVENTS[held];maps=[simulate(ev,p,args.runs,s)>=.5 for s in args.confirm_seeds]
            row.update(heldout_iou=float(np.mean([iou(m,ev['target']) for m in maps])),heldout_per_seed=[iou(m,ev['target']) for m in maps],
                       all_burned=float(ev['target'].mean()),observed_cells=int(ev['target'].sum()),predicted_cells=float(np.mean([m.sum() for m in maps])))
            np.save(str(args.output/('heldout_'+held+'.npy')),maps[0])
        else:row['in_sample_per_event']={e:iou(simulate(EVENTS[e],p,args.runs,43)>=.5,EVENTS[e]['target']) for e in names}
        folds[held]=row
        print(json.dumps({held:{k:(round(v,3) if isinstance(v,float) else v) for k,v in row.items() if k!='heldout_per_seed'}}),flush=True)
    pool.close()
    h=np.array([folds[e]['heldout_iou'] for e in names])
    summary={'kind':'leave_one_fire_out','model':'Alexandridis-type stochastic CA','events':names,'heldout_mean':float(h.mean()),'heldout_sd':float(h.std()),
             'heldout_min':float(h.min()),'all_burned_mean':float(np.mean([folds[e]['all_burned'] for e in names])),'folds':folds,
             'search':{'bounds':dict(zip(NAMES,BOUNDS)),'runs':args.runs,'maxiter':args.maxiter,'popsize':args.popsize},'minutes':(time.perf_counter()-started)/60}
    (args.output/'ca_cross_event.json').write_text(json.dumps(summary,indent=1),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('heldout_mean','heldout_sd','heldout_min','all_burned_mean')}),flush=True)

if __name__=='__main__':main()
