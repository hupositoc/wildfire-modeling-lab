"""Thesis section 4 calibration: simplex LHS, then Bayesian optimisation of the reward weights
and the diffusion scale C_f, maximising the Jaccard index against the burned-area target.

Footprints: 'policy' runs the trained DQN as a stochastic policy over the horizon (thesis 3.5,
Fig. 4.4); 'visits' keeps cells visited in a fraction tau of training episodes (thesis p.15).
With --ji-reward, entering a target cell adds +lambda and a non-target cell -lambda during
training (thesis p.14). Every score here is fitted to the same event, as the thesis's 0.726 was.

Beyond the thesis (each off by default): --thetas sets the ignition reward threshold (A),
--barrier/--tree-mins block cells whose WorldCover tree fraction is below a cut (B), and
--sharpness makes ignition probabilistic around theta (E), --residences burns a cell out after that many
steps (F). All are grid-searched after training.
"""
import csv
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.stats import norm
from .experiment import QNetwork
from .model import Config, Dataset, FirePathEnv, iou, reward_table
from .rollout import fast_spread, q_table

CF_RANGE=(.005,.5)
TAUS=np.round(np.r_[np.arange(.002,.01,.001),np.arange(.01,.62,.01)],3)
TEMPERATURES=(.03,.06,.125,.25,.5,1.,2.,4.,8.)
START=(6,10)

def train_visits(data,cfg,episodes,seed,start=START,bonus=None):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1)
    rng=np.random.default_rng(seed);n=cfg.size
    network=QNetwork();target=QNetwork();target.load_state_dict(network.state_dict())
    optimizer=torch.optim.Adam(network.parameters(),lr=.001)
    capacity=20000;size=0;cursor=0;updates=0
    obs=np.empty((capacity,8),np.float32);nxt=np.empty_like(obs)
    actions=np.empty(capacity,np.int64);rewards=np.empty(capacity,np.float32);dones=np.empty(capacity,np.float32)
    episodes_seen=np.zeros((n,n));reward_sum=np.zeros((n,n))
    for episode in range(episodes):
        env=FirePathEnv(data,cfg,start,bonus);state=env.reset();done=False
        seen=np.zeros((n,n),bool);seen[start[1],start[0]]=True
        epsilon=max(.08,1-episode/(episodes*.8))
        while not done:
            if rng.random()<epsilon:action=int(rng.integers(9))
            else:
                with torch.no_grad():action=int(network(torch.from_numpy(state)).argmax().item())
            next_state,reward,done,info=env.step(action)
            for x,y,r in info['path']:seen[y,x]=True;reward_sum[y,x]+=r
            obs[cursor]=state;nxt[cursor]=next_state;actions[cursor]=action;rewards[cursor]=reward;dones[cursor]=done
            cursor=(cursor+1)%capacity;size=min(capacity,size+1);state=next_state
            if size>=64:
                ix=rng.choice(size,64,replace=False)
                q=network(torch.from_numpy(obs[ix])).gather(1,torch.from_numpy(actions[ix,None])).squeeze(1)
                with torch.no_grad():
                    values=target(torch.from_numpy(nxt[ix])).max(1).values
                    expected=torch.from_numpy(rewards[ix])+cfg.gamma*(1-torch.from_numpy(dones[ix]))*values
                loss=nn.functional.mse_loss(q,expected)
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(network.parameters(),10.);optimizer.step()
                updates+=1
                if updates%100==0:target.load_state_dict(network.state_dict())
        episodes_seen+=seen
    return episodes_seen/episodes,reward_sum,network

def footprint(frequency,reward_sum,tau,start=START):
    mask=(frequency>=tau)&(reward_sum>0);mask[start[1],start[0]]=True
    return mask

def region_iou(pred,target,region):
    return iou(pred&region,target&region)

def visit_scores(frequency,reward_sum,target,region):
    return {float(t):region_iou(footprint(frequency,reward_sum,t),target,region) for t in TAUS}

def knobs(spread):
    out=[dict(temperature=t,theta_q=h,tree_min=m,sharpness=k,**({'residence':d} if d is not None else {})) for t in spread['temperatures'] for h in spread['thetas']
            for m in spread['tree_mins'] for k in spread['sharpness'] for d in spread.get('residences',[None])]
    return [dict(k,spot=s) if s else k for k in out for s in spread.get('spots',[None])]

def reward_stats(rewards,spread):
    fuel=rewards[rewards>-1]
    return {'theta':{h:float(np.quantile(fuel,h)) if h else 0. for h in spread['thetas']},'std':float(fuel.std())}

def run_spread(data,cfg,q,rewards,seed,knob,tree=None,stats=None):
    """theta_q is a quantile of the fuel-cell reward distribution so it means the same under any weights;
    sharpness is per reward standard deviation."""
    stats=stats or reward_stats(rewards,{'thetas':[knob['theta_q']]})
    theta=stats['theta'][knob['theta_q']];sharpness=None if knob['sharpness'] is None else knob['sharpness']/stats['std']
    barrier=None if knob['tree_min'] is None else tree<knob['tree_min']
    return fast_spread(data,cfg,q,rewards,START,seed,knob['temperature'],theta=theta,barrier=barrier,sharpness=sharpness,residence=knob.get('residence'),
                       spot=knob.get('spot'),spot_m=STATE.get('spread',{}).get('spot_m',500.))[-1]

def policy_scores(data,cfg,network,target,spread,regions,tree=None,rollouts=3):
    q=q_table(data,network);rewards=reward_table(data,cfg);stats=reward_stats(rewards,spread);out=[{} for _ in regions]
    for k in knobs(spread):
        maps=[run_spread(data,cfg,q,rewards,s,k,tree,stats) for s in range(rollouts)]
        for o,region in zip(out,regions):o[json.dumps(k)]=float(np.mean([region_iou(m,target,region) for m in maps]))
    return out

def holdout_regions(kind,n=50,block=10):
    """Fit region (JI training reward and search objective) and held-out region (reported only)."""
    if kind=='none':return np.ones((n,n),bool),None
    yy,xx=np.mgrid[:n,:n];fit=((yy//block+xx//block)%2)==0
    return fit,~fit

def load_tree(path,n=50):
    tree=np.full((n,n),np.nan)
    for row in csv.DictReader(open(path,encoding='utf-8')):tree[int(row['y']),int(row['x'])]=float(row['tree_fraction'])
    return tree

STATE={}

def init_worker(dataset,mask,ndvi_mode,episodes,footprint_mode,ji_reward,spread,barrier,holdout='none',consumed_fuel=False):
    torch.set_num_threads(1)
    data=Dataset.load(dataset);target=np.load(str(mask)).astype(bool)
    horizon,size,_=data.fields['ndvi'].shape;fit,held=holdout_regions(holdout,size)
    # The JI reward never reaches held-out cells, so their score measures generalisation.
    STATE.update(data=data,target=target,cfg=Config(size=size,horizon=horizon,ndvi_mode=ndvi_mode,consumed_fuel=consumed_fuel,wind_ref_ms=spread.get('wind_ref')),episodes=episodes,
                 footprint=footprint_mode,bonus=ji_reward*(2*target-1.)*fit if ji_reward else None,fit=fit,held=held,
                 spread=spread,tree=load_tree(barrier) if barrier else None)

def score_point(job):
    x,seed=job;s=STATE
    cfg=replace(s['cfg'],weights=tuple(float(v) for v in x[:-1]),diffusion_scale=float(x[-1]))
    frequency,reward_sum,network=train_visits(s['data'],cfg,s['episodes'],seed,START,s['bonus'])
    regions=[s['fit']]+([s['held']] if s['held'] is not None else [])
    if s['footprint']=='visits':scores=[visit_scores(frequency,reward_sum,s['target'],r) for r in regions]
    else:scores=policy_scores(s['data'],cfg,network,s['target'],s['spread'],regions,s['tree'])
    return scores+[None]*(2-len(scores))

K=4  # reward weights searched; 5 with --fuel-type

def from_unit(u):return np.exp(np.log(CF_RANGE[0])+u*(np.log(CF_RANGE[1])-np.log(CF_RANGE[0])))

def simplex_lhs(n,rng):
    strata=lambda:(rng.permutation(n)+rng.random(n))/n
    cube=np.stack([strata() for _ in range(K)],1)
    weights=np.diff(np.hstack([np.zeros((n,1)),np.sort(cube[:,:K-1],1),np.ones((n,1))]),axis=1)
    return np.hstack([weights,from_unit(cube[:,K-1])[:,None]])

def random_candidates(n,rng):
    return np.hstack([rng.dirichlet(np.ones(K),n),from_unit(rng.random(n))[:,None]])

def scaled(x):
    x=np.atleast_2d(x).copy();x[:,K]=(np.log(x[:,K])-np.log(CF_RANGE[0]))/(np.log(CF_RANGE[1])-np.log(CF_RANGE[0]));return x

def expected_improvement(gp,x,best):
    mu,sd=gp.predict(scaled(x),return_std=True);sd=np.maximum(sd,1e-9)
    z=(mu-best-.005)/sd
    return (mu-best-.005)*norm.cdf(z)+sd*norm.pdf(z)

def batch(gp,rng,best,size):
    candidates=random_candidates(6000,rng);ei=expected_improvement(gp,candidates,best);chosen=[]
    for i in np.argsort(-ei):
        if all(np.linalg.norm(scaled(candidates[i])-scaled(c))>.15 for c in chosen):chosen.append(candidates[i])
        if len(chosen)==size:break
    return chosen

def main():
    global CF_RANGE,K
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--mask',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ndvi-mode',choices=['raw','literal'],default='raw')
    parser.add_argument('--footprint',choices=['policy','visits'],default='policy')
    parser.add_argument('--ji-reward',type=float,default=0.,help='lambda for the thesis p.14 Jaccard training reward; 0 disables it')
    parser.add_argument('--lhs',type=int,default=16);parser.add_argument('--rounds',type=int,default=6);parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--episodes',type=int,default=300);parser.add_argument('--seeds',type=int,nargs='+',default=[7,19])
    parser.add_argument('--confirm-seeds',type=int,nargs='+',default=[43,101,202]);parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--search-seed',type=int,default=0)
    parser.add_argument('--temperatures',type=float,nargs='+',default=list(TEMPERATURES))
    parser.add_argument('--thetas',type=float,nargs='+',default=[0.],help='(A) ignition threshold as a quantile of fuel-cell rewards; 0 keeps the thesis rule r>=0')
    parser.add_argument('--barrier',type=Path,help='(B) extras.csv with WorldCover tree_fraction')
    parser.add_argument('--tree-mins',type=float,nargs='+',default=[],help='(B) cells below this tree fraction cannot burn')
    parser.add_argument('--residences',type=int,nargs='+',default=[],help='(F) burnout time in steps; omitted keeps cells active forever')
    parser.add_argument('--spots',type=float,nargs='+',default=[],help='(G) spotting chance per burning cell per step; omitted disables spotting')
    parser.add_argument('--spot-m',type=float,default=500.,help='(G) mean spotting distance in metres at 5 m/s wind')
    parser.add_argument('--wind-ref',type=float,help='(H) scale the wind reward term by wind speed / this value (m/s)')
    parser.add_argument('--sharpness',type=float,nargs='+',default=[],help='(E) P(ignite)=sigmoid(k*(r-theta)/std(r)); omitted means hard threshold')
    parser.add_argument('--holdout',choices=['none','checker'],default='none',help='checker: JI reward and search objective use alternate 10x10 blocks; the rest is reported as held-out')
    parser.add_argument('--consumed-fuel',action='store_true',help='re-entering a cell burned earlier in the episode earns nothing (beyond the thesis)')
    parser.add_argument('--fuel-type',action='store_true',help='search a fifth weight on the conifer_fraction field (data/add_forest_type.py)')
    parser.add_argument('--cf-range',type=float,nargs=2,default=list(CF_RANGE),help='C_f search bounds; e.g. a lower bound from observed arrival times (tools/viirs_arrival.py)')
    args=parser.parse_args();rng=np.random.default_rng(args.search_seed);history=[];started=time.perf_counter()
    CF_RANGE=tuple(args.cf_range);K=5 if args.fuel_type else 4
    if args.tree_mins and not args.barrier:parser.error('--tree-mins needs --barrier')
    spread={'temperatures':args.temperatures,'thetas':args.thetas,'tree_mins':[None]+args.tree_mins,'sharpness':[None]+args.sharpness,'residences':args.residences or [None],
            'spots':args.spots or [None],'spot_m':args.spot_m,'wind_ref':args.wind_ref}
    setup=(args.dataset,args.mask,args.ndvi_mode,args.episodes,args.footprint,args.ji_reward,spread,args.barrier,args.holdout,args.consumed_fuel)
    init_worker(*setup)
    pool=ProcessPoolExecutor(args.workers,initializer=init_worker,initargs=setup)
    def evaluate(points,phase,seeds,fixed=None):
        jobs=[(x,s) for x in points for s in seeds];results=list(pool.map(score_point,jobs))
        rows=[]
        for i,x in enumerate(points):
            per=results[i*len(seeds):(i+1)*len(seeds)]
            mean={k:float(np.mean([p[0][k] for p in per])) for k in per[0][0]};knob=max(mean,key=mean.get) if fixed is None else fixed
            row={'phase':phase,'seeds':seeds,'weights':[float(v) for v in x[:K]],'diffusion_scale':float(x[K]),
                 'knob':json.loads(knob) if args.footprint=='policy' else knob,
                 'jaccard':mean[knob],'per_seed':[p[0][knob] for p in per]}
            # Reported only; never used to pick the knob or the next search point.
            if per[0][1] is not None:row['heldout_jaccard']=float(np.mean([p[1][knob] for p in per]))
            rows.append(row);print(json.dumps({k:(round(v,4) if isinstance(v,float) else v) for k,v in row.items()}),flush=True)
        return rows
    history+=evaluate(list(simplex_lhs(args.lhs,rng)),'simplex-lhs',args.seeds)
    for r in range(args.rounds):
        X=np.array([h['weights']+[h['diffusion_scale']] for h in history]);y=np.array([h['jaccard'] for h in history])
        gp=GaussianProcessRegressor(Matern(length_scale=np.ones(K+1),nu=2.5)+WhiteKernel(1e-3),normalize_y=True,n_restarts_optimizer=3,random_state=0).fit(scaled(X),y)
        history+=evaluate(batch(gp,rng,y.max(),args.batch),'bayesian-%d'%(r+1),args.seeds)
        print(json.dumps({'round':r+1,'best_so_far':round(float(max(h['jaccard'] for h in history)),4),'elapsed_min':round((time.perf_counter()-started)/60,1)}),flush=True)
    best=max(history,key=lambda h:h['jaccard'])
    x=np.array(best['weights']+[best['diffusion_scale']])
    # Unseen seeds are scored at the knob chosen during the search, not re-optimised.
    confirm=evaluate([x],'confirm',args.confirm_seeds,json.dumps(best['knob']) if args.footprint=='policy' else best['knob'])[0]
    pool.shutdown()
    s=STATE;cfg=replace(s['cfg'],weights=tuple(best['weights']),diffusion_scale=best['diffusion_scale'])
    frequency,reward_sum,network=train_visits(s['data'],cfg,args.episodes,args.seeds[0],START,s['bonus'])
    if args.footprint=='visits':predicted=footprint(frequency,reward_sum,best['knob'])
    rewards=reward_table(s['data'],cfg)
    if args.footprint=='policy':predicted=run_spread(s['data'],cfg,q_table(s['data'],network),rewards,0,best['knob'],s['tree'])
    # The random policy keeps the selected threshold and burnout time, so only the action choice differs.
    random_knob=dict(best['knob'],temperature=1e9) if args.footprint=='policy' else {'temperature':1e9,'theta_q':0,'tree_min':None,'sharpness':None}
    random_maps=[run_spread(s['data'],cfg,q_table(s['data'],network),rewards,k,random_knob,s['tree']) for k in range(5)]
    baseline=float(np.mean([region_iou(m,s['target'],s['fit']) for m in random_maps]))
    held={}
    if s['held'] is not None:
        held={'heldout_jaccard_recomputed':region_iou(predicted,s['target'],s['held']),
              'heldout_random_policy_jaccard':float(np.mean([region_iou(m,s['target'],s['held']) for m in random_maps]))}
    args.output.mkdir(parents=True,exist_ok=True)
    np.save(str(args.output/'predicted.npy'),predicted);torch.save(network.state_dict(),str(args.output/'best_network.pt'))
    result={'kind':'in_sample_calibration','event_id':s['data'].metadata.get('event_id'),'thesis_reported_jaccard':.726,
            'best':best,'best_jaccard_recomputed':region_iou(predicted,s['target'],s['fit']),'holdout':args.holdout,**held,
            'random_policy_jaccard_same_config':baseline,
            'dqn_gain_over_random':best['jaccard']-baseline,
            'confirm_unseen_seeds':confirm,'observed_cells':int(s['target'].sum()),'predicted_cells':int(predicted.sum()),
            'footprint':args.footprint,'knob':'tau (visit fraction)' if args.footprint=='visits' else 'temperature, theta_q, tree_min, sharpness',
            'beyond_thesis':{'thetas':args.thetas,'barrier':str(args.barrier) if args.barrier else None,'tree_mins':args.tree_mins,'sharpness':args.sharpness,
                           'residences':args.residences,'spots':args.spots,'spot_m':args.spot_m,'wind_ref':args.wind_ref},
            'ji_reward_lambda':args.ji_reward,'mask':str(args.mask),
            'search':{'simplex_lhs':args.lhs,'bayesian_rounds':args.rounds,'batch':args.batch,'episodes':args.episodes,'search_seeds':args.seeds,
                      'cf_range_log':CF_RANGE,'tau_grid':[float(TAUS[0]),float(TAUS[-1])],'temperatures':args.temperatures},
            'config':asdict(cfg),'history':history,'seconds':time.perf_counter()-started,
            'note':'Weights, C_f and the footprint knob are fitted to the same Uljin target, as in the thesis. Not held-out accuracy.'+
                   (' The JI training reward also exposes the target during training (thesis p.14); the footprint is judged on environmental reward only.' if args.ji_reward else '')+
                   (' heldout_* scores cover checkerboard blocks that neither the JI reward nor the search objective saw.' if held else '')}
    (args.output/'calibration.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'best':best['jaccard'],'confirm_unseen_seeds':confirm['per_seed'],'confirm_mean':confirm['jaccard'],'random_policy_same_config':baseline,**held}),flush=True)

if __name__=='__main__':main()
