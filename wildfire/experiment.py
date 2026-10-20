"""Run a genuine DQN implementation smoke benchmark; not thesis score reproduction."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from .model import Config, Dataset, FirePathEnv, synthetic

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(8,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,9))
    def forward(self,x):return self.layers(x)

def run(seed,mode,episodes,output,dataset=None):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(seed);cfg=Config(ndvi_mode=mode)
    if dataset is not None:
        cfg.horizon,cfg.size,_=dataset.fields['ndvi'].shape
    network=QNetwork();target=QNetwork();target.load_state_dict(network.state_dict())
    optimizer=torch.optim.Adam(network.parameters(),lr=.001)
    capacity=20000;size=0;cursor=0;updates=0
    obs=np.empty((capacity,8),np.float32);nxt=np.empty_like(obs)
    actions=np.empty(capacity,np.int64);rewards=np.empty(capacity,np.float32);dones=np.empty(capacity,np.float32)
    trace=[];started=time.perf_counter()
    for episode in range(episodes):
        env=FirePathEnv(dataset if dataset is not None else synthetic(int(rng.integers(0,100000)),cfg),cfg)
        state=env.reset();done=False;total=0.;losses=[]
        epsilon=max(.08,1-episode/(episodes*.8))
        while not done:
            if rng.random()<epsilon:action=int(rng.integers(9))
            else:
                with torch.no_grad():action=int(network(torch.from_numpy(state)).argmax().item())
            next_state,reward,done,_=env.step(action);total+=reward
            obs[cursor]=state;nxt[cursor]=next_state;actions[cursor]=action;rewards[cursor]=reward;dones[cursor]=done
            cursor=(cursor+1)%capacity;size=min(capacity,size+1);state=next_state
            if size>=64:
                ix=rng.choice(size,64,replace=False)
                q=network(torch.from_numpy(obs[ix])).gather(1,torch.from_numpy(actions[ix,None])).squeeze(1)
                with torch.no_grad():
                    values=target(torch.from_numpy(nxt[ix])).max(1).values
                    # Terminal transitions do not bootstrap, including horizon.
                    expected=torch.from_numpy(rewards[ix])+cfg.gamma*(1-torch.from_numpy(dones[ix]))*values
                loss=nn.functional.mse_loss(q,expected)
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(network.parameters(),10.);optimizer.step()
                updates+=1;losses.append(float(loss.detach()))
                if updates%100==0:target.load_state_dict(network.state_dict())
        trace.append({'episode':episode,'return':total,'epsilon':epsilon,'loss':float(np.mean(losses)) if losses else None})
    # Held-out environment seeds are fixed before testing; no best-seed selection.
    evaluation=[]
    for data_seed in ([] if dataset is not None else [200001,200002,200003,200004,200005]):
        data=synthetic(data_seed,cfg)
        for policy in ['dqn','random','reward-greedy']:
            local=np.random.default_rng(data_seed+seed);env=FirePathEnv(data,cfg);state=env.reset();done=False;total=0.;steps=0
            while not done:
                if policy=='random':a=int(local.integers(9))
                elif policy=='dqn':
                    with torch.no_grad():a=int(network(torch.from_numpy(state)).argmax().item())
                else:
                    from .model import ACTIONS
                    candidates=[]
                    for a in range(9):
                        dest=np.clip(env.pos+ACTIONS[a],0,cfg.size-1)
                        candidates.append(data.reward(dest,a,min(env.t,cfg.horizon-1),cfg))
                    a=int(np.argmax(candidates))
                state,reward,done,_=env.step(a);total+=reward;steps+=1
            evaluation.append({'environment_seed':data_seed,'policy':policy,'return':total,'steps':steps})
    output.mkdir(parents=True,exist_ok=True)
    weights_path=output/('dqn_'+mode+'_seed'+str(seed)+'.pt')
    torch.save(network.state_dict(),str(weights_path))
    result={'kind':'synthetic_implementation_benchmark' if dataset is None else 'observational_reconstruction_training',
            'dataset_provenance':None if dataset is None else dataset.metadata,
            'thesis_reproduction':False,'observed_mask_available':False,
            'iou':None,'target_iou':.726,'target_used_for_training':False,'seed':seed,'config':asdict(cfg),
            'architecture':[8,64,64,9],'optimizer':'Adam','learning_rate':.001,'batch_size':64,'replay_capacity':capacity,
            'target_sync_updates':100,'gradient_clip':10.,'episodes':episodes,'updates':updates,
            'training_seconds':time.perf_counter()-started,'training_trace':trace,'evaluation':evaluation,
            'weights_sha256':hashlib.sha256(weights_path.read_bytes()).hexdigest(),
            'runtime':{'torch':torch.__version__,'numpy':np.__version__,'device':'cpu'},
            'assumptions':['64x64 hidden layers not specified in thesis','trajectory training is separated from branching-front simulation','NDVI interpretation explicitly selected','floor used for diffusion coefficient','synthetic environments have no geographic identity; no held-out evaluation for single supplied observational event']}
    path=output/('result_'+mode+'_seed'+str(seed)+'.json');path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'mode':mode,'seed':seed,'episodes':episodes,'updates':updates,'seconds':round(result['training_seconds'],2),'iou':None}),flush=True)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--episodes',type=int,default=160)
    parser.add_argument('--seeds',type=int,nargs='+',default=[7,19,43]);parser.add_argument('--output',type=Path,default=Path('results/implementation-benchmark'))
    parser.add_argument('--dataset',type=Path,help='Optional real environmental NPZ with explicit JSON provenance. No target mask enters training.')
    args=parser.parse_args()
    if args.episodes<1:parser.error('Episodes must be positive.')
    dataset=Dataset.load(args.dataset) if args.dataset else None
    if dataset is not None and dataset.metadata['kind']!='observational':parser.error('Use the built-in synthetic path for synthetic experiments.')
    if dataset is not None and args.output==Path('results/implementation-benchmark'):parser.error('Supply a separate --output folder for real-event reconstruction training.')
    for mode in ['literal','raw']:
        for seed in args.seeds:run(seed,mode,args.episodes,args.output,dataset)

if __name__=='__main__':main()
