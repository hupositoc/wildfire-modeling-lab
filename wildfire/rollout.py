"""Parallel ignition-front reconstruction; unreported choices stay explicit."""
import numpy as np
import torch
from .model import ACTIONS,diffusion_steps,iou

def spread(data,cfg,network=None,policy='dqn',start=(6,10),seed=0):
    if policy not in ('dqn','random','reward-greedy'):raise ValueError('Unknown policy.')
    if policy=='dqn' and network is None:raise ValueError('DQN network is required.')
    rng=np.random.default_rng(seed)
    burned=np.zeros((cfg.size,cfg.size),bool);burned[start[1],start[0]]=True
    active={tuple(start)};frames=[]
    for t in range(cfg.horizon):
        speed=float(data.fields['wind_speed_ms'][t].mean())
        for _ in range(diffusion_steps(speed,cfg)):
            positions=sorted(active)
            if not positions:break
            if policy=='dqn':
                observations=np.stack([data.observation(p,t) for p in positions])
                with torch.no_grad():choices=network(torch.from_numpy(observations)).argmax(1).cpu().numpy()
            elif policy=='random':choices=rng.integers(0,9,size=len(positions))
            else:
                choices=[]
                for p in positions:
                    values=[]
                    for a in range(9):
                        dest=np.clip(np.array(p)+ACTIONS[a],0,cfg.size-1)
                        values.append(data.reward(dest,a,t,cfg))
                    choices.append(int(np.argmax(values)))
            newborn=set()
            for p,a in zip(positions,choices):
                x,y=p
                if x in (0,cfg.size-1) or y in (0,cfg.size-1):continue
                dest=np.array(p)+ACTIONS[int(a)];nx,ny=map(int,dest)
                if not(0<=nx<cfg.size and 0<=ny<cfg.size):continue
                if data.fields['ndvi'][t,ny,nx]<=0:continue
                # Thesis p.15 describes positive-reward cells as spread cells.
                if data.reward(dest,int(a),t,cfg)<0:continue
                burned[ny,nx]=True;newborn.add((nx,ny))
            # Assumption: ignited sources remain active. No residence time was supplied.
            active.update(newborn)
        frames.append(burned.copy())
    return np.stack(frames)

def evaluate_observed(predicted,observed,metadata):
    """Reject synthetic/unknown labels for a real-event reproduction result."""
    if metadata.get('kind')!='observational':raise ValueError('Observed event provenance is required.')
    required=['source_url','event_id','mask_method','time_window','grid_bounds','grid_origin','version']
    if any(not metadata.get(key) for key in required):raise ValueError('Incomplete observed-mask provenance.')
    if metadata.get('is_example') or any('REPLACE_' in str(metadata.get(key,'')) for key in required):raise ValueError('Example provenance is not an observed event.')
    return {'iou':iou(predicted,observed),'event_id':metadata['event_id'],'provenance':metadata}
