"""Parallel ignition-front reconstruction; unreported choices stay explicit."""
import numpy as np
import torch
from .model import ACTIONS,diffusion_steps,iou,observation_table

def spread(data,cfg,network=None,policy='dqn',start=(6,10),seed=0,temperature=1.):
    if policy not in ('dqn','softmax','random','reward-greedy'):raise ValueError('Unknown policy.')
    if policy in ('dqn','softmax') and network is None:raise ValueError('DQN network is required.')
    rng=np.random.default_rng(seed)
    burned=np.zeros((cfg.size,cfg.size),bool);burned[start[1],start[0]]=True
    active={tuple(start)};frames=[]
    for t in range(cfg.horizon):
        speed=float(data.fields['wind_speed_ms'][t].mean())
        for _ in range(diffusion_steps(speed,cfg)):
            positions=sorted(active)
            if not positions:break
            if policy in ('dqn','softmax'):
                observations=np.stack([data.observation(p,t) for p in positions])
                with torch.no_grad():q=network(torch.from_numpy(observations)).cpu().numpy()
                if policy=='dqn':choices=q.argmax(1)
                else:
                    # Thesis 3.5: the policy is a probability distribution over actions.
                    prob=np.exp((q-q.max(1,keepdims=True))/temperature);prob/=prob.sum(1,keepdims=True)
                    choices=[rng.choice(9,p=p) for p in prob]
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

def q_table(data,network):
    obs=observation_table(data)
    with torch.no_grad():return network(torch.from_numpy(obs.reshape(-1,8))).numpy().reshape(obs.shape[:3]+(9,))

def fast_spread(data,cfg,q,rewards,start=(6,10),seed=0,temperature=1.,policy='softmax',theta=0.,barrier=None,sharpness=None,residence=None,
                spot=None,spot_m=500.,spot_ref_ms=5.):
    """Vectorised spread with the same rules as spread(); fully enclosed sources are skipped.

    Beyond the thesis: theta raises the ignition reward threshold (A), barrier marks cells that
    cannot burn (B), and sharpness turns the threshold into P(ignite)=sigmoid(sharpness*(r-theta)) (E).
    residence is a burnout time in steps: a cell can ignite neighbours only for that many steps after
    it caught fire, so a front that meets poor conditions dies out instead of waiting indefinitely.
    spot is the chance per step that a burning cell throws a firebrand (spotting). It lands downwind,
    direction wind_to_deg plus N(0, 15 deg), at an exponential distance with mean spot_m scaled by
    wind speed / spot_ref_ms, and ignites the cell if its fuel reward (stay action) reaches theta.
    """
    rng=np.random.default_rng(seed);n=cfg.size
    burned=np.zeros((n,n),bool);burned[start[1],start[0]]=True
    active=np.zeros((n,n),bool);active[start[1],start[0]]=True
    ignited=np.full((n,n),-1);ignited[start[1],start[0]]=0
    fuel=data.fields['ndvi']>0;frames=[]
    if barrier is not None:fuel=fuel&~barrier
    for t in range(cfg.horizon):
        speed=float(data.fields['wind_speed_ms'][t].mean())
        if residence is not None:active&=ignited>t-residence
        if spot:
            ys,xs=np.nonzero(active&(rng.random((n,n))<spot))
            if len(xs):
                angle=np.deg2rad(data.fields['wind_to_deg'][t,ys,xs]+rng.normal(0,15,len(xs)))
                d=rng.exponential(spot_m/cfg.cell_m,len(xs))*data.fields['wind_speed_ms'][t,ys,xs]/spot_ref_ms
                nx=np.rint(xs+d*np.cos(angle)).astype(int);ny=np.rint(ys+d*np.sin(angle)).astype(int)
                keep=(nx>=0)&(nx<n)&(ny>=0)&(ny<n);nx,ny=nx[keep],ny[keep]
                ok=fuel[t,ny,nx]&(rewards[t,ny,nx,8]>=theta)&~burned[ny,nx]
                ignited[ny[ok],nx[ok]]=t;burned[ny[ok],nx[ok]]=True;active[ny[ok],nx[ok]]=True
        for _ in range(diffusion_steps(speed,cfg)):
            open_=np.pad(~burned,1)
            free=np.zeros((n,n),bool)
            for dx,dy in ACTIONS[:8]:free|=open_[1+dy:n+1+dy,1+dx:n+1+dx]
            ys,xs=np.nonzero(active&free)
            inner=(xs>0)&(xs<n-1)&(ys>0)&(ys<n-1);ys,xs=ys[inner],xs[inner]
            if not len(xs):break
            qs=q[t,ys,xs]
            if policy=='dqn':choices=qs.argmax(1)
            else:
                prob=np.exp((qs-qs.max(1,keepdims=True))/temperature);prob/=prob.sum(1,keepdims=True)
                choices=(prob.cumsum(1)<rng.random(len(xs))[:,None]).sum(1).clip(0,8)
            nx=xs+ACTIONS[choices,0];ny=ys+ACTIONS[choices,1]
            r=rewards[t,ny,nx,choices]
            if sharpness is None:ok=fuel[t,ny,nx]&(r>=theta)
            else:ok=fuel[t,ny,nx]&(rng.random(len(xs))<.5+.5*np.tanh(.5*sharpness*(r-theta)))
            fresh=~burned[ny,nx]&ok;ignited[ny[fresh],nx[fresh]]=t
            burned[ny[ok],nx[ok]]=True;active[ny[ok],nx[ok]]=True
        frames.append(burned.copy())
    return np.stack(frames)

def evaluate_observed(predicted,observed,metadata):
    """Reject synthetic/unknown labels for a real-event reproduction result."""
    if metadata.get('kind')!='observational':raise ValueError('Observed event provenance is required.')
    required=['source_url','event_id','mask_method','time_window','grid_bounds','grid_origin','version']
    if any(not metadata.get(key) for key in required):raise ValueError('Incomplete observed-mask provenance.')
    if metadata.get('is_example') or any('REPLACE_' in str(metadata.get(key,'')) for key in required):raise ValueError('Example provenance is not an observed event.')
    return {'iou':iou(predicted,observed),'event_id':metadata['event_id'],'provenance':metadata}
