from dataclasses import dataclass
import math
from pathlib import Path
import json
import numpy as np

# Coordinates: x increases east, y increases north, origin at lower-left.
ACTIONS=np.array([[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1],[0,0]],dtype=int)

@dataclass
class Config:
    size:int=50
    horizon:int=100
    dt_seconds:float=1800.
    cell_m:float=100.
    diffusion_scale:float=.1
    gamma:float=1.
    # Order follows Table 4.2: moisture, vegetation, slope, wind. A fifth weight (beyond the thesis)
    # scores the conifer share from the forest type map and needs the conifer_fraction field.
    weights:tuple=(.264,.113,.140,.483)
    ndvi_mode:str='literal'
    rounding:str='floor'
    emc_max:float=20.
    # Beyond the thesis: a cell already burned in this episode has no fuel left, so re-entering
    # it (including the stay action) earns nothing. Off reproduces the printed reward.
    consumed_fuel:bool=False
    # Beyond the thesis: when set, the wind term is scaled by wind speed / wind_ref_ms, so wind
    # direction matters more in strong wind and little in calm air. None keeps the printed reward.
    wind_ref_ms:float=None

def emc_percent(temp_c,rh):
    """Hailwood-Horrobin temperature-dependent form, T in Celsius, RH fraction.

    Coefficients follow the USDA Wood Handbook temperature-Celsius form.
    Model implementation is separate from the thesis's unspecified code.
    """
    t=np.asarray(temp_c,dtype=float);h=np.asarray(rh,dtype=float)
    if np.any((h<0)|(h>1)):raise ValueError('Relative humidity must be a fraction in [0,1].')
    w=349+1.29*t+.0135*t*t
    k=.805+.000736*t-.00000273*t*t
    k1=6.27-.00938*t-.000303*t*t
    k2=1.91+.0407*t-.000293*t*t
    kh=k*h
    return 1800/w*(kh/(1-kh)+(k1*kh+2*k1*k2*kh*kh)/(1+k1*kh+k1*k2*kh*kh))

def angular_alignment(action,angle_deg):
    delta=ACTIONS[int(action)]
    norm=np.linalg.norm(delta)
    if not norm:return 0.0
    theta=np.deg2rad(angle_deg)
    return float((delta[0]*np.cos(theta)+delta[1]*np.sin(theta))/norm)

def diffusion_steps(speed,cfg):
    raw=cfg.diffusion_scale*float(speed)*cfg.dt_seconds/cfg.cell_m
    return int(math.floor(raw) if cfg.rounding=='floor' else math.ceil(raw))

def fuel_moisture(f,ix=Ellipsis):
    """EMC in percent, plus the optional terrain offset when the dataset carries one."""
    emc=emc_percent(f['temperature_c'][ix],f['rh_fraction'][ix])
    if 'moisture_offset_pct' in f:emc=np.maximum(emc+f['moisture_offset_pct'][ix],0.)
    return emc

class Dataset:
    def __init__(self,fields,metadata):
        self.fields=fields;self.metadata=metadata
        required=('temperature_c','rh_fraction','wind_to_deg','wind_speed_ms','ndvi','upslope_deg')
        # Beyond the thesis: an optional terrain correction to fuel moisture (see data/add_terrain_moisture.py).
        optional=('moisture_offset_pct','conifer_fraction')
        if not set(required)<=set(fields)<=set(required+optional):raise ValueError('Expected the six named environmental fields.')
        shape=next(iter(fields.values())).shape
        if len(shape)!=3 or any(v.shape!=shape for v in fields.values()):raise ValueError('Fields must share shape [time,y,x].')
        if shape[1]!=shape[2] or shape[0]<2:raise ValueError('Expected at least 2 time steps on a square grid.')
        if any(not np.isfinite(v).all() for v in fields.values()):raise ValueError('NaN/Inf environmental input.')
        if np.any((fields['ndvi'] < -1)|(fields['ndvi']>1)):raise ValueError('NDVI must be raw [-1,1].')
        emc_percent(fields['temperature_c'],fields['rh_fraction'])
        if np.any(fields['wind_speed_ms']<0):raise ValueError('Negative wind speed.')
    @classmethod
    def load(cls,path):
        path=Path(path)
        metadata=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        if metadata.get('kind') not in ('synthetic','observational'):raise ValueError('Explicit provenance kind is required.')
        with np.load(str(path),allow_pickle=False) as data:
            return cls({k:data[k] for k in data.files},metadata)
    def observation(self,pos,t):
        x,y=map(int,pos);n=self.fields['ndvi'].shape[-1]
        f=self.fields;t=min(int(t),f['ndvi'].shape[0]-1)
        # Eight inputs: grid coordinates and six environment variables (thesis p.13).
        return np.array([x/(n-1),y/(n-1),f['temperature_c'][t,y,x]/40.,f['rh_fraction'][t,y,x],
                         f['wind_to_deg'][t,y,x]/360.,f['wind_speed_ms'][t,y,x]/15.,
                         f['ndvi'][t,y,x],f['upslope_deg'][t,y,x]/360.],dtype=np.float32)
    def reward(self,pos,action,t,cfg):
        x,y=map(int,pos);f=self.fields;t=min(int(t),f['ndvi'].shape[0]-1)
        ndvi=float(f['ndvi'][t,y,x])
        # Thesis says non-vegetated cells do not spread and receive -1.
        if ndvi<=0:return -1.
        moisture=1-float(fuel_moisture(f,(t,y,x)))/cfg.emc_max
        vegetation=ndvi/50.-1 if cfg.ndvi_mode=='literal' else ndvi
        slope=angular_alignment(action,f['upslope_deg'][t,y,x])
        wind=angular_alignment(action,f['wind_to_deg'][t,y,x])
        if cfg.wind_ref_ms:wind*=float(f['wind_speed_ms'][t,y,x])/cfg.wind_ref_ms
        terms=[moisture,vegetation,slope,wind]
        if len(cfg.weights)==5:terms.append(float(f['conifer_fraction'][t,y,x]))
        return float(np.dot(cfg.weights,terms))

def observation_table(data):
    f=data.fields;t,n,_=f['ndvi'].shape
    yy,xx=np.meshgrid(np.arange(n),np.arange(n),indexing='ij')
    grid=lambda a:np.broadcast_to(a/(n-1),(t,n,n))
    return np.stack([grid(xx),grid(yy),f['temperature_c']/40.,f['rh_fraction'],f['wind_to_deg']/360.,
                     f['wind_speed_ms']/15.,f['ndvi'],f['upslope_deg']/360.],-1).astype(np.float32)

def reward_table(data,cfg):
    """Reward for entering each [t,y,x] cell with each action; same values as Dataset.reward."""
    f=data.fields;ndvi=f['ndvi'].astype(float)
    moisture=1-fuel_moisture(f)/cfg.emc_max
    vegetation=ndvi/50.-1 if cfg.ndvi_mode=='literal' else ndvi
    norms=np.linalg.norm(ACTIONS,axis=1);unit=np.divide(ACTIONS,norms[:,None],out=np.zeros(ACTIONS.shape),where=norms[:,None]>0)
    align=lambda deg:np.cos(np.deg2rad(deg))[...,None]*unit[:,0]+np.sin(np.deg2rad(deg))[...,None]*unit[:,1]
    w=cfg.weights
    wind=align(f['wind_to_deg'])
    if cfg.wind_ref_ms:wind=wind*(f['wind_speed_ms']/cfg.wind_ref_ms)[...,None]
    table=w[0]*moisture[...,None]+w[1]*vegetation[...,None]+w[2]*align(f['upslope_deg'])+w[3]*wind
    if len(w)==5:table=table+w[4]*f['conifer_fraction'][...,None]
    return np.where(ndvi[...,None]<=0,-1.,table)

class FirePathEnv:
    """One training path; full spread is a separate branching rollout.

    Boundary arrival terminates. Raw data use y-up coordinates. Wind means
    direction of travel, not meteorological wind-from. No target mask is read.
    """
    def __init__(self,data,cfg,start=(6,10),bonus=None):
        self.data=data;self.cfg=cfg;self.start=np.array(start,dtype=int);self.bonus=bonus
        if data.fields['ndvi'].shape!=(cfg.horizon,cfg.size,cfg.size):raise ValueError('Config/data shape mismatch.')
        if not np.isclose(sum(cfg.weights),1) or any(w<0 for w in cfg.weights):raise ValueError('Weights must lie on the simplex.')
        if len(cfg.weights)==5 and 'conifer_fraction' not in data.fields:raise ValueError('A fifth weight needs the conifer_fraction field.')
        if len(cfg.weights) not in (4,5):raise ValueError('Expected four or five weights.')
        if cfg.ndvi_mode not in ('literal','raw'):raise ValueError('Unknown NDVI interpretation.')
        if cfg.rounding not in ('floor','ceil'):raise ValueError('Unknown rounding interpretation.')
        if np.any(self.start<0) or np.any(self.start>=cfg.size):raise ValueError('Ignition outside AOI.')
    def reset(self):
        self.pos=self.start.copy();self.t=0;self.done=False
        self.burned=np.zeros((self.cfg.size,self.cfg.size),bool);self.burned[self.start[1],self.start[0]]=True
        return self.data.observation(self.pos,0)
    def step(self,action):
        if not 0<=int(action)<len(ACTIONS):raise ValueError('Unknown action.')
        if self.done or self.t>=self.cfg.horizon:raise RuntimeError('Reset a completed episode.')
        x,y=self.pos;speed=self.data.fields['wind_speed_ms'][self.t,y,x]
        moves=diffusion_steps(speed,self.cfg);reward=0.;path=[]
        # Every microstep is adjacent; no leap across non-fuel cells.
        for _ in range(moves):
            dest=self.pos+ACTIONS[int(action)]
            if np.any(dest<0) or np.any(dest>=self.cfg.size):break
            spent=self.cfg.consumed_fuel and self.burned[dest[1],dest[0]]
            env=0. if spent else self.data.reward(dest,action,self.t,self.cfg);gained=env
            # Thesis p.14: agreement with the actual burned map is an additional training reward.
            if self.bonus is not None and not spent:gained+=float(self.bonus[dest[1],dest[0]])
            self.burned[dest[1],dest[0]]=True
            reward+=gained
            # The path keeps the environmental reward only: the author's code used the JI term for
            # training, never for judging the footprint, so the target cannot leak into the map.
            path.append((int(dest[0]),int(dest[1]),env))
            if self.data.fields['ndvi'][self.t,dest[1],dest[0]]<=0:break
            self.pos=dest
            if np.any(self.pos==0) or np.any(self.pos==self.cfg.size-1):break
        self.t+=1
        done=self.t>=self.cfg.horizon or np.any(self.pos==0) or np.any(self.pos==self.cfg.size-1)
        self.done=bool(done)
        return self.data.observation(self.pos,min(self.t,self.cfg.horizon-1)),reward,bool(done),{'position':self.pos.copy(),'microsteps_budget':moves,'path':path}

def iou(pred,target):
    p=np.asarray(pred,dtype=bool);t=np.asarray(target,dtype=bool)
    if p.shape!=t.shape:raise ValueError('Mask shape mismatch.')
    union=np.logical_or(p,t).sum()
    # Both-empty is undefined for positive-class footprint evaluation.
    return float(np.logical_and(p,t).sum()/union) if union else float('nan')

def synthetic(seed,cfg):
    """Explicit synthetic fields for software verification, never observed wildfire."""
    from scipy.ndimage import gaussian_filter
    rng=np.random.default_rng(seed);n=cfg.size;h=cfg.horizon
    fuel=np.clip(.55+.4*gaussian_filter(rng.normal(size=(n,n)),2),.05,.95)
    fuel[:,n//2:n//2+1]=-.2
    fuel[n//2-3:n//2+3,n//2]=.5  # a narrow fuel bridge
    xx,yy=np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n))
    def repeat(a):return np.broadcast_to(a,(h,n,n)).copy().astype(np.float32)
    theta=float(rng.uniform(0,360))
    fields={'temperature_c':repeat(22+4*xx),'rh_fraction':repeat(.3+.15*yy),
            'wind_to_deg':repeat(np.full((n,n),theta)), 'wind_speed_ms':repeat(np.full((n,n),3.9)),
            'ndvi':repeat(fuel),'upslope_deg':repeat(np.full((n,n),45.))}
    return Dataset(fields,{'kind':'synthetic','seed':int(seed),'generator':'wildfire.model.synthetic','not_observational':True})
