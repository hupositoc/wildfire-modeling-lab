"""Build one event's inputs, burn mask and VIIRS arrival with the same rules for every fire.

Grid: 50x50 cells of 100 m. The AOI is fixed before any burn map is read: its centre lies 1.5 km
downwind of the reported ignition village, using the ERA5 mean wind of the first six hours, so the
ignition sits 1 km in from the upwind edge.
Horizon: ignition to the reported main-fire-out time, at most 100 half-hour steps (the thesis horizon).
Inputs: data/fetch_open_sources.py (NDVI of the previous year, AW3D30, WorldCover),
data/synth_weather.py (ERA5 + MicroMet), data/prepare_dataset.py.
Mask (as v3 of Uljin): post = first scene with AOI cloud <=5% from 21 days after containment;
pre = the clear scene of the previous year closest to post - 365 days, so phenology cancels;
pixel dNBR >= 0.16, cell burned when >= 50% of valid pixels are burned.
Ignition cell: the reported village is a centroid, not the origin, so it is moved to the nearest
burned cell (reported with the distance). This reads the mask only to place the start.
"""
import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import requests
from rasterio.enums import Resampling
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import fetch_open_sources as fos
import synth_weather as sw
import prepare_dataset
import build_burn_mask as bbm
import viirs_arrival as va

N=50;CELL_M=100.;OFFSET_M=1500.;MAX_STEPS=100;KST=timezone(timedelta(hours=9))
FIRMS='https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/{y}/viirs-snpp_{y}_Republic_of_Korea.csv'

def bounds_at(lat,lon):
    dlat=N*CELL_M/110574.;dlon=N*CELL_M/(111320.*np.cos(np.radians(lat)))
    return (lon-dlon/2,lat-dlat/2,lon+dlon/2,lat+dlat/2)

def aoi(ev,t0):
    d=sw.era5_at(ev['lat'],ev['lon'],t0.date().isoformat(),(t0+timedelta(days=1)).date().isoformat())
    h=d['hourly'];stamps=np.array([np.datetime64(s) for s in h['time']]);start=np.datetime64(t0.replace(tzinfo=None),'m')
    k=(stamps>=start-np.timedelta64(30,'m'))&(stamps<start+np.timedelta64(6,'h'))
    rad=np.radians(np.array(h['wind_direction_10m'])[k]);speed=np.array(h['wind_speed_10m'])[k]
    # wind_direction is where the wind comes from; the fire moves the other way.
    east=-np.mean(speed*np.sin(rad));north=-np.mean(speed*np.cos(rad));to=np.degrees(np.arctan2(north,east))
    lat=ev['lat']+OFFSET_M*np.sin(np.radians(to))/110574.
    lon=ev['lon']+OFFSET_M*np.cos(np.radians(to))/(111320.*np.cos(np.radians(ev['lat'])))
    return bounds_at(lat,lon),{'wind_to_deg_ccw_from_east':round(float(to),1),'mean_speed_ms':round(float(np.hypot(east,north)),2)}

def cell_of(b,lat,lon):
    return int((lon-b[0])/(b[2]-b[0])*N),int((lat-b[1])/(b[3]-b[1])*N)

def clear_days(days,order):
    for day in order:
        scl=fos.mosaic(days[day],'scl',resampling=Resampling.nearest)
        cloud=float(np.nanmean(np.isin(scl,fos.CLOUD_CLASSES)))
        if cloud<=.05:return day,cloud,scl
    raise RuntimeError('No clear scene in window.')

def burn_mask(contained,out,threshold=.16):
    post_from=contained+timedelta(days=21)
    post=fos.by_date(fos.search_s2(post_from.isoformat(),(post_from+timedelta(days=90)).isoformat()))
    post_day,post_cloud,post_scl=clear_days(post,sorted(post))
    anchor=date.fromisoformat(post_day)-timedelta(days=365)
    pre=fos.by_date(fos.search_s2((anchor-timedelta(days=30)).isoformat(),(anchor+timedelta(days=30)).isoformat()))
    pre_day,pre_cloud,pre_scl=clear_days(pre,sorted(pre,key=lambda d:abs((date.fromisoformat(d)-anchor).days)))
    _,nbr_pre=bbm.indices(pre[pre_day]);ndvi_post,nbr_post=bbm.indices(post[post_day])
    dnbr=nbr_pre-nbr_post
    valid=np.isfinite(dnbr)&np.isfinite(ndvi_post)&~np.isin(post_scl,fos.CLOUD_CLASSES+(6,))&~np.isin(pre_scl,fos.CLOUD_CLASSES+(6,))
    burned=np.full(dnbr.shape,np.nan);burned[valid]=dnbr[valid]>=threshold
    fraction=fos.cells(burned);mask=np.nan_to_num(fraction)>=.5
    # cells() flips rows so row 0 is the southern edge, matching the model grid.
    np.save(str(out),mask)
    return mask,{'time_window':pre_day+' (pre) / '+post_day+' (post)','pre_aoi_cloud':pre_cloud,'post_aoi_cloud':post_cloud,
                 'pre_items':[i['id'] for i in pre[pre_day]],'post_items':[i['id'] for i in post[post_day]],
                 'water_cells':int((np.nan_to_num(fos.cells((post_scl==6).astype(float)))>=.5).sum())}

def snap(mask,cell,max_cells=25):
    ys,xs=np.nonzero(mask)
    if not len(xs):raise RuntimeError('Empty burn mask.')
    d=np.hypot(xs-cell[0],ys-cell[1]);i=int(d.argmin())
    if d[i]>max_cells:raise RuntimeError('No burned cell within %d cells of the reported village.'%max_cells)
    return (int(xs[i]),int(ys[i])),float(d[i])*CELL_M

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('event');parser.add_argument('--root',type=Path,default=Path('data/events'))
    parser.add_argument('--registry',type=Path,default=Path(__file__).resolve().parent/'events.json')
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    ev=json.loads(args.registry.read_text(encoding='utf-8'))[args.event]
    t0=datetime.fromisoformat(ev['ignition_kst']).replace(tzinfo=KST).astimezone(timezone.utc)
    t0=t0.replace(minute=t0.minute//30*30,second=0)
    end=datetime.fromisoformat(ev['contained_kst']).replace(tzinfo=KST).astimezone(timezone.utc)
    steps=int(min(MAX_STEPS,np.ceil((end-t0).total_seconds()/1800)+1))
    out=args.root/args.event;raw=out/'raw';raw.mkdir(parents=True,exist_ok=True)
    b,wind=aoi(ev,t0);fos.BOUNDS=b;va.BOUNDS=b
    print(json.dumps({'event':args.event,'bounds':[round(v,5) for v in b],'start_utc':t0.isoformat(),'steps':steps,**wind}),flush=True)
    if not (raw/'static.csv').exists():fos.main(['--out',str(raw),'--skip-weather','--year',str(t0.year-1),'--workers',str(args.workers)])
    lat=(b[1]+b[3])/2;sw.CENTER=(lat,(b[0]+b[2])/2);sw.DY=(b[3]-b[1])/N*110574.;sw.DX=(b[2]-b[0])/N*111320.*np.cos(np.radians(lat))
    day=t0.date();last=(t0+timedelta(minutes=30*steps)).date()
    if not (raw/'weather_micromet.csv').exists():
        sw.main(['--extras',str(raw/'extras.csv'),'--output',str(raw/'weather_micromet.csv'),'--start',(day-timedelta(days=1)).isoformat(),'--end',(last+timedelta(days=1)).isoformat()])
    mask_path=out/'observed.npy'
    mask,mask_meta=burn_mask(datetime.fromisoformat(ev['contained_kst']).date(),mask_path)
    reported=cell_of(b,ev['lat'],ev['lon']);ignition,moved_m=snap(mask,reported)
    meta={'kind':'observational','event_id':args.event,'version':'multi-event-v1','source_url':'https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a',
          'mask_method':'pixel dNBR>=0.16 (pre: previous-year clear scene nearest post-365 d); cell burned if >=50% of valid pixels burned; water/cloud excluded via SCL',
          **mask_meta,'grid_bounds':list(b),'grid_origin':'lower-left','shape':[N,N],'burned_cells':int(mask.sum()),
          'mask_sha256':hashlib.sha256(mask_path.read_bytes()).hexdigest(),'built_utc':datetime.now(timezone.utc).isoformat()}
    mask_path.with_suffix('.json').write_text(json.dumps(meta,indent=1),encoding='utf-8')
    dataset=out/'dataset.npz'
    prepare_dataset.main(['--static',str(raw/'static.csv'),'--weather',str(raw/'weather_micromet.csv'),'--scenes',str(raw/'scenes.csv'),
                          '--output',str(dataset),'--steps',str(steps),'--start',t0.isoformat(),'--weather-meta',str(raw/'weather_micromet.json'),
                          '--event-id',args.event,'--bounds',*map(str,b),'--ignition-cell',*map(str,ignition)])
    firms=args.root/('viirs-snpp_%d.csv'%t0.year)
    if not firms.exists():firms.write_bytes(requests.get(FIRMS.format(y=t0.year),timeout=300).content)
    va.START=ignition;arrival=va.arrival(firms,t0);np.save(str(out/'viirs_arrival.npy'),arrival)
    summary={'event':args.event,'bounds':list(b),'start_utc':t0.isoformat(),'steps':steps,'aoi_wind':wind,'reported_cell':list(reported),
             'ignition_cell':list(ignition),'ignition_moved_m':moved_m,'burned_cells':int(mask.sum()),'burned_fraction':round(float(mask.mean()),3),
             'viirs_cells':int(np.isfinite(arrival).sum()),'mask_window':mask_meta['time_window']}
    (out/'event.json').write_text(json.dumps(summary,indent=1),encoding='utf-8')
    print(json.dumps(summary),flush=True)
    for y in range(N-1,-1,-2):print(''.join('S' if (x,y)==ignition or (x,y-1)==ignition else ('#' if mask[y,x] else '.') for x in range(N)))

if __name__=='__main__':main()
