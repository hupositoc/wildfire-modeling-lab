"""Build a provenance-bearing environmental tensor from genuine GEE CSV exports.

This does not create a burned-area target or choose data to reach a target score.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--static',type=Path,required=True);parser.add_argument('--weather',type=Path,required=True)
    parser.add_argument('--scenes',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,choices=[99,100],required=True,help='Author must resolve the paper endpoint/count inconsistency.')
    parser.add_argument('--start',default='2022-03-04T02:00:00+00:00')
    args=parser.parse_args();n=50
    start=datetime.fromisoformat(args.start)
    if start.tzinfo is None:parser.error('Start timestamp requires a UTC offset.')
    requested=start.timestamp()*1000+np.arange(args.steps)*1800000
    static=list(csv.DictReader(args.static.open(encoding='utf-8-sig',newline='')))
    if len(static)!=n*n:raise ValueError('Expected exactly 2500 static grid cells.')
    maps={key:np.full((n,n),np.nan) for key in ['ndvi','slope','aspect_sin','aspect_cos']};seen=set()
    for row in static:
        x,y=int(float(row['x'])),int(float(row['y']))
        if not(0<=x<n and 0<=y<n) or (x,y) in seen:raise ValueError('Invalid/duplicate static cell.')
        seen.add((x,y))
        for key in maps:maps[key][y,x]=float(row[key])
    # GEE aspect is downhill, clockwise from north. Convert to uphill,
    # counterclockwise from east for the model's x-east / y-north convention.
    aspect=np.degrees(np.arctan2(maps['aspect_sin'],maps['aspect_cos']))%360
    upslope=(270-aspect)%360
    groups={}
    for row in csv.DictReader(args.weather.open(encoding='utf-8-sig',newline='')):
        t=float(row['time_ms']);x,y=int(float(row['x'])),int(float(row['y']))
        if not(0<=x<n and 0<=y<n):raise ValueError('Invalid weather cell.')
        group=groups.setdefault(t,{})
        if (x,y) in group:raise ValueError('Duplicate weather time/cell.')
        group[(x,y)]=[float(row[k]) for k in ['temperature_2m','dewpoint_temperature_2m','u_component_of_wind_10m','v_component_of_wind_10m']]
    times=np.array(sorted(groups))
    if len(times)<2 or requested[0]<times[0] or requested[-1]>times[-1]:raise ValueError('Weather does not bracket the requested interval; extrapolation is forbidden.')
    if any(len(groups[t])!=n*n for t in times):raise ValueError('Incomplete hourly weather grid.')
    hourly=np.array([[groups[t][(x,y)] for y in range(n) for x in range(n)] for t in times])
    interpolated=np.empty((args.steps,n*n,4))
    for cell in range(n*n):
        for band in range(4):interpolated[:,cell,band]=np.interp(requested,times,hourly[:,cell,band])
    temp,dew,u,v=[interpolated[:,:,k].reshape(args.steps,n,n) for k in range(4)]
    temp=temp-273.15;dew=dew-273.15
    rh=np.clip(np.exp(17.625*dew/(243.04+dew)-17.625*temp/(243.04+temp)),0,1)
    wind_to=(np.degrees(np.arctan2(v,u))+360)%360
    def repeat(a):return np.broadcast_to(a,(args.steps,n,n)).copy()
    fields={'temperature_c':temp,'rh_fraction':rh,'wind_to_deg':wind_to,'wind_speed_ms':np.hypot(u,v),'ndvi':repeat(maps['ndvi']),'upslope_deg':repeat(upslope)}
    if any(not np.isfinite(a).all() for a in fields.values()):raise ValueError('Missing/nonfinite input; no automatic filling is performed.')
    if np.any((fields['ndvi']<-1)|(fields['ndvi']>1)):raise ValueError('Raw NDVI outside [-1,1].')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(str(args.output),**{k:v.astype(np.float32) for k,v in fields.items()})
    metadata={'kind':'observational','event_id':'uljin-2022','version':'reconstruction-v1','original_thesis_export':False,
              'grid_bounds':[129.29,37.03,129.356,37.075],'grid_origin':'lower-left','shape':[args.steps,n,n],
              'start_utc':start.isoformat(),'step_seconds':1800,'last_timestamp_ms':float(requested[-1]),
              'collections':['COPERNICUS/S2_SR_HARMONIZED','JAXA/ALOS/AW3D30/V4_1','ECMWF/ERA5_LAND/HOURLY'],
              'source_files_sha256':{'static':digest(args.static),'weather':digest(args.weather),'scenes':digest(args.scenes)},
              'processing':['2021 annual mean scene NDVI, scene cloud fraction <=20%','cell-mean terrain aspect represented by circular components','hourly u/v/T/dewpoint linearly interpolated to 30-minute steps','RH derived with documented Magnus approximation','no observed burned-area mask generated','spatial sampling does not increase native weather resolution'],
              'source_url':'https://developers.google.com/earth-engine/datasets/catalog',
              'reproduction_status':'reconstructed inputs; original scene/classifier/hyperparameter choices unresolved'}
    args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps({'shape':metadata['shape'],'observed_burn_mask_created':False,'original_thesis_export':False}))

if __name__=='__main__':main()
