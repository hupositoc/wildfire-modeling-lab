"""Terrain-downscaled ERA5 weather for when the ERA5-Land CDS route is unavailable.

ERA5 (0.25 deg) comes from the Open-Meteo archive with downscaling disabled, so every AOI cell
shares one grid value. MicroMet (Liston & Elder 2006, J. Hydrometeorol. 7:217) then adds terrain:
March lapse rates for temperature and vapour pressure, and Liston & Sturm (1998) wind weighting
W = 1 + coupling*(gs*Os + gc*Oc) with direction diversion -0.5*coupling*Os*sin(2(aspect-dir)).
The burned-area target is never read. Output matches data/prepare_dataset.py weather.csv.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import requests

N=50
CENTER=(37.0525,129.323)
ARCHIVE='https://archive-api.open-meteo.com/v1/archive'
T_LAPSE=7.1e-3
VP_LAPSE=.40e-3
DY=.045/N*111320.
DX=.066/N*111320.*np.cos(np.radians(CENTER[0]))

def era5(start,end):
    return era5_at(CENTER[0],CENTER[1],start,end)

def era5_at(lat,lon,start,end):
    params=dict(latitude=lat,longitude=lon,start_date=start,end_date=end,timezone='GMT',models='era5',
                elevation='nan',wind_speed_unit='ms',hourly='temperature_2m,dew_point_2m,wind_speed_10m,wind_direction_10m')
    d=requests.get(ARCHIVE,params=params,timeout=120).json()
    if 'hourly' not in d:raise RuntimeError(d)
    return d

def load_elevation(path):
    z=np.full((N,N),np.nan)
    for row in csv.DictReader(open(path,encoding='utf-8')):z[int(row['y']),int(row['x'])]=float(row['elevation_m'])
    return z

def scaled(a):
    return .5*a/np.abs(a).max()

def curvature(z,eta):
    p=np.pad(z,eta,mode='edge');c=lambda dx,dy:p[eta+dy:eta+dy+N,eta+dx:eta+dx+N]
    d=2*eta*np.hypot(DX,DY)/np.sqrt(2)
    return scaled(((z-(c(0,eta)+c(0,-eta))/2)/(2*eta*DY)+(z-(c(eta,0)+c(-eta,0))/2)/(2*eta*DX)
                   +(z-(c(eta,eta)+c(-eta,-eta))/2)/(2*d)+(z-(c(-eta,eta)+c(eta,-eta))/2)/(2*d))/4)

def downscale(z,z_ref,temp_c,dew_c,speed,wind_from,coupling,eta):
    dz=z-z_ref
    temp=temp_c-T_LAPSE*dz
    e=6.112*np.exp(17.67*dew_c/(dew_c+243.5))*np.exp(-VP_LAPSE*dz)
    dew=np.minimum(243.5*np.log(e/6.112)/(17.67-np.log(e/6.112)),temp)
    dz_north,dz_east=np.gradient(z,DY,DX)
    slope=np.arctan(np.hypot(dz_east,dz_north));aspect=np.arctan2(-dz_east,-dz_north)
    omega_c=curvature(z,eta);direction=np.radians(wind_from)
    omega_s=scaled(slope*np.cos(direction-aspect)) if slope.any() else slope
    weight=1+coupling*(.5*omega_s+.5*omega_c)
    turned=direction-.5*coupling*omega_s*np.sin(2*(aspect-direction))
    s=speed*weight/weight.mean()
    return temp,dew,-s*np.sin(turned),-s*np.cos(turned)

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--extras',type=Path,default=Path('data/raw/extras.csv'))
    parser.add_argument('--output',type=Path,default=Path('data/raw/weather_micromet.csv'))
    parser.add_argument('--start',default='2022-03-04');parser.add_argument('--end',default='2022-03-06')
    parser.add_argument('--coupling',type=float,default=1.,help='0 keeps uniform ERA5, 1 is the MicroMet default')
    parser.add_argument('--eta',type=int,default=5,help='curvature length scale in cells')
    args=parser.parse_args(argv)
    z=load_elevation(args.extras);d=era5(args.start,args.end);h=d['hourly']
    with args.output.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['time_ms','x','y','temperature_2m','dewpoint_temperature_2m','u_component_of_wind_10m','v_component_of_wind_10m'])
        for i,stamp in enumerate(h['time']):
            ms=int(np.datetime64(stamp,'ms').astype(np.int64))
            temp,dew,u,v=downscale(z,d['elevation'],h['temperature_2m'][i],h['dew_point_2m'][i],h['wind_speed_10m'][i],h['wind_direction_10m'][i],args.coupling,args.eta)
            for y in range(N):
                for x in range(N):w.writerow([ms,x,y,temp[y,x]+273.15,dew[y,x]+273.15,u[y,x],v[y,x]])
    meta={'kind':'synthetic','generator':'data/synth_weather.py','base':'ERA5 0.25 deg via Open-Meteo archive, elevation=nan (raw grid value)',
          'grid_point':[d['latitude'],d['longitude']],'grid_elevation_m':d['elevation'],'downscaling':'MicroMet (Liston & Elder 2006), March lapse rates',
          'coupling':args.coupling,'eta_cells':args.eta,'not_era5_land':True,'target_mask_used':False}
    args.output.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta))

if __name__=='__main__':main()
