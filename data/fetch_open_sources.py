"""Rebuild the thesis inputs from open archives instead of Google Earth Engine.

Static: Sentinel-2 L2A (Earth Search) 2021 scenes with AOI cloud <=20%, AW3D30 (Planetary Computer).
Weather: ERA5-Land hourly through the Copernicus CDS API (needs the user's own ~/.cdsapirc).
Outputs match data/prepare_dataset.py: static.csv, scenes.csv, weather.csv.
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import requests
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT

BOUNDS=(129.29,37.03,129.356,37.075)
N=50
PX=12
EARTH_SEARCH='https://earth-search.aws.element84.com/v1/search'
PC_SEARCH='https://planetarycomputer.microsoft.com/api/stac/v1/search'
PC_SIGN='https://planetarycomputer.microsoft.com/api/sas/v1/sign'
CLOUD_CLASSES=(3,8,9,10,11)

def grid(px=PX):
    return from_bounds(*BOUNDS,N*px,N*px),N*px

def warp(href,px=PX,resampling=Resampling.bilinear):
    transform,size=grid(px)
    env=dict(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',AWS_NO_SIGN_REQUEST='YES',GDAL_HTTP_MAX_RETRY='4',GDAL_HTTP_RETRY_DELAY='2')
    with rasterio.Env(**env),rasterio.open(href) as src,WarpedVRT(src,crs='EPSG:4326',transform=transform,width=size,height=size,resampling=resampling,nodata=0) as vrt:
        a=vrt.read(1).astype(np.float64)
    a[a==0]=np.nan
    return a

def cells(a):
    n=a.shape[0]//N
    return np.nanmean(a.reshape(N,n,N,n).swapaxes(1,2).reshape(N,N,-1),axis=2)[::-1]

def search_s2(start,end):
    body={'collections':['sentinel-2-l2a'],'bbox':list(BOUNDS),'datetime':start+'T00:00:00Z/'+end+'T23:59:59Z','limit':100}
    url,features=EARTH_SEARCH,[]
    while url:
        page=requests.post(url,json=body,timeout=120).json()
        if 'features' not in page:raise RuntimeError('STAC search failed: '+json.dumps(page)[:300])
        features+=page['features']
        nxt=[l for l in page.get('links',[]) if l.get('rel')=='next']
        url,body=(nxt[0]['href'],nxt[0].get('body',body)) if nxt else (None,None)
    return features

def by_date(features):
    days={}
    for f in features:days.setdefault(f['properties']['datetime'][:10],[]).append(f)
    return days

def mosaic(items,band,px=PX,resampling=Resampling.bilinear):
    out=None
    for item in items:
        a=warp(item['assets'][band]['href'],px,resampling)
        out=a if out is None else np.where(np.isnan(out),a,out)
    return out

def scene(day,items):
    scl=mosaic(items,'scl',resampling=Resampling.nearest)
    cloud=float(np.nanmean(np.isin(scl,CLOUD_CLASSES)))
    if cloud>.2:return day,cloud,None
    red=mosaic(items,'red');nir=mosaic(items,'nir')
    return day,cloud,(nir-red)/(nir+red)

def annual_ndvi(year,workers):
    days=by_date(search_s2(str(year)+'-01-01',str(year)+'-12-31'))
    with ThreadPoolExecutor(workers) as pool:results=list(pool.map(lambda d:scene(d,days[d]),sorted(days)))
    kept=[r for r in results if r[2] is not None]
    if not kept:raise RuntimeError('No scene passed the AOI cloud filter.')
    stack=np.stack([r[2] for r in kept])
    manifest=[{'date':d,'aoi_cloud_fraction':round(c,4),'used':c<=.2,'items':';'.join(i['id'] for i in days[d])} for d,c,_ in results]
    return np.nanmean(stack,axis=0),manifest

def terrain():
    item=requests.post(PC_SEARCH,json={'collections':['alos-dem'],'bbox':list(BOUNDS)},timeout=120).json()['features'][0]
    href=requests.get(PC_SIGN,params={'href':item['assets']['data']['href']},timeout=120).json()['href']
    z=warp(href,px=PX,resampling=Resampling.bilinear)
    transform,size=grid()
    dy=abs(transform.e)*111320.
    dx=transform.a*111320.*np.cos(np.radians(np.mean(BOUNDS[1::2])))
    dz_north,dz_east=np.gradient(z,dy,dx)
    dz_north=-dz_north
    slope=np.degrees(np.arctan(np.hypot(dz_east,dz_north)))
    aspect=np.radians((np.degrees(np.arctan2(-dz_east,-dz_north))+360)%360)
    return slope,np.sin(aspect),np.cos(aspect),item['id'],z

def landcover():
    """ESA WorldCover 2021 (10 m); class 10 is tree cover. Used only by the beyond-thesis fuel barrier."""
    item=requests.post(PC_SEARCH,json={'collections':['esa-worldcover'],'bbox':list(BOUNDS),'datetime':'2021-01-01/2021-12-31'},timeout=120).json()['features'][0]
    href=requests.get(PC_SIGN,params={'href':item['assets']['map']['href']},timeout=120).json()['href']
    classes=warp(href,px=PX,resampling=Resampling.nearest)
    return np.where(np.isnan(classes),np.nan,classes==10),item['id']

def weather(path):
    import cdsapi
    cdsapi.Client().retrieve('reanalysis-era5-land',{
        'variable':['2m_temperature','2m_dewpoint_temperature','10m_u_component_of_wind','10m_v_component_of_wind'],
        'year':'2022','month':'03','day':['04','05','06'],'time':['%02d:00'%h for h in range(24)],
        'area':[37.3,129.0,36.8,129.6],'data_format':'netcdf','download_format':'unarchived'},str(path))

def read_era5(path):
    import h5py
    with h5py.File(str(path),'r') as f:
        names={'t2m':'temperature_2m','d2m':'dewpoint_temperature_2m','u10':'u_component_of_wind_10m','v10':'v_component_of_wind_10m'}
        time=f['valid_time'][:] if 'valid_time' in f else f['time'][:]
        lat=f['latitude'][:];lon=f['longitude'][:]
        data={}
        for key,name in names.items():
            v=f[key];a=v[:].astype(np.float64)
            fill=v.attrs.get('_FillValue')
            if fill is not None:a[a==fill[0]]=np.nan
            scale=v.attrs.get('scale_factor');offset=v.attrs.get('add_offset')
            if scale is not None:a=a*scale[0]+(offset[0] if offset is not None else 0)
            data[name]=a.reshape(len(time),len(lat),len(lon))
    return time.astype(np.int64),lat,lon,data

def to_cells(lat,lon,field):
    lons=BOUNDS[0]+(np.arange(N)+.5)*(BOUNDS[2]-BOUNDS[0])/N
    lats=BOUNDS[1]+(np.arange(N)+.5)*(BOUNDS[3]-BOUNDS[1])/N
    gy,gx=np.meshgrid(lat,lon,indexing='ij')
    valid=np.isfinite(field)
    out=np.empty((N,N))
    for y,la in enumerate(lats):
        for x,lo in enumerate(lons):
            d=np.hypot(gy-la,(gx-lo)*np.cos(np.radians(la)))
            d=np.where(valid,d,np.inf);near=np.argsort(d,axis=None)[:4]
            w=1/np.maximum(d.flat[near],1e-6)**2
            out[y,x]=np.sum(w*field.flat[near])/np.sum(w)
    return out

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('data/raw'))
    parser.add_argument('--skip-static',action='store_true');parser.add_argument('--skip-weather',action='store_true')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--year',type=int,default=2021,help='NDVI year (the year before the fire)')
    parser.add_argument('--skip-extras',action='store_true',help='skip elevation and WorldCover tree fraction (extras.csv)')
    args=parser.parse_args(argv);args.out.mkdir(parents=True,exist_ok=True)
    if not args.skip_static:
        ndvi,manifest=annual_ndvi(args.year,args.workers)
        slope,asin,acos,dem_id,_=terrain()
        maps={'ndvi':cells(ndvi),'slope':cells(slope),'aspect_sin':cells(asin),'aspect_cos':cells(acos)}
        with (args.out/'static.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['x','y']+list(maps))
            for y in range(N):
                for x in range(N):w.writerow([x,y]+[float(maps[k][y,x]) for k in maps])
        with (args.out/'scenes.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
        (args.out/'static_sources.json').write_text(json.dumps({'sentinel2':'earth-search sentinel-2-l2a %d,'%args.year+' per-scene NDVI(B08,B04), AOI SCL cloud fraction <=0.2','dem':'planetary-computer alos-dem '+dem_id,'scenes_used':sum(m['used'] for m in manifest),'scenes_total':len(manifest),'built_utc':datetime.now(timezone.utc).isoformat()},indent=2),encoding='utf-8')
        print('static ok: scenes used',sum(m['used'] for m in manifest),'of',len(manifest))
    if not args.skip_extras:
        _,_,_,dem_id,z=terrain();tree,wc_id=landcover()
        maps={'elevation_m':cells(z),'tree_fraction':cells(tree)}
        with (args.out/'extras.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['x','y']+list(maps))
            for y in range(N):
                for x in range(N):w.writerow([x,y]+[float(maps[k][y,x]) for k in maps])
        print('extras ok:',dem_id,wc_id)
    if not args.skip_weather:
        nc=args.out/'era5_land.nc'
        if not nc.exists():weather(nc)
        time,lat,lon,data=read_era5(nc)
        with (args.out/'weather.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);names=list(data);w.writerow(['time_ms','x','y']+names)
            for i,t in enumerate(time):
                maps=[to_cells(lat,lon,data[k][i]) for k in names]
                for y in range(N):
                    for x in range(N):w.writerow([int(t)*1000,x,y]+[float(m[y,x]) for m in maps])
        print('weather ok: hours',len(time))

if __name__=='__main__':main()
