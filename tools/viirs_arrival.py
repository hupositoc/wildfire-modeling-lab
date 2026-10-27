"""VIIRS first-detection step per AOI cell and the C_f lower bound it implies (beyond the thesis).

Input: the keyless FIRMS country archive, e.g.
https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/2022/viirs-snpp_2022_Republic_of_Korea.csv
Step 0 is the dataset start (2022-03-04 02:00 UTC), 30-minute steps. A detection marks every cell
whose centre lies inside its scan x track footprint. The C_f bound uses only arrival times and
wind speed: burned cells seen on the first pass must be reachable from the ignition cell by then,
moving at most one cell per microstep. The burn mask is used only to pick reliable cells.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np

BOUNDS=(129.29,37.03,129.356,37.075);N=50;START=(6,10)

def arrival(path,t0,n=N):
    w,s,e,nn=BOUNDS
    cx=w+(np.arange(n)+.5)*(e-w)/n;cy=s+(np.arange(n)+.5)*(nn-s)/n
    out=np.full((n,n),np.inf)
    for r in csv.DictReader(open(path,encoding='utf-8')):
        lat,lon=float(r['latitude']),float(r['longitude'])
        if not (s-.01<=lat<=nn+.01 and w-.01<=lon<=e+.01):continue
        t=datetime.strptime(r['acq_date']+r['acq_time'],'%Y-%m-%d%H%M').replace(tzinfo=timezone.utc)
        step=(t-t0).total_seconds()/1800
        if step<0 or step>336:continue
        hx=float(r['scan'])*.5/(111.32*np.cos(np.deg2rad(lat)));hy=float(r['track'])*.5/110.57
        m=np.outer(np.abs(cy-lat)<=hy,np.abs(cx-lon)<=hx);out[m]=np.minimum(out[m],step)
    return out

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--firms',type=Path,required=True);parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--mask',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--quantile',type=float,default=50.)
    args=parser.parse_args()
    meta=json.loads(args.dataset.with_suffix('.json').read_text(encoding='utf-8'))
    t0=datetime.fromisoformat(meta['start_utc']);a=arrival(args.firms,t0)
    np.save(str(args.output),a)
    mask=np.load(str(args.mask)).astype(bool);first_step=float(np.nanmin(a[np.isfinite(a)]))
    first=(a<=first_step+.1)&mask;ys,xs=np.nonzero(first)
    distance=float(np.percentile(np.maximum(abs(xs-START[0]),abs(ys-START[1])),args.quantile))
    with np.load(str(args.dataset)) as d:speed=float(d['wind_speed_ms'][:int(np.ceil(first_step))].mean())
    # microsteps per step = C_f * speed * 1800 / 100 (equation 3.12)
    bound=distance/first_step/(speed*18.)
    print(json.dumps({'first_pass_step':first_step,'first_pass_burned_cells':int(first.sum()),'distance_cells_q%g'%args.quantile:distance,
                      'mean_wind_ms':round(speed,2),'cf_lower_bound':round(bound,4)}))

if __name__=='__main__':main()
