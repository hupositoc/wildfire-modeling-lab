"""Add a terrain fuel-moisture offset to a prepared tensor (beyond the thesis).

Sun-exposed south slopes hold drier fine fuel than shaded north slopes. Rothermel (1983,
INT-143, fine dead fuel moisture corrections) puts the daytime north/south difference on
steep slopes at a few moisture points. Here offset = K * cos(aspect) * min(slope/30, 1) with
K = 2 percentage points: north-facing steep cells +2, south-facing -2, flat cells 0.
K is fixed from that table, not fitted to any burned-area mask.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

K_PCT=2.

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--static',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with np.load(str(args.dataset)) as data:fields={k:data[k] for k in data.files}
    t,n,_=fields['ndvi'].shape
    aspect_cos=np.full((n,n),np.nan);slope=np.full((n,n),np.nan)
    for row in csv.DictReader(open(args.static,encoding='utf-8')):
        x,y=int(row['x']),int(row['y']);aspect_cos[y,x]=float(row['aspect_cos']);slope[y,x]=float(row['slope'])
    if not (np.isfinite(aspect_cos).all() and np.isfinite(slope).all()):raise ValueError('Static terrain has gaps.')
    offset=K_PCT*aspect_cos*np.minimum(slope/30.,1.)
    fields['moisture_offset_pct']=np.broadcast_to(offset,(t,n,n)).astype(np.float32)
    np.savez_compressed(str(args.output),**fields)
    meta=json.loads(args.dataset.with_suffix('.json').read_text(encoding='utf-8'))
    meta.setdefault('processing',[]).append('terrain fuel-moisture offset K*cos(aspect)*min(slope/30,1), K=%g pct (Rothermel 1983), beyond the thesis'%K_PCT)
    args.output.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps({'offset_range_pct':[float(offset.min()),float(offset.max())],'output':str(args.output)}))

if __name__=='__main__':main()
