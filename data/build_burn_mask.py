"""Observed Uljin 2022 burned-area mask from Sentinel-2 with ISODATA, following thesis section 4.

The burned-class rule is fixed before any model comparison: ISODATA clusters whose mean dNBR
is at least 0.1 (USGS lower bound of low severity) are burned; a grid cell is burned when at
least half of its valid pixels are burned. Nothing here reads model output or a target score.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from rasterio.enums import Resampling
from fetch_open_sources import BOUNDS, CLOUD_CLASSES, N, by_date, cells, mosaic, search_s2

def isodata(x,k=6,iterations=20,min_size=.01,split_sd=.6,merge_dist=.5,seed=0):
    rng=np.random.default_rng(seed)
    centers=x[rng.choice(len(x),k,replace=False)]
    for _ in range(iterations):
        labels=np.argmin(((x[:,None,:]-centers[None])**2).sum(-1),axis=1)
        keep=[c for c in range(len(centers)) if (labels==c).mean()>=min_size]
        centers=np.array([x[labels==c].mean(0) for c in keep])
        labels=np.argmin(((x[:,None,:]-centers[None])**2).sum(-1),axis=1)
        sd=np.array([x[labels==c].std(0) for c in range(len(centers))])
        if len(centers)<2*k:
            split=[]
            for c in range(len(centers)):
                if sd[c].max()>split_sd:
                    d=np.zeros(x.shape[1]);d[sd[c].argmax()]=sd[c].max()/2
                    split+=[centers[c]+d,centers[c]-d]
                else:split.append(centers[c])
            centers=np.array(split)
        if len(centers)>k//2:
            dist=np.linalg.norm(centers[:,None]-centers[None],axis=-1);np.fill_diagonal(dist,np.inf)
            i,j=np.unravel_index(dist.argmin(),dist.shape)
            if dist[i,j]<merge_dist:centers=np.vstack([np.delete(centers,[i,j],0),(centers[i]+centers[j])/2])
    labels=np.argmin(((x[:,None,:]-centers[None])**2).sum(-1),axis=1)
    return labels,centers

def clear_day(days,prefer_last):
    for day in sorted(days,reverse=prefer_last):
        scl=mosaic(days[day],'scl',resampling=Resampling.nearest)
        cloud=float(np.nanmean(np.isin(scl,CLOUD_CLASSES)))
        if cloud<=.05:return day,cloud,scl
    raise RuntimeError('No clear scene found.')

def indices(items):
    b=lambda name:mosaic(items,name)
    red,nir,nir08,swir=b('red'),b('nir'),b('nir08'),b('swir22')
    return (nir-red)/(nir+red),(nir08-swir)/(nir08+swir)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pre',default='2022-01-15/2022-03-03');parser.add_argument('--post',default='2022-03-14/2022-04-30')
    parser.add_argument('--out',type=Path,default=Path('data/uljin_observed.npy'))
    args=parser.parse_args()
    pre_days=by_date(search_s2(*args.pre.split('/')));post_days=by_date(search_s2(*args.post.split('/')))
    pre_day,pre_cloud,pre_scl=clear_day(pre_days,prefer_last=True)
    post_day,post_cloud,post_scl=clear_day(post_days,prefer_last=False)
    ndvi_pre,nbr_pre=indices(pre_days[pre_day]);ndvi_post,nbr_post=indices(post_days[post_day])
    dnbr=nbr_pre-nbr_post
    valid=np.isfinite(dnbr)&np.isfinite(ndvi_post)&~np.isin(post_scl,CLOUD_CLASSES+(6,))&~np.isin(pre_scl,CLOUD_CLASSES+(6,))
    feats=np.stack([nbr_post[valid],ndvi_post[valid],dnbr[valid]],1)
    z=(feats-feats.mean(0))/feats.std(0)
    labels,_=isodata(z)
    burned_class=[c for c in np.unique(labels) if feats[labels==c,2].mean()>=.1]
    burned=np.full(dnbr.shape,np.nan);burned[valid]=np.isin(labels,burned_class)
    fraction=cells(burned)
    mask=np.nan_to_num(fraction)>=.5
    args.out.parent.mkdir(parents=True,exist_ok=True)
    np.save(str(args.out),mask)
    meta={'kind':'observational','event_id':'uljin-2022','version':'reconstruction-v1',
          'source_url':'https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a',
          'mask_method':'ISODATA on standardized [post NBR, post NDVI, dNBR] at ~10 m; clusters with mean dNBR>=0.1 are burned; cell burned if >=50% of valid pixels burned; water/cloud excluded via SCL',
          'time_window':pre_day+' (pre) / '+post_day+' (post)','pre_aoi_cloud':pre_cloud,'post_aoi_cloud':post_cloud,
          'pre_items':[i['id'] for i in pre_days[pre_day]],'post_items':[i['id'] for i in post_days[post_day]],
          'grid_bounds':list(BOUNDS),'grid_origin':'lower-left','shape':[N,N],
          'burned_cells':int(mask.sum()),'clusters':int(len(np.unique(labels))),'burned_clusters':len(burned_class),
          'thesis_difference':'thesis used VIIRS reference [25] and GEE ISODATA with unreported settings; this is an independent reconstruction',
          'mask_sha256':hashlib.sha256(args.out.read_bytes()).hexdigest(),'built_utc':datetime.now(timezone.utc).isoformat()}
    args.out.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps({k:meta[k] for k in ['time_window','burned_cells','clusters','burned_clusters']}))
    for row in mask[::-1][::2]:print(''.join('#' if v else '.' for v in row))

if __name__=='__main__':main()
