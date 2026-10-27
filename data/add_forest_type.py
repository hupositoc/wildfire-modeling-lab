"""Per-cell forest-type fractions from the KFS 1:5,000 forest type map (beyond the thesis).

Input: the 2019 edition (pre-fire) for Uljin-gun from the FGIS download service
(map.forest.go.kr, 나무지도(대축척 임상도), 시군구 47930). Each 100 m cell is sampled
on a 10x10 sub-grid of points and the share of points in each class is stored.
FRTP_CD: 1 conifer, 2 broadleaf, 3 mixed, 4 bamboo, 0 non-forest. KOFTR_GROU 11: Pinus densiflora.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.geometry import box

BOUNDS=(129.29,37.03,129.356,37.075);N=50;SUB=10

def sample_points(n=N,sub=SUB):
    w,s,e,nn=BOUNDS;k=n*sub
    lon=w+(np.arange(k)+.5)*(e-w)/k;lat=s+(np.arange(k)+.5)*(nn-s)/k
    xx,yy=np.meshgrid(lon,lat)  # row 0 is the southern edge, matching the y-up grid
    return gpd.GeoDataFrame(geometry=gpd.points_from_xy(xx.ravel(),yy.ravel()),crs=4326)

def fractions(shp):
    aoi=gpd.GeoSeries([box(*BOUNDS)],crs=4326).to_crs(5179)
    poly=gpd.read_file(shp,bbox=tuple(aoi.buffer(200).total_bounds)).to_crs(5179)
    pts=sample_points().to_crs(5179)
    hit=gpd.sjoin(pts,poly[['FRTP_CD','KOFTR_GROU','HEIGHT','DNST_CD','geometry']],how='left',predicate='within')
    hit=hit[~hit.index.duplicated()]
    frtp=hit['FRTP_CD'].fillna('0').astype(int).to_numpy().reshape(N*SUB,N*SUB)
    group=hit['KOFTR_GROU'].fillna('0').astype(int).to_numpy().reshape(N*SUB,N*SUB)
    height=hit['HEIGHT'].fillna('0').astype(float).to_numpy().reshape(N*SUB,N*SUB)
    cell=lambda a:a.reshape(N,SUB,N,SUB).mean((1,3))
    return {'conifer':cell(frtp==1),'broadleaf':cell(frtp==2),'mixed':cell(frtp==3),'nonforest':cell(frtp==0),
            'pine':cell(group==11),'height_m':cell(height)}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--shp',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,help='also write a copy of this dataset with a conifer_fraction field')
    args=parser.parse_args()
    out=fractions(args.shp);np.savez_compressed(str(args.output),**out)
    for k,v in out.items():print('%-10s mean %.3f'%(k,v.mean()))
    if args.dataset:
        with np.load(str(args.dataset)) as d:fields={k:d[k] for k in d.files}
        fields['conifer_fraction']=np.broadcast_to(out['conifer'],fields['ndvi'].shape).astype(np.float32)
        target=args.output.with_name(args.dataset.stem+'_forest.npz');np.savez_compressed(str(target),**fields)
        meta=json.loads(args.dataset.with_suffix('.json').read_text(encoding='utf-8'))
        meta['conifer_fraction']='KFS 1:5,000 forest type map, 2019 edition (pre-fire), FRTP_CD==1 share per cell'
        target.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=1),encoding='utf-8')
        print('wrote',target)

if __name__=='__main__':main()
