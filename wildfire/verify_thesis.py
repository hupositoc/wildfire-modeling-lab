"""Evaluate a frozen model against a separately supplied observed event mask."""
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .experiment import QNetwork
from .model import Config,Dataset
from .rollout import spread,evaluate_observed

def main():
    parser=argparse.ArgumentParser()
    for name in ['dataset','mask','mask-provenance','checkpoint','training-record','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    data=Dataset.load(args.dataset)
    if data.metadata['kind']!='observational':parser.error('A synthetic input cannot establish real-event reproduction.')
    provenance=json.loads(args.mask_provenance.read_text(encoding='utf-8'))
    record=json.loads(args.training_record.read_text(encoding='utf-8'))
    if record.get('kind')!='observational_reconstruction_training':parser.error('Synthetic implementation checkpoints cannot be presented as real-event thesis reconstructions.')
    if data.metadata.get('grid_origin')!='lower-left' or provenance.get('grid_origin')!='lower-left':parser.error('Expected explicit lower-left grid origin for both input and mask.')
    if data.metadata.get('grid_bounds')!=provenance.get('grid_bounds'):parser.error('Input and mask grid bounds differ.')
    digest=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if digest!=record['weights_sha256']:parser.error('Checkpoint hash differs from the training record.')
    cfg=Config(**{k:v for k,v in record['config'].items() if k in {f.name for f in fields(Config)}})
    model=QNetwork();model.load_state_dict(torch.load(str(args.checkpoint),map_location='cpu',weights_only=True));model.eval()
    predicted=spread(data,cfg,network=model,seed=record['seed'])[-1]
    target=np.load(str(args.mask),allow_pickle=False)
    if target.dtype!=np.bool_ and not np.isin(target,[0,1]).all():parser.error('Mask must be boolean or binary.')
    result=evaluate_observed(predicted,target,provenance)
    result.update({'reported_score':.726,'difference_from_reported':result['iou']-.726,'checkpoint_sha256':digest,
                   'dataset_sha256':hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                   'mask_sha256':hashlib.sha256(args.mask.read_bytes()).hexdigest(),
                   'interpretation':'reconstruction score under documented assumptions, not proof of exact original-code reproduction',
                   'assumptions':record['assumptions']})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    np.save(str(args.output.with_suffix('.predicted.npy')),predicted)
    print(json.dumps({'iou':result['iou'],'difference_from_reported':result['difference_from_reported']}))

if __name__=='__main__':main()
