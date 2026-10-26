"""Draw observed vs calibrated footprint and the search trace from calibration.json."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--calibration',type=Path,required=True);parser.add_argument('--mask',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=json.loads((args.calibration/'calibration.json').read_text(encoding='utf-8'))
    observed=np.load(str(args.mask)).astype(bool);predicted=np.load(str(args.calibration/'predicted.npy')).astype(bool)
    overlay=np.zeros(observed.shape+(3,))+.95
    overlay[observed&~predicted]=(.87,.2,.07);overlay[predicted&~observed]=(.15,.43,.96);overlay[observed&predicted]=(.12,.13,.14)
    fig,axes=plt.subplots(1,3,figsize=(13,4.4))
    meta=args.mask.with_suffix('.json');source=json.loads(meta.read_text(encoding='utf-8')) if meta.exists() else {}
    label='Thesis Fig. 4.5(a) target' if source.get('kind')=='thesis_reported_target' else 'Observed (Sentinel-2 ISODATA)'
    for ax,image,title in [(axes[0],observed,label),(axes[1],predicted,'Calibrated DQN footprint')]:
        ax.imshow(image,origin='lower',cmap='Greys',vmin=0,vmax=1.4);ax.plot(6,10,'r*',ms=12);ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
    axes[2].imshow(overlay,origin='lower');axes[2].plot(6,10,'r*',ms=12)
    axes[2].set_title('Overlap  JI = %.3f'%result['best_jaccard_recomputed']);axes[2].set_xticks([]);axes[2].set_yticks([])
    fig.text(.5,.02,'black: both   red: observed only   blue: predicted only   star: ignition (7,11)',ha='center')
    fig.tight_layout(rect=(0,.05,1,1));args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(str(args.output/'uljin-footprint.png'),dpi=130);plt.close(fig)
    history=result['history']
    fig,ax=plt.subplots(figsize=(7,3.6))
    colors=['#6d7882' if h['phase']=='simplex-lhs' else '#256ef4' for h in history]
    ax.scatter(range(1,len(history)+1),[h['jaccard'] for h in history],c=colors,s=22)
    ax.plot(range(1,len(history)+1),np.maximum.accumulate([h['jaccard'] for h in history]),color='#1e2124',lw=1)
    ax.axhline(.726,color='#de3412',ls='--',lw=1);ax.text(1,.733,'thesis 0.726',color='#de3412',fontsize=8)
    ax.set_xlabel('evaluation (grey: simplex LHS, blue: Bayesian optimisation)');ax.set_ylabel('Jaccard index');ax.set_ylim(0,1)
    fig.tight_layout();fig.savefig(str(args.output/'uljin-search.png'),dpi=130);plt.close(fig)

if __name__=='__main__':main()
