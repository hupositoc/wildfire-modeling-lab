"""Render actual run logs; never draw invented score or accuracy claims."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results/implementation-benchmark'
ASSETS=ROOT/'docs/assets'
ASSETS.mkdir(parents=True,exist_ok=True)
items=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(RESULTS.glob('result*.json'))]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':16,'axes.labelsize':12,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#8b95a1','text.color':'#191f28','axes.labelcolor':'#4e5968','xtick.color':'#4e5968','ytick.color':'#4e5968','figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
fig,axes=plt.subplots(1,2,figsize=(13.5,4.8),constrained_layout=True)
for ax,mode in zip(axes,['literal','raw']):
    runs=[r for r in items if r['config']['ndvi_mode']==mode]
    curves=np.array([[e['return'] for e in r['training_trace']] for r in runs])
    window=15;kernel=np.ones(window)/window
    smoothed=np.array([np.convolve(c,kernel,mode='valid') for c in curves])
    x=np.arange(window-1,curves.shape[1])
    for y in smoothed:ax.plot(x,y,color='#1b64da',alpha=.23,lw=1)
    mean=smoothed.mean(0);std=smoothed.std(0,ddof=1)
    ax.plot(x,mean,color='#1b64da',lw=2.5,label='Mean of 3 training seeds')
    ax.fill_between(x,mean-std,mean+std,color='#1b64da',alpha=.12,label='+/- 1 SD across seeds')
    ax.set_title('Printed NDVI equation' if mode=='literal' else 'Raw-NDVI interpretation')
    ax.set_xlabel('Training episode (15-episode moving average)');ax.set_ylabel('Undiscounted environment return')
    ax.grid(alpha=.16);ax.legend(fontsize=9)
fig.suptitle('DQN implementation check on synthetic environments',fontsize=19,fontweight='bold')
fig.savefig(ASSETS/'training-returns.png',dpi=180);fig.savefig(ASSETS/'training-returns.pdf');plt.close(fig)

summary=[]
for mode in ['literal','raw']:
    for policy in ['dqn','random','reward-greedy']:
        values=[e['return'] for r in items if r['config']['ndvi_mode']==mode for e in r['evaluation'] if e['policy']==policy]
        summary.append({'ndvi_mode':mode,'policy':policy,'mean_return':float(np.mean(values)),'sd_return':float(np.std(values,ddof=1)),'evaluation_pairs':len(values)})
fig,axes=plt.subplots(1,2,figsize=(13.5,4.8),constrained_layout=True)
for ax,mode in zip(axes,['literal','raw']):
    rows=[r for r in summary if r['ndvi_mode']==mode]
    bars=ax.bar(['DQN','Random','One-step reward'],[r['mean_return'] for r in rows],color=['#1b64da','#4e5968','#028450'],width=.6)
    for bar,hatch in zip(bars,['','///','...']):bar.set_hatch(hatch)
    ax.errorbar(range(3),[r['mean_return'] for r in rows],yerr=[r['sd_return'] for r in rows],fmt='none',ecolor='#191f28',capsize=5)
    ax.axhline(0,color='#8b95a1',lw=.8)
    ax.set_title('Printed NDVI equation' if mode=='literal' else 'Raw-NDVI interpretation')
    ax.set_ylabel('Environment return, mean +/- sample SD')
    ax.text(.02,.98,'3 model seeds x 5 fixed held-out maps\nSD is descriptive, not a confidence interval',transform=ax.transAxes,va='top',fontsize=9)
fig.suptitle('Policy comparison: return is not burned-area accuracy',fontsize=18,fontweight='bold')
fig.savefig(ASSETS/'policy-returns.png',dpi=180);fig.savefig(ASSETS/'policy-returns.pdf');plt.close(fig)

result={'runs':len(items),'total_episodes':sum(r['episodes'] for r in items),'total_updates':sum(r['updates'] for r in items),'training_seconds':sum(r['training_seconds'] for r in items),'summaries':summary,'reported_thesis_iou':.726,'reproduced_real_event_iou':None,'synthetic_is_not_observation':True}
(RESULTS/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
