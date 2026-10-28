import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
O=Path('/tmp/jointdual_eight_20261022');r=json.loads((O/'analysis.json').read_text());g=json.loads((O/'gradients.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2})
fig,axes=plt.subplots(4,2,figsize=(12,12),sharex=True)
for i,(ds,rep,names) in enumerate([('Main','c',['jdkin_c','jdworld_c']),('Main','tau_r',['jdkin_tau_r','jdworld_tau_r_1']),('Multiworld','c',['jdkin_c_multi','jdworld_c_multi']),('Multiworld','tau_r',['jdkin_tau_r_multi','jdworld_tau_r_multi'])]):
 for j,key in enumerate(['p','z']):
  ax=axes[i,j]
  for name,color,label in zip(names,['#2468b4','#dc7624'],['KIN','WORLD']):
   run=r['runs'][name];v=[v for v in run['validation'] if v['step']>=20000]
   ax.plot([v['step']/1000 for v in v],[v[key] for v in v],color=color,label=label,lw=1.8)
   best=run['best'][key];ax.plot(best['step']/1000,best[key],'o',color=color,ms=5)
  ax.set_title(f'{ds} / {rep}: '+('spatial noise MSE' if key=='p' else 'timing noise MSE'))
  ax.set_xlim(20,200);ax.legend(loc='best',frameon=False)
  if i==3:ax.set_xlabel('Optimizer steps (thousands)')
fig.suptitle('JointDual: full-validation EMA losses through 200k steps\nDots mark each component minimum. Different datasets / timing coordinates are not directly rankable.',fontsize=13)
fig.tight_layout(rect=(0,0,1,.95));fig.savefig(O/'validation_curves.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4.6));labels=['KIN c','KIN tau_r','WORLD c','WORLD tau_r'];x=np.arange(4)
for ax,ds,names in zip(axes,['Main','Multiworld'],[['jdkin_c','jdkin_tau_r','jdworld_c','jdworld_tau_r_1'],['jdkin_c_multi','jdkin_tau_r_multi','jdworld_c_multi','jdworld_tau_r_multi']]):
 for shift,stage,color in [(-.18,'best','#2468b4'),(.18,'final','#dc7624')]:
  values=[g['results'][n+'_'+stage]['groups']['all_shared']['median_ratio'] for n in names]
  bars=ax.bar(x+shift,values,.34,color=color,label='Best total validation' if stage=='best' else '200k')
  ax.bar_label(bars,fmt='%.2f',fontsize=9)
 ax.set_xticks(x,labels);ax.set_title(ds);ax.set_ylim(0,13);ax.set_ylabel('Median ||grad Lz|| / ||grad Lp||');ax.legend(frameon=False)
fig.suptitle('Shared-parameter raw gradients: EMA eval, 8 effective batches of 256\nSame family-balanced samples and noise across checkpoints within each dataset',fontsize=12)
fig.tight_layout(rect=(0,0,1,.88));fig.savefig(O/'gradient_ratios.png',dpi=170);plt.close(fig)
