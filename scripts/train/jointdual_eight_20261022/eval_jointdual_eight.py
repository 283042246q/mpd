import json,collections,gc,time
from pathlib import Path
import numpy as np
import torch
from mpd.joint_training.trainer import load_joint_checkpoint
from mpd.datasets.warehouse_joint_cache import CachedWarehouseJointDataset
from mpd.datasets.warehouse_joint_dataset import collate_joint

torch.set_num_threads(4)
ROOT=Path('/home/eric/Projects/MotionPlanningDiffusion/mpd');OUT=Path('/tmp/jointdual_eight_20261022')
a=json.loads((OUT/'analysis.json').read_text());result={'protocol':'Best-total EMA checkpoint; full validation, CPU, batch64; fixed CPU noise seed20261022+917 shared within dataset; unweighted and family-balanced summaries; NULL ablation shares exact inputs/noise. CPU noise differs from training CUDA validation, so values are not exact log reproductions.','runs':{}}
for name,run in a['runs'].items():
 start=time.monotonic();cfg=run['config'];rep=cfg['representation'];isworld=cfg['variant']=='jd_world';step=run['best']['loss']['step']
 ds=CachedWarehouseJointDataset(cfg['snapshot'],cfg['tensor_cache_path'],split='validation',representation=rep,include_world=True)
 famcounts=collections.Counter(e['family'] for e in ds.entries)
 w=np.array([1/famcounts[e['family']] for e in ds.entries]);counts=[];ks=[];losses={mode:[] for mode in (['known','null'] if isworld else ['kin'])}
 model,payload=load_joint_checkpoint(ROOT/'data_trained_models/jointdual'/name/('step-%08d.pt'%step));del payload
 gen=torch.Generator().manual_seed(20261022+917)
 with torch.no_grad():
  for i in range(0,len(ds),64):
   b=collate_joint([ds[j] for j in range(i,min(i+64,len(ds)))]);p,z=b['p'],b['z'];k=torch.randint(model.num_diffusion_steps,(len(p),),generator=gen)
   ep=torch.randn(p.shape,generator=gen);ez=torch.randn(z.shape,generator=gen);al=model.alphas_cumprod[k]
   pk=al[:,None,None].sqrt()*p+(1-al[:,None,None]).sqrt()*ep;zk=al[:,None].sqrt()*z+(1-al[:,None]).sqrt()*ez
   counts.extend((b['world_mask'].sum(-1)//32).tolist());ks.extend(k.tolist())
   for mode in losses:
    state=torch.full_like(b['world_state'],2) if mode=='null' else b['world_state']
    pp,pz=model.denoiser(pk,zk,k,b['task'],b['world_features'],b['world_mask'],state)
    lp=(pp-ep).square().mean((1,2));lz=(pz-ez).square().mean(1)
    losses[mode].append(torch.stack([lp,lz,(pz[:,0]-ez[:,0]).square(),(pz[:,1:]-ez[:,1:]).square().mean(1)],1).numpy())
 counts=np.array(counts);ks=np.array(ks);vals={m:np.concatenate(v,axis=0) for m,v in losses.items()}
 def stat(v,mask=None,weights=None):
  if mask is None:mask=np.ones(len(v),bool)
  avg=np.average(v[mask].astype('float64'),axis=0,weights=weights[mask] if weights is not None else None)
  keys=['p','z','tau','r'] if rep=='tau_r' else ['p','z']
  return dict(n=int(mask.sum()),loss=float(avg[0]+avg[1]),**{k:float(avg[j]) for j,k in enumerate(keys)})
 s={mode:dict(all=stat(v),family_balanced=stat(v,weights=w),dynamic_only=stat(v,counts>0),by_obstacles={int(c):stat(v,counts==c) for c in np.unique(counts)},by_noise_step={str(j*20)+'-'+str(j*20+19):stat(v,ks//20==j) for j in range(5)}) for mode,v in vals.items()}
 if isworld:
  delta=vals['null']-vals['known'];s['null_minus_known']=dict(all=stat(delta),by_obstacles={int(c):stat(delta,counts==c) for c in np.unique(counts)},by_noise_step={str(j*20)+'-'+str(j*20+19):stat(delta,ks//20==j) for j in range(5)})
 result['runs'][name]={'step':step,**s}
 (OUT/'stratified_validation.json').write_text(json.dumps(result,indent=2))
 print(name,step,json.dumps({m:{k:v for k,v in s[m].items() if k in ('all','dynamic_only','family_balanced')} for m in s}),round(time.monotonic()-start,1),flush=True)
 ds.close();del model,ds,vals;gc.collect()
