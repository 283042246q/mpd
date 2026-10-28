import json, math, gc, time
from pathlib import Path
import numpy as np
import torch
from mpd.joint_training.trainer import load_joint_checkpoint, JointBatchSampler
from mpd.datasets.warehouse_joint_cache import CachedWarehouseJointDataset
from mpd.datasets.warehouse_joint_dataset import collate_joint

torch.set_num_threads(4)
ROOT=Path('/home/eric/Projects/MotionPlanningDiffusion/mpd')
OUT=Path('/tmp/jointdual_eight_20261022')
a=json.loads((OUT/'analysis.json').read_text())
GROUPS={'spatial_encoder':{'input','encoder','down'},'timing_and_task':{'time','task','timing'},'fusion':{'fusion'},'world':{'world_encoder','spatial_attention','timing_attention'}}
NB=8; MICRO=64; ACC=4; SEED=20261022
result={'protocol':{'weights':'EMA','mode':'eval (no world condition dropout)','micro_batch':MICRO,'accumulate':ACC,'effective_batch':MICRO*ACC,'batches':NB,'examples_per_checkpoint':NB*MICRO*ACC,'seed':SEED,'split':'family-balanced train','note':'Raw pre-clipping gradients on shared parameters, not AdamW-preconditioned updates.'},'results':{},'world_ablation':{}}
for name,run in a['runs'].items():
 cfg=run['config']; rep=cfg['representation'];isworld=cfg['variant']=='jd_world'
 ds=CachedWarehouseJointDataset(cfg['snapshot'],cfg['tensor_cache_path'],split='train',representation=rep,include_world=isworld)
 batches=[collate_joint([ds[i] for i in inds]) for inds in JointBatchSampler(ds.entries,MICRO,SEED,1000000,1000000+NB*ACC)]
 for stage,step in [('best',run['best']['loss']['step']),('final',200000)]:
  start=time.monotonic();path=ROOT/'data_trained_models/jointdual'/name/('step-%08d.pt'%step)
  model,payload=load_joint_checkpoint(path); assert payload['dataset_snapshot_hash']==ds.snapshot['identity'];assert payload['normalization']==ds.normalization
  model.requires_grad_(True).eval(); del payload
  names=[];params=[];gpmap={g:[] for g in GROUPS}
  for n,p in model.named_parameters():
   group=next((g for g,v in GROUPS.items() if n.split('.')[1] in v),None)
   if group:
    gpmap[group].append(len(params));names.append(n);params.append(p)
  gpmap={g:idx for g,idx in gpmap.items() if idx};gpmap['all_shared']=list(range(len(params)))
  gen=torch.Generator().manual_seed(SEED+917); measures={g:[] for g in gpmap};losses=[]
  for j in range(NB):
   sums=[[torch.zeros_like(p) for p in params] for _ in range(2)]
   lp_total=lz_total=0.
   for batch in batches[j*ACC:(j+1)*ACC]:
    p,z=batch['p'],batch['z'];k=torch.randint(model.num_diffusion_steps,(len(p),),generator=gen)
    ep=torch.randn(p.shape,generator=gen);ez=torch.randn(z.shape,generator=gen);al=model.alphas_cumprod[k]
    pp,pz=model.denoiser(al[:,None,None].sqrt()*p+(1-al[:,None,None]).sqrt()*ep,al[:,None].sqrt()*z+(1-al[:,None]).sqrt()*ez,k,batch['task'],batch.get('world_features'),batch.get('world_mask'),batch.get('world_state'))
    lp=(pp-ep).square().mean();lz=(pz-ez).square().mean();lp_total+=float(lp)/ACC;lz_total+=float(lz)/ACC
    for t,loss in enumerate((lp,lz)):
     grads=torch.autograd.grad(loss/ACC,params,retain_graph=t==0,allow_unused=True)
     with torch.no_grad():
      for v,grad in zip(sums[t],grads):
       if grad is not None:v.add_(grad)
    del grads,lp,lz,pp,pz
   losses.append({'p':lp_total,'z':lz_total})
   for g,inds in gpmap.items():
    p2=sum(float(sums[0][i].square().sum()) for i in inds);z2=sum(float(sums[1][i].square().sum()) for i in inds)
    dot=sum(float((sums[0][i]*sums[1][i]).sum()) for i in inds)
    measures[g].append(dict(p_norm=math.sqrt(p2),z_norm=math.sqrt(z2),ratio=math.sqrt(z2/max(p2,1e-30)),cosine=dot/max(math.sqrt(p2*z2),1e-30)))
   del sums
  summary={g:dict(median_ratio=float(np.median([r['ratio'] for r in rows])),mean_cosine=float(np.mean([r['cosine'] for r in rows])),min_cosine=min(r['cosine'] for r in rows),max_cosine=max(r['cosine'] for r in rows),negative_batches=sum(r['cosine']<0 for r in rows),strong_negative_batches=sum(r['cosine']<-.2 for r in rows),raw=rows) for g,rows in measures.items()}
  result['results'][name+'_'+stage]=dict(step=step,groups=summary,losses=losses)
  (OUT/'gradients.json').write_text(json.dumps(result,indent=2))
  print(name,stage,step,json.dumps({g:{k:v for k,v in vals.items() if k!='raw'} for g,vals in summary.items()}),'seconds',round(time.monotonic()-start,1),flush=True)
  del model,params;gc.collect()
 ds.close();del batches,ds;gc.collect()
