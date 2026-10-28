import json, collections, math
from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ROOT=Path('/home/eric/Projects/MotionPlanningDiffusion/mpd')
OUT=Path('/tmp/jointdual_eight_20261022'); OUT.mkdir(exist_ok=True)
NAMES=['jdkin_c','jdkin_tau_r','jdworld_c','jdworld_tau_r_1','jdkin_c_multi','jdkin_tau_r_multi','jdworld_c_multi','jdworld_tau_r_multi']
result={'runs':{},'datasets':{}}
for name in NAMES:
 d=ROOT/'data_trained_models/jointdual'/name
 cfg=json.loads((d/'config.json').read_text())
 ea=EventAccumulator(str(d/'tensorboard'),size_guidance={'scalars':0}); ea.Reload()
 tags=ea.Tags()['scalars']
 vals={}
 for tag in tags:
  if tag.startswith('validation/'):
   key=tag.split('/',1)[1]
   for e in ea.Scalars(tag):
    if e.step<=200000: vals.setdefault(e.step,{'step':e.step})[key]=e.value
 rows=[v for k,v in sorted(vals.items()) if 'loss' in v]
 best={k:min(rows,key=lambda v:v[k]) for k in ('loss','p','z','tau','r') if k in rows[0]}
 pmin=best['p']['p'];zmin=best['z']['z']
 compromise=min(rows,key=lambda v:max(v['p']/pmin-1,v['z']/zmin-1))
 pareto=[v for v in rows if not any(q['p']<=v['p'] and q['z']<=v['z'] and (q['p']<v['p'] or q['z']<v['z']) for q in rows)]
 tail={};bad=0; jsonvals={};count=0;seen=set();dupes=0
 with (d/'metrics.jsonl').open() as f:
  for line in f:
   try:r=json.loads(line)
   except (ValueError,TypeError):bad+=1;continue
   step=r.get('step',0)
   if step>200000:continue
   count+=1
   if step in seen:dupes+=1
   seen.add(step)
   if step>180000:tail[step]=r
   if 'validation_loss' in r:jsonvals[step]=r
 tailmean={k:float(np.mean([r[k] for r in tail.values() if r['step']>195000 and k in r])) for k in ('loss','p','z','tau','r','grad_norm') if k in next(iter(tail.values()))}
 tailmean['clip_fraction']=float(np.mean([r['grad_norm']>cfg['training']['gradient_clip_norm'] for r in tail.values() if r['step']>195000]))
 last=rows[-1]
 valtail=[r for r in rows if r['step']>180000]
 slopes={k:float(np.polyfit([r['step']/10000 for r in valtail],[r[k] for r in valtail],1)[0]) for k in ('loss','p','z')}
 diff=max(abs(r[k]-jsonvals[r['step']]['validation_'+k]) for r in rows if r['step'] in jsonvals for k in ('loss','p','z'))
 run=dict(config=cfg,validation=rows,best=best,compromise=compromise,pareto=pareto,final=last,tail5k=tailmean,tail20k_slopes_per10k=slopes,duplicates=dupes,invalid_lines=bad,records=count,tb_json_maxdiff=diff)
 result['runs'][name]=run
 print(name,json.dumps({k:run[k] for k in ('best','compromise','final','tail5k','tail20k_slopes_per10k','duplicates','invalid_lines','tb_json_maxdiff')}),flush=True)
fams={}
for label in ['main','multiworld']:
 snap=json.loads((ROOT/('data_trajectories_spacetime/jointdual-'+label+'-snapshot.json')).read_text())
 entries=snap['entries']; groups=collections.defaultdict(collections.Counter)
 for e in entries:groups[e['split']][e['family']]+=1
 fams[label]=groups
 ds={k:snap[k] for k in ('identity','split_counts','normalization','sources')}
 ds['family_stats']={s:dict(families=len(counts),samples=sum(counts.values()),samples_per_family=dict(min=min(counts.values()),max=max(counts.values()),mean=np.mean(list(counts.values())),hist=dict(collections.Counter(counts.values()))),expected_draws_per_family=51200000/len(counts)) for s,counts in groups.items()}
 ds['within_split_family_overlap']={a+'_'+b:len(set(groups[a])&set(groups[b])) for a,b in [('train','validation'),('train','test'),('validation','test')]}
 for split,stats in ds['family_stats'].items():
  if split!='train':stats.pop('expected_draws_per_family',None)
 cfg=result['runs']['jdkin_c' if label=='main' else 'jdkin_c_multi']['config']
 cache=Path(cfg['tensor_cache_path']);offsets=np.load(cache/'world_offsets.npy',mmap_mode='r');state=np.load(cache/'world_state.npy',mmap_mode='r')
 obstacle_counts=np.where(state==1,0,np.diff(offsets)//32)
 ds['obstacle_counts']={split:dict(collections.Counter(int(obstacle_counts[i]) for i,e in enumerate(entries) if e['split']==split)) for split in groups}
 result['datasets'][label]=ds
 print('DATASET',label,json.dumps(ds),flush=True)
result['cross_dataset_family_overlap']={a+'_'+b:len(set(fams['main'][a])&set(fams['multiworld'][b])) for a in fams['main'] for b in fams['multiworld']}
print('CROSS',result['cross_dataset_family_overlap'],flush=True)
(OUT/'analysis.json').write_text(json.dumps(result,indent=2))
