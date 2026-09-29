#!/usr/bin/env python3
"""Run diagnostic precision/backend/sorting ablations after baseline validation."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'artifacts/experiments/so400m_hard_negative_lora_20260923T203613Z'
OLD=ROOT/'artifacts/experiments/encoder_bakeoff_20260920T143928Z/models/siglip2_so400m_384'
MODEL_ID='google/siglip2-so400m-patch14-384'; REV='e8e487298228002f3d8a82e0cd5c8ea9c567f57f'
BENCHES=['hard_near_duplicate_dev_v2','generated_stress_dev_pilot32','synthetic_dev']

def rows(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def write(path,rs):
 with Path(path).open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]) if rs else []);w.writeheader();w.writerows(rs)
def rank(scores,slugs): return sorted(range(len(slugs)),key=lambda i:(-float(scores[i]),slugs[i]))
def encode(model,processor,path,device,mode):
 with Image.open(path) as im: image=im.convert('RGB')
 px=processor(images=[image],return_tensors='pt')['pixel_values'].to(device=device,dtype=next(model.parameters()).dtype)
 ctx=torch.autocast(device_type='cuda',dtype=torch.float16,enabled=(mode=='fp16_autocast' and device.type=='cuda'))
 with torch.inference_mode(),ctx:
  x=model.get_image_features(pixel_values=px)
  if hasattr(x,'pooler_output'):x=x.pooler_output
  projection=getattr(model,'visual_projection',None)
  if projection is not None and x.shape[-1]==projection.in_features:x=projection(x)
  x=torch.nn.functional.normalize(x.float(),p=2,dim=-1,eps=1e-12)[0]
 return x.cpu().numpy()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-dir',default='artifacts/experiments/frozen_baseline_forensics_20260924T044708Z');args=ap.parse_args()
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 out=(ROOT/args.run_dir).resolve()
 sys.path.insert(0,str(ROOT/'src'))
 from recognition.frozen_baseline_forensics import require_valid_baseline
 base=json.loads((out/'baseline_reproduction.json').read_text())
 require_valid_baseline(base['after_single_query_fix'])
 ab=json.loads((Path('/tmp/forensic_batch_ablation.json')).read_text())
 selected={(r['dataset'],r['query_id']):r for r in ab}
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 processor=AutoImageProcessor.from_pretrained(MODEL_ID,revision=REV,local_files_only=True)
 cache=torch.load(ROOT/'artifacts/reference_embeddings/siglip2_so400m_384/embeddings.pt',map_location='cpu',weights_only=True)
 ref=torch.nn.functional.normalize(cache['embeddings'].float(),p=2,dim=1).to(device)
 slugs=json.loads((ROOT/'artifacts/reference_embeddings/siglip2_so400m_384/slugs.json').read_text())
 slug_i={s:i for i,s in enumerate(slugs)}
 precision_rows=[]; embeddings_by_mode={}
 modes=[('fp16_no_autocast',torch.float16,None),('fp16_autocast',torch.float16,None),('fp32',torch.float32,None)]
 if device.type=='cuda' and torch.cuda.is_bf16_supported():modes.append(('bf16',torch.bfloat16,None))
 for name,dtype,_ in modes:
  print('precision mode',name,flush=True)
  try:
   model=AutoModel.from_pretrained(MODEL_ID,revision=REV,dtype=dtype,local_files_only=True).to(device).eval()
   qvec={}
   for (bench,qid),_probe in selected.items():
    pred=next(r for r in rows(OLD/bench/'predictions.csv') if r['query_id']==qid)
    qpath=next(r['query_path'] for r in rows(ROOT/'data/benchmarks'/bench/'manifest.csv') if r['query_id']==qid)
    v=encode(model,processor,ROOT/qpath,device,name);qvec[(bench,qid)]=v
    sims=(torch.as_tensor(v,dtype=torch.float32,device=device)@ref.T).cpu().numpy()
    order=rank(sims,slugs); old_top=json.loads(pred['top5_slugs']);new_top=[slugs[i] for i in order[:5]]
    old_scores=[float(x) for x in json.loads(pred['top5_scores'])]
    delta=[abs(float(sims[slug_i[s]])-old_scores[k]) for k,s in enumerate(old_top)]
    baseq=np.load(out/'query_embeddings'/f'{bench}.npy',mmap_mode='r')
    allids=[x['query_id'] for x in rows(OLD/bench/'predictions.csv')]; qrow=allids.index(qid); basev=np.asarray(baseq[qrow])
    precision_rows.append({'mode':name,'dataset':bench,'query_id':qid,'role':'mismatch' if _probe['is_baseline_mismatch'] else 'control',
      'batch_size':1,'top5_exact_old':old_top==new_top,'top5_set_same_old':set(old_top)==set(new_top),'top1_changed_old':old_top[0]!=new_top[0],
      'old_top5_candidate_max_score_abs_diff':max(delta),'top1_top2_margin':float(sims[order[0]]-sims[order[1]]),
      'historical_rank5_minus_rank4':old_scores[3]-old_scores[4],'query_embedding_cosine_vs_baseline_fp16':float(np.dot(v,basev)/(np.linalg.norm(v)*np.linalg.norm(basev))),
      'embedding_max_abs_diff_vs_baseline_fp16':float(np.max(np.abs(v-basev)))})
   embeddings_by_mode[name]=qvec
   del model;torch.cuda.empty_cache() if device.type=='cuda' else None
  except Exception as exc:
   precision_rows.append({'mode':name,'dataset':'ALL','query_id':'','role':'','batch_size':1,'top5_exact_old':'not_run','top5_set_same_old':'','top1_changed_old':'','old_top5_candidate_max_score_abs_diff':'','top1_top2_margin':'','historical_rank5_minus_rank4':'','query_embedding_cosine_vs_baseline_fp16':'','embedding_max_abs_diff_vs_baseline_fp16':f'ERROR: {type(exc).__name__}: {exc}'})
 write(out/'precision_ablation.csv',precision_rows)
 # Compare eager attention to the default SDPA backend on the same diagnostic set.
 attention_rows=[]
 try:
  model=AutoModel.from_pretrained(MODEL_ID,revision=REV,dtype=torch.float16,attn_implementation='eager',local_files_only=True).to(device).eval()
  for (bench,qid),probe in selected.items():
   pred=next(r for r in rows(OLD/bench/'predictions.csv') if r['query_id']==qid)
   qpath=next(r['query_path'] for r in rows(ROOT/'data/benchmarks'/bench/'manifest.csv') if r['query_id']==qid)
   v=encode(model,processor,ROOT/qpath,device,'fp16_no_autocast')
   sims=(torch.as_tensor(v,dtype=torch.float32,device=device)@ref.T).cpu().numpy();order=rank(sims,slugs)
   old_top=json.loads(pred['top5_slugs']);new_top=[slugs[i] for i in order[:5]]
   sdpa=embeddings_by_mode.get('fp16_no_autocast',{}).get((bench,qid))
   attention_rows.append({'historical_backend':'unknown','current_backend':'sdpa','ablation_backend':'eager','dataset':bench,'query_id':qid,
    'role':'mismatch' if probe['is_baseline_mismatch'] else 'control','old_top5_exact':old_top==new_top,'old_top5_set_same':set(old_top)==set(new_top),
    'cosine_eager_vs_sdpa_fp16':float(np.dot(v,sdpa)/(np.linalg.norm(v)*np.linalg.norm(sdpa))) if sdpa is not None else '',
    'max_abs_embedding_diff_eager_vs_sdpa_fp16':float(np.max(np.abs(v-sdpa))) if sdpa is not None else ''})
  del model;torch.cuda.empty_cache() if device.type=='cuda' else None
 except Exception as exc:
  attention_rows.append({'historical_backend':'unknown','current_backend':'sdpa','ablation_backend':'eager','dataset':'ALL','query_id':'','role':'','old_top5_exact':'not_run','old_top5_set_same':'','cosine_eager_vs_sdpa_fp16':'','max_abs_embedding_diff_eager_vs_sdpa_fp16':f'ERROR: {type(exc).__name__}: {exc}'})
 write(out/'attention_backend_ablation.csv',attention_rows)
 # Sorting implementation comparison on every pre-fix score vector.
 sort_rows=[]
 for bench in BENCHES:
  q=np.load(RUN/'benchmarks'/bench/'query_embeddings/frozen_base.npy',mmap_mode='r')
  current=np.load(RUN/'benchmarks'/bench/'frozen_full_ranking_indices.npy',mmap_mode='r')
  qslugs=json.loads((RUN/'benchmarks'/bench/'ranking_reference_slugs.json').read_text())
  oldpred=rows(OLD/bench/'predictions.csv'); equal=ties=0
  for i in range(len(oldpred)):
   sims=(torch.as_tensor(np.asarray(q[i],dtype=np.float32),device=device)@ref.T).cpu().numpy()
   by_slug=rank(sims,qslugs)
   stable=np.argsort(-sims,kind='stable').tolist()
   equal+=by_slug[:5]==stable[:5]
   vals,counts=np.unique(sims,return_counts=True);ties+=int((counts>1).any())
  sort_rows.append({'dataset':bench,'queries':len(oldpred),'historical_score_desc_slug_asc_vs_current_stable_index_same_top5':equal,
                    'queries_with_exact_score_ties':ties,'reference_slugs_sorted_lexicographically':qslugs==sorted(qslugs)})
 write(out/'sorting_ablation.csv',sort_rows)
 print(json.dumps({'precision_rows':len(precision_rows),'attention_rows':len(attention_rows),'sorting':sort_rows},indent=2),flush=True)
if __name__=='__main__':main()
