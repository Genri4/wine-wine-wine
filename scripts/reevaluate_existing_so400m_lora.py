#!/usr/bin/env python3
"""Re-evaluate the already frozen epoch-5 LoRA only after canonical baseline passes."""
from __future__ import annotations
import csv,hashlib,json,os,shutil,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
BASELINE_DIR=ROOT/'artifacts/experiments/frozen_baseline_forensics_20260924T044708Z'
SOURCE_RUN=ROOT/'artifacts/experiments/so400m_hard_negative_lora_20260923T203613Z'
OUT=BASELINE_DIR/'lora_evaluation'
BENCHES=('hard_near_duplicate_dev_v2','generated_stress_dev_pilot32','synthetic_dev')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def jwrite(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True,default=str)+'\n',encoding='utf-8')
def cwrite(path,rows,fields=None):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if fields is None:fields=list(rows[0]) if rows else []
 with path.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def main():
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
 from recognition.frozen_baseline_forensics import require_valid_baseline,canonical_fingerprint
 import run_so400m_hard_negative_lora as runner
 # Hard prerequisite: no model/checkpoint is loaded before all canonical splits pass.
 baseline=json.loads((BASELINE_DIR/'baseline_reproduction.json').read_text())
 require_valid_baseline(baseline['after_single_query_fix'])
 selected=json.loads((SOURCE_RUN/'selected_model.json').read_text())
 if int(selected['best_epoch'])!=5:raise RuntimeError('frozen checkpoint is not the requested epoch 5')
 checkpoint=SOURCE_RUN/'checkpoints/epoch_005.pt'
 checkpoint_file_sha=sha(checkpoint)
 expected_lora_sha=selected['lora_sha256']
 OUT.mkdir(parents=True,exist_ok=True)
 shutil.copy2(SOURCE_RUN/'embedding_margin_analysis.csv',OUT/'embedding_margin_analysis.csv')
 (OUT/'selected_model.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 if device.type!='cuda':raise RuntimeError('epoch-5 controlled inference requires CUDA')
 catalog,slugs,base_refs,base_meta,image_hashes=runner.load_catalog(ROOT)
 model,processor,lora_modules,_=runner.load_model(device)
 payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
 runner.load_lora_state_dict(model,payload['lora_state'])
 actual_lora_sha=runner.lora_sha256(model)
 if actual_lora_sha!=expected_lora_sha:raise RuntimeError('epoch-5 LoRA SHA-256 mismatch; refusing evaluation')
 runner.set_lora_enabled(model,True);model.eval();model.vision_model.eval()
 checkpoint_validation={"selected_epoch":5,"checkpoint_path":str(checkpoint.relative_to(ROOT)),"checkpoint_file_sha256":checkpoint_file_sha,
                        "expected_lora_state_sha256":expected_lora_sha,"loaded_lora_state_sha256":actual_lora_sha,"sha_match":True,
                        "trainable_parameter_count":sum(p.numel() for p in model.parameters() if p.requires_grad),"inference_only":True}
 jwrite(OUT/'checkpoint_validation.json',checkpoint_validation)
 # Include all cache identity inputs requested by the milestone.
 processor_obj=getattr(processor,'image_processor',processor)
 preprocess={"class":type(processor_obj).__name__,"size":getattr(processor_obj,'size',None),"resample":getattr(processor_obj,'resample',None),
             "image_mean":getattr(processor_obj,'image_mean',None),"image_std":getattr(processor_obj,'image_std',None),"do_center_crop":getattr(processor_obj,'do_center_crop',None)}
 code_paths=[Path(__file__),ROOT/'scripts/run_so400m_hard_negative_lora.py',ROOT/'src/recognition/so400m_lora.py',ROOT/'src/recognition/ocr_reranker.py',ROOT/'src/recognition/geometric_reranker.py']
 code_revision=canonical_fingerprint({str(p.relative_to(ROOT)):sha(p) for p in code_paths})
 base_weights=json.loads((BASELINE_DIR/'model_fingerprint_diff.json').read_text())['weights_current_sha256_by_file']
 base_hash=canonical_fingerprint({"revision":runner.BASE_REVISION,"weight_files":base_weights})
 cache_fingerprint=canonical_fingerprint({"base_model_hash":base_hash,"lora_checkpoint_hash":actual_lora_sha,
    "preprocess_fingerprint":canonical_fingerprint(preprocess),"dtype":str(next(model.vision_model.parameters()).dtype),
    "code_revision":code_revision,"references":sorted((slug,image_hashes[slug]) for slug in slugs),"batch_size":1})
 refdir=OUT/'adapted_reference_cache';refdir.mkdir(parents=True,exist_ok=True)
 refpath=refdir/'embeddings.npy';progress=refdir/'progress.json';expected_shape=(len(slugs),runner.HIDDEN_SIZE);done=0
 if refpath.exists() and progress.exists():
  p=json.loads(progress.read_text())
  if p.get('fingerprint')==cache_fingerprint and p.get('completed_rows')==len(slugs) and np.load(refpath,mmap_mode='r').shape==expected_shape:done=len(slugs)
 if not done:
  vectors=np.lib.format.open_memmap(refpath,mode='w+',dtype=np.float32,shape=expected_shape);vectors.flush()
 else:vectors=np.load(refpath,mmap_mode='r+')
 by_slug={r['slug']:r for r in catalog}
 for i in range(done,len(slugs)):
  item=by_slug[slugs[i]]
  with Image.open(ROOT/item['reference_image_path']) as image:
   pixels=runner._processor_pixels(processor,[image.convert('RGB')],model,device)
  with torch.inference_mode():v=runner._features(model,pixels,device)
  vectors[i]=v[0].detach().cpu().numpy()
  if (i+1)%64==0 or i+1==len(slugs):
   vectors.flush();jwrite(progress,{"completed_rows":i+1,"row_count":len(slugs),"fingerprint":cache_fingerprint,"batch_size":1,"updated_at_utc":datetime.now(timezone.utc).isoformat()})
  if (i+1)%256==0 or i+1==len(slugs):print(f'[lora references] {i+1}/{len(slugs)}',flush=True)
 vectors.flush();adapted_refs=np.asarray(vectors,dtype=np.float32).copy();del vectors
 if not np.allclose(np.linalg.norm(adapted_refs,axis=1),1.0,atol=2e-4):raise RuntimeError('adapted reference cache contains non-normalized vectors')
 jwrite(refdir/'metadata.json',{"base_model_id":runner.BASE_MODEL_ID,"base_revision":runner.BASE_REVISION,"base_model_hash":base_hash,
    "lora_checkpoint_sha256":actual_lora_sha,"preprocess":preprocess,"preprocess_fingerprint":canonical_fingerprint(preprocess),
    "dtype":str(next(model.vision_model.parameters()).dtype),"code_revision":code_revision,"batch_size":1,
    "fingerprint":cache_fingerprint,"embedding_dim":runner.HIDDEN_SIZE,"reference_count":len(slugs),"catalog_reference_image_hashes":image_hashes})
 (refdir/'slugs.json').write_text(json.dumps(slugs,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 # Verify exactly the already validated query identities/order. Copy canonical base query vectors and infer LoRA vectors only now.
 records,_=runner.load_external_inputs(ROOT)
 old_ids={b:[r['query_id'] for r in runner.read_csv(ROOT/'artifacts/experiments/encoder_bakeoff_20260920T143928Z/models/siglip2_so400m_384'/b/'predictions.csv')] for b in BENCHES}
 for bench in BENCHES:
  current=records[bench]
  if set(r['query_id'] for r in current)!=set(old_ids[bench]):raise RuntimeError(f'{bench}: query identity mismatch after baseline gate')
  old_rows=runner.read_csv(ROOT/'artifacts/experiments/encoder_bakeoff_20260920T143928Z/models/siglip2_so400m_384'/bench/'predictions.csv')
  oldidx={row['query_id']:i for i,row in enumerate(old_rows)}
  canonical=np.load(BASELINE_DIR/'query_embeddings'/f'{bench}.npy',mmap_mode='r')
  bdir=OUT/'benchmarks'/bench/'query_embeddings';bdir.mkdir(parents=True,exist_ok=True)
  baseq=np.lib.format.open_memmap(bdir/'frozen_base.npy',mode='w+',dtype=np.float32,shape=(len(current),runner.HIDDEN_SIZE))
  for i,row in enumerate(current):baseq[i]=canonical[oldidx[row['query_id']]]
  baseq.flush();del baseq
  fingerprint=runner._query_embedding_fingerprint(ROOT,bench,current,actual_lora_sha)
  pfile=bdir/'progress.json';done=0
  if pfile.exists() and (bdir/'selected_lora.npy').is_file():
   p=json.loads(pfile.read_text())
   if p.get('fingerprint')==fingerprint and p.get('completed_rows')==len(current) and np.load(bdir/'selected_lora.npy',mmap_mode='r').shape==(len(current),runner.HIDDEN_SIZE):done=len(current)
  if not done:loraq=np.lib.format.open_memmap(bdir/'selected_lora.npy',mode='w+',dtype=np.float32,shape=(len(current),runner.HIDDEN_SIZE))
  else:loraq=np.load(bdir/'selected_lora.npy',mmap_mode='r+')
  for i in range(done,len(current)):
   with Image.open(ROOT/str(current[i]['query_path'])) as image:
    pixels=runner._processor_pixels(processor,[image.convert('RGB')],model,device)
   with torch.inference_mode():v=runner._features(model,pixels,device)
   loraq[i]=v[0].detach().cpu().numpy()
   if (i+1)%64==0 or i+1==len(current):
    loraq.flush();jwrite(pfile,{"benchmark":bench,"fingerprint":fingerprint,"completed_rows":i+1,"row_count":len(current),"embedding_dim":runner.HIDDEN_SIZE,"dtype":"float32","base_checkpoint_sha256":actual_lora_sha,"inference_batch_size":1,"updated_at_utc":datetime.now(timezone.utc).isoformat()})
   if (i+1)%256==0 or i+1==len(current):print(f'[{bench}] epoch-5 query inference {i+1}/{len(current)}',flush=True)
  loraq.flush();del loraq
 model.eval();model.vision_model.eval()
 selected_cache=json.loads((refdir/'metadata.json').read_text())
 # These assertions are evaluated before production scoring; the reference/query checkpoint identities must match.
 runner.assert_same_lora_checkpoint(actual_lora_sha,selected_cache['lora_checkpoint_sha256'])
 adapted_refs=np.load(refpath,mmap_mode='r')
 external=runner.run_external_benchmarks(ROOT,OUT,catalog,slugs,base_refs,adapted_refs,model,processor,device,selected,selected_cache)
 if not external['base_encoder_query_top5_reproduced']:raise RuntimeError('canonical baseline gate failed during LoRA evaluation')
 jwrite(OUT/'external_metrics.json',external['metrics'])
 jwrite(OUT/'latency.json',external['latency'])
 rows=[]
 transition=runner.read_csv(OUT/'transition_summary.csv')
 for bench in BENCHES:
  m=external['metrics'][bench];im=m['adapted_image_only'];frozen=m['frozen_image_only']
  overall=next((x for x in transition if x['benchmark']==bench and x['slice']=='overall'),{})
  rows.append({"dataset":bench,"queries":m['queries'],"frozen_image_top1":f"{frozen['top1_correct']}/{frozen['queries']}",
   "lora_image_top1":f"{im['top1_correct']}/{im['queries']}","lora_image_r5":f"{im['recall_at_5_correct']}/{im['queries']}",
   "lora_image_mrr":im['mrr'],"frozen_production_top1":f"{m['frozen_production_top1_correct']}/{m['queries']}",
   "lora_production_top1":f"{m['adapted_production_top1_correct']}/{m['queries']}","rescued":overall.get('rescued',''),"broken":overall.get('broken',''),
   "net_gain":overall.get('net_gain',''),"baseline_reproduced_exactly":m['exact_frozen_image_top5_reproduction']==m['queries']})
 cwrite(BASELINE_DIR/'lora_re_evaluation.csv',rows)
 summary={"verdict":"pending_review","baseline_validated_before_lora_inference":True,"checkpoint_validation":checkpoint_validation,
          "adapted_reference_cache_fingerprint":cache_fingerprint,"adapted_reference_code_revision":code_revision,
          "metrics":external['metrics'],"production_transitions":transition,"remaining_generated_error_oracle":json.loads((OUT/'remaining_generated_error_oracle.json').read_text()),
          "report_artifacts":str((BASELINE_DIR/'lora_re_evaluation.csv').relative_to(ROOT))}
 jwrite(BASELINE_DIR/'lora_re_evaluation.json',summary)
 print(json.dumps({"lora_re_evaluation":rows,"remaining_generated_error_oracle":summary['remaining_generated_error_oracle'],"cache_fingerprint":cache_fingerprint},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
