# Frozen baseline reproducibility forensics

ROOT CAUSE: Current post-freeze query inference batched 16 images; historical benchmark encoded each query as a single-image batch. FP16 shape-dependent execution caused small embedding/score drift, changing near-tied rankings.

| Dataset | Queries | Exact order old/current | Same Top-5 set | Different Top-1 | Target entered/exited Top-5 |
|---|---:|---:|---:|---:|---:|
| hard_near_duplicate_dev_v2 | 1150 | 1134/1150 → 1150/1150 | 1150/1150 | 1 before; 0 after | 0 before; 0 after |
| generated_stress_dev_pilot32 | 128 | 128/128 → 128/128 | 128/128 | 0 before; 0 after | 0 before; 0 after |
| synthetic_dev | 4084 | 4022/4084 → 4084/4084 | 4084/4084 | 1 before; 0 after | 0 before; 0 after |

## Model, data and ranking sources

| Source | Old | Current | Same? | Evidence |
|---|---|---|---|---|
| Model weights | google/siglip2-so400m-patch14-384@e8e487298228002f3d8a82e0cd5c8ea9c567f57f | google/siglip2-so400m-patch14-384@e8e487298228002f3d8a82e0cd5c8ea9c567f57f | revision: yes | immutable revision recorded in old metrics and current model cache; no old checkpoint checksum |
| Preprocessing | {'class': 'SiglipImageProcessor', 'size': SizeDict(height=384, width=384, longest_edge=None, shortest_edge=None, max_height=None, max_width=None, min_pixels=None, max_pixels=None), 'resample': 2, 'image_mean': (0.5, 0.5, 0.5), 'image_std': (0.5, 0.5, 0.5), 'do_center_crop': None, 'do_resize': True} | same processor config | config: yes | old and current tensors not both persisted |
| Query files | historical paths and IDs | {'hard_near_duplicate_dev_v2': 1150, 'generated_stress_dev_pilot32': 128, 'synthetic_dev': 4084} queries, SHA-256 stored | historical bytes unverified | old artifact omitted per-file hashes |
| Reference files | 2,042 references | 2,042 current SHA-256 records | old bytes unverified | files/hashes in dataset_hashes.json |
| Reference embeddings | persisted cache fingerprint 25a02b3da1c586579619efb0823ef12227e2ba6d76cde1c3f4f97c825d9e66ec | same cache file SHA-256 c339cf05f472c57cd0b9c172af8d177ec072bf372cf56c1815b88c79e9b270ee | yes | old run says reference_cache=loaded; no separate cache copy |
| Dtype | fp16 model / fp32 normalized vectors | fp16 model / fp32 normalized vectors | yes | current runtime and stored cache metadata |
| Attention backend | not recorded | sdpa | unknown | old environment artifact absent |
| Ranking implementation | score desc, slug asc | score desc, slug asc after fix | yes | `full_ranking` historical helper; tie order is deterministic |
| Software versions | not recorded | see environment_diff.json | unknown | no historical lock/environment dump |

## Full baseline after correction

| Dataset | Historical Top-1 | Reproduced Top-1 | Exact Top-5 order | Same Top-5 set | Max score diff |
|---|---:|---:|---:|---:|---:|
| hard_near_duplicate_dev_v2 | 999/1150 | 999/1150 | 1150/1150 | 1150/1150 | 0 |
| generated_stress_dev_pilot32 | 94/128 | 94/128 | 128/128 | 128/128 | 0 |
| synthetic_dev | 3884/4084 | 3884/4084 | 4084/4084 | 4084/4084 | 0 |

## Forensic limits

Historical query embeddings, preprocessed tensors, rank-6 scores, per-image hashes, exact package versions, attention backend, and model-file checksums were not retained. The observed batch-size effect is directly measured, but these missing artifacts prevent independent proof that every historical input byte and software component was identical.

The existing LoRA checkpoint was verified by SHA-256. Its re-evaluation is recorded separately in `artifacts/experiments/frozen_baseline_forensics_20260924T044708Z/lora_re_evaluation.csv` after the baseline gate passed. No retraining or weight modification occurred.

## Mismatch classes before the single-query correction

| Dataset | Type A: same set, order only | Type B: set changed | Type C: Top-1 changed, same set | Type D: target crossed Top-5 |
|---|---:|---:|---:|---:|
| hard_near_duplicate_dev_v2 | 6 | 9 | 1 | 0 |
| generated_stress_dev_pilot32 | 0 | 0 | 0 | 0 |
| synthetic_dev | 40 | 21 | 1 | 0 |

Type A excludes Type C so the category counts are mutually exclusive; Type D is reported independently. A and C both require identical Top-5 candidate sets.

## Numerical and implementation ablations

A 138-query diagnostic subset contained all original hard and synthetic mismatches plus 20 matching controls per split. The same single-image batch and the same references were used across precision modes.

| Mode | Exact old Top-5 | Same set | Top-1 changed | Maximum old-candidate score delta |
|---|---:|---:|---:|---:|
| bf16 | 88/138 | 110/138 | 1 | 0.0066409111 |
| fp16_autocast | 138/138 | 138/138 | 0 | 2.3841858e-07 |
| fp16_no_autocast | 138/138 | 138/138 | 0 | 2.3841858e-07 |
| fp32 | 107/138 | 131/138 | 0 | 0.0010557175 |

Eager attention on the same subset matched old order in 109/138 and set in 132/138; historical backend is not recorded. Eager-vs-SDPA embedding cosine minimum was 0.99985373, max coordinate difference 0.0032210965.

The historical full sort uses score descending and slug ascending on ties. Current stable index sort matched it on all benchmark queries; the reference slugs are already lexicographically ordered. This rules sorting out as the observed mismatch source.

The old Top-5 score CSV values were reproduced with maximum absolute delta 0.0 across all historical Top-5 candidates. Historical rank-6 scores were not stored, so old boundary margins cannot be measured directly. Current batch-16 rank-5/rank-6 margins and matched-control distributions are in `tie_margin_analysis.csv` and `forensic_summary.json`.

Current processor tensors are deterministic on repeated calls; shape/dtype/range/mean/std for the mismatch/control sample are in `preprocess_tensor_samples.csv`. Historical pixel tensors were not retained, so there is no byte/pixel-level old-vs-new tensor diff.

## Artifacts

All required inventories and per-query tables are under `artifacts/experiments/frozen_baseline_forensics_20260924T044708Z/`. The baseline inference checkpoint state is under `query_embeddings/<split>.progress.json`; completed queries can be resumed from the saved arrays.

### Boundary and embedding deltas

| Dataset | Class | Count | Old inverted-pair gap p50 | Batch-16 gap p50 | Candidate-boundary score gap p50 |
|---|---|---:|---:|---:|---:|
| hard_v2 | A (same set/order only) | 6 queries, 7 inversions | 0.0000581 | 0.0000553 | — |
| hard_v2 | B (set changed) | 9 | — | — | 0.0000541 |
| hard_v2 | C (Top-1 changed, same set) | 1 | 0.0001836 | 0.0000597 | — |
| synthetic | A (same set/order only) | 40 queries, 41 inversions | 0.0000581 | 0.0000593 | — |
| synthetic | B (set changed) | 21 | — | — | 0.0000520 |
| synthetic | C (Top-1 changed, same set) | 1 | 0.0001836 | 0.0000597 | — |

The query vectors from the corrected batch-1 rerun and the earlier batch-16 run are close but not identical: hard cosine median 0.9999971 (minimum 0.9999307), synthetic median 0.9999970 (minimum 0.9995640), generated median 0.9999940 (minimum 0.9993401). Maximum absolute coordinate differences were 0.00159, 0.00360, and 0.00412 respectively. Historical query vectors were not saved, so these are batch-1 vs prior batch-16 comparisons, not direct old-vector comparisons. Yet the corrected batch-1 path exactly reproduces the historical stored Top-5 scores and orders.
## Existing epoch-5 LoRA re-evaluation

The selected checkpoint was loaded for inference only after the exact frozen baseline gate passed. Its expected and loaded LoRA-state SHA-256 both equal `b3dfce35160cc4c745090f7ec03dd783f86a4c0a82d7566521e7aa21e6a04a04`; the whole checkpoint file SHA-256 is `7ff3bba0cc416629ee159f1692a221295da322480d1c7b41f06d9eea1c41bb28`. All query and reference images were encoded with the same adapted encoder. The adapted reference cache fingerprint includes the base model, LoRA state, preprocessing, dtype, code revision and reference image hashes.

### Production Top-1 with the unchanged OCR and SIFT reranker (SIFT weight 0.40)

| Method | Hard / 1150 | Generated / 128 | Generated-hard / 64 | Representative / 64 | Synthetic / 4084 |
|---|---:|---:|---:|---:|---:|
| Canonical frozen + OCR + SIFT | 1057 (91.91%) | 103 (80.47%) | 40 (62.50%) | 63 (98.44%) | 3967 (97.14%) |
| Existing epoch-5 LoRA + unchanged OCR + SIFT | 1067 (92.78%) | 107 (83.59%) | 43 (67.19%) | 64 (100%) | 3984 (97.55%) |

### Image retrieval before reranking

| Dataset | Frozen Top-1 | LoRA Top-1 | Frozen R@5 | LoRA R@5 | Frozen MRR | LoRA MRR |
|---|---:|---:|---:|---:|---:|---:|
| Hard / 1150 | 999 | 1051 | 1148/1150 | 1150/1150 | 0.93138 | 0.95623 |
| Generated / 128 | 94 | 104 | 128/128 | 128/128 | 0.85026 | 0.90625 |
| Synthetic / 4084 | 3884 | 3953 | 4082/4084 | 4084/4084 | 0.97447 | 0.98359 |

Production transition counts are measured against the valid canonical baseline: hard 13 rescued / 3 broken; generated overall 5 / 1; generated representative 1 / 0; generated hard 4 / 1; generated vintage 0 / 0; generated subtype 4 / 1; synthetic 21 / 4. The full per-slice CSV is `transition_summary.csv`.

The original set of 25 canonical generated errors was retained. The adapted target score beat / lost to / tied the frozen production incumbent in **11 / 2 / 12** cases. Thus the old diagnostic 11/25 wins signal is preserved. The oracle calculation is diagnostic only and does not affect retrieval, candidate selection, or training.

## Answers to the milestone questions

1. **Why did the baseline differ?** The historical benchmark invokes the encoder once per query (batch 1). The later frozen query path used batch 16. On this fp16 model, input batch shape changes floating-point execution enough to perturb close rankings.
2. **Were model weights the same?** The model ID, immutable Hub revision and config match. The current `model.safetensors` hash is recorded, but the historical checkpoint SHA/state hash was not retained; therefore a byte-for-byte historical weight proof is unavailable.
3. **Was preprocessing the same?** Recorded processor configs match: RGB, 384×384 resize, resample 2, mean/std 0.5, no center crop. Historical pixel tensors are absent, so a direct pixel-by-pixel comparison cannot be made.
4. **Were query/reference bytes identical?** Current hashes cover 5,362 query rows and 2,042 references. The historical run did not retain per-file hashes, so old-to-current byte identity cannot be independently established. Both runs loaded the same persisted reference embedding cache artifact.
5. **How much did embeddings differ?** Historical query vectors were not saved. Batch-1 versus batch-16 vectors have cosine medians 0.999997 (hard), 0.999994 (generated), and 0.999997 (synthetic); minimum cosines are 0.999931, 0.999340, and 0.999564. These are a comparison of current batch shapes, not old vectors versus current vectors.
6. **How many were order-only ties?** Type A: 6 hard and 40 synthetic (46 total). Type C adds 1 hard and 1 synthetic Top-1 change while retaining the same set.
7. **How many changed the Top-5 set?** Type B: 9 hard and 21 synthetic (30 total); no target crossed the Top-5 boundary.
8. **Was rank 5/rank 6 near-tied?** The old rank-6 score was not saved. In the original batch-16 mismatches, the reconstructed incoming/outgoing boundary gap median was 5.41e-5 for hard and 5.20e-5 for synthetic; same-set inversions had median old gap 5.81e-5. These small gaps support numerical sensitivity, without defining a blanket tie tolerance.
9. **Did dtype, batch or attention matter?** Yes for diagnostics. Batch 1 exactly reproduced all stored rankings; batch 16 did not on the prior mismatch subset. On a 138-query precision subset, fp16 reproduced 138/138, fp32 107/138, bf16 88/138. Eager attention matched 109/138 old orders versus 138/138 with the current SDPA path; the historical backend is unknown, so the attention ablation does not establish a historical difference.
10. **Was sorting/top-k defective?** No observed bug. Score-descending then slug-ascending ranking and stable index ranking agreed for all benchmark queries because reference slugs are already sorted; deterministic tie and top-k tests were added.
11. **What was fixed?** Query inference was restored to the historical batch size 1, and the cache fingerprint now includes batch size and query-code identity. We did not change model weights, processor, OCR, SIFT, or fusion weight.
12. **How is reproduction validity defined?** Require exact Top-5 order on all queries using the pinned model revision, batch 1, the recorded reference cache and slug tie-break. Since all historical Top-5 scores also matched with max absolute delta 0, no numerical tolerance is introduced.
13. **Hard baseline restored?** Yes: exact order 1150/1150; Top-1 999/1150; R@5 1148/1150.
14. **Generated baseline restored?** Yes: exact order 128/128; Top-1 94/128; R@5 128/128.
15. **Synthetic baseline restored?** Yes: exact order 4084/4084; Top-1 3884/4084; R@5 4082/4084.
16. **Was the existing LoRA re-evaluated?** Yes, after the baseline passed, with the verified epoch-5 LoRA state and no training.
17. **New valid generated result?** 107/128 production Top-1 (image-only 104/128, R@5 128/128, MRR 0.90625).
18. **Generated-hard result?** 43/64, up from 40/64.
19. **Rescued/broken?** Generated overall 5/1; hard 13/3; synthetic 21/4. Representative generated slice: 1/0.
20. **Did the 11/25 oracle signal remain?** Yes: 11 wins, 2 losses, 12 ties on the unchanged original error set.
21. **Tests?** `.venv/bin/python -m pytest -q -k 'not BaselineReproductionTests'`: 319 passed, 2 deselected, 3 pre-existing scikit-learn deprecation warnings. The model-download integration class was excluded.

## Final verdict

**A. BASELINE_FIXED_LORA_GAIN_VALID** — the historical frozen retrieval is exactly reproduced with batch-1 inference and zero delta on saved Top-5 scores. The existing epoch-5 LoRA keeps a measurable gain on the valid external production benchmarks: +4 generated, +10 hard and +17 synthetic correct queries, with the original 11/25 oracle wins preserved. This completes only this forensic milestone; it does not start another training run.
