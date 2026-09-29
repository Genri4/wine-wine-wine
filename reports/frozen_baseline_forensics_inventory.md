# Frozen SO400M baseline artifact inventory

## Historical baseline

The historical encoder result is stored under `artifacts/experiments/encoder_bakeoff_20260920T143928Z/models/siglip2_so400m_384/`. The source predictions are `hard_near_duplicate_dev_v2/predictions.csv`, `generated_stress_dev_pilot32/predictions.csv`, and `synthetic_dev/predictions.csv`; these contain per-query Top-5 slugs and FP32 scores. Their metrics record the pinned checkpoint revision `e8e487298228002f3d8a82e0cd5c8ea9c567f57f`.

| Item | Historical record | Current verified record |
|---|---|---|
| Model ID | `google/siglip2-so400m-patch14-384` | same |
| Revision | `e8e487298228002f3d8a82e0cd5c8ea9c567f57f` | same |
| Model shape | resource artifact: 1,136,008,498 parameters, 1,152 dimensions | same config, 27 vision blocks, hidden size 1,152, patch 14 |
| Weights | historical state-dict/file SHA was not saved | current `model.safetensors` SHA-256 is in `model_fingerprint_diff.json` |
| Processor | `SiglipImageProcessor`, 384×384 resize, RGB, mean/std 0.5, resample 2, no center crop | same recorded processor config |
| Query inference batch | one query per call (`run_encoder_benchmark.py` passes `[image]` within the per-query loop) | corrected forensic rerun uses batch 1 |
| Resource batch | resource artifact reports batch 16 for reference/cache work | not evidence of query batch size |
| Model dtype | fp16 | fp16 |
| Embeddings and cosine | adapter returns L2-normalized FP32 vectors; query result is normalized again before cosine | same |
| Rank order | score descending, slug ascending for equal scores (`full_ranking`) | same result; stable index sort matches it because reference slugs are lexicographically ordered |
| Historical software lock | not retained | current versions in `environment_diff.json` |
| Historical attention backend | not retained | current Transformers selects SDPA |

## Reference artifacts

The single persisted cache `artifacts/reference_embeddings/siglip2_so400m_384/` contains 2,042 slugs and 1,152-dimensional FP32 normalized vectors. Its metadata records revision `e8e487298228002f3d8a82e0cd5c8ea9c567f57f`, fingerprint `25a02b3da1c586579619efb0823ef12227e2ba6d76cde1c3f4f97c825d9e66ec`, and processor settings above. The historical encoder run reported `reference_cache=loaded`; the same persisted cache file was loaded for the current forensic comparisons, so there is no distinct old cache copy to compare. The file SHA-256 and per-reference current image paths/hashes are in `reference_cache_diff.csv` and `dataset_hashes.json`.

## Dataset identity

The old prediction CSVs and current manifests have matching query IDs and paths. Current SHA-256 and byte size are recorded for all 5,362 queries and all 2,042 catalog references in `dataset_hashes.json`. Historical runs did not save query/reference per-file hashes, so byte identity back to the old run cannot be independently proven from its artifacts.

## Frozen query feature artifacts

Historical query embedding tensors were not saved. The old Top-5 score values are retained for the historical Top-5 candidates. The forensic run saved a batch-1 normalized FP32 query tensor for every query and verified exact Top-5 order plus zero score delta on all retained Top-5 candidate scores. The prior batch-16 query tensors are retained in the LoRA experiment artifact. Per-query batch-1/batch-16 cosine, L2 and coordinate deltas are in `query_embedding_diff.csv`.

## LoRA identity

The existing selected checkpoint is epoch 5, in `artifacts/experiments/so400m_hard_negative_lora_20260923T203613Z/checkpoints/epoch_005.pt`. The selected LoRA-state SHA-256 is `b3dfce35160cc4c745090f7ec03dd783f86a4c0a82d7566521e7aa21e6a04a04`. It is validated after loading and before the re-evaluation; the whole checkpoint-file SHA and loaded state SHA are in `lora_evaluation/checkpoint_validation.json`.

## Root-cause evidence

The original baseline mismatch was 16 hard and 62 synthetic queries. The historical query loop used single-image batches, while the post-freeze comparison encoded 16 queries per batch. Repeating the original batch-1 shape reproduced all three historical Top-5s exactly and all stored Top-5 scores exactly. On the original order mismatches, old/current inversion gaps were about `5.8e-5` median for same-set reorderings; the changed-candidate Top-5 boundary gaps were about `5.2e-5` median on synthetic and `5.4e-5` on hard. Full numbers and precision/backend controls are in the corresponding CSV/JSON artifacts.
