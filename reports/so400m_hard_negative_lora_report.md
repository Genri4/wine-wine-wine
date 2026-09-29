# Partial SO400M hard-negative LoRA milestone

Run: `artifacts/experiments/so400m_hard_negative_lora_20260923T203613Z`
Selected checkpoint: `artifacts/experiments/so400m_hard_negative_lora_20260923T203613Z/checkpoints/best.pt`
Selected checkpoint SHA-256: `b3dfce35160cc4c745090f7ec03dd783f86a4c0a82d7566521e7aa21e6a04a04`
Internal-selection epoch: 5

## Benchmark results

| Method | Hard Top-1 | Generated Top-1 | Generated correct /128 | Generated hard /64 | Representative /64 | R@5 (hard / generated) | Rescued / broken (generated) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen production (SO400M + OCR + SIFT 0.40) | 1057/1150 (91.91%) | 103/128 (80.47%) | 103/128 | 40/64 | 63/64 | 1148/1150 / 128/128 | — |
| Adapted SO400M image-only | 1051/1150 (91.39%) | 104/128 (81.25%) | 104/128 | 40/64 | 64/64 | 1150/1150 / 128/128 | 7 / 6 |
| Adapted SO400M + existing OCR + SIFT 0.40 | 1066/1150 (92.70%) | 107/128 (83.59%) | 107/128 | 43/64 | 64/64 | 1150/1150 / 128/128 | 5 / 1 |

Encoder-only retrieval (full-catalog ranking):

| Split | Frozen image Top-1 | Adapted image Top-1 | Adapted R@5 | Adapted R@10 | Adapted MRR |
|---|---:|---:|---:|---:|---:|
| hard_near_duplicate_dev_v2 | 1000/1150 (86.96%) | 1051/1150 (91.39%) | 1150/1150 | 1150/1150 | 0.9562 |
| generated_stress_dev_pilot32 | 94/128 (73.44%) | 104/128 (81.25%) | 128/128 | 128/128 | 0.9062 |
| synthetic_dev | 3885/4084 (95.13%) | 3953/4084 (96.79%) | 4084/4084 | 4084/4084 | 0.9836 |

Hard-v2 production by family type:

| Hard-v2 family type | Frozen production | Adapted production | Net gain |
|---|---:|---:|---:|
| other | 289/326 (88.65%) | 289/326 (88.65%) | +0 |
| subtype | 711/756 (94.05%) | 718/756 (94.97%) | +7 |
| vintage | 57/68 (83.82%) | 59/68 (86.76%) | +2 |

## Internal and external family margins

| Model | Internal full catalog | Internal family accuracy | Internal family margin | Generated hard | External family margin (hard_v2 / generated) |
|---|---:|---:|---:|---:|---:|
| Frozen SO400M | 56.22% | 75.53% | 0.0273 | 40/64 | 0.0439 / 0.0093 |
| Selected LoRA SO400M | 80.93% | 92.40% | 0.1910 | 43/64 | 0.2438 / 0.0388 |

## Model, training data, and integrity

- Checkpoint `google/siglip2-so400m-patch14-384` at revision `e8e487298228002f3d8a82e0cd5c8ea9c567f57f`; Transformers 5.17.0; vision blocks 27, hidden size 1152, 16 heads.
- LoRA: rank 8, alpha 16, dropout 0.05; blocks 23–26, targets `q_proj, k_proj, v_proj, out_proj`; 294,912 trainable parameters. Text tower trainable parameters: 0; all other checkpoint parameters frozen.
- Train/validation split: 8 / 2 views per SKU; 16,336 / 4,084 rows.
- Capture-v2 uses random placement and scale, perspective warps, procedural subdued backgrounds, exposure and color-temperature shifts, uneven lighting, mild glare and shadows, defocus/motion blur, down/up-sampling, sensor noise, and JPEG artifacts. Train/validation transform ranges differ. Local images outside benchmark/reference paths were catalog-mapping review candidates, not usable scene backgrounds. With no reliable bottle masks, the whole reference photo is placed as a card; the code does not claim bottle cutout compositing.
- Negatives come only from same-family catalog metadata, frozen SO400M reference Top-20 neighbors, and seeded random catalog SKUs. Training never reads benchmark images, query IDs, target ranks, error pairs, or screenshots.
- Re-embedded frozen image Top-5 agreement: hard_near_duplicate_dev_v2 1134/1150, generated_stress_dev_pilot32 128/128, synthetic_dev 4022/4084. The exact baseline guard failed, so adapted external metrics are diagnostic and the milestone is marked INVALID. The SIFT reference cache fingerprint `10f9514172c9068e0e0671a804f684c88f8e2da334f85758d14e0e5ee016e6e6` matched the unchanged catalog images.
- Adapted reference-cache fingerprint includes LoRA checkpoint SHA-256: `b3dfce35160cc4c745090f7ec03dd783f86a4c0a82d7566521e7aa21e6a04a04`.
- Post-freeze query embeddings and progress checkpoints are stored under `benchmarks/<split>/query_embeddings/`; fingerprints cover benchmark IDs, source image hashes, base revision, and selected LoRA SHA-256.

## Internal generalization and cost

- Frozen → LoRA validation: candidate-set 66.53% → 90.18%; full catalog 56.22% → 80.93%; same-family 75.53% → 92.40%; family margin 0.0273 → 0.1910.
- Selected train → validation candidate Top-1: 93.50% → 90.18%; full-catalog Top-1: 86.52% → 80.93%.
- Per-bucket validation results (`clean-ish`, `perspective`, `background_scale`, `blur`, `glare`) are in `internal_validation_buckets.json`.
- Training: 6 epochs, best epoch 5; summed epoch duration 246.4 minutes; peak VRAM 1056 MB; microbatch 4, accumulation 4.
- Generated query latency: frozen encoder p95 40.2 ms; adapted encoder p95 40.0 ms; adapted OCR+SIFT full pipeline p95 356.7 ms; SLA <3 s: True; peak inference VRAM 876 MB.
- Frozen generated error oracle: 25 errors; adapted target similarity beats the frozen production incumbent in 11 cases (Phase A metric-adapter result was 5/25).
- Synthetic post-selection sanity: frozen production 3967/4084 (97.14%); adapted production 3983/4084 (97.53%); adapted image-only R@5 4084/4084.

## Transitions and verdict

- Hard_v2 rescued/broken: adapted image-only 17/23; adapted production 13/4.
- Generated hard rescued/broken: 4/1; representative rescued/broken: 1/0.
- Generated production 107/128; target ≥116/128: **NO**. Hard_v2 guard ≥90.9%: **PASS**.
- **E. INVALID / LEAKAGE / TRAINING FAILURE** — Frozen SO400M query embeddings did not exactly reproduce the stored Top-5 baseline: hard_near_duplicate_dev_v2 1134/1150, generated_stress_dev_pilot32 128/128, synthetic_dev 4022/4084. Adapted metrics are diagnostic and do not qualify as a valid milestone result.
- Scenario, representative/hard, vintage/subtype/other, and transition tables: `scenario_metrics.csv`, `family_metrics.csv`, `transition_summary.csv`.
- Frozen and adapted full-catalog rank-index matrices are under `benchmarks/<split>/`; the corresponding `ranking_reference_slugs.json` maps each index to a catalog SKU.
- Resume from the latest completed epoch with `python scripts/run_so400m_hard_negative_lora.py --max-epochs 6 --resume artifacts/experiments/so400m_hard_negative_lora_20260923T203613Z`.
