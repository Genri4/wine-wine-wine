# Hard-negative metric adapter milestone

Run: `hard_negative_metric_adapter_20260923T183443Z`. Verdict: **C. FROZEN_FEATURES_INSUFFICIENT**.

## Primary results

| Method | Hard Top1 | Generated Top1 | Generated correct/128 | Generated-hard correct/64 | Rescued | Broken |
|---|---:|---:|---:|---:|---:|---:|
| current + SIFT w=0.40 | 91.91% | 80.47% | 103/128 | 40/64 | — | — |
| current + SIFT + adapter w=0.10 | 92.00% (1058/1150) | 80.47% | 103/128 | 40/64 | 0 | 0 |
| adapter only | 1041/1150 | 66.41% | 85/128 | 27/64 | — | — |

## Embedding diagnostics

| State | Positive sim | Hard-negative sim | Margin | Random-negative sim |
|---|---:|---:|---:|---:|
| Before | 0.8828 | 0.8344 | 0.0485 | 0.6079 |
| After | 0.7836 | 0.5159 | 0.2579 | 0.0014 |

## Training and checks

- Training views: 16,336; validation views: 4,084.
- Hard negatives: catalog metadata same-family candidates first, frozen SO400M reference Top-20 neighbors next, plus two deterministic random catalog negatives; no benchmark error mining.
- Adapter: 1152 → 256 → 1152 residual MLP, identity initialized; 591,232 parameters; SO400M remains frozen.
- Internal validation candidate Top-1: 86.95% → 95.59%; selected epoch 15.
- Family guard: hard_v2 minimum 1046/1150; representative minimum 63/64. Selected policy guards: True.
- Hard_v2 guard result: 1058/1150; generated representative: 63/64 (rescued 0, broken 0).
- Generated-hard: baseline 40/64 → 40/64 (rescued 0, broken 0).
- Adapter oracle on frozen generated errors: 5/25 target scores beat the incumbent. Oracle covers the 25 frozen generated baseline errors.
- Synthetic sanity (post-selection): 3969/4084 = 97.18%; baseline current+SIFT is 97.14%.
- Added adapter stage p95: 0.703 ms; estimated total pipeline p95: 514.799 ms (SLA <3000 ms: True).
- Overfit flag: False; phase B started: no.
- Reached at least 116/128 with guards: NO.

## Family slices

| Benchmark slice | Top1 | Correct / queries |
|---|---:|---:|
| hard_near_duplicate_dev_v2 · other | 88.76% | 300/338 |
| hard_near_duplicate_dev_v2 · subtype | 94.22% | 701/744 |
| hard_near_duplicate_dev_v2 · vintage | 83.82% | 57/68 |
| hard_near_duplicate_dev_v2 · within_family | 92.00% | 1058/1150 |
| generated_stress_dev_pilot32 · hard_subtype | 59.38% | 19/32 |
| generated_stress_dev_pilot32 · hard_vintage | 65.62% | 21/32 |
| generated_stress_dev_pilot32 · subtype | 59.38% | 19/32 |
| generated_stress_dev_pilot32 · vintage | 65.62% | 21/32 |

Transition details by overall, representative, hard, vintage, subtype, and within-family slices are in `transition_summary.csv`.

## Checkpoint and resume

`checkpoints/latest.pt` is written after each epoch; `checkpoints/best.pt` is selected only on internal validation. Train and validation view embedding memmaps flush after each encoded batch with adjacent progress JSON, so restart with the same `--run-dir` resumes unfinished preprocessing or the next whole epoch. Benchmark query embeddings are also resumable and were created only after the frozen checkpoint metadata.

Artifacts: `/mnt/d/project/my_wine/artifacts/experiments/hard_negative_metric_adapter_20260923T183443Z`. Frozen checkpoint metadata: `checkpoint_metadata.json`.
