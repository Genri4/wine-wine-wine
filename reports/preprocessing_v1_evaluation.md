# preprocessing_v1 evaluation (frozen SigLIP2)

Controlled comparison of `baseline_full` vs `preprocessing_v1` on the frozen benchmarks.
preprocessing_v1 = full + center 85% + center 70% views of the query, encoded with the
same frozen SigLIP2 in one batched forward pass, ranked by mean similarity against the
unchanged cached reference embeddings. One fixed strategy for every query; no target,
scenario, or subset information is used during inference.

Fixed reasoning (from the pilot32 crop diagnostics, not reinterpreted):

- full image keeps overall context;
- crop85 gives a mild zoom-in;
- crop70 further reduces background influence;
- crop55 is excluded: the diagnostics showed it helps distance_crop but clearly degrades glare/normal cases;
- mean aggregation is preferred over max as the more stable strategy.

## Main comparison

| Benchmark | Method | Top1 | R@5 | R@10 | MRR | MedianRank | MeanLatency ms |
|---|---|---|---|---|---|---|---|
| synthetic_dev | baseline_full | 89.40% | 99.83% | 99.95% | 0.9416 | 1.0 | 12.2 |
| synthetic_dev | preprocessing_v1 | 85.63% | 98.92% | 99.63% | 0.9168 | 1.0 | 25.3 |
| hard_near_duplicate_dev_v2 | baseline_full | 68.43% | 99.65% | 99.83% | 0.8280 | 1.0 | 11.6 |
| hard_near_duplicate_dev_v2 | preprocessing_v1 | 66.17% | 98.35% | 99.39% | 0.8083 | 1.0 | 24.6 |
| generated_stress_dev_pilot32 | baseline_full | 23.44% | 47.66% | 54.69% | 0.3496 | 6.5 | 15.2 |
| generated_stress_dev_pilot32 | preprocessing_v1 | 29.69% | 62.50% | 71.88% | 0.4473 | 3.0 | 29.2 |

## Generated stress pilot32: per-scenario

| Scenario | baseline Top1 | v1 Top1 | baseline R@5 | v1 R@5 |
|---|---|---|---|---|
| distance_crop | 18.75% | 18.75% | 34.38% | 53.12% |
| glare_bad_light | 25.00% | 34.38% | 56.25% | 65.62% |
| handheld | 25.00% | 34.38% | 50.00% | 65.62% |
| slight_angle | 25.00% | 31.25% | 50.00% | 65.62% |

## Generated stress pilot32: per-subset

| Subset | baseline Top1 | v1 Top1 | baseline R@5 | v1 R@5 |
|---|---|---|---|---|
| representative | 42.19% | 50.00% | 57.81% | 76.56% |
| hard | 4.69% | 9.38% | 37.50% | 48.44% |

## Generated stress pilot32: family metrics (hard subset)

| Method | family_type | exact Top1 | family Top1 | family R@5 | within-family disambiguation |
|---|---|---|---|---|---|
| baseline_full | subtype | 6.25% | 21.88% | 40.62% | 28.57% |
| baseline_full | vintage | 3.12% | 9.38% | 53.12% | 33.33% |
| baseline_full | ALL | 4.69% | 15.62% | 46.88% | 30.00% |
| preprocessing_v1 | subtype | 15.62% | 37.50% | 56.25% | 41.67% |
| preprocessing_v1 | vintage | 3.12% | 3.12% | 53.12% | 100.00% |
| preprocessing_v1 | ALL | 9.38% | 20.31% | 54.69% | 46.15% |

## Transitions (baseline_full -> preprocessing_v1)

| Benchmark | Top1 wrong->correct | correct->wrong | unchanged correct | unchanged wrong | Top5 out->in | in->out | stayed in | stayed out | rank improved | degraded | unchanged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| synthetic_dev | 78 | 232 | 3419 | 355 | 1 | 38 | 4039 | 6 | 104 | 312 | 3668 |
| hard_near_duplicate_dev_v2 | 60 | 86 | 701 | 303 | 0 | 15 | 1131 | 4 | 81 | 148 | 921 |
| generated_stress_dev_pilot32 | 8 | 0 | 30 | 90 | 20 | 1 | 60 | 47 | 75 | 4 | 49 |

## Latency

| Benchmark | Method | mean ms | p50 ms | p95 ms |
|---|---|---|---|---|
| synthetic_dev | baseline_full | 12.2 | 11.3 | 16.5 |
| synthetic_dev | preprocessing_v1 | 25.3 | 24.9 | 28.7 |
| hard_near_duplicate_dev_v2 | baseline_full | 11.6 | 11.1 | 15.0 |
| hard_near_duplicate_dev_v2 | preprocessing_v1 | 24.6 | 24.2 | 27.7 |
| generated_stress_dev_pilot32 | baseline_full | 15.2 | 14.2 | 21.9 |
| generated_stress_dev_pilot32 | preprocessing_v1 | 29.2 | 29.3 | 31.4 |

## Acceptance criteria assessment

- PASS: generated stress R@5 growth vs baseline (47.66% -> 62.50%)
- FAIL: synthetic_dev minimal degradation (Top-1 drop <= 1pp and R@5 drop <= 0.2pp) (Top1 89.40% -> 85.63%, R@5 99.83% -> 98.92%)
- FAIL: hard_v2 no substantial regression (Top-1 drop <= 1pp and R@5 drop <= 0.2pp) (Top1 68.43% -> 66.17%, R@5 99.65% -> 98.35%)
- PASS: latency stays far below the 3 s hackathon SLA (v1 mean latency 25.3 ms (baseline 12.2 ms))

Note: metrics use full-catalog target ranks. Frozen-run MRR values published earlier for
pilot32 (0.3251) were computed from Top-5 truncated rankings; full-rank baseline MRR here is
directly comparable only to other full-rank rows of this report.

## H. Synthetic regression check (critical)

| Metric | baseline_full | preprocessing_v1 | Delta |
|---|---|---|---|
| Top-1 | 89.40% | 85.63% | **-3.77pp** |
| R@5 | 99.83% | 98.92% | -0.91pp |
| R@10 | 99.95% | 99.63% | -0.32pp |
| MRR (full rank) | 0.9416 | 0.9168 | -0.0248 |
| Top-1 flips | - | 78 wrong->correct | **232 correct->wrong** |

synthetic_dev degrades clearly more than the pre-declared 1pp guard. The mechanism: synthetic
queries are reference-derived, so the full view is already an almost exact match; averaging
with crop views dilutes the exact-match signal and compresses the margin that separated the
target from near-duplicates. The damage is concentrated in Top-1 ordering (R@5 barely moves).

## I. Hard benchmark check

| Metric | baseline_full | preprocessing_v1 | Delta |
|---|---|---|---|
| Top-1 | 68.43% | 66.17% | **-2.26pp** |
| R@5 | 99.65% | 98.35% | -1.30pp |
| MRR (full rank) | 0.8280 | 0.8083 | -0.0197 |
| Top-1 flips | - | 60 wrong->correct | 86 correct->wrong |

Same margin-compression effect on clean, well-framed hard queries: multi-crop helps capture
robustness but slightly blurs fine-grained discrimination between near-duplicates.

## Verdict (evidence-based)

- preprocessing_v1 is **not adopted as the unconditional default** inference preprocessing:
  it fails the pre-declared regression guards (synthetic_dev -3.77pp Top-1, hard_v2 -2.26pp
  Top-1 vs the allowed 1pp).
- The pilot32 profile is genuinely safe and useful (8 gained / 0 lost Top-1; 20 gained / 1
  lost Top-5), so the multi-crop signal itself is valuable exactly where the query framing
  deviates from the reference framing.
- The data motivates a *conditional* policy (for example: run baseline full view first and
  apply the multi-crop fallback only when the full-view result is not confident), to be
  designed and evaluated as a separate controlled milestone. It also aligns with the
  Smart Retry feature contract. No such gating was implemented or tuned in this milestone.

Run directory: `artifacts/experiments/siglip2_preprocessing_v1_20260919T194507Z/`.

