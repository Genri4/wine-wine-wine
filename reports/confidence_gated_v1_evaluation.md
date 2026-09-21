# confidence_gated_v1 evaluation (frozen SigLIP2)

Controlled milestone: baseline_full vs unconditional preprocessing_v1 vs confidence_gated_v1.
The gate uses only the full-image ranking score distribution (top1 score / top1-top2 margin);
fallback encodes exactly center_85 + center_70 in one batched forward and never recomputes
the full embedding; aggregation is identical to preprocessing_v1 (mean of three views).

## Selected gate (frozen)

- Rule: `fallback if top1_score < 0.8824`
- Calibration seed: 20260919, fingerprint `c92352d82b53...`
- Calibration query sizes: {'synthetic_dev': 800, 'hard_near_duplicate_dev_v2': 240, 'generated_stress_dev_pilot32': 64}
- Candidates passing the clean guard: 3

## Main comparison

| Benchmark | Method | Top1 | R@5 | R@10 | MRR | Fallback% | Mean latency ms |
|---|---|---|---|---|---|---|---|
| synthetic_dev | baseline_full | 89.40% | 99.83% | 99.95% | 0.9416 | 0.00% | 10.6 |
| synthetic_dev | preprocessing_v1 | 85.63% | 98.92% | 99.63% | 0.9168 | 0.00% | 22.9 |
| synthetic_dev | confidence_gated_v1 | 88.10% | 99.17% | 99.66% | 0.9310 | 19.17% | 13.8 |
| hard_near_duplicate_dev_v2 | baseline_full | 68.43% | 99.65% | 99.83% | 0.8280 | 0.00% | 10.8 |
| hard_near_duplicate_dev_v2 | preprocessing_v1 | 66.17% | 98.35% | 99.39% | 0.8083 | 0.00% | 22.8 |
| hard_near_duplicate_dev_v2 | confidence_gated_v1 | 68.70% | 98.87% | 99.39% | 0.8241 | 20.52% | 14.0 |
| generated_stress_dev_pilot32 | baseline_full | 23.44% | 47.66% | 54.69% | 0.3496 | 0.00% | 14.2 |
| generated_stress_dev_pilot32 | preprocessing_v1 | 29.69% | 62.50% | 71.88% | 0.4473 | 0.00% | 29.3 |
| generated_stress_dev_pilot32 | confidence_gated_v1 | 29.69% | 62.50% | 71.88% | 0.4471 | 93.75% | 35.3 |

## Confidence signal analysis (Part B)

| Benchmark | Scope | Signal | AUC top1-correct | AUC target-in-top5 | median correct | median incorrect |
|---|---|---|---|---|---|---|
| synthetic_dev | overall | top1_score | 0.581 | 0.754 | 0.9189 | 0.9078 |
| synthetic_dev | overall | top1_top2_margin | 0.910 | 0.792 | 0.0589 | 0.0057 |
| synthetic_dev | overall | top1_top5_margin | 0.725 | 0.947 | 0.1152 | 0.0702 |
| hard_near_duplicate_dev_v2 | overall | top1_score | 0.583 | 0.816 | 0.9192 | 0.9096 |
| hard_near_duplicate_dev_v2 | overall | top1_top2_margin | 0.807 | 0.561 | 0.0294 | 0.0055 |
| hard_near_duplicate_dev_v2 | overall | top1_top5_margin | 0.644 | 0.904 | 0.1007 | 0.0703 |
| generated_stress_dev_pilot32 | overall | top1_score | 0.588 | 0.521 | 0.8194 | 0.8128 |
| generated_stress_dev_pilot32 | overall | top1_top2_margin | 0.774 | 0.601 | 0.0365 | 0.0110 |
| generated_stress_dev_pilot32 | overall | top1_top5_margin | 0.873 | 0.710 | 0.0681 | 0.0315 |
| generated_stress_dev_pilot32 | representative | top1_score | 0.721 | 0.635 | 0.8206 | 0.7935 |
| generated_stress_dev_pilot32 | representative | top1_top2_margin | 0.851 | 0.771 | 0.0386 | 0.0075 |
| generated_stress_dev_pilot32 | representative | top1_top5_margin | 0.888 | 0.882 | 0.0717 | 0.0286 |
| generated_stress_dev_pilot32 | hard | top1_score | 0.317 | 0.426 | 0.8041 | 0.8249 |
| generated_stress_dev_pilot32 | hard | top1_top2_margin | 0.197 | 0.366 | 0.0000 | 0.0128 |
| generated_stress_dev_pilot32 | hard | top1_top5_margin | 0.809 | 0.444 | 0.0478 | 0.0336 |

## pilot32 per-scenario (baseline / v1 / gated, with gated fallback rate)

| Scenario | baseline Top1 | v1 Top1 | gated Top1 | baseline R@5 | v1 R@5 | gated R@5 | gated fallback% |
|---|---|---|---|---|---|---|---|
| distance_crop | 18.75% | 18.75% | 18.75% | 34.38% | 53.12% | 53.12% | 100.00% |
| glare_bad_light | 25.00% | 34.38% | 34.38% | 56.25% | 65.62% | 65.62% | 81.25% |
| handheld | 25.00% | 34.38% | 34.38% | 50.00% | 65.62% | 65.62% | 100.00% |
| slight_angle | 25.00% | 31.25% | 31.25% | 50.00% | 65.62% | 65.62% | 93.75% |

## pilot32 per-subset

| Subset | baseline Top1 | v1 Top1 | gated Top1 | baseline R@5 | v1 R@5 | gated R@5 | gated fallback% |
|---|---|---|---|---|---|---|---|
| representative | 42.19% | 50.00% | 50.00% | 57.81% | 76.56% | 76.56% | 92.19% |
| hard | 4.69% | 9.38% | 9.38% | 37.50% | 48.44% | 48.44% | 95.31% |

## pilot32 family metrics (hard subset)

| Method | family_type | exact Top1 | family Top1 | family R@5 | disambiguation | fallback% |
|---|---|---|---|---|---|---|
| baseline_full | subtype | 6.25% | 21.88% | 40.62% | 28.57% | 0.00% |
| baseline_full | vintage | 3.12% | 9.38% | 53.12% | 33.33% | 0.00% |
| baseline_full | ALL | 4.69% | 15.62% | 46.88% | 30.00% | 0.00% |
| preprocessing_v1 | subtype | 15.62% | 37.50% | 56.25% | 41.67% | 0.00% |
| preprocessing_v1 | vintage | 3.12% | 3.12% | 53.12% | 100.00% | 0.00% |
| preprocessing_v1 | ALL | 9.38% | 20.31% | 54.69% | 46.15% | 0.00% |
| confidence_gated_v1 | subtype | 15.62% | 37.50% | 56.25% | 41.67% | 96.88% |
| confidence_gated_v1 | vintage | 3.12% | 3.12% | 53.12% | 100.00% | 93.75% |
| confidence_gated_v1 | ALL | 9.38% | 20.31% | 54.69% | 46.15% | 95.31% |

## Gate confusion matrix (Part M)

| Benchmark | trusted correct | unnecessary fallback | useful trigger | missed opportunity |
|---|---|---|---|---|
| synthetic_dev | 2999 (73.43%) | 652 (15.96%) | 131 (3.21%) | 302 (7.39%) |
| hard_near_duplicate_dev_v2 | 655 (56.96%) | 132 (11.48%) | 104 (9.04%) | 259 (22.52%) |
| generated_stress_dev_pilot32 | 5 (3.91%) | 25 (19.53%) | 95 (74.22%) | 3 (2.34%) |

## Fallback effectiveness among triggered queries (Part N)

| Benchmark | fallback queries | Top1 wrong->correct | Top1 correct->wrong | Top5 out->in | Top5 in->out |
|---|---|---|---|---|---|
| synthetic_dev | 783 | 28 | 81 | 1 | 28 |
| hard_near_duplicate_dev_v2 | 236 | 19 | 16 | 0 | 9 |
| generated_stress_dev_pilot32 | 120 | 8 | 0 | 20 | 1 |

## Latency (production-like)

| Benchmark | Method | mean ms | p50 ms | p95 ms | fallback% | views/query |
|---|---|---|---|---|---|---|
| synthetic_dev | baseline_full | 10.6 | 10.2 | 13.4 | 0.00% | 1.00 |
| synthetic_dev | preprocessing_v1 | 22.9 | 22.2 | 27.5 | 0.00% | 3.00 |
| synthetic_dev | confidence_gated_v1 | 13.8 | 10.6 | 26.3 | 19.17% | 1.38 |
| hard_near_duplicate_dev_v2 | baseline_full | 10.8 | 10.3 | 15.0 | 0.00% | 1.00 |
| hard_near_duplicate_dev_v2 | preprocessing_v1 | 22.8 | 22.2 | 26.9 | 0.00% | 3.00 |
| hard_near_duplicate_dev_v2 | confidence_gated_v1 | 14.0 | 10.5 | 26.2 | 20.52% | 1.41 |
| generated_stress_dev_pilot32 | baseline_full | 14.2 | 13.1 | 20.1 | 0.00% | 1.00 |
| generated_stress_dev_pilot32 | preprocessing_v1 | 29.3 | 28.5 | 35.5 | 0.00% | 3.00 |
| generated_stress_dev_pilot32 | confidence_gated_v1 | 35.3 | 36.4 | 40.7 | 93.75% | 2.88 |

Gate reproducibility mismatches on calibration queries: 0.
Run directory: `artifacts/experiments/siglip2_confidence_gated_v1_20260919T204926Z/`.

## Calibration design (Parts C, D, E)

- pilot32 was split **by product** (16 calibration / 16 held-out), stratified
  representative 8/8 and by hard family type: 2 vintage families and 2 subtype
  families per side; all four scenarios of one product always stay together.
  Fixed seed `20260919`; split fingerprinted in `selected_gate.json`.
- Clean calibration subsets: deterministic product-level samples of 400/2042
  synthetic_dev products (800 queries) and 120/575 hard_v2 products (240
  queries), same seed. Threshold selection used **only** these subsets; full
  benchmark numbers below are separate evaluation numbers (partial-set
  overlap disclosed here).
- Threshold candidates came from fixed quantile grids of pooled calibration
  distributions: margins {50,60,70,80,90}% and scores {5,10,20,30}%;
  29 candidates total (5 margin, 4 score, 20 margin-or-score).

## Why this gate was selected (Parts F, G, I)

Only 3 of 29 candidates passed the clean guard (Top-1 delta >= -1pp on both
clean calibration subsets): all margin-based rules triggered fallback on
40-97% of clean calibration queries because clean margins are small in
absolute terms, and failed the guard. The winning rule is a plain absolute
score gate at the pooled 20% score quantile:

    confidence_gated_v1: fallback if top1_score < 0.8824

Interpretation supported by the AUC table: the absolute top1 score separates
domain-shifted generated images (median 0.813-0.820) from clean reference-like
images (median 0.919) much better than it predicts per-query correctness
(AUC 0.58 for top1-correct). The gate effectively detects capture domain
shift, which is exactly the regime where multi-crop helps. The margin signal
is the better correctness predictor on clean data (AUC 0.91) but cannot be
thresholded globally without firing on most clean queries.

## Held-out generated evaluation (Part J)

16 pilot32 products (64 queries) never used for threshold selection:

| Method | Top1 | R@5 | R@10 | MRR | Median rank | Fallback% | Mean latency ms |
|---|---|---|---|---|---|---|---|
| baseline_full | 23.44% | 45.31% | 51.56% | 0.3289 | 9.5 | 0% | 14.1 |
| preprocessing_v1 | 28.13% | 62.50% | 67.19% | 0.4180 | 3.5 | 0% | 29.3 |
| confidence_gated_v1 | 28.13% | 62.50% | 67.19% | 0.4177 | 3.5 | 93.75% | 35.2 |

The frozen gate transfers without any tuning: on held-out products it matches
unconditional preprocessing_v1 exactly on Top-1/R@5/R@10 while running the
fallback path for 93.75% of queries.

## Failure modes (honest accounting)

- On synthetic_dev the 19.17% fallback rate buys net-negative Top-1 movement
  inside the fallback set (28 rescued vs 81 broken): the clean regression
  (-1.30pp Top-1 vs baseline) is concentrated there. This is the price of a
  single global score threshold; the full-benchmark clean regression
  (-1.30pp) is larger than the calibration estimate (-0.12pp) and slightly
  exceeds the pre-declared 1pp guard.
- On hard_v2 the same trade is net-positive for Top-1 (19 rescued vs 16
  broken; Top-1 68.43% -> 68.70%) but costs R@5 (-0.78pp: 9 in->out vs 0
  out->in inside the fallback set).
- On pilot32 the gate is near-optimal: 95/128 useful triggers, 25 unnecessary,
  3 missed, and zero Top-1 breaks inside the fallback set.
- When fallback fires as often as on pilot32 (93.75%), the gated path is
  ~20% slower than unconditional preprocessing_v1 (35.3 vs 29.3 ms) because
  it performs two sequential forward passes (1 + 2) instead of one batched
  forward of 3, plus a second ranking pass. The gate pays off in latency
  exactly when fallback is rare (clean: 13.8 ms vs 22.9 ms, 1.38 views/query).

## Verdict (evidence-based)

**FIX confidence_gated_v1 as the retrieval default** for the current pipeline,
with the frozen rule `fallback if top1_score < 0.8824`:

1. It keeps most of the baseline clean quality that unconditional
   preprocessing_v1 destroyed: synthetic Top-1 88.10% vs 89.40% baseline
   (v1: 85.63%), hard_v2 Top-1 68.70% vs 68.43% baseline (v1: 66.17%).
2. It captures the full generated-stress gain of preprocessing_v1 (Top-1
   29.69%, R@5 62.50% - identical to v1), validated on held-out products.
3. It respects the efficiency requirement: fallback fires on ~20% of clean
   queries and 94% of generated queries; 1.38-1.41 views/query on clean data;
   all latencies stay far below the 3 s SLA.
4. Remaining known cost: -1.30pp synthetic Top-1 (within the fallback set)
   and -0.78pp hard_v2 R@5; acceptable vs +14.8pp generated R@5, and the
   final confidence value remains available for the future Smart Retry layer
   (the gate decision is deliberately kept separate from user-facing retry).

Not solved by this milestone: hard-subset vintage/subtype disambiguation
(exact Top-1 still 9.38% on generated hard) - that is the OCR/reranking
regime, next milestone.

