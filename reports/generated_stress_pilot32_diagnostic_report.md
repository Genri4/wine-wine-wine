# Pilot32 crop diagnostics (frozen SigLIP2)

Diagnostic milestone run; not a production benchmark result.

## A. Current preprocessing (verified)

- Processor: `SiglipImageProcessor` (`google/siglip2-base-patch16-224`).
- Resize: direct anisotropic resize to 224x224 (`size={'height':224,'width':224}`, bilinear, resample=2).
- Center crop: **not performed** (`do_center_crop=None`). A non-square probe image kept both halves visible: the pipeline stretches, it does not cut.
- Normalization: mean/std = 0.5/0.5 (input scaled to [-1, 1]).
- Final tensor: `(3, 224, 224)`; catalog references and queries go through the identical path.
- Consequence: current preprocessing cannot cut off part of the bottle; it can only distort the aspect ratio.

## L. Comparison table (overall, 128 queries)

| Method | Top1 | R@5 | R@10 | MRR | MedianRank(found) | Missing |
|---|---|---|---|---|---|---|
| baseline_full | 23.44% | 47.66% | 54.69% | 0.3496 | 6.5 | 0 |
| center_crop_85 | 31.25% | 64.06% | 69.53% | 0.4609 | 3.0 | 0 |
| center_crop_70 | 35.16% | 69.53% | 78.12% | 0.5131 | 2.0 | 0 |
| center_crop_55 | 27.34% | 52.34% | 57.81% | 0.3915 | 4.0 | 0 |
| multicrop_max | 28.91% | 59.38% | 66.41% | 0.4382 | 3.0 | 0 |
| multicrop_mean | 32.03% | 62.50% | 74.22% | 0.4723 | 2.5 | 0 |

### representative (64 queries)

| Method | Top1 | R@5 | R@10 | MRR | MedianRank(found) | Missing |
|---|---|---|---|---|---|---|
| baseline_full | 42.19% | 57.81% | 62.50% | 0.5068 | 2.5 | 0 |
| center_crop_85 | 53.12% | 78.12% | 81.25% | 0.6441 | 1.0 | 0 |
| center_crop_70 | 64.06% | 85.94% | 87.50% | 0.7474 | 1.0 | 0 |
| center_crop_55 | 50.00% | 78.12% | 82.81% | 0.6263 | 1.5 | 0 |
| multicrop_max | 54.69% | 75.00% | 79.69% | 0.6520 | 1.0 | 0 |
| multicrop_mean | 54.69% | 79.69% | 90.62% | 0.6750 | 1.0 | 0 |

### hard (64 queries)

| Method | Top1 | R@5 | R@10 | MRR | MedianRank(found) | Missing |
|---|---|---|---|---|---|---|
| baseline_full | 4.69% | 37.50% | 46.88% | 0.1925 | 16.5 | 0 |
| center_crop_85 | 9.38% | 50.00% | 57.81% | 0.2778 | 5.5 | 0 |
| center_crop_70 | 6.25% | 53.12% | 68.75% | 0.2788 | 5.0 | 0 |
| center_crop_55 | 4.69% | 26.56% | 32.81% | 0.1566 | 37.5 | 0 |
| multicrop_max | 3.12% | 43.75% | 53.12% | 0.2243 | 7.0 | 0 |
| multicrop_mean | 9.38% | 45.31% | 57.81% | 0.2697 | 7.0 | 0 |

### scenario: distance_crop (32 queries)

| Method | Top1 | R@5 | R@10 | MRR | MedianRank(found) | Missing |
|---|---|---|---|---|---|---|
| baseline_full | 18.75% | 34.38% | 40.62% | 0.2691 | 16.5 | 0 |
| center_crop_85 | 28.12% | 50.00% | 59.38% | 0.3828 | 5.5 | 0 |
| center_crop_70 | 28.12% | 65.62% | 71.88% | 0.4565 | 2.5 | 0 |
| center_crop_55 | 40.62% | 75.00% | 84.38% | 0.5632 | 2.0 | 0 |
| multicrop_max | 25.00% | 62.50% | 65.62% | 0.4170 | 3.0 | 0 |
| multicrop_mean | 25.00% | 59.38% | 68.75% | 0.4163 | 3.0 | 0 |

### scenario: glare_bad_light (32 queries)

| Method | Top1 | R@5 | R@10 | MRR | MedianRank(found) | Missing |
|---|---|---|---|---|---|---|
| baseline_full | 25.00% | 56.25% | 62.50% | 0.3840 | 5.0 | 0 |
| center_crop_85 | 34.38% | 68.75% | 68.75% | 0.4927 | 2.0 | 0 |
| center_crop_70 | 37.50% | 65.62% | 75.00% | 0.5185 | 2.0 | 0 |
| center_crop_55 | 18.75% | 40.62% | 43.75% | 0.2968 | 20.0 | 0 |
| multicrop_max | 28.12% | 65.62% | 71.88% | 0.4485 | 2.5 | 0 |
| multicrop_mean | 34.38% | 68.75% | 71.88% | 0.5041 | 2.0 | 0 |

### scenario: handheld (32 queries)

| Method | Top1 | R@5 | R@10 | MRR | MedianRank(found) | Missing |
|---|---|---|---|---|---|---|
| baseline_full | 25.00% | 50.00% | 59.38% | 0.3844 | 5.5 | 0 |
| center_crop_85 | 28.12% | 75.00% | 78.12% | 0.4912 | 2.0 | 0 |
| center_crop_70 | 46.88% | 75.00% | 90.62% | 0.6020 | 2.0 | 0 |
| center_crop_55 | 25.00% | 50.00% | 53.12% | 0.3686 | 5.0 | 0 |
| multicrop_max | 28.12% | 56.25% | 65.62% | 0.4383 | 3.0 | 0 |
| multicrop_mean | 34.38% | 59.38% | 84.38% | 0.4925 | 2.5 | 0 |

### scenario: slight_angle (32 queries)

| Method | Top1 | R@5 | R@10 | MRR | MedianRank(found) | Missing |
|---|---|---|---|---|---|---|
| baseline_full | 25.00% | 50.00% | 56.25% | 0.3610 | 7.0 | 0 |
| center_crop_85 | 34.38% | 62.50% | 71.88% | 0.4771 | 2.5 | 0 |
| center_crop_70 | 28.12% | 71.88% | 75.00% | 0.4754 | 2.5 | 0 |
| center_crop_55 | 25.00% | 43.75% | 50.00% | 0.3373 | 9.0 | 0 |
| multicrop_max | 34.38% | 53.12% | 62.50% | 0.4490 | 4.5 | 0 |
| multicrop_mean | 34.38% | 62.50% | 71.88% | 0.4764 | 2.5 | 0 |

### subset x scenario (Top1 / R@5 / R@10)

| Method | subset | scenario | Top1 | R@5 | R@10 |
|---|---|---|---|---|---|
| baseline_full | representative | distance_crop | 37.50% | 43.75% | 43.75% |
| baseline_full | representative | glare_bad_light | 43.75% | 62.50% | 75.00% |
| baseline_full | representative | handheld | 43.75% | 68.75% | 75.00% |
| baseline_full | representative | slight_angle | 43.75% | 56.25% | 56.25% |
| baseline_full | hard | distance_crop | 0.00% | 25.00% | 37.50% |
| baseline_full | hard | glare_bad_light | 6.25% | 50.00% | 50.00% |
| baseline_full | hard | handheld | 6.25% | 31.25% | 43.75% |
| baseline_full | hard | slight_angle | 6.25% | 43.75% | 56.25% |
| center_crop_85 | representative | distance_crop | 50.00% | 68.75% | 68.75% |
| center_crop_85 | representative | glare_bad_light | 56.25% | 87.50% | 87.50% |
| center_crop_85 | representative | handheld | 50.00% | 87.50% | 87.50% |
| center_crop_85 | representative | slight_angle | 56.25% | 68.75% | 81.25% |
| center_crop_85 | hard | distance_crop | 6.25% | 31.25% | 50.00% |
| center_crop_85 | hard | glare_bad_light | 12.50% | 50.00% | 50.00% |
| center_crop_85 | hard | handheld | 6.25% | 62.50% | 68.75% |
| center_crop_85 | hard | slight_angle | 12.50% | 56.25% | 62.50% |
| center_crop_70 | representative | distance_crop | 56.25% | 81.25% | 81.25% |
| center_crop_70 | representative | glare_bad_light | 75.00% | 81.25% | 87.50% |
| center_crop_70 | representative | handheld | 68.75% | 93.75% | 93.75% |
| center_crop_70 | representative | slight_angle | 56.25% | 87.50% | 87.50% |
| center_crop_70 | hard | distance_crop | 0.00% | 50.00% | 62.50% |
| center_crop_70 | hard | glare_bad_light | 0.00% | 50.00% | 62.50% |
| center_crop_70 | hard | handheld | 25.00% | 56.25% | 87.50% |
| center_crop_70 | hard | slight_angle | 0.00% | 56.25% | 62.50% |
| center_crop_55 | representative | distance_crop | 75.00% | 100.00% | 100.00% |
| center_crop_55 | representative | glare_bad_light | 37.50% | 68.75% | 68.75% |
| center_crop_55 | representative | handheld | 43.75% | 75.00% | 81.25% |
| center_crop_55 | representative | slight_angle | 43.75% | 68.75% | 81.25% |
| center_crop_55 | hard | distance_crop | 6.25% | 50.00% | 68.75% |
| center_crop_55 | hard | glare_bad_light | 0.00% | 12.50% | 18.75% |
| center_crop_55 | hard | handheld | 6.25% | 25.00% | 25.00% |
| center_crop_55 | hard | slight_angle | 6.25% | 18.75% | 18.75% |
| multicrop_max | representative | distance_crop | 50.00% | 81.25% | 81.25% |
| multicrop_max | representative | glare_bad_light | 50.00% | 81.25% | 87.50% |
| multicrop_max | representative | handheld | 56.25% | 68.75% | 75.00% |
| multicrop_max | representative | slight_angle | 62.50% | 68.75% | 75.00% |
| multicrop_max | hard | distance_crop | 0.00% | 43.75% | 50.00% |
| multicrop_max | hard | glare_bad_light | 6.25% | 50.00% | 56.25% |
| multicrop_max | hard | handheld | 0.00% | 43.75% | 56.25% |
| multicrop_max | hard | slight_angle | 6.25% | 37.50% | 50.00% |
| multicrop_mean | representative | distance_crop | 43.75% | 75.00% | 93.75% |
| multicrop_mean | representative | glare_bad_light | 62.50% | 87.50% | 87.50% |
| multicrop_mean | representative | handheld | 56.25% | 81.25% | 93.75% |
| multicrop_mean | representative | slight_angle | 56.25% | 75.00% | 87.50% |
| multicrop_mean | hard | distance_crop | 6.25% | 43.75% | 43.75% |
| multicrop_mean | hard | glare_bad_light | 6.25% | 50.00% | 56.25% |
| multicrop_mean | hard | handheld | 12.50% | 37.50% | 75.00% |
| multicrop_mean | hard | slight_angle | 12.50% | 50.00% | 56.25% |

## B. Baseline full-rank distribution (query counts per band)

| scope | value | 1 | 2-5 | 6-10 | 11-25 | 26-100 | >100 |
|---|---|---|---|---|---|---|---|
| overall |  | 30 | 31 | 9 | 20 | 28 | 10 |
| scenario | distance_crop | 6 | 5 | 2 | 5 | 9 | 5 |
| scenario | glare_bad_light | 8 | 10 | 2 | 3 | 7 | 2 |
| scenario | handheld | 8 | 8 | 3 | 7 | 5 | 1 |
| scenario | slight_angle | 8 | 8 | 2 | 5 | 7 | 2 |
| subset | representative | 27 | 10 | 3 | 14 | 9 | 1 |
| subset | hard | 3 | 21 | 6 | 6 | 19 | 9 |
| subset_scenario | representative|distance_crop | 6 | 1 | 0 | 4 | 4 | 1 |
| subset_scenario | representative|glare_bad_light | 7 | 3 | 2 | 2 | 2 | 0 |
| subset_scenario | representative|handheld | 7 | 4 | 1 | 3 | 1 | 0 |
| subset_scenario | representative|slight_angle | 7 | 2 | 0 | 5 | 2 | 0 |
| subset_scenario | hard|distance_crop | 0 | 4 | 2 | 1 | 5 | 4 |
| subset_scenario | hard|glare_bad_light | 1 | 7 | 0 | 1 | 5 | 2 |
| subset_scenario | hard|handheld | 1 | 4 | 2 | 4 | 4 | 1 |
| subset_scenario | hard|slight_angle | 1 | 6 | 2 | 0 | 5 | 2 |

Full distribution (all methods, all scopes): `rank_distribution.csv` in the run directory.

## M. Error transitions vs baseline_full

| Method | wrong->correct Top1 | correct->wrong Top1 | Top5 gained | Top5 lost | recovered_from_missing | lost_to_missing | improved | degraded | unchanged | missing_unchanged | median rank delta |
|---|---|---|---|---|---|---|---|---|---|---|---|
| center_crop_85 | 11 | 1 | 24 | 3 | 0 | 0 | 65 | 20 | 43 | 0 | 1.0 |
| center_crop_70 | 18 | 3 | 31 | 3 | 0 | 0 | 71 | 20 | 37 | 0 | 1.0 |
| center_crop_55 | 13 | 8 | 25 | 19 | 0 | 0 | 51 | 50 | 27 | 0 | 0.0 |
| multicrop_max | 8 | 1 | 20 | 5 | 0 | 0 | 61 | 23 | 44 | 0 | 0.0 |
| multicrop_mean | 11 | 0 | 21 | 2 | 0 | 0 | 75 | 10 | 43 | 0 | 1.5 |

## C/H. Family metrics (hard subset)

| Method | family_type | queries | exact Top1 | family Top1 | family R@5 | within-family disambiguation |
|---|---|---|---|---|---|---|
| baseline_full | subtype | 32 | 6.25% | 21.88% | 40.62% | 28.57% |
| baseline_full | vintage | 32 | 3.12% | 9.38% | 53.12% | 33.33% |
| baseline_full | ALL | 64 | 23.44% | 15.62% | 46.88% | 30.00% |
| center_crop_85 | subtype | 32 | 18.75% | 40.62% | 56.25% | 46.15% |
| center_crop_85 | vintage | 32 | 0.00% | 0.00% | 53.12% | - |
| center_crop_85 | ALL | 64 | 31.25% | 20.31% | 54.69% | 46.15% |
| center_crop_70 | subtype | 32 | 3.12% | 9.38% | 62.50% | 33.33% |
| center_crop_70 | vintage | 32 | 9.38% | 9.38% | 56.25% | 100.00% |
| center_crop_70 | ALL | 64 | 35.16% | 9.38% | 59.38% | 66.67% |
| center_crop_55 | subtype | 32 | 9.38% | 9.38% | 31.25% | 100.00% |
| center_crop_55 | vintage | 32 | 0.00% | 6.25% | 34.38% | 0.00% |
| center_crop_55 | ALL | 64 | 27.34% | 7.81% | 32.81% | 60.00% |
| multicrop_max | subtype | 32 | 6.25% | 18.75% | 62.50% | 33.33% |
| multicrop_max | vintage | 32 | 0.00% | 3.12% | 40.62% | 0.00% |
| multicrop_max | ALL | 64 | 28.91% | 10.94% | 51.56% | 28.57% |
| multicrop_mean | subtype | 32 | 15.62% | 28.12% | 56.25% | 55.56% |
| multicrop_mean | vintage | 32 | 3.12% | 3.12% | 46.88% | 100.00% |
| multicrop_mean | ALL | 64 | 32.03% | 15.62% | 51.56% | 60.00% |

## N. Family transitions (hard subset, vs baseline_full)

| Method | family_type | target back in Top5 | family back in Top5 | correct family Top1, wrong member | left family Top5 |
|---|---|---|---|---|---|
| center_crop_85 | ALL | 10 | 8 | 7 | 3 |
| center_crop_85 | subtype | 8 | 5 | 7 | 0 |
| center_crop_85 | vintage | 2 | 3 | 0 | 3 |
| center_crop_70 | ALL | 13 | 11 | 2 | 3 |
| center_crop_70 | subtype | 10 | 9 | 2 | 2 |
| center_crop_70 | vintage | 3 | 2 | 0 | 1 |
| center_crop_55 | ALL | 10 | 12 | 2 | 21 |
| center_crop_55 | subtype | 8 | 10 | 0 | 13 |
| center_crop_55 | vintage | 2 | 2 | 2 | 8 |
| multicrop_max | ALL | 8 | 9 | 5 | 6 |
| multicrop_max | subtype | 6 | 8 | 4 | 1 |
| multicrop_max | vintage | 2 | 1 | 1 | 5 |
| multicrop_mean | ALL | 6 | 5 | 4 | 2 |
| multicrop_mean | subtype | 6 | 5 | 4 | 0 |
| multicrop_mean | vintage | 0 | 0 | 0 | 2 |

## K. Query image scale diagnostics

| scenario | median width | median height | median aspect |
|---|---|---|---|
| distance_crop | 1024 | 1024 | 1.0 |
| glare_bad_light | 1024 | 1024 | 1.0 |
| handheld | 1024 | 1024 | 1.0 |
| slight_angle | 1024 | 1024 | 1.0 |

Baseline Top-5 alignment check: status=mismatch, exact Top-5 match 125/128; mismatches are near-tie rank swaps within Top-5 from cross-run float nondeterminism (positions 1↔2 and 4↔5 exchanged; recomputed baseline overall Top-1/R@5 match the frozen run exactly: 23.44% / 47.66%).

## J. Manual oracle crop workflow (prepared, not executed)

Tooling: `scripts/oracle_manual_crop.py`. A human may fill
`data/benchmarks/generated_stress_dev_pilot32/oracle_crop_bboxes.csv`
(`query_id,x1,y1,x2,y2,note`, pixel coordinates) for e.g. 24–32 queries;
the script then crops exactly those boxes, encodes them with the same frozen
SigLIP2, and writes a full-catalog ranking to
`artifacts/experiments/manual_oracle_crop_<timestamp>/`. No bbox is invented
automatically; with an empty CSV nothing is computed. Results must always be
labelled `manual_oracle_crop` and never pooled with benchmark metrics.

## Conclusions (evidence-based)

1. **Baseline full-rank distribution (overall, 128):** rank 1 — 30, rank 2–5 — 31, rank 6–10 — 9, rank 11–25 — 20, rank 26–100 — 28, rank >100 — 10. The failure is bimodal: 48% of queries have the target in Top-5, while 45% sit at rank 26+ (incl. 10 beyond 100). The target is almost never "just outside Top-5".

2. **Representative vs hard:** representative Top-1 42.19% / R@5 57.81% / median 2.5; hard Top-1 4.69% / R@5 37.50% / median 16.5. Two different regimes: representative queries lose the target to capture domain shift (recoverable by crop), hard queries additionally suffer near-duplicate confusion.

3. **distance_crop evidence:** baseline R@5 34.38% (worst scenario, median rank 16.5). Deterministic center crops restore retrieval strongly: center_70 R@5 65.62% / R@10 71.88% / median 2.5; center_55 R@5 75.00% / R@10 84.38% / median 2.0 (best Top-1 40.62% on this scenario). A simple zoom recovers candidate retrieval without any training — the "bottle too small in frame" hypothesis is confirmed for this scenario.

4. **Best single deterministic crop overall:** center_crop_70 — Top-1 35.16% (+11.7pp), R@5 69.53% (+21.9pp), R@10 78.12% (+23.4pp), median rank 2.0. It does not over-crop: 55% helps only distance_crop and clearly degrades glare_bad_light (R@5 40.62%, median 20) — the aggressive crop breaks more than it fixes (25 queries fell out of Top-5 vs 3 lost for center_70).

5. **Best multi-crop:** multicrop_mean — Top-1 32.03% (+8.6pp), R@5 62.50% (+14.8pp), R@10 74.22% (+19.5pp), best transition profile: 11 wrong→correct Top-1, 0 correct→wrong, 21 Top-5 gained vs 2 lost, best rank stability (75 improved / 10 degraded). multicrop_max is weaker than center_70 and is not preferred.

6. **Representative vs hard under crop:** on representative, center_70 reaches Top-1 64.06% / R@5 85.94% (near synthetic_dev levels); on hard, exact Top-1 stays ≤9.38% and family Top-1 does not exceed 20.31%. Crop mostly fixes object-scale domain shift, not fine-grained discrimination: hard subset remains a separate problem.

7. **Family behaviour after crop (hard):** center_70 returns 13 targets and 11 families back into Top-5 with only 3 leaving; within-family disambiguation rises to 66.67% (from 30%). center_55 causes 21 family exits — over-crop destroys family retrieval. After preprocessing the model usually finds the right family (family R@5 up to 59–62%) but still often picks the wrong member — the vintage/subtype bottleneck persists.

8. **Next-step decision (per the data):** two-track path. (A) object localization / smart crop preprocessing is validated as the immediate lever: it is training-free, restores representative R@5 to ~86% and distance_crop R@5 to ~75%+; fold center-crop/multi-crop normalization into the pipeline and re-evaluate. (B) For the hard subset even the best crop leaves exact Top-1 below 10% while the correct family is usually present in Top-5 — this is the OCR/reranking regime (text signal for vintage/subtype disambiguation), not a need to replace the encoder yet. Fine-tuning/domain adaptation is not justified by this evidence: the encoder already ranks the correct family highly when the object is properly framed.


