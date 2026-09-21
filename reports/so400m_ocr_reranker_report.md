# SO400M OCR reranker evaluation

Reranker: `reference_ocr_blend_alpha0.30_bonus0.05_penalty0.05_margin0.05` — frozen on calibration subsets only (seed 20260920).

## Baseline reproduction (image_only)

| Benchmark | Frozen Top-1 | Local Top-1 | Agreement | Reproduced |
| --- | --- | --- | --- | --- |
| generated_stress_dev_pilot32 | 0.7344 | 0.7344 | 1.0000 | True |
| hard_near_duplicate_dev_v2 | 0.8687 | 0.8687 | 1.0000 | True |
| synthetic_dev | 0.9510 | 0.9510 | 1.0000 | True |

## Main table

| Benchmark | SO400M Top1 | OCR Rerank Top1 | Delta pp | Rescued | Broken |
| --- | --- | --- | --- | --- | --- |
| synthetic_dev | 0.9510 | 0.9527 | +0.17 | 22 | 15 |
| hard_near_duplicate_dev_v2 | 0.8687 | 0.8791 | +1.04 | 16 | 4 |
| generated_stress_dev_pilot32 | 0.7344 | 0.7344 | +0.00 | 0 | 0 |

## Generated stress: methods

| Method | Overall Top1 | Representative | Hard | Vintage | Subtype |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.7344 | 0.8594 | 0.6094 | 0.65625 | 0.5625 |
| reranked | 0.7344 | 0.8594 | 0.6094 | 0.65625 | 0.5625 |

## OCR signal

| Dataset | Records | Any token | Vintage detected | Median tokens |
| --- | --- | --- | --- | --- |
| synthetic_dev | 4084 | 0.9829 | 0.4483 | 6 |
| hard_near_duplicate_dev_v2 | 1150 | 0.9913 | 0.4209 | 6 |
| generated_stress_dev_pilot32 | 128 | 0.9922 | 0.4766 | 9 |

## Oracles (diagnostic, not production metrics)

| Benchmark | Top-5 contains target | Text-oracle Top-1: metadata / reference / combined |
| --- | --- | --- |
| synthetic_dev | 0.9995 | 0.6824 / 0.8411 / 0.8477 |
| hard_near_duplicate_dev_v2 | 0.9983 | 0.6157 / 0.753 / 0.767 |
| generated_stress_dev_pilot32 | 1.0000 | 0.5703 / 0.7188 / 0.7266 |

## Transitions

| Benchmark | wrong→correct | correct→wrong | same | wrong→wrong | rescued/broken |
| --- | --- | --- | --- | --- | --- |
| synthetic_dev | 22 | 15 | 3869 | 178 | 1.47 |
| hard_near_duplicate_dev_v2 | 16 | 4 | 995 | 135 | 4.0 |
| generated_stress_dev_pilot32 | 0 | 0 | 94 | 34 | inf |

## Latency

| Benchmark | Retrieval mean ms | OCR probe mean ms | Rerank mean ms | Pipeline mean ms | Pipeline p95 ms | SLA<3s |
| --- | --- | --- | --- | --- | --- | --- |
| synthetic_dev | 41.3 | 237.0 | 0.16 | 278.5 | 262.6 | True |
| hard_near_duplicate_dev_v2 | 40.6 | 237.0 | 0.16 | 277.8 | 262.5 | True |
| generated_stress_dev_pilot32 | 42.34 | 237.0 | 0.17 | 279.5 | 264.0 | True |

## Policy ablation on FULL benchmark sets (diagnostic, Part H/R)

| Benchmark | Policy | alpha | Top-1 | Rescued | Broken |
| --- | --- | --- | --- | --- | --- |
| synthetic_dev | image_only |  | 0.951 | 0 | 0 |
| synthetic_dev | metadata_text_blend | 0.3 | 0.9527 | 24 | 17 |
| synthetic_dev | metadata_text_blend | 0.4 | 0.953 | 27 | 19 |
| synthetic_dev | reference_ocr_blend | 0.2 | 0.952 | 14 | 10 |
| synthetic_dev | reference_ocr_blend | 0.3 | 0.9527 | 22 | 15 |
| synthetic_dev | combined_text_blend | 0.3 | 0.954 | 20 | 8 |
| synthetic_dev | combined_text_blend | 0.4 | 0.954 | 28 | 16 |
| synthetic_dev | combined_vintage_blend | 0.3 | 0.954 | 20 | 8 |
| synthetic_dev | combined_vintage_blend | 0.4 | 0.954 | 28 | 16 |
| hard_near_duplicate_dev_v2 | image_only |  | 0.8687 | 0 | 0 |
| hard_near_duplicate_dev_v2 | metadata_text_blend | 0.3 | 0.873 | 16 | 11 |
| hard_near_duplicate_dev_v2 | metadata_text_blend | 0.4 | 0.8748 | 19 | 12 |
| hard_near_duplicate_dev_v2 | reference_ocr_blend | 0.2 | 0.8739 | 9 | 3 |
| hard_near_duplicate_dev_v2 | reference_ocr_blend | 0.3 | 0.8791 | 16 | 4 |
| hard_near_duplicate_dev_v2 | combined_text_blend | 0.3 | 0.8783 | 14 | 3 |
| hard_near_duplicate_dev_v2 | combined_text_blend | 0.4 | 0.8826 | 22 | 6 |
| hard_near_duplicate_dev_v2 | combined_vintage_blend | 0.3 | 0.8783 | 14 | 3 |
| hard_near_duplicate_dev_v2 | combined_vintage_blend | 0.4 | 0.8826 | 22 | 6 |
| generated_stress_dev_pilot32 | image_only |  | 0.7344 | 0 | 0 |
| generated_stress_dev_pilot32 | metadata_text_blend | 0.3 | 0.7344 | 0 | 0 |
| generated_stress_dev_pilot32 | metadata_text_blend | 0.4 | 0.7344 | 0 | 0 |
| generated_stress_dev_pilot32 | reference_ocr_blend | 0.2 | 0.7344 | 0 | 0 |
| generated_stress_dev_pilot32 | reference_ocr_blend | 0.3 | 0.7344 | 0 | 0 |
| generated_stress_dev_pilot32 | combined_text_blend | 0.3 | 0.7344 | 0 | 0 |
| generated_stress_dev_pilot32 | combined_text_blend | 0.4 | 0.7344 | 0 | 0 |
| generated_stress_dev_pilot32 | combined_vintage_blend | 0.3 | 0.7344 | 0 | 0 |
| generated_stress_dev_pilot32 | combined_vintage_blend | 0.4 | 0.7344 | 0 | 0 |
| hard_v2_by_family_kind:subtype_or_other | image_only |  | 0.8752 |  |  |
| hard_v2_by_family_kind:vintage | image_only |  | 0.7647 |  |  |
| hard_v2_by_family_kind:subtype_or_other | reference_ocr_blend | 0.3 | 0.8863 |  |  |
| hard_v2_by_family_kind:vintage | reference_ocr_blend | 0.3 | 0.7647 |  |  |

## OCR coverage groups (generated pilot32)

```json
{
 "scenario:distance_crop": {
  "median_token_count": 13,
  "queries": 32,
  "share_name_detected": 0.5312,
  "share_winery_detected": 0.4375,
  "share_with_any_token": 1.0,
  "share_with_detected_vintage": 0.5
 },
 "scenario:glare_bad_light": {
  "median_token_count": 7,
  "queries": 32,
  "share_name_detected": 0.375,
  "share_winery_detected": 0.4688,
  "share_with_any_token": 1.0,
  "share_with_detected_vintage": 0.5
 },
 "scenario:handheld": {
  "median_token_count": 7,
  "queries": 32,
  "share_name_detected": 0.4062,
  "share_winery_detected": 0.4062,
  "share_with_any_token": 1.0,
  "share_with_detected_vintage": 0.5
 },
 "scenario:slight_angle": {
  "median_token_count": 7,
  "queries": 32,
  "share_name_detected": 0.4375,
  "share_winery_detected": 0.4375,
  "share_with_any_token": 0.9688,
  "share_with_detected_vintage": 0.4062
 },
 "subset:hard": {
  "median_token_count": 7,
  "queries": 64,
  "share_name_detected": 0.3906,
  "share_winery_detected": 0.5,
  "share_with_any_token": 1.0,
  "share_with_detected_vintage": 0.3906
 },
 "subset:representative": {
  "median_token_count": 11,
  "queries": 64,
  "share_name_detected": 0.4844,
  "share_winery_detected": 0.375,
  "share_with_any_token": 0.9844,
  "share_with_detected_vintage": 0.5625
 }
}
```

## Reference OCR audit (2042 references)

- empty OCR: 37 (1.81%)
- median tokens: 7
- references with detected year: 982 (48.09%)
- suspicious failures listed: 27 (see reference_ocr_audit.json)
- visual sample: artifacts/ocr_cache/.../catalog_references/reference_ocr_sample.html

## Frozen selection evidence

Selection was lexicographic on calibration subsets (seed 20260920): pilot32 held-out Top-1 first —
ALL 32 grid configs and image_only tie at 0.7812 there, so the decision fell to hard_v2 calibration
Top-1, won by reference_ocr_blend alpha 0.30 (0.8978 vs 0.8817 image_only). Synthetic guard passed
(0.9600 vs 0.9600 calibration Top-1, 0.00pp regression).

On the full sets the selected reranker gives hard_v2 +1.04pp (16 rescued / 4 broken) and synthetic
+0.17pp (22 / 15) with pilot32 unchanged. The diagnostic ablation shows combined_text_blend alpha 0.40
would have scored higher on full hard_v2 (0.8826) — a lexicographic cost of freezing on calibration
subsets, reported honestly. Vintage subsets did not move under any policy.
