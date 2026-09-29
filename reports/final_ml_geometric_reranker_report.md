# Final ML geometric reranker report

## Table 1 — Error audit

Taxonomy uses frozen predictions, target rank, catalog/family metadata, pair-level reference-image similarity evidence, the accepted scenario field, and frozen OCR diagnostics. The HTML gallery exposes every query, target, current Top-1 and candidate references for manual review. `D` and `E` partition target-in-Top5 reranking errors versus retrieval failures; explanatory tags A/B/C/F/G/H can overlap. Category assignment never uses a filename as evidence.

| Category | Hard count | Generated count | % errors (hard / generated) | Target in Top5 % (hard / generated) |
|---|---:|---:|---:|---:|
| Correct family, wrong vintage | 11 | 10 | 7.9% / 29.4% | 90.9% / 100.0% |
| Correct family, wrong subtype / grape / subline | 104 | 6 | 74.8% / 17.6% | 100.0% / 100.0% |
| Visually near-identical packaging | 114 | 16 | 82.0% / 47.1% | 99.1% / 100.0% |
| Target is in Top-5, wrong Top-1 | 137 | 34 | 98.6% / 100.0% | 100.0% / 100.0% |
| Target outside Top-5 (retrieval failure) | 2 | 0 | 1.4% / 0.0% | 0.0% / 0.0% |
| Capture/domain-shift evidence | 0 | 34 | 0.0% / 100.0% | 0.0% / 100.0% |
| OCR/text evidence misleading | 4 | 0 | 2.9% / 0.0% | 100.0% / 0.0% |
| Unclear / mixed | 18 | 0 | 12.9% / 0.0% | 94.4% / 0.0% |

## Retrieval versus reranking

- hard_v2: 137/139 errors (98.6%) have target in Top-5; 2 retrieval failures.
- generated pilot32: 34/34 errors (100.0%) have target in Top-5; 0 retrieval failures.
- Near-identical package evidence (C): hard 114; generated 16.
- Scenario/domain-shift evidence (F): hard 0; generated 34. F means an accepted generated stress scenario and is contextual, not a causal claim.

The image gallery includes the query, target, current Top-1, all five candidates, SO400M scores, frozen OCR-reranker scores, family/year/subtype metadata, query OCR text, and assigned tags.

[Open visual error gallery](final_ml_error_audit.html)

## Table 2 — Main results

| Method | Hard Top-1 | Generated Top-1 | Hard rescued/broken | Generated rescued/broken | Added SIFT latency (hard / generated) |
|---|---:|---:|---:|---:|---:|
| A_SO400M_image_only | 0.8687 | 0.7344 | 4/16 | 0/0 | — |
| B_current_production | 0.8791 | 0.7344 | 0/0 | 0/0 | — |
| C_SIFT_geometry_only | 0.8974 | 0.7656 | 62/41 | 18/14 | +65 / +217 ms mean |
| D_SO400M_plus_geometry_w20 | 0.9096 | 0.7266 | 38/3 | 1/2 | +65 / +217 ms mean |
| E_current_plus_geometry_w0.10 | 0.9026 | 0.7344 | 27/0 | 1/1 | +65 / +217 ms mean |
| E_current_plus_geometry_w0.20 | 0.9130 | 0.7344 | 40/1 | 1/1 | +65 / +217 ms mean |
| E_current_plus_geometry_w0.30 | 0.9183 | 0.7500 | 47/2 | 4/2 | +65 / +217 ms mean |
| E_current_plus_geometry_w0.40 | 0.9191 | 0.8047 | 51/5 | 12/3 | +65 / +217 ms mean |
| conservative_policy | 0.8800 | 0.7344 | 2/1 | 0/0 | +65 / +217 ms mean |

## Top-5-conditioned exact accuracy

| Benchmark | Current production | Selected SIFT fusion | Target in Top-5 |
|---|---:|---:|---:|
| hard_near_duplicate_dev_v2 | 0.8807 | 0.9207 | 1148/1150 |
| generated_stress_dev_pilot32 | 0.7344 | 0.8047 | 128/128 |

## Geometry diagnostics

| Benchmark | Pair homography success | Target geo rank 1 among current errors | Target geo rank ≤2 | Median target−best-wrong geo margin | Geometry-only Top-1 |
|---|---:|---:|---:|---:|---:|
| hard_near_duplicate_dev_v2 | 85.1% (4891/5750) | 62/137 (45.3%) | 134/137 (97.8%) | 0.0000 | 0.8974 |
| generated_stress_dev_pilot32 | 54.8% (351/640) | 18/34 (52.9%) | 30/34 (88.2%) | 0.0485 | 0.7656 |

## Hard family and generated breakdowns

### hard_near_duplicate_dev_v2

- B_current_production: family_type:vintage: n=68, Top-1=76.5%, family_type:subtype: n=756, Top-1=90.7%, family_type:other: n=326, Top-1=83.7%, within_family_disambiguation: n=1071, Top-1=87.4%
- E_current_plus_geometry_w0.40: family_type:vintage: n=68, Top-1=83.8%, family_type:subtype: n=756, Top-1=94.0%, family_type:other: n=326, Top-1=88.7%, within_family_disambiguation: n=1071, Top-1=91.4%

### generated_stress_dev_pilot32

- B_current_production: scenario:distance_crop: n=32, Top-1=75.0%, scenario:glare_bad_light: n=32, Top-1=71.9%, scenario:handheld: n=32, Top-1=75.0%, scenario:slight_angle: n=32, Top-1=71.9%, subset:representative: n=64, Top-1=85.9%, subset:hard: n=64, Top-1=60.9%, family_type:vintage: n=32, Top-1=65.6%, family_type:subtype: n=32, Top-1=56.2%, family_type:other: n=0, Top-1=0.0%
- E_current_plus_geometry_w0.40: scenario:distance_crop: n=32, Top-1=81.2%, scenario:glare_bad_light: n=32, Top-1=84.4%, scenario:handheld: n=32, Top-1=81.2%, scenario:slight_angle: n=32, Top-1=75.0%, subset:representative: n=64, Top-1=98.4%, subset:hard: n=64, Top-1=62.5%, family_type:vintage: n=32, Top-1=65.6%, family_type:subtype: n=32, Top-1=59.4%, family_type:other: n=0, Top-1=0.0%

## Latency

| Benchmark | Query SIFT mean | 5-candidate matching mean | Geometry total mean / p95 | Estimated full pipeline mean / p95 | SLA <3s |
|---|---:|---:|---:|---:|---|
| hard_near_duplicate_dev_v2 | 42.6 ms | 22.1 ms | 64.7 / 153.7 ms | 342.5 / 431.5 ms | True |
| generated_stress_dev_pilot32 | 172.3 ms | 44.9 ms | 217.2 / 278.7 ms | 496.7 / 558.2 ms | True |
| synthetic_dev | 47.6 ms | 34.4 ms | 82.0 / 207.7 ms | 360.5 / 486.2 ms | True |

Reference SIFT features are precomputed offline for the full usable 2042-reference catalog; online latency excludes offline extraction and cache load. Matching latency includes all five candidates per query. Estimated total adds measured query extraction + five matchings to the frozen pipeline mean from the prior OCR-reranker run.

## Decision

**FIX_GEOMETRIC_RERANKER** — Fixed-grid fusion met the declared stop criterion and did not regress either primary benchmark.

## Synthetic regression sanity check

After selecting the fixed fusion on hard_v2 + generated only, synthetic_dev was 0.9527 Top-1 / 0.9995 R@5 / 0.9755 MRR; selected fusion is 0.9714 / 0.9995 / 0.9851. This is a post-selection check, not a tuning target.

Selected fixed-grid weight: 0.40; selected method: `E_current_plus_geometry_w0.40`.

### Required answers

1. Remaining errors: hard 139 and generated 34; taxonomy and per-query evidence are in the audit CSV/gallery.
2. Target already in Top-5: hard 137/139; generated 34/34.
3–12. SIFT separation, geometry-only/fused Top-1, deltas, rescues/breaks, vintage/subtype metrics, homography rate and added latency are in the tables above.
13. Integrate SIFT: yes, fixed weight 0.4.
14. Continue ML: yes, only this validated SIFT fusion.

**Final recommendation:** CONTINUE ML

The hard_v2 set reuses synthetic-derived query images and the generated pilot has only 32 products repeated across four scenarios. Treat these as strong diagnostic evidence for this fixed signal; collect real field queries before exploring additional ML methods.

[Error audit](final_ml_error_audit.md) · [Error gallery](final_ml_error_audit.html) · [Geometry failure gallery](../artifacts/experiments/final_ml_geometric_reranker_20260923T082259Z/geometry_failure_gallery.html)

[Experiment artifacts](../artifacts/experiments/final_ml_geometric_reranker_20260923T082259Z)
