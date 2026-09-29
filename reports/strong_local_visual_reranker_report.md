# Strong local visual Top-5 reranking report

Run: `strong_local_visual_reranker_final_20260923T1000Z`. Frozen candidate generation and OCR pipeline; every local signal reranks exactly the existing five candidates.

## Baseline reproduction

Current OCR reranker reproduced exactly: hard 1150/1150 and generated 128/128. The fixed SIFT 0.40 ranking was also recomputed and matched the previous Top-5 order for every query.

## Table 1 — Primary results

| Method | Hard Top-1 | Generated Top-1 | Generated hard | Rescued / broken vs SIFT | Estimated p95 |
|---|---:|---:|---:|---:|---:|
| `B_current_production` | 87.91% | 73.44% | 60.94% (64) | 3 / 12 | — |
| `current_plus_sift_w0.40` | 91.91% | 80.47% | 62.50% (64) | 0 / 0 | 514 ms |
| `current_plus_sift_plus_aligned_so400m_w0.20` | 92.09% | 80.47% | 60.94% (64) | 1 / 1 | 1077 ms |
| `current_plus_sift_plus_aligned_pe_core_w0.20` | 92.09% | 80.47% | 60.94% (64) | 1 / 1 | 1800 ms |
| `current_plus_sift_plus_multiview_mean_w0.20` | 91.74% | 76.56% | 60.94% (64) | 0 / 5 | 584 ms |
| `current_plus_sift_plus_multiview_max_w0.20` | 91.74% | 78.91% | 60.94% (64) | 0 / 2 | 584 ms |
| `combined_local_signals_w0.10` | 92.00% | 80.47% | 62.50% (64) | 0 / 0 | 2314 ms |

Frozen SIFT baseline: hard 91.91%, generated 80.47% (103/128); the generated target requires at least 116/128, or +13 net correct queries. R@5 is unchanged because candidate sets are fixed.

## Matcher comparison and homography coverage

| Matcher | Hard valid homography | Generated valid homography | Geometry-only Top-1 (hard / gen) | Current+SIFT Top-1 (hard / gen) | Generated rescued / broken | p95 (generated) |
|---|---:|---:|---:|---:|---:|---:|
| SIFT | 4891/5750 (85.06%) | 351/640 (54.84%) | 89.74% / 76.56% | 91.91% / 80.47% | 0 / 0 | 514 ms |
| LightGlue + SIFT | not measured | not measured | not measured | not measured | — | — |
| LightGlue + ALIKED | not measured | not measured | not measured | not measured | — | — |
| LightGlue + DISK | not measured | not measured | not measured | not measured | — | — |

LightGlue variants remain unmeasured because the official source/weights download timed out; their missing scores are not imputed.

## Table 2 — Oracle coverage of remaining generated errors

| Signal | Remaining-error target wins | Union contribution/status |
|---|---:|---:|
| `sift_baseline_target_beats_incumbent` | 9 | measured |
| `lightglue_sift` | not measured | not_run_github_transfer_timeout |
| `lightglue_aliked` | not measured | not_run_github_transfer_timeout |
| `lightglue_disk` | not measured | not_run_github_transfer_timeout |
| `aligned_so400m` | 8 | measured |
| `aligned_pe_core` | 7 | measured |
| `multiview_mean` | 4 | measured |
| `multiview_max` | 2 | measured |
| `oracle_union_measured_signals` | 12 | 12 |

Oracle win means the signal gives the target a raw score above the frozen current+SIFT Top-1. It is an optimistic ceiling, not a realized rescue. Signals are measured within the same five candidates. Of 25 remaining generated errors, the measured union covers 12; reaching 90% requires 13 net correct queries, so these signals alone have an 89.84% oracle ceiling. LightGlue was omitted, so the measured union excludes it.

## Generated scenarios and subsets

| Method | Representative | Hard | Distance crop | Glare | Handheld | Slight angle |
|---|---:|---:|---:|---:|---:|---:|
| `B_current_production` | 85.94% (n=64) | 60.94% (n=64) | 75.00% (n=32) | 71.88% (n=32) | 75.00% (n=32) | 71.88% (n=32) |
| `current_plus_sift_w0.40` | 98.44% (n=64) | 62.50% (n=64) | 81.25% (n=32) | 84.38% (n=32) | 81.25% (n=32) | 75.00% (n=32) |
| `current_plus_sift_plus_aligned_so400m_w0.20` | 100.00% (n=64) | 60.94% (n=64) | 81.25% (n=32) | 84.38% (n=32) | 78.12% (n=32) | 78.12% (n=32) |
| `current_plus_sift_plus_aligned_pe_core_w0.20` | 100.00% (n=64) | 60.94% (n=64) | 81.25% (n=32) | 84.38% (n=32) | 78.12% (n=32) | 78.12% (n=32) |
| `current_plus_sift_plus_multiview_mean_w0.20` | 92.19% (n=64) | 60.94% (n=64) | 78.12% (n=32) | 78.12% (n=32) | 78.12% (n=32) | 71.88% (n=32) |
| `current_plus_sift_plus_multiview_max_w0.20` | 96.88% (n=64) | 60.94% (n=64) | 78.12% (n=32) | 81.25% (n=32) | 81.25% (n=32) | 75.00% (n=32) |
| `combined_local_signals_w0.10` | 98.44% (n=64) | 62.50% (n=64) | 81.25% (n=32) | 84.38% (n=32) | 81.25% (n=32) | 75.00% (n=32) |

## Generated error transitions

Counts are `rescued / broken` relative to current+SIFT. `Vintage` and `Subtype` are the generated `family_type` slices.

| Method | Overall | Representative | Hard | Vintage | Subtype |
|---|---:|---:|---:|---:|---:|
| `B_current_production` | 3 / 12 | 0 / 8 | 3 / 4 | 2 / 2 | 1 / 2 |
| `current_plus_sift_w0.40` | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| `current_plus_sift_plus_aligned_so400m_w0.20` | 1 / 1 | 1 / 0 | 0 / 1 | 0 / 0 | 0 / 1 |
| `current_plus_sift_plus_aligned_pe_core_w0.20` | 1 / 1 | 1 / 0 | 0 / 1 | 0 / 0 | 0 / 1 |
| `current_plus_sift_plus_multiview_mean_w0.20` | 0 / 5 | 0 / 4 | 0 / 1 | 0 / 0 | 0 / 1 |
| `current_plus_sift_plus_multiview_max_w0.20` | 0 / 2 | 0 / 1 | 0 / 1 | 0 / 0 | 0 / 1 |
| `combined_local_signals_w0.10` | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

## Matcher inventory and execution limits

Official upstream: [LightGlue README](https://github.com/cvg/LightGlue/blob/main/README.md) and [LightGlue license](https://github.com/cvg/LightGlue/blob/main/LICENSE). The README documents SIFT, ALIKED and DISK support, Apache-2.0 code/matcher weights, Apache-2.0 DISK weights, and BSD-3-Clause ALIKED weights ([ALIKE license](https://github.com/Shiaoming/ALIKE/blob/main/LICENSE)). Runtime inventory: `official source clone and codeload archive fetch timed out; not installed, no matcher metrics`. The official Git clone and archive download both stalled/timed out in this shell, so no LightGlue variant was silently replaced with another implementation. Strong-matcher coverage is therefore unavailable for this run.

This machine reports RTX 4060 8 GB and PyTorch CUDA 13.0 (`torch 2.14.0+cu130`). Upstream publishes performance on RTX 3080, not a direct RTX 4060 validation. The source/API claims support CUDA, but an actual RTX 4060 LightGlue compatibility/latency probe could not be performed without fetching the official code/weights.

## Latency

Latency adds the frozen pipeline estimate to measured query SIFT, five matches, overlap warp, local encoding and/or two crop encodings. Reference embeddings are precomputed. The selected combined policy's estimated total generated p95 is 2,314 ms, under the 3 s guard; the total combines the previously measured frozen-pipeline p95 with per-query stage measurements from this run. Aligned SO400M and PE-Core tied on generated Top-1 (80.47%, 1 rescued / 1 broken for each individual method); SO400M had lower p95 (1,077 ms vs 1,800 ms). Multiview mean/max reached 76.56%/78.91%, both below the SIFT baseline’s 80.47%. See `latency_summary.csv` for means, p95s, and each component.

## Decision

**KEEP_SIFT** — The measured local signals did not add enough honest rescue potential to justify replacing the frozen SIFT policy.

The best fixed-grid entry was `combined_local_signals_w0.10`, but it tied SIFT at generated Top-1 with 0/0 transitions. Production recommendation remains `current_plus_sift_w0.40`. The post-selection synthetic sanity result for this unchanged SIFT 0.40 policy is 97.14% Top-1 / 99.95% R@5, recorded in `reports/final_ml_geometric_reranker_report.md`; no additional synthetic tuning was performed.

The query/gallery and artifacts are diagnostic on the generated pilot32 (32 products repeated across four stress scenarios) and hard_v2 (synthetic-derived queries). Do not treat them as real field validation. No follow-on fine-tuning or integration was started.

Artifacts: `artifacts/experiments/strong_local_visual_reranker_final_20260923T1000Z/`. Error examples: [reports/local_visual_reranker_errors.html](local_visual_reranker_errors.html).
