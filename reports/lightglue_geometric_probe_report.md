# LightGlue geometric matching probe

Run: RUN_ID. The probe stopped at the external-access gate before installing LightGlue or extracting any LightGlue features. It uses the previously frozen SIFT run only as the required control; no LightGlue scores are imputed.

## Baseline reproduction

The frozen baseline run reproduced exact candidate order for all 1,150 hard_v2 queries and all 128 generated_pilot32 queries. current_plus_sift_w0.40 is 1,057/1,150 (91.91%) on hard_v2 and 103/128 (80.47%) on generated. Generated-hard is 40/64 (62.50%). Generated R@5 remains 128/128; hard R@5 is 1,148/1,150. The same Top-5 was used and ground truth was not an inference input.

## Matcher/source gate

The official upstream identifies SIFT, ALIKED and DISK as supported feature families, and states LightGlue code/matcher weights are Apache-2.0, DISK follows Apache-2.0, and ALIKED is BSD-3-Clause (https://github.com/cvg/LightGlue/blob/main/README.md, https://github.com/cvg/LightGlue/blob/main/LICENSE). The official checkpoint source pattern appears in the upstream matcher implementation (https://github.com/cvg/LightGlue/blob/main/lightglue/lightglue.py); the ALIKED wrapper (https://github.com/cvg/LightGlue/blob/main/lightglue/aliked.py) and DISK wrapper (https://github.com/cvg/LightGlue/blob/main/lightglue/disk.py) describe their extractor weight paths.

The browser-accessible official README/source pages confirm support and licensing, but the execution environment could not retrieve the implementation or any checkpoint. Earlier attempts timed out on official shallow clone (90 s) and codeload archive (60 s). This continuation also timed out on git ls-remote (25 s), GitHub REST commit lookup, raw LICENSE fetch, and an official release checkpoint HEAD request (connect timeouts around 7 s). The exact commit SHA therefore could not be resolved; only the upstream main branch was observed. No unofficial code or checkpoints were used. The environment has Python 3.12.13, PyTorch 2.14.0+cu130, CUDA available, and an RTX 4060 8 GB, but LightGlue CUDA compatibility was not exercised.

## Results available before stop

| Method | Hard Top-1 | Generated Top-1 | Generated-hard | Homography valid hard / generated | Generated rescued / broken vs SIFT |
|---|---:|---:|---:|---:|---:|
| SIFT geometry-only | 89.74% | 76.56% | unavailable | 4,891/5,750 (85.06%) / 351/640 (54.84%) | — |
| current + SIFT w=0.40 | 91.91% | 80.47% (103/128) | 40/64 (62.50%) | same SIFT pair coverage | 0 / 0 (self control) |
| SIFT + LightGlue | not measured | not measured | not measured | not measured | not measured |
| ALIKED + LightGlue | not measured | not measured | not measured | not measured | not measured |
| DISK + LightGlue | not measured | not measured | not measured | not measured | not measured |

Generated scenario SIFT pair homography coverage from the saved pair diagnostics:

| Scenario | SIFT | LightGlue variants |
|---|---:|---:|
| distance_crop | 85/160 (53.12%) | not measured |
| glare_bad_light | 90/160 (56.25%) | not measured |
| handheld | 89/160 (55.62%) | not measured |
| slight_angle | 87/160 (54.37%) | not measured |

The SIFT-only geometric score beat the incumbent Top-1 on 9 of the 25 remaining generated errors. LightGlue target rank, target-vs-wrong margin, and the SIFT∪LightGlue oracle union are unknown. No fusion grid values or LightGlue error transitions can be reported.

## Runtime and resources

- LightGlue query extraction mean/p50/p95: not measured.
- Five LightGlue match calls mean/p50/p95: not measured.
- LightGlue RANSAC/homography mean/p95 and total extra-stage/full-pipeline p95: not measured.
- SIFT control extraction+matching (combined, saved probe): hard mean 64.94 ms / p95 172.01 ms; generated mean 175.62 ms / p95 234.60 ms. Estimated full-pipeline generated p95 for the frozen SIFT policy is 514.10 ms.
- LightGlue peak VRAM: not measured. Host GPU is RTX 4060 8 GB; production coexistence was not measured.
- Offline LightGlue reference caches for 2,042 references: not created because the run stopped before extractor initialization.

## Tests

An additional assertion now checks the available five-candidate SIFT control extracts query features once per query. Existing geometric-reranker tests cover fixed Top-5 boundaries, cache/image/extractor fingerprint checks, no ground-truth parameters in inference APIs, homography validation, deterministic fusion, and transition counts. Full suite: 286 passed, 3 existing scikit-learn FutureWarnings, 182.29 seconds. These checks do not substitute for LightGlue-specific runtime tests, which were not executable without the official package.

## Decision

**D. LIGHTGLUE_COULD_NOT_BE_EVALUATED_DUE_TO_EXTERNAL_ACCESS**

The result does not support replacing SIFT or estimating LightGlue's value. The exact external blocker is recorded in environment.json; all experiment artifacts distinguish the measured SIFT control from unrun matcher configurations. No fine-tuning or follow-on experiment was started.
