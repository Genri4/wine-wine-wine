# Experiment history

## 2026-09-20 — so400m_ocr_reranker milestone

- Implemented local OCR + text/metadata reranking over the frozen `siglip2_so400m_384` Top-5: OCR never participates in candidate generation; it only supplies conservative evidence inside the shortlist. New modules: `text_normalization.py` (NFKC/casefold/ё→е/punctuation-safe normalization preserving 4-digit years, transliteration reuse from catalog, deterministic vintage range filter 1900–2030), `text_signals.py` (7 bounded signals via RapidFuzz over Cyrillic+transliterated keys, confidence cutoff 0.5), `ocr_reranker.py` (5 fixed policies, per-query min-max image normalization, global alpha, bounded vintage bonus/penalty, text-margin guard that keeps the image winner without strong evidence), `ocr_engine.py` (PaddleOCR wrapper).
- OCR engine (Part A): PaddleOCR 3.7.0 / paddlex 3.7.2 with paddlepaddle-gpu 3.3.1 (cu126), PP-OCRv5_server_det + eslav_PP-OCRv5_mobile_rec, GPU, fully local; CPU mode required `enable_mkldnn=False` (paddle 3.3.1 PIR/oneDNN crash) and was ~6x slower. Installing the GPU wheel overwrote torch's `nvidia-nccl-cu13` library (shared `nvidia/nccl/lib` path) — restored by reinstalling `nvidia-nccl-cu13==2.30.7`.
- OCR caches built once and never re-run: 2042 references (1.81% empty, median 7 tokens, 48.09% with a detected year; 50-reference visual sample) + 5362 query images across three benchmarks (98.3–99.2% with tokens, 42–48% with detected vintage, 28–44% name detection — stylized wine-name fonts read worst). Coverage per scenario in `ocr_coverage.json`.
- Baseline reproduction (policy image_only) reproduced the frozen bake-off exactly: Top-1 agreement 1.0000 on all three benchmarks; 95.10% / 86.87% / 73.44%, R@5 99.95% / 99.83% / 100% — no alignment drift.
- Calibration (product-level pilot32 16/16 stratified, family-level hard_v2 128/128, 400 synthetic products; seed 20260920, fingerprinted in `calibration_split.csv`): 32 grid configs, pre-declared ≤1pp synthetic guard, lexicographic objective. All configs tied image_only on pilot32 held-out Top-1 (78.12%), so selection fell to hard_v2 calibration Top-1 — frozen: `reference_ocr_blend, alpha 0.30, vintage ±0.05, text margin 0.05`.
- Full-set results: hard_v2 86.87% → 87.91% (+1.04pp, 16 rescued / 4 broken), synthetic 95.10% → 95.27% (+0.17pp, 22 / 15), pilot32 73.44% unchanged (0 / 0). MRR improved on all three. R@5/R@10 invariant by construction (Top-5 permutation only).
- Diagnostic ablation on full sets: `combined_text_blend alpha 0.40` would score higher on hard_v2 (88.26%, 22/6) and synthetic (95.40%, 20/8 at 0.30) — a lexicographic cost of freezing on calibration halves, kept as evidence, not adopted. Vintage subsets did not move under ANY policy (pilot32 vintage 65.63% unchanged, hard_v2 vintage 76.47% unchanged): all hard_v2 gain is subtype disambiguation (87.52% → 88.63%).
- Oracle ceilings explain the generated-stress outcome: a perfect text judge reaches only 72.7% (combined) / 75.0% (with vintage) Top-1 on pilot32 vs 73.44% image baseline — the OCR signal is roughly at parity, not above it, so conservative fusion correctly refuses to reorder. OCR reads generated stress images fine (99.2% tokens, median 9) but not discriminatively.
- Latency: retrieval ~41 ms + OCR ~237 ms probe mean (p50 189 ms) + rerank 0.16 ms ≈ 280 ms pipeline mean, p95 ~264 ms — far below the 3 s SLA. Peak VRAM: retrieval 2226.9 MB (torch), OCR ~2.1 GB device-wide.
- Verdict: OCR reranking FIXed as a conservative post-retrieval stage for near-duplicate disambiguation (net positive on clean/hard, strictly neutral on generated, SLA-safe); vintage disambiguation remains unsolved — the bottleneck is discriminative OCR of stylized name/vintage typography, not fusion weights. Artifacts: `artifacts/experiments/so400m_ocr_reranker_20260920T193925Z/` (+ retrieval dumps `..._192004Z/`); reports: `reports/so400m_ocr_reranker_report.md`, `reports/ocr_reranker_error_analysis.html`. 66 new tests (219 total pass).

## 2026-09-19 — confidence_gated_v1 milestone

- Implemented a confidence-gated multi-crop fallback (`src/recognition/gated_strategy.py`): full-image ranking first, then crop85+crop70 fallback in one batch=2 forward (full embedding reused, aggregation identical to preprocessing_v1); gate reads only the full-ranking score distribution.
- Product-level calibration split (seed 20260919, fingerprinted): pilot32 16/16 products stratified representative/vintage/subtype, plus deterministic 400-product synthetic and 120-product hard_v2 calibration subsets; 29 bounded gate candidates evaluated on calibration subsets only, lexicographic objective (clean guard -> generated gain -> fallback rate).
- Frozen gate: `fallback if top1_score < 0.8824` (3/29 candidates passed the clean guard; margin rules failed because clean margins are small in absolute terms).
- Results: synthetic Top-1 89.40% -> 88.10% (v1: 85.63%), fallback 19.2%, 13.8 ms; hard_v2 Top-1 68.43% -> 68.70% (v1: 66.17%), fallback 20.5%; pilot32 Top-1 29.69% and R@5 62.50% - identical to unconditional v1 with 93.75% fallback; held-out products confirmed the full gain transfer (R@5 45.31% -> 62.50%).
- Gate diagnostics: pilot32 95 useful triggers / 25 unnecessary / 3 missed; synthetic fallback set is net-negative (28 rescued vs 81 broken) - the honest cost of the global threshold (-1.30pp synthetic Top-1, slightly above the pre-declared 1pp guard).
- Verdict: confidence_gated_v1 fixed as the retrieval default; hard-subset vintage/subtype disambiguation remains the next (OCR/reranking) milestone. 23 new tests (136 total pass).
- Artifacts: `artifacts/experiments/siglip2_confidence_gated_v1_20260919T204926Z/`; report: `reports/confidence_gated_v1_evaluation.md`.

## 2026-09-19 — preprocessing_v1 controlled comparison

- Added reusable query view strategies (`src/recognition/view_strategy.py`): `baseline_full` (one full view) and `preprocessing_v1` (full + center 85% + center 70%, one batched forward per query, mean similarity). Reference embeddings and benchmark datasets unchanged.
- Ran a controlled comparison on all three frozen benchmarks (`scripts/run_preprocessing_v1_comparison.py`): 4084 synthetic_dev, 1150 hard_v2, 128 pilot32 queries, both strategies, full-catalog ranks, real per-query latency.
- pilot32 improved safely: Top-1 23.44% -> 29.69%, R@5 47.66% -> 62.50%, R@10 54.69% -> 71.88%, median rank 6.5 -> 3.0; transitions 8 wrong->correct / 0 correct->wrong Top-1, 20 gained / 1 lost Top-5.
- But synthetic_dev regressed (Top-1 89.40% -> 85.63%, 232 correct->wrong flips) and hard_v2 regressed (68.43% -> 66.17%): mean-with-crops dilutes exact-match margins on reference-framed queries.
- Latency: 12.2 -> 25.3 ms mean (2.08x), p95 28.7 ms - far below the 3 s SLA.
- Verdict: preprocessing_v1 is NOT adopted as unconditional default (fails pre-declared 1pp regression guards); the data motivates a confidence-gated conditional policy as a separate future milestone. 22 new tests (113 total pass).
- Artifacts: `artifacts/experiments/siglip2_preprocessing_v1_20260919T194507Z/`; report: `reports/preprocessing_v1_evaluation.md`.

## 2026-09-13 — baseline implementation

- Added a pretrained visual-embedding retrieval baseline and offline catalog index builder.
- Added prediction and evaluation runners with configurable, uncalibrated cosine threshold.
- Quality metrics were not measured: the organizer dataset, target metric, split, and catalog format are not available yet.

## 2026-09-13 — baseline encoder comparison support

- Kept ResNet-18 and ResNet-50 and added DINOv2 (`facebook/dinov2-base`) and SigLIP (`google/siglip-base-patch16-224`).
- All four models are selected by one `model_name` value and produce L2-normalized embeddings with model-specific preprocessing.
- Migrated catalog embeddings from JSON to PyTorch `.pt` indexes with tensor embeddings and adjacent item metadata.
- Added run JSON fields for model name, accuracy, known top-1 accuracy, top-K recall, unknown detection accuracy, average latency, and sample count.
- No quality comparison was run: the organizer dataset, split, and target metric are still unavailable.

## 2026-09-19 — pilot32 crop diagnostics milestone

- Added a diagnostic-only frozen-encoder experiment for the pilot32 generated-stress failure: full-catalog target ranks, fixed center-crop ablation (85/70/55) and fixed multi-crop ensembles (max/mean), no training/OCR/reranker/detector and no target usage in crop policy.
- Verified the current SigLIP2 preprocessing: direct square stretch to 224x224 without center crop; it distorts but never cuts the bottle.
- Full-rank baseline on 128 queries: rank 1 - 30, 2-5 - 31, 6-10 - 9, 11-25 - 20, 26-100 - 28, >100 - 10; representative vs hard split confirmed as two different regimes (42.19% vs 4.69% Top-1).
- center_crop_70 was the best single preprocessing (Top-1 35.16%, R@5 69.53%), multicrop_mean had the safest transition profile (11 wrong-to-correct, 0 correct-to-wrong Top-1); center_crop_55 helped only distance_crop and destroyed glare_bad_light/hard retrieval.
- Family diagnostics: crop restores family retrieval (family R@5 up to 59.38%) but exact within-family Top-1 stays low; hard-subtype disambiguation remains the bottleneck.
- Added `src/recognition/crop_diagnostics.py`, `scripts/run_pilot32_crop_diagnostics.py`, `scripts/oracle_manual_crop.py` (manual bbox oracle tooling, not executed), 32 new tests (91 total pass); artifacts in `artifacts/experiments/siglip2_generated_stress_pilot32_crop_diagnostics_20260919T163711Z/`; reports in `reports/generated_stress_pilot32_diagnostic_report.md` and `reports/generated_stress_pilot32_error_analysis.html`.

## 2026-09-14 — CUDA environment and experiment tooling

- Replaced the local CPU PyTorch pair with `torch==2.14.0+cu130` and `torchvision==0.29.0+cu130`; CUDA inference was verified on an NVIDIA GeForce RTX 4060.
- Added `scripts/cache_models.py` to initialize and cache all four pretrained baseline encoders sequentially.
- Added dataset diagnostics (`scripts/analyze_dataset.py`) and a sequential four-model benchmark (`scripts/benchmark.py`).
- Evaluation now keeps a separate error artifact with the query image, expected/predicted item, and top-5 candidates with scores.
- No real-dataset metrics were recorded: the organizer dataset format, split, target metric, and evaluation rules remain unknown.
