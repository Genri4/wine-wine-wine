# Experiment history

## 2026-09-25 — desktop product UI and Smart Retry checkpoint

- Added personal wine collection: save a recognized wine as “want to try” or “tried” with a 1–5 rating; browse, edit and remove saved items. Data persists in this browser's local storage only.
- Added sequential Compare: save one recognized bottle, scan a second, and view their catalog attributes side by side. Selection is session-only; no recognition or model behavior changed.
- Simplified the desktop presentation: removed the large decorative scanner art and outer app frame, reduced repeated copy, and collapsed Match for You until requested. Recognition and retry behavior are unchanged.
- Implemented desktop-only scanner screens with local image upload, Smart Retry guidance, one-card success and no-match states. Mobile adaptation remains explicitly out of scope.
- Smart Retry now checks image decoding, minimum resolution and strong blur locally, then uses the frozen R8 + OCR + SIFT runtime for recognition and evidence-specific retry guidance. The demo outcome selector is removed.
- Added Match for You: up to three catalog cards ranked by exact category/grape/region matches from the processed manifest; percentage means selected-attribute match, not wine quality. No backend persistence.
- ML experiments remain frozen; validated R8 epoch 5 is the selected runtime. The desktop MVP is integrated and running locally. Human review of the new 100-image dataset is still needed before open-set thresholds can be field-calibrated.
- Durable continuation notes: `reports/product_ui_checkpoint_20260925.md`; preview: `http://127.0.0.1:8765/web/`.

## 2026-09-22 — vintage disambiguation / targeted OCR milestone

- Built a diagnostic vintage challenge slice v1 (hard_v2: 46 queries / 17 vintage families; generated: 23 queries / 8 products): target in a vintage family AND in SO400M Top-5 AND a same-family competitor with a different known year is also in Top-5. Not a headline benchmark.
- Candidate year evidence with provenance (catalog_metadata / product_title / reference_ocr / multiple_sources_agree; conflicts -> unknown_conflict): 934/2042 products have a year; hard vintage-family products 33/34; pilot32 vintage 8/8. Nothing invented.
- Oracle ceiling (fixed double-counting semantics): a perfect query year rescues ALL baseline-wrong vintage queries - 8/8 on hard_v2, 6/6 on generated; ceiling 100% on both slices (7/46 hard queries ambiguous: two family members share the target year).
- Detector-box retry (re-recognize query's own PP-OCRv5 det boxes at 1x/2x/4x with the frozen eslav recognizer): correct-year 56.5% -> 58.7% (hard, 4x) and 47.8% -> 52.2% (generated) - marginal; wrong-year stays 0-3; correct-year rate itself is the bottleneck.
- Conservative family-scoped vintage rerank (swap only same-family members with different known years, confident query year): +1 rescued / 0 broken on hard_v2 challenge, 0/0 on generated.
- Reference-guided year crop (SIFT+ratio-test+RANSAC, validity gates): alignment success 100% hard / 95.7% generated (median 227/29 good matches); projected-crop year reads 53/61 aligned hard crops, 30 matching the candidate's own year. Conservative candidate-specific evidence allowed 1 swap: rescued the fanagoriya-primum-alveus-brut-2014 case (2016 product at Top-1, projected crops read 2014 on both target and top1) - +1/-0.
- Latency: full-image detection 362 ms mean / SIFT alignment 486 ms per candidate; the stage is conditional (same-family vintage ambiguity only, ~10% of hard queries), pipeline stays far below the 3 s SLA.
- Verdict: ADD REFERENCE-GUIDED VINTAGE STAGE as a conditional conservative layer (C). Gain is small but strictly non-negative (+1/-0 hard, 0/-0 generated); scales to the full +8/-0 hard ceiling if correct-year OCR ever improves. Not adopted as unconditional default; production trigger not wired into the API path. 23 new tests (257 total pass).
- Artifacts: `artifacts/experiments/vintage_disambiguation_20260922T1/`; report: `reports/vintage_disambiguation_report.md`.

## 2026-09-21 — OCR + reranker bake-off

- Built OCR config registry (`ocr_engine.OCR_CONFIGS`: current_eslav / cyrillic / paddleocr_vl) with per-config deterministic cache keys; eslav key unchanged (backward compatible). Build script generalized (`--ocr-config`), incremental append+resume caches (crash-safe).
- Built cyrillic cache (2042 refs + 5362 queries); VL cache: references + hard_v2 + generated (synthetic skipped - ~4.9 s/image on RTX 4060, ~8.5 h for a full cache; disclosed, not silently substituted). VL wrapper parses `parsing_res_list` blocks (confidence 1.0, no per-line confidences in VL).
- Discriminative diagnostics (not just coverage): inside the frozen SO400M Top-5, target-vs-best-wrong text margin - eslav 0.054/59.9% beats-best-wrong on hard_v2, cyrillic identical in usefulness (57.6%), VL far weaker (0.003 margin, 41.1%); VL coverage 75.8% vs 99.2% non-empty on generated.
- Vintage audit: 103/2042 (5.0%) products have a title year; ref-OCR year coverage 48.1%; title-year x ref-OCR-year agreement 94.9% (n=78) - ref-OCR years usable as derived evidence with provenance.
- Structured reranker: 21 bounded tabular features (image margins, fuzzy/token/IDF-weighted overlap, numeric overlap, vintage, query quality) + standardized LogisticRegression; trained only on calibration units (product-level generated, family-level hard_v2, product-level synthetic; seed 20260920).
- BGE cross-encoder (BAAI/bge-reranker-v2-m3, apache-2.0, fp16): sigmoid scores over (query OCR text, candidate doc) pairs; fusion grid image/text 0.9/0.1-0.7/0.3 on hard+generated.
- Matrix results (Top-1): hard_v2 - eslav+blend 87.91% (16/4), cyrillic+BGE0.3 88.17% (60/45, dirty profile), eslav+structured 87.30% (49/44), VL+blend 86.26% (4/11 net-negative); synthetic - structured up to 95.54%, blend 95.27%; generated - 73.44% unchanged except VL+structured 74.22% (40/64 generated-hard, +1 query at 4.1 s/image latency).
- Feature importance: image_margin_top1 dominates (-9.09), then ref_ocr_year_match (+0.88), query_has_year (-0.66) - the model leans on image margin and year consistency, OCR adds a smaller bounded correction.
- Latency: eslav OCR 0.27 s mean (p95 0.23), cyrillic 0.17 s, VL 4.14 s, BGE +133 ms/query; pipeline mean ~0.4 s (SLA 3 s safe); peak VRAM BGE 1108 MB.
- Verdict: KEEP CURRENT OCR + CURRENT RERANKER (eslav + reference_ocr_blend alpha 0.30). OCR upgrade gives no gain; structured LR does not beat the hand-written blend on rescued/broken; BGE's +0.26pp hard comes with generated regression and latency. Vintage/subtype disambiguation bottleneck is OCR's inability to read stylized vintage typography better - a VLM/fine-tuned-recognition question, outside this milestone. 15 new tests (234 total pass).
- Artifacts: `artifacts/experiments/ocr_reranker_bakeoff_20260921T1/` (combos, signal_cache, bge, structured), diagnostics `artifacts/experiments/ocr_reranker_bakeoff_diag/`; report `reports/ocr_reranker_bakeoff_report.md`.

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

## 2026-09-23 — candidate-constrained vintage recognition milestone

- Rebuilt SIFT-aligned year crops deterministically from the frozen reference_guided_per_query.csv (69 queries: 61+21 query crops, 16+3 reference crops). Constrained task framing: choose one allowed year from the same-family Top-5 candidates' catalog years; no target identity at inference.
- Methods compared on decided queries: current eslav OCR 12/12+12/12; SO400M crop matching 25/25+12/12 (100%); PE-Core 25/25+12/12; DINOv2 local-patch 64-67% with 11 broken (local-matching hypothesis rejected); digit-only CRNN trained on 2988 synthetic year crops with constrained CTC decoding 56-67% (synthetic-to-real domain gap); VLM ceiling Qwen2-VL-2B-Instruct (apache-2.0, fp16, local) 24/25+12/12 = 96-100%.
- DECISIVE finding: the bottleneck is crop availability, not recognition. 21/46 hard and 11/23 generated challenge queries have no year crop (eslav never read the reference year, so no reference year box existed). 6/8 hard and 6/6 generated baseline-wrong are in this no_crop group and unreachable by ANY crop method.
- Family-safe swap: hard rescued 1 / 0 broken (2014/2016 fanagoriya case); the ekstra-2017 swap is correctly blocked (both members share year 2017 - subtype case). Generated 0/0 (no crops for wrong queries).
- Latency: crop matching 12-15 ms/query, digit ~2 ms, VLM 190-820 ms (ceiling only). Peak VRAM VLM ~2.5 GB. Fixed environment: nvidia-cudnn-cu13 reinstalled (paddle install had stubbed libcudnn.so.9, breaking torch LSTM).
- Verdict: USE VISUAL REFERENCE-CROP MATCHING (SO400M/PE-Core) as the chooser inside the family-scoped vintage stage, PLUS an offline VLM pass over vintage reference images to create missing year boxes - that moves 6/8 hard and 6/6 generated baseline-wrong queries from unreachable to reachable. VLM as production runtime rejected (latency), but its 96-100% accuracy validates that crops carry the signal. 257 tests pass.
- Artifacts: artifacts/experiments/candidate_constrained_vintage_20260922T1/; report reports/candidate_constrained_vintage_report.md; modules src/recognition/constrained_vintage.py, year_recognizer.py, vlm_year.py.

## 2026-09-23 — final ML sanity check: error audit + Top-5 SIFT geometry

- Reproduced the frozen current production candidate exactly from saved Top-5 signals and the selected reference-OCR fusion: 1150/1150 hard_v2 and 128/128 generated pilot32 Top-5 orders matched. No candidate generation changed and no ground truth enters inference.
- Audited 173 current Top-1 errors: hard_v2 139 (137 target-in-Top-5, 2 retrieval failures); generated pilot32 34 (34 target-in-Top-5, 0 retrieval failures). Pair-level reference-image evidence flags near-identical packaging in 114 hard and 16 generated errors; misleading OCR evidence in 4 hard errors. Generated F tags are scenario context, not proof of cause.
- Precomputed SIFT descriptors offline for all 2042 usable references with an OpenCV/config/image-SHA256 cache fingerprint. Scored exactly five current candidates per query: hard 5750 pairs, generated 640 pairs. Valid homography rates: 4891/5750 (85.1%) hard; 351/640 (54.8%) generated.
- Fixed-grid winner `current production + 0.40 × normalized geometry`: hard_v2 87.91% -> 91.91% (+4.00 pp); generated pilot32 73.44% -> 80.47% (+7.03 pp). R@5 stayed 99.83% / 100%. Transitions: hard 51 rescued / 5 broken; generated 12 / 3; combined 63 / 8 (7.9:1).
- Geometry-only Top-1: 89.74% hard and 76.56% generated. Among current Top-1 errors with target in Top-5, target ranked first by geometry in 45.3%/52.9%, and in the top two in 97.8%/88.2% (hard/generated).
- Family effects for selected fusion: hard vintage 76.5% -> 83.8%, subtype 90.7% -> 94.0%, other near-duplicate families 83.7% -> 88.7%; generated vintage 65.6% -> 65.6%, subtype 56.2% -> 59.4%. Synthetic post-selection regression check: 95.27% -> 97.14% Top-1; R@5 unchanged at 99.95%.
- Added online geometry latency (query extraction + all five matchings): 64.7 ms mean / 153.7 ms p95 hard; 217.2 / 278.7 ms generated. Estimated full pipeline p95: 431.5 / 558.2 ms, below the 3 s SLA. Full test suite: 274 passed (3 unrelated sklearn deprecation warnings).
- Verdict: **FIX GEOMETRIC RERANKER**, fixed weight 0.40. Recommendation: **CONTINUE ML only to wire this validated signal into the recognition path**; do not start another model/feature research track before collecting field queries. The app production path is not yet wired to this experiment output.
- Artifacts: `artifacts/experiments/final_ml_geometric_reranker_20260923T082259Z/`; reports: `reports/final_ml_error_audit.md`, `reports/final_ml_error_audit.html`, `reports/final_ml_geometric_reranker_report.md`; implementation: `src/recognition/geometric_reranker.py`, `scripts/run_final_ml_sanity_check.py`; tests: `tests/test_geometric_reranker.py`.

## 2026-09-23 — strong local visual Top-5 reranking

- Reproduced the frozen baseline and SIFT 0.40 rankings exactly (1150/1150 hard_v2, 128/128 generated). Candidate generation, OCR and Top-5 stayed fixed.
- Recomputed SIFT homography coverage: 85.06% hard (4891/5750 pairs), 54.84% generated (351/640). Aligned SO400M/PE-Core and fixed full/center85/center70 SO400M mean/max were scored over those same candidates.
- Generated accuracy remained 80.47% (103/128) under the selected global `combined_local_signals_w0.10`; generated-hard 62.50%, hard 92.00%, R@5 100%, transitions 0 rescued / 0 broken. The highest fixed-grid alternatives did not yield a net gain; multiview alone regressed generated Top-1.
- Among 25 remaining generated errors, aligned SO400M/PE-Core/multiview mean/max target wins were 8/7/4/2; measured new-signal oracle union 12/25, ceiling 89.84%, below the 116/128 success target. Estimated selected-policy total p95 was 2.314 s, under 3 s.
- LightGlue’s official docs and licenses were reviewed, but official source/weight downloads timed out; SIFT/ALIKED/DISK+LightGlue were not run. The measured oracle excludes them.
- Verdict: **KEEP_SIFT** for this milestone; do not integrate the measured local fusion and do not begin fine-tuning automatically. The result does not establish whether LightGlue can help.
- Production recommendation stays `current_plus_sift_w0.40`; its prior post-selection synthetic sanity check was reused (synthetic_dev 97.14% Top-1 / 99.95% R@5) because this verdict does not change SIFT. Synthetic data was not used for local-weight selection.
- Full test suite: 286 passed (3 existing sklearn deprecation warnings). Artifacts: `artifacts/experiments/strong_local_visual_reranker_final_20260923T1000Z/`; report/gallery: `reports/strong_local_visual_reranker_report.md`, `reports/local_visual_reranker_errors.html`.

## 2026-09-25 — Smart Retry desktop MVP complete

- Frozen candidate resolved for product wiring: validated R8 epoch 5 + current eslav reference-OCR blend + SIFT weight 0.40. R16 is not adopted. No fine-tuning or new ML experiment was started.
- Added `src/recognition/smart_retry.py`: catalog retrieval from the saved R8 adapted reference embeddings, live query OCR, Top-5 SIFT fusion, image/retrieval corroboration, and evidence-specific retry guidance. It validates checkpoint, reference checksums, OCR coverage, SIFT config and OpenCV version before serving.
- Added `scripts/serve_smart_retry.py`: localhost web/API service with rich `/api/recognize`, flat evaluator `/api/predict`, and `/api/health`. GPU OCR is the default for latency; CPU OCR remains an explicit fallback. No raw upload persistence.
- Removed the demo-state control and hard-coded successful wine card; the UI now renders the actual catalog result and enables retry/no-match from recognition responses. Desktop-only; Match for You and Compare are available after recognition, with the personal collection added in the current product pass.
- The agreement and photo-quality thresholds are explicitly heuristics, not field-calibrated probabilities. Human slug/no-match review of the new 100-photo dataset remains needed for a calibrated open-set threshold.
- CPU OCR failed its first integration smoke on this 10 GB RAM host: one 1,336×2,000 catalog image exceeded 30 s and forced 6.1 GB swap. The server was stopped; default switched to the already selected local GPU OCR to meet the SLA. This is product runtime integration, not an OCR model change.
- End-to-end API smoke using the catalog reference `zb-vajn-spumante-bryut-beloe` returned the expected slug with image+valid-SIFT agreement; flat `/api/predict` returned exactly `{"slug":"zb-vajn-spumante-bryut-beloe"}` in 1.10 s when warm. First cold recognition took 8.97 s, so GPU mode now performs one reference-image warm-up before opening the listener. `/api/health` and `/web/` both returned HTTP 200. The integration pass did not run the suite; the separate regression audit is recorded below.
- Manual API checks also returned `retry/resolution` for a 320×240 image and `retry/unreadable` for corrupt bytes, verifying those targeted reason paths.
- Full regression rerun: 334 tests passed, 2 pretrained-model reproduction tests deselected, 3 existing sklearn deprecation warnings. An unfiltered attempt was interrupted after 77 passes when it blocked on a socket read. Static UI checks and live recognition API smokes passed; browser click-through was unavailable because the Windows UI bridge rejected the WSL workspace URI.
- Completion checkpoint: `reports/product_ui_checkpoint_20260925.md`. The desktop MVP and local integration are complete. Field-calibrated open-set/no-match thresholds remain dependent on human labels for the new 100-photo dataset.
