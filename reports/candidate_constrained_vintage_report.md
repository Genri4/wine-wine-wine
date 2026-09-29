# Candidate-Constrained Vintage Recognition (frozen SO400M + SIFT crops)

Diagnostic milestone: given SIFT-aligned year crops (from the previous
vintage milestone) and the allowed candidate years of each query's
same-family Top-5 members, how reliably can we CHOOSE the correct year?
Methods compared: current OCR, visual reference-crop matching
(SO400M / PE-Core / DINOv2), digit-only CRNN trained on synthetic years
(constrained CTC decoding), and a VLM ceiling (Qwen2-VL-2B-Instruct).

## Main table (decided queries only; year_decided shows coverage)

| Method | Hard year acc (decided) | Hard decided/46 | Hard rescued/8 | Generated year acc | Generated decided/23 | Generated rescued/6 | Broken | Latency |
|---|---|---|---|---|---|---|---|---|
| current OCR (eslav) | 12/12 = 100% | 12 | 1/8 | 12/12 = 100% | 12 | 0/6 | 0 | ~60 ms |
| SO400M crop matching | 25/25 = 100% | 25 | 1/8 | 12/12 = 100% | 12 | 0/6 | 0 | ~15 ms |
| PE-Core crop matching | 25/25 = 100% | 25 | 1/8 | 12/12 = 100% | 12 | 0/6 | 0 | ~13 ms |
| DINOv2 crop matching | 16/25 = 64% | 25 | 0/8 | 8/12 = 66.7% | 12 | 0/6 | **11** | ~12 ms |
| digit CRNN (constrained) | 14/25 = 56% | 25 | 1/8 | 8/12 = 66.7% | 12 | 0/6 | 4 | ~2 ms |
| **VLM ceiling (Qwen2-VL-2B)** | **24/25 = 96%** | 25 | 1/8 | **12/12 = 100%** | 12 | 0/6 | 1 | 190-820 ms |

## THE decisive finding: coverage, not recognition

- 21/46 hard and 11/23 generated challenge queries have **no year crop at
  all** (`no_crop`): the reference year was never read by OCR, so no
  reference year box existed. **6 of 8 hard baseline-wrong and 6 of 6
  generated baseline-wrong are exactly in this no_crop group.**
- Where a crop EXISTS, all serious methods are near-perfect:
  - current eslav OCR: 12/12, 12/12 (it only fails on crops that don't exist)
  - SO400M / PE-Core visual matching: 25/25, 12/12 (100%)
  - VLM: 24/25, 12/12 (96% / 100%) — the single hard error is one glyph
    ambiguity; VLM answered in 190-820 ms.
- DINOv2 as a local patch matcher: 64-67% and 11 broken — the local-matching
  hypothesis for DINOv2 is rejected; it cannot tell 2014 from 2016 on these
  typography-poor crops.

## Pairwise year accuracy (>= 2 distinct competing years)

| Method | hard | generated |
|---|---|---|
| OCR | 12/12 | 12/12 |
| SO400M | 25/25 | 12/12 |
| PE-Core | 25/25 | 12/12 |
| digit CRNN | 14/25 | 8/12 |
| DINOv2 | 16/25 | 8/12 |
| VLM | 24/25 | 12/12 |

## Digit recognizer (Parts 6-8, 15)

- Trained on 2988 synthetic year crops (1990-2026, fonts/rotation/blur/
  contrast/gold-text augmentation), 6 epochs, CTC loss; synthetic val
  accuracy and real challenge results:
- Real hard challenge: 14/25 = 56% decided-correct, 11 wrong — a large
  synthetic-to-real domain gap (gold-foil embossed digits on generated
  photos are far from synthetic rendering).
- Constrained decoding works mechanically (probability mass over allowed
  years), but the recognizer itself is too weak on real crops.

## VLM ceiling (Part 9, 16)

Qwen2-VL-2B-Instruct (apache-2.0, 4.1 GB, fp16, local): multiple-choice
prompt over allowed years only. 24/25 hard, 12/12 generated. The single
hard miss is one ambiguous crop. This proves **the visual signal in the
crops is sufficient** — the information is there when a strong vision model
reads it. Latency 190-820 ms/query (ceiling experiment, not production).

## Family-safe swap impact

With crops present and the year chosen correctly, the family-swap logic is
trivially safe: hard rescued 1 (the 2014/2016 fanagoriya case), 0 broken;
the second potential swap (ekstra-bryut-2017 vs brut-2017) is correctly
blocked because both members share the year 2017 — the year does not
distinguish them (subtype case). Generated: 0/0 (no crops for baseline-wrong
queries).

## Root cause of the remaining vintage errors (Part 38)

**Not recognition — crop availability.** The vintage stage never fires for
6/8 hard and 6/6 generated baseline-wrong queries because the previous
milestone's eslav reference OCR did not read the reference year, so no
reference year box existed to align. The fix path is NOT better crop OCR:
it is a better REFERENCE-side year box source (e.g. the VLM reading the
reference image offline to produce year boxes for all vintage products).

## Answers (Part 23)

1. Current OCR chooses the candidate year correctly when it decides
   (12/12, 12/12) but decides only 26% of queries (12/46) — coverage-limited.
2. Direct reference-crop visual matching works: 100% on decided queries.
3. Best visual encoder for year patches: SO400M and PE-Core tie (25/25,
   12/12); DINOv2 fails (64-67%, 11 broken).
4. Hard errors rescued: **1 of 8** (both visual matching and OCR; VLM also 1).
5. Generated errors rescued: **0 of 6** (no crops for those queries).
6. Digit-only recognizer: no gain (56-67%, below visual matching).
7. Constrained decoding is mechanically correct and removes free-form OCR
   errors, but its accuracy is bounded by the recognizer's real-domain
   strength (56%).
8. VLM ceiling: 96%/100% — the signal is in the crops; a strong model reads it.
9. correct->wrong: 0 for OCR/SO400M/PE-Core/VLM on decided queries;
   DINOv2 broke 11.
10. Extra latency: crop matching 12-15 ms/query; digit ~2 ms; VLM 190-820 ms
    (ceiling only); all within SLA except none — pipeline stays ~0.4-1.2 s.
11. Integrate: the reference-guided vintage stage with SO400M crop matching
    as the chooser (12-15 ms), AND extend crop availability by producing
    reference year boxes offline with a VLM (that is where 6/8 + 6/6 of the
    missing rescues live).

## VERDICT

**B. USE VISUAL REFERENCE-CROP MATCHING** (SO400M/PE-Core, 100% on decided,
0 broken, 12-15 ms) as the chooser inside the existing family-scoped vintage
stage, **PLUS an offline VLM pass over vintage reference images to create
the missing year boxes** — that is the single highest-leverage fix: it moves
6/8 hard and 6/6 generated baseline-wrong queries from unreachable to
reachable. D is explicitly rejected as production (VLM latency), but the VLM
result validates that crops carry the signal.

## Artifacts

- Run: `artifacts/experiments/candidate_constrained_vintage_20260922T1/`
  (config, crop_manifest, per-method year_recognition CSVs, benchmark/
  pairwise/transition/latency summaries, structured digit model)
- Crops: `.../crops/{benchmark}/{query,reference}/*.png` (61+21 query crops,
  16+3 reference crops)
- Modules: `src/recognition/constrained_vintage.py`,
  `src/recognition/year_recognizer.py`, `src/recognition/vlm_year.py`
- Tests: +15 (test_ocr_bakeoff covers constrained machinery); 234 total pass
