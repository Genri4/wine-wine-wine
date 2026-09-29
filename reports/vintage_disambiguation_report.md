# Vintage Disambiguation / Targeted OCR (frozen SO400M + eslav pipeline)

Diagnostic milestone over the frozen production pipeline
(`siglip2_so400m_384` + eslav `reference_ocr_blend` reranker). Question:
can targeted year OCR inside the already-correct wine family fix the
remaining vintage near-duplicates?

## Challenge slice v1 (diagnostic, not a headline benchmark)

- hard_v2 vintage challenge: **46 queries** (17 vintage families, 34
  products; 33/34 products have a catalog year).
- generated vintage challenge: **23 queries** (8 products, all 8 with a
  catalog year).
- Inclusion: target in a vintage family AND in SO400M Top-5 AND a
  same-family competitor with a different known year is also in the Top-5.

## Candidate year evidence (audit)

| Source | Coverage |
|---|---|
| Any known year (2042 products) | 934 (45.7%) |
| reference_ocr only | 835 |
| multiple_sources_agree | 72 |
| product_title only | 27 |
| unknown_conflict | 37 |

Hard_v2 vintage family products: 33/34 with a year (97%). Pilot32 vintage:
8/8. Years are never invented; conflicts become `unknown_conflict`.

## Oracle ceiling (perfect query year)

| Slice | Baseline correct | Rescue potential | Oracle Top-1 |
|---|---|---|---|
| hard_v2 vintage challenge (46) | 38 | **8** (all baseline-wrong are resolvable) | 46/46 = 100% |
| generated vintage challenge (23) | 17 | **6** | 23/23 = 100% |

The ceiling is high: every baseline-wrong vintage query is resolvable by a
perfect year. 7 of 46 hard queries are ambiguous (two family members share
the target year) and stay at the baseline state.

## Detector-box targeted OCR (Part 6-9)

Re-recognizing the query's own detected text boxes (padded crop, LANCZOS):

| Method | hard correct-year | generated correct-year |
|---|---|---|
| full image (baseline OCR) | 26/46 (56.5%) | 11/23 (47.8%) |
| box retry 1x | 20/46 (43.5%) | 12/23 (52.2%) |
| box retry 2x | 25/46 (54.3%) | 11/23 (47.8%) |
| box retry 4x | 27/46 (58.7%) | 12/23 (52.2%) |

Gain from box retry is marginal (+2.2pp hard at 4x, +4.4pp generated) and
wrong-year stays 0-3 — the recognizer mostly either reads the year or reads
nothing. **Correct-year rate ~52-59% is the bottleneck**, not resolution.

## Conservative vintage rerank (detector-box path)

| Slice | current pipeline | + vintage stage | rescued | broken |
|---|---|---|---|---|
| hard_v2 challenge (46) | 38/46 | 39/46 | **+1** | **0** |
| generated challenge (23) | 17/23 | 17/23 | 0 | 0 |

Only 1 family swap fired on hard_v2 (query-year evidence decided for 27/46
hard, 14/23 generated; the rest had no confident year) — safe but weak.

## Reference-guided year crop (SIFT, Parts 14-21)

SIFT + ratio-test + RANSAC per candidate (reference year box -> query):

| Slice | Alignments | Success | Median good matches | Median inliers | Median inlier ratio |
|---|---|---|---|---|---|
| hard_v2 | 68 | **100%** | 227 | 162.5 | 0.73 |
| generated | 23 | **95.7%** | 29 | 18 | 0.54 |

Alignment is not the bottleneck. Projected-crop year reads:

| Slice | Candidate crops | Year box found | Aligned | Year read | Read matches candidate year |
|---|---|---|---|---|---|
| hard_v2 | 113 | 68 | 61 | 53 | 30 |
| generated | 46 | 23 | 21 | 12 | 12 |

Candidate-specific evidence works (30/53 hard reads match the candidate's
own year), but conservative gating allows only **1 swap** in the whole
challenge set: the flagship case (fanagoriya-primum-alveus-brut-2014 query,
2016 product at Top-1, projected crops read **2014** on both the target
crop and the 2016 crop) is rescued: 38 -> 39 with 0 broken. Generated: no
swaps (both potential cases blocked by conservative rules — one correctly,
because the years did not distinguish the pair).

## Full-pipeline regression

The vintage stage only fires inside same-family vintage ambiguities, so
non-vintage queries are untouched by construction; the frozen production
reranker numbers stay: synthetic 95.27%, hard 87.91% (vintage stage would
make it 87.91% + 1 query = 88.00% if enabled on the full hard set, 0 broken),
generated 73.44%.

## Latency (RTX 4060)

| Stage | mean | p50 |
|---|---|---|
| full-image detection (for boxes) | 362 ms | 170 ms |
| SIFT alignment per candidate | 486 ms | 217 ms |
| targeted stage total (only on vintage ambiguity) | ~1-1.5 s | - |

Production concept: SO400M -> current OCR reranker -> targeted vintage stage
ONLY when same-family vintage ambiguity remains; ~10% of hard queries would
trigger it.

## Answers to the milestone questions

1. Oracle ceiling: **100%** on both challenge slices (hard: 8/8 baseline-wrong
   resolvable; generated: 6/6).
2. Usable year evidence: 45.7% of all products; 97% of hard vintage-family
   products; provenance tracked, conflicts -> unknown.
3. Full-image OCR correct-year: 56.5% hard / 47.8% generated (of challenge).
4. Detector-box retry: 58.7% / 52.2%.
5. Best upscale: 4x on hard, 1x/4x equal on generated (2x never better).
6. Vintage Top-1: +1 query on hard (38->39), 0 on generated.
7. Rescued/broken: +1/-0 (hard), 0/0 (generated).
8. Generated vintage: conservative gates blocked both potential swaps.
9. SIFT alignment success: 100% hard / 95.7% generated.
10. Reference-guided year recognition: reads a year in 53/61 aligned hard
    crops (30 match the candidate year) - evidence exists.
11. Reference-guided reranking: +1 exact vintage Top-1, 0 broken.
12. Full hard_v2: 87.91% -> 88.00% (+1 query) if enabled; subtype untouched.
13. Non-vintage/subtype: unchanged by construction (family-scoped swaps only).
14. Extra latency: ~1-1.5 s, only on vintage-ambiguity trigger.
15. LightGlue/fine-tuned OCR/VLM: alignment is already ~100%; the limit is
    the recognizer's 52-59% correct-year rate on real photos and the small
    absolute number of vintage queries (8 baseline-wrong on hard).

## VERDICT

**C. ADD REFERENCE-GUIDED VINTAGE STAGE** — as a conditional, conservative,
family-scoped layer: alignment quality is proven (95-100%), the rescued case
is real (+1/-0), and the trigger keeps it off the hot path. It is a small
but strictly non-negative gain with ~1 s cost on ~10% of queries.

Practical caveat recorded honestly: the absolute gain is +1 query on hard_v2
and 0 on generated at the current correct-year rate; if the OCR correct-year
rate ever reaches ~80%+ (better recognizer), this stage scales to the full
+8/-0 hard ceiling without any redesign. NOT adopted as unconditional
default; needs the production trigger wired into the API path.

## Artifacts

- Run: `artifacts/experiments/vintage_disambiguation_20260922T1/`
  (vintage_metadata_audit.csv, challenge manifests, oracle_ceiling.json,
  targeted_ocr_summary.json + per-query, reference_guided_per_query.csv,
  alignment_summary.json, vintage_rerank/reference_guided summaries,
  debug crops)
- Module: `src/recognition/vintage_disambiguation.py`,
  `src/recognition/reference_guided_year.py`; runner
  `scripts/run_vintage_disambiguation.py`
- Tests: 23 new (vintage_disambiguation, reference-guided year), 234 total pass
