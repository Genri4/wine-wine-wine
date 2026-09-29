# Final ML error audit

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
