# `new_data_open_set_v1`

Separate open-set benchmark for the 100 unlabeled photos in `data/new_data/`.
The images stay in their original directory; this benchmark does not copy or
modify them. Image provenance is not confirmed.

## Label contract

Review each photo against the 2042 usable catalog slugs in
`data/processed/catalog_manifest.csv` and select exactly one outcome in the
HTML review page:

- `catalog_match`: set `verified_slug` to the exact catalog slug;
- `no_catalog_match`: leave `verified_slug` empty;
- `uncertain` or `unreviewed`: do not score this row as ground truth.

The page is `artifacts/experiments/new_data_slug_audit_20260924/slug_candidate_review.html`.
Export the reviewed CSV from that page. Model candidates are suggestions only;
they are not ground-truth labels. A complete benchmark score requires all
images to be resolved and at least one example from each class.

## New metric

The primary metric is **Open-Set Retrieval AUC** (`open_set_retrieval_auc`), a
threshold-free score derived from the retrieval ranking:

- confidence is the maximum cosine similarity (the Top-1 score);
- at each threshold, unknown false-accept rate is the fraction of
  `no_catalog_match` photos whose Top-1 score reaches the threshold;
- known correct-classification rate is the fraction of `catalog_match` photos
  whose Top-1 slug is exactly right and whose score reaches the threshold;
- the metric is the area under correct-classification rate vs unknown
  false-accept rate.

The report also includes known-set Top-1 accuracy and Recall@5. Frozen SO400M
and selected R16 are evaluated separately, using the already generated
image-only rankings. This does not represent OCR/SIFT or the full production
pipeline. The score cannot be calculated from the current unlabeled data.

Run after exporting reviewed labels:

```bash
.venv/bin/python scripts/evaluate_new_data_open_set.py \
  --review-csv /path/to/new_data_open_set_review.csv
```

Outputs are written under
`artifacts/experiments/new_data_slug_audit_20260924/open_set_metric/`.
Partial labels are reported as provisional; unresolved photos are excluded and
their counts remain visible.
