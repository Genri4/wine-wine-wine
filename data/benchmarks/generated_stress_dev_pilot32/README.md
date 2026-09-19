# generated_stress_dev_pilot32

Budgeted generated stress pilot with **16 representative** and **16 hard**
products, **8 hard families × 2 products**, and **128**
planned generations. This directory is isolated from the existing
`generated_stress_dev` 150-product plan.

No paid generation is run by the builder. All rows in `generation_manifest.csv`
start as `generation_status=pending`.

## Inspect and later generate

First inspect the exact scope without an API call:

```bash
.venv/bin/python scripts/generate_stress_aitunnel.py \
  --manifest data/benchmarks/generated_stress_dev_pilot32/generation_manifest.csv \
  --all \
  --generated-dir data/benchmarks/generated_stress_dev_pilot32/generated_raw \
  --runs-dir data/benchmarks/generated_stress_dev_pilot32/generation_runs \
  --dry-run
```

If generation is explicitly authorized later, remove `--dry-run` from the same
command. The generator reads `AITUNNEL_API_KEY` only from the environment.

## Review and scoring workflow

```bash
.venv/bin/python scripts/import_generated_stress.py \
  --generation-manifest data/benchmarks/generated_stress_dev_pilot32/generation_manifest.csv \
  --selected-products data/benchmarks/generated_stress_dev_pilot32/selected_products.csv \
  --generated-dir data/benchmarks/generated_stress_dev_pilot32/generated_raw \
  --review-csv data/benchmarks/generated_stress_dev_pilot32/review.csv \
  --review-html data/benchmarks/generated_stress_dev_pilot32/review.html

# manually set only identity-preserving rows to accepted
.venv/bin/python scripts/build_generated_stress_benchmark.py \
  --generation-manifest data/benchmarks/generated_stress_dev_pilot32/generation_manifest.csv \
  --review-csv data/benchmarks/generated_stress_dev_pilot32/review.csv \
  --catalog-manifest data/processed/catalog_manifest.csv \
  --output-dir data/benchmarks/generated_stress_dev_pilot32

.venv/bin/python scripts/run_siglip2_baseline.py --benchmark generated_stress_dev_pilot32
.venv/bin/python scripts/evaluate_benchmark.py \
  --benchmark generated_stress_dev_pilot32 \
  --predictions artifacts/experiments/<run_id>/predictions.csv
```

The scored manifest can contain only `accepted` and valid, non-reference-copy
images. Family fields in the manifest enable family-aware diagnostics for the
hard half.
