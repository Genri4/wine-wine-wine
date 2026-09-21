# Architecture boundaries

Текущий воспроизводимый DEV pipeline проекта:

```text
canonical catalog
→ synthetic DEV
→ visual encoder (pretrained SigLIP2)
→ cached catalog embeddings
→ cosine retrieval (matrix multiplication)
→ evaluator
→ error analysis
```

## Query view strategies

Query-side preprocessing is a frozen-encoder inference concern and is defined
in `src/recognition/view_strategy.py`. Exactly one fixed strategy is applied
to every query of a run; views and aggregation never depend on the target,
scenario, or subset:

- `baseline_full` — one full-image view; the frozen reference baseline.
- `preprocessing_v1` — full + center 85% + center 70% views of the query,
  encoded in one batched forward pass, ranked by mean similarity. Evaluated
  in `reports/preprocessing_v1_evaluation.md`: strong safe recovery on
  generated-stress queries (+14.8pp R@5 with 0 correct→wrong Top-1 flips) but
  a real synthetic_dev (−3.8pp Top-1) and hard_v2 (−2.3pp Top-1) regression.
  It is therefore **not** the unconditional default; a conditional
  (confidence-gated) policy is a separate future decision.
- `confidence_gated_v1` — the current retrieval default: run the full-image
  ranking first, then fall back to crop85+crop70 (reusing the full embedding,
  mean aggregation identical to preprocessing_v1) only when
  `top1_score < 0.8824`. Frozen by product-level calibration
  (`reports/confidence_gated_v1_evaluation.md`): captures the full
  generated-stress gain on held-out products, keeps clean quality close to
  the baseline, ~1.4 views/query on clean data. The gate reads only the
  full-ranking score distribution; the confidence value stays available for
  a future user-facing Smart Retry layer, which is a separate concern.

The catalog/reference side is unchanged: one embedding per canonical
reference image from the validated shared cache.

## OCR reranking stage (frozen so400m backbone)

A conservative post-retrieval stage sits between image Top-5 and the final
answer; OCR never participates in candidate generation:

```text
query image
→ siglip2_so400m_384 retrieval (Top-5)
→ cached query OCR (PaddleOCR 3.7 local, PP-OCRv5 server det + eslav rec)
→ text signals per candidate (RapidFuzz, Cyrillic + transliterated keys)
→ conservative fusion (global alpha, text-margin guard)
→ final Top-1
```

Boundaries fixed in `src/recognition/ocr_reranker.py` and frozen by
calibration (see `reports/so400m_ocr_reranker_report.md`):

- Policy set is closed: `image_only`, `metadata_text_blend`,
  `reference_ocr_blend`, `combined_text_blend`, `combined_vintage_blend`.
  Frozen default: `reference_ocr_blend, alpha 0.30`.
- `alpha` is strictly global — one value per run, never per scenario,
  family or product. No learned reranker, no LLM, no cloud APIs.
- Reranking permutes the Top-5 only: the candidate set, Recall@5 and
  Recall@10 are invariant by construction.
- The text-margin guard keeps the image Top-1 unless the challenging
  candidate's text evidence beats the image winner's by ≥ 0.05; empty OCR
  can never reorder anything.
- Query OCR and reference OCR are separate caches under
  `artifacts/ocr_cache/<ocr_model>/`; OCR runs exactly once per unique image
  (dedup by sha256) and is never re-run by reranking experiments.
- Calibration is split-level: pilot32 by product (all 4 scenarios of one
  product share a split), hard_v2 by whole family, synthetic by product,
  fixed seed fingerprinted in `calibration_split.csv`.
- Known limits (honest negatives): vintage disambiguation did not move
  under any policy; the text oracle is at parity with the image baseline on
  generated stress, so OCR there is non-discriminative rather than unread
  (99% token coverage, median 9 tokens).

Отдельная ветка generated stress отделена от baseline и вызывает image API
только по явной команде пользователя:

```text
canonical catalog
→ deterministic category×region selection
→ generation_manifest + AITUNNEL image-edit pilot
→ generated_raw/<output_filename>
→ generation_runs/<run_id> runtime artifacts
→ import/validation
→ mandatory manual review
→ accepted generated_stress_dev manifest
→ same encoder/evaluator with per-scenario metrics
```

До появления accepted images scored `generated_stress_dev/manifest.csv` не
создаётся. Pending, rejected, corrupt и exact reference-copy samples не могут
попасть в evaluation.

## Текущие факты

- Catalog v1 содержит 2 042 usable products с reference image; полный
  canonical manifest содержит 2 103 products.
- Synthetic DEV создаёт 2 query variants на product (4 084 queries) с
  фиксированным seed. Это только внутренний сравнительный benchmark: он не
  является официальной оценкой и не моделирует реальный пользовательский
  снимок полностью.
- Reference index строится только по исходным canonical reference images;
  generated queries лежат отдельно, leakage assertion выполняется до baseline.
- SigLIP2 используется zero-shot/pretrained: без fine-tuning, OCR,
  inference augmentation и reranking.
- Catalog embeddings кэшируются с model id/version, catalog checksum и
  preprocessing config. При несовпадении metadata cache не используется.
- Retrieval — точное normalized cosine через обычное matrix multiplication;
  FAISS, pgvector и vector DB не требуются.
- Query latency измеряется отдельно для embedding, retrieval и total; catalog
  indexing не входит в per-query latency.

## Будущие границы и открытые решения

- Официальный eval dataset и его окончательный split/metric contract ещё не
  подтверждены этим DEV benchmark.
- Preprocessing/normalization, OCR, additional signals и reranking — отдельные
  будущие эксперименты, которые нельзя считать улучшением без общей evaluation.
- UI, backend, database и production deployment пока не входят в milestone.

## Benchmark taxonomy

```text
synthetic_dev
  → laboratory/regression benchmark; queries derived from references
hard_near_duplicate_dev
  → fine-grained confusion benchmark; v1 reuses synthetic queries
hard_near_duplicate_dev_v2
  → strict v1 refinement; ≥2 strong signals and evidence-carrying families
generated_stress_dev
  → AITUNNEL/external image-edit plan; mandatory import/review gate before scoring
generated_stress_dev_pilot32
  → isolated 32-product generated-stress pilot; 16 representative products and
    8 evidence-carrying hard families, with family-aware diagnostics
web_extra_dev
  → independent web-image benchmark only after exact identity and duplicate checks
```

Эти наборы оценивают разные свойства и не сводятся в одну headline-метрику.
Ни один из них не является accuracy на private test организаторов.
`web_extra_dev` можно называть independent только для samples, прошедших
reference-duplicate/leakage validation. Подробности collection и rejected
кандидатов находятся в `data/benchmarks/web_extra_dev/README.md`.
Текущая official-site выборка содержит 5 accepted samples; остальные
кандидаты не входят в scored manifest после conservative duplicate checks.

`generated_stress_dev_pilot32` строится скриптом
`scripts/build_generated_stress_pilot32.py` в отдельной директории
`data/benchmarks/generated_stress_dev_pilot32/`. Его 128 planned generations
не считаются scored, пока внешний output не пройдет importer и обязательную
ручную проверку. После accept общий evaluator reports overall/per-scenario
metrics, а также `family_top1`, `family_recall_at_5`,
`within_family_disambiguation_top1`, accuracy per family и
внутрисемейные confusion pairs для hard-половины. Это производный stress
benchmark, а не real-world independent dataset и не оценка private test.

Подробное проектирование database, frontend, Docker, microservices и production deployment на этом этапе не требуется.

## AITUNNEL generation boundary

`scripts/generate_stress_aitunnel.py` читает только существующий
`generation_manifest.csv`. Идентификаторы, prompts, slug и scenario не
переписываются; runtime status хранится в отдельном
`data/benchmarks/generated_stress_dev/generation_runs/<run_id>/`.

Без `--dry-run` credential обязателен в env `AITUNNEL_API_KEY`. Сначала
выбираются первые 5 products из существующего plan (20 requests); полный scope
возможен только через явный `--all`. На каждый запрос отправляются reference
image reference и prompt в `POST /v1/images/generations` через
`input_references` с model
`gpt-image-2.5-sunburst`, quality `low`, size `1024x1024`. Output проходит
atomic write и image validation. Error policy: retry только transport/408/429/
5xx и transient malformed responses с ограничением backoff; 401/402/403 и
unsupported model/parameter останавливают run. API key никогда не попадает в
runtime CSV/JSON.

После генерации importer всё равно оставляет rows `pending`; только человек
может выставить `accepted`. Generated images являются производными от
reference catalog и не должны использоваться для training/fine-tuning frozen
benchmark.
