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
