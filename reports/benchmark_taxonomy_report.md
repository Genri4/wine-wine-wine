# Benchmark taxonomy report

Дата фиксации: 2026-09-16. Все benchmark-ы оцениваются раздельно; pooled
headline metric не используется.

## Реализовано

- Универсальный evaluator `scripts/evaluate_benchmark.py` принимает
  `--benchmark synthetic_dev|hard_near_duplicate_dev|web_extra_dev`, проверяет
  полное совпадение `query_id` и `target_slug` с manifest и считает Top-1,
  Recall@5, MRR, query count, unique targets и latency.
- `scripts/run_siglip2_baseline.py` запускает один и тот же неизменённый
  pretrained SigLIP2 baseline для каждого benchmark-типа.
- `scripts/build_hard_near_duplicate.py` строит воспроизводимые disjoint
  families из canonical metadata и observed confusion signals. V1 не заявляет
  visual independence: query images переиспользованы из `synthetic_dev`.
- `scripts/collect_web_extra.py` собирает только exact-slug entries из
  официального каталога `https://vino-svoe.ru/` и его sitemap. `manifest.csv`
  является единственным scored set.
- `generated_stress_dev` только зарезервирован design note-ом; изображения и
  генерация не выполнялись.

## Метрики одного baseline

| benchmark | queries | unique targets | Top-1 | Recall@5 | MRR | mean latency |
|---|---:|---:|---:|---:|---:|---:|
| synthetic_dev | 4084 | 2042 | 89.40% | 99.83% | 0.9413 | 19.90 ms |
| hard_near_duplicate_dev | 3444 | 1722 | 87.78% | 99.80% | 0.9327 | 33.72 ms |
| web_extra_dev | 5 | 5 | 0.00% | 40.00% | 0.1500 | 561.30 ms |

`web_extra_dev` слишком мал для общего quality claim. Его текущий результат —
проверка pipeline и leakage policy, а не оценка качества на полевых данных.

## Hard benchmark

- 498 families, 1722 products, 3444 queries; sizes: 2 — 229, 3 — 105,
  4 — 56, 5 — 42, 6 — 24, 7 — 14, 8 — 9, 9 — 5, 10 — 6, 12 — 8.
- 325 families сформированы metadata-only правилами, 173 используют observed
  SigLIP2 confusion signal.
- Rank distribution: rank 1 — 3023, rank 2 — 323, rank 3 — 66, rank 4 — 17,
  rank 5 — 8, not found — 7.
- 421 Top-1 errors; 346 случаев, где target и prediction оказались в одной
  hard family.
- Ограничение: family membership — воспроизводимый diagnostic selection, а не
  доказательство визуального сходства без ручной проверки.

## Web benchmark и dedup

- Источник identity: официальная страница `/wines/{slug}` из
  `https://vino-svoe.ru/wines-sitemap.xml`; exact slug совпадает с canonical
  slug. Все принятые и отклонённые image URLs пришли с `api.vino-svoe.ru`.
- 1981 exact-slug candidate, 5 accepted, 1976 rejected, 0 manual review.
  61 usable catalog product не имели exact sitemap slug.
- Rejection breakdown: 1967 `perceptual_duplicate`, 9
  `target_reference_duplicate`. Generated images не использовались.
- Для каждой пары считаются SHA-256, 32x32 grayscale average hash full image и
  center crop, aspect/content aspect ratio и 64x64 thumbnail distance.
  Политика намеренно консервативная: принимается только кандидат без exact
  reference hash и без conservative near-duplicate match.
- Финальная проверка accepted set: 5 уникальных query IDs, 5 уникальных
  targets, все файлы существуют, exact SHA-256 intersection с 2042 reference
  images — 0.
- Визуально проверены representative accepted pairs; например, accepted
  `muskat-pozdnego-sbora-rozovyj` и `novyj-svet-shardone-kyuve-de-prestizh`
  отличаются от локальных reference render-ов по bottle/label variant, а
  не только размером файла.

## Артефакты

- `data/benchmarks/synthetic_dev/manifest.csv`
- `data/benchmarks/hard_near_duplicate_dev/manifest.csv`
- `data/benchmarks/hard_near_duplicate_dev/families.csv`
- `data/benchmarks/web_extra_dev/manifest.csv`
- `data/benchmarks/web_extra_dev/rejected.csv`
- `data/benchmarks/web_extra_dev/review_candidates.csv`
- `data/benchmarks/web_extra_dev/metadata.json`
- `data/benchmarks/web_extra_dev/README.md`
- `reports/hard_near_duplicate_dev_report.md`
- `data/benchmarks/generated_stress_dev/README.md`
- `artifacts/evaluations/*_siglip2*/metrics.json` и `errors.csv`
- baseline predictions: `artifacts/experiments/siglip2_<benchmark>_*/`

## Воспроизводимые команды

```bash
.venv/bin/python scripts/build_hard_near_duplicate.py
.venv/bin/python scripts/collect_web_extra.py --workers 4

.venv/bin/python scripts/run_siglip2_baseline.py --benchmark synthetic_dev
.venv/bin/python scripts/run_siglip2_baseline.py --benchmark hard_near_duplicate_dev
.venv/bin/python scripts/run_siglip2_baseline.py --benchmark web_extra_dev

.venv/bin/python scripts/evaluate_benchmark.py \
  --benchmark web_extra_dev \
  --predictions artifacts/experiments/<run_id>/predictions.csv \
  --output-dir artifacts/evaluations/web_extra_dev_siglip2
```

`run_siglip2_baseline.py` использует существующий catalog embedding cache и не
меняет модель, raw data или canonical catalog. Перед запуском web collector
нужно соблюдать robots/anti-bot ограничения источника.

## Тесты и блокеры

Unit tests покрывают ranking metrics, leakage assertion, deterministic synthetic
transform, hard-family reproducibility, exact-copy rejection и evaluator split
alignment. Полная команда: `.venv/bin/python -m pytest -q -s`.

Текущий блокер для следующего quality milestone — отсутствие достаточного
набора независимых разрешённых web images: строгая dedup policy оставила 5
samples. Нельзя компенсировать это synthetic queries или объединением метрик.
Нужны новые независимые фотографии/источник либо подтверждённый public/private
eval split организаторов.
