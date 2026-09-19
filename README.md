# My Wine

## Что это за проект

Проект для хакатона по улучшению сканера винных этикеток платформы «Своё Вино».
Официальный публичный каталог: [vino-svoe.ru](https://vino-svoe.ru/).

## Главная задача

По фотографии этикетки найти точную карточку вина в каталоге и вернуть её `slug`.

## Текущий статус

Каталог v1 и reproducible synthetic DEV benchmark подготовлены. Выполнен
первый zero-shot SigLIP2 retrieval baseline; benchmark является внутренним
сравнительным и не заменяет официальный eval.

## Контекст

- `context/hackaton_context.md` — официальное ТЗ, контракт и критерии;
- `context/data_context.md` — полученные данные и непроверенные наблюдения;
- `context/feature.md` — дополнительные функции и порядок их реализации.

## Ожидаемый API-контракт

```json
{"slug":"wine-slug"}
```

## Основные официальные цели

- 90–100% совпадений;
- SLA < 3 sec;
- local inference/demo.

## Запуск DEV benchmark и baseline

Из корня проекта:

```bash
.venv/bin/python scripts/freeze_catalog_v1.py
.venv/bin/python scripts/create_synthetic_dev.py
.venv/bin/python scripts/run_siglip2_baseline.py --benchmark synthetic_dev
```

Фиксация каталога создаёт `data/processed/catalog_v1.json`, генератор —
`data/benchmarks/synthetic_dev/manifest.csv` и отдельные query images.
Baseline использует pretrained `google/siglip2-base-patch16-224`, CUDA при
наличии и CPU fallback, а catalog embeddings сохраняет в
`artifacts/catalog_embeddings/`.

Любой готовый `predictions.csv` можно независимо оценить:

```bash
.venv/bin/python scripts/evaluate_benchmark.py \
  --predictions artifacts/experiments/<run_id>/predictions.csv
```

Все результаты запуска находятся в `artifacts/experiments/<run_id>/`:
`config.json`, `metrics.json`, `predictions.csv`, `errors.csv` и
`error_report.html`.

## Taxonomy benchmark-ов

- `synthetic_dev` — laboratory/regression benchmark на reference-derived queries.
- `hard_near_duplicate_dev` — fine-grained confusion benchmark; первая версия
  также переиспользует synthetic queries.
- `hard_near_duplicate_dev_v2` — строгий hard subset v1: pair должен иметь
  минимум 2 из 4 сильных сигналов; также переиспользует synthetic queries.
- `generated_stress_dev` — generative image-edit stress benchmark для capture /
  domain robustness. Plan подготовлен; генерация выполняется отдельной
  явной командой через AITUNNEL и обязательный manual-review gate.
- `generated_stress_dev_pilot32` — изолированный budgeted pilot той же оси:
  16 representative products + 16 products из 8 evidence-carrying hard
  families, 128 planned generations. Он не изменяет основной 150-product plan;
  для hard-половины evaluator дополнительно считает family-level metrics.
- `web_extra_dev` — experimental web-image benchmark; сейчас слишком мал для
  quality conclusions, не использовать как основной quality signal.

Метрики этих benchmark-ов никогда не объединяются в одну headline accuracy и
не выдаются за accuracy на private test организаторов.

Сборка дополнительных benchmark-ов:

```bash
.venv/bin/python scripts/build_hard_near_duplicate.py
.venv/bin/python scripts/build_hard_near_duplicate_v2.py
.venv/bin/python scripts/build_generated_stress_plan.py --num-products 150
.venv/bin/python scripts/build_generated_stress_pilot32.py
.venv/bin/python scripts/collect_web_extra.py --workers 4
```

Запуск общего evaluator для конкретного набора:

```bash
.venv/bin/python scripts/evaluate_benchmark.py \
  --benchmark synthetic_dev \
  --predictions artifacts/experiments/<run_id>/predictions.csv
```

Для `hard_near_duplicate_dev` evaluator дополнительно считает rank
distribution, score margins, accuracy per family и confusion pairs.
Те же hard diagnostics доступны для `hard_near_duplicate_dev_v2`; v2
сохраняет numerical selection evidence в `families.csv`.

Один и тот же baseline можно запускать для каждого набора отдельно:

```bash
.venv/bin/python scripts/run_siglip2_baseline.py --benchmark hard_near_duplicate_dev
.venv/bin/python scripts/run_siglip2_baseline.py --benchmark hard_near_duplicate_dev_v2
.venv/bin/python scripts/run_siglip2_baseline.py --benchmark generated_stress_dev
.venv/bin/python scripts/run_siglip2_baseline.py --benchmark generated_stress_dev_pilot32
.venv/bin/python scripts/run_siglip2_baseline.py --benchmark web_extra_dev
```

Для `hard_near_duplicate_dev` query images в v1 переиспользуются из
`synthetic_dev`; это диагностический, а не независимый benchmark. Для
`hard_near_duplicate_dev_v2` query ids/bytes также переиспользуются из
`synthetic_dev`; текущий v2 artifact содержит projection замороженных
synthetic predictions, без повторного запуска модели. Для
`web_extra_dev` scored manifest содержит только 5 принятых official-site
изображений после строгой проверки на копии canonical references. Детали
происхождения, отклонённых кандидатов и policy находятся в
`data/benchmarks/web_extra_dev/README.md`.

### generated_stress_dev workflow

План создаётся без image API:

```bash
.venv/bin/python scripts/build_generated_stress_plan.py --num-products 150
```

Перед платным запуском сначала проверить ровно пять первых products (20
requests, 4 scenario на product):

```bash
.venv/bin/python scripts/generate_stress_aitunnel.py \
  --pilot-products 5 --dry-run
```

Генератор использует документированный [AITUNNEL image-edit endpoint](https://aitunnel.ru/docs/images)
`https://api.aitunnel.ru/v1/images/generations` и JSON с полями
`model`, `prompt`, `input_references`, `quality`, `size`, `output_format`.
Локальная reference передаётся как base64 data-URI. Credential читается только из
`AITUNNEL_API_KEY`; ключ не записывается в код, конфигурацию или logs. По
умолчанию используются model `gpt-image-2.5-sunburst`, quality `low`, size
`1024x1024`. Ответ сохраняется из `data[0].b64_json` (URL-ответ также
поддерживается как compatibility path).

После проверки dry-run пользователь явно запускает только pilot:

```bash
.venv/bin/python scripts/generate_stress_aitunnel.py --pilot-products 5
```

В Windows PowerShell можно использовать безопасную одноразовую обёртку,
которая сама запросит ключ masked, передаст его в WSL и очистит окружение:

```powershell
.\scripts\run_stress_aitunnel.ps1
```

По умолчанию это один product × четыре scenarios. Обёртка не умеет запускать
полный 600-row plan.

Для более дешёвого сравнительного пилота можно выбрать поддерживающую image-edit
модель `gpt-image-1-mini` и отдельную папку, чтобы не перезаписывать удачные
результаты текущей модели:

```powershell
.\scripts\run_stress_aitunnel.ps1 `
  -Model gpt-image-1-mini `
  -GeneratedDir data/benchmarks/generated_stress_dev/generated_raw_gpt_image_1_mini
```

Для этой отдельной папки importer нужно запускать с отдельными review-файлами:

```bash
.venv/bin/python scripts/import_generated_stress.py \
  --generated-dir data/benchmarks/generated_stress_dev/generated_raw_gpt_image_1_mini \
  --review-csv data/benchmarks/generated_stress_dev/review_gpt_image_1_mini.csv \
  --review-html data/benchmarks/generated_stress_dev/review_gpt_image_1_mini.html
```

Результаты сохраняются строго в
`data/benchmarks/generated_stress_dev/generated_raw/<output_filename>`.
Валидные уже существующие outputs пропускаются; `--overwrite` требуется для
замены. Runtime-статусы и usage/cost artifacts находятся в
`data/benchmarks/generated_stress_dev/generation_runs/<run_id>/`:
`requests.csv` и `summary.json`. Cost считается неизвестным, если provider не
вернул его явно. Полный scope доступен только через явный `--all`.

Если outputs генерировались внешней proprietary model, контракт тот же:
положить каждый результат из `generation_manifest.csv` в
`data/benchmarks/generated_stress_dev/generated_raw/` с тем же
`output_filename`. Затем:

```bash
.venv/bin/python scripts/import_generated_stress.py
# вручную отредактировать data/benchmarks/generated_stress_dev/review.csv
.venv/bin/python scripts/build_generated_stress_benchmark.py
.venv/bin/python scripts/run_siglip2_baseline.py --benchmark generated_stress_dev
.venv/bin/python scripts/evaluate_benchmark.py \
  --benchmark generated_stress_dev \
  --predictions artifacts/experiments/<run_id>/predictions.csv
```

Только `review_status=accepted`, валидные изображения без exact reference copy
попадают в scored `manifest.csv`. Pending/rejected rows не оцениваются. Этот
benchmark является производным от reference catalog, не real-world independent
dataset и не может трактоваться как expected private-test accuracy.
