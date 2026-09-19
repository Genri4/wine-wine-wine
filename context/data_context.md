# Контекст полученных данных

Этот файл фиксирует только полученные наборы и статус сведений о них.

## Получены

- `strapi_output0709.csv`;
- многотомный архив `prod-svoe-vino-strapi`;
- `eval.zip`.

## Ожидается по официальному ТЗ

- каталог вин;
- эталонные изображения;
- публичные полевые фотографии.

## Непроверенные наблюдения участников Telegram

Следующие сообщения не подтверждены самостоятельной проверкой и помечены как **НЕПРОВЕРЕННЫЕ**:

- сообщалось о большом количестве дублей `media`;
- сообщалось примерно о 5842 уникальных файлах;
- сообщалось примерно о 1844 duplicate `slug`;
- возможны несовпадения имён файлов и значений в CSV;
- архив похож на сырой dump Strapi;
- CSV некоторые участники считают проблемным, другие читают через pandas без ошибок.

Эти утверждения нельзя использовать как ground truth.

## Подтверждено первичным data audit и построением canonical catalog

- CSV читается как UTF-8 без BOM; в нём 4147 строк и 9 колонок.
- Найдено 2103 уникальных `Slug`. Удалены только 2044 полных повторных строк; raw CSV не изменён.
- В multipart RAR по listing найдено 15803 файла, из них 15770 изображений.
- `eval.zip` содержит 3 query-изображения (2 JPEG и 1 WebP), manifest `queries.tsv` и не содержит labels/явных target identifiers; целевые карточки по нему самостоятельно не назначались.
- Первый canonical manifest pass на 2103 продукта дал 1628 matched, 27 ambiguous и 448 unmatched; однозначное покрытие было 77.41%.
- Для первого pass выборочно извлечены reference images в `data/processed/reference_images/`; итоговый набор пересобран после ручных overrides.
- 13 имён фото используются 26 canonical products; это сохранено как отдельный issue-класс и не объявлено ошибкой автоматически.
- Mapping выполнен только детерминированной нормализацией имени файла; fuzzy matching, CV/OCR/ML и изменение raw data не выполнялись.

## Подтверждено восстановлением unresolved mappings

- Детерминированный каскад `strict → transliteration → punctuation sanitization` увеличил matched с 1628 до 1920 продуктов: +292; покрытие выросло с 77.41% до 91.30%.
- Новые однозначные mappings: 205 через `transliterated_normalized_original`, 69 через `transliterated_punctuation_normalized_original` и 18 через CRC+size-identical originals.
- Текущий остаток: 44 ambiguous и 139 unmatched. Для 44 ambiguous originals отличаются по CRC/size, поэтому автоматический выбор не выполнялся.
- Из 349 прежних Cyrillic/non-ASCII product cases восстановлены 265; оставшиеся 84 сохранены для ручной проверки или остались unmatched.
- Metadata candidate generation использует только точное пересечение токенов slug/title/winery/photo_name и не назначает ground truth автоматически. Результат сохранён в `data/processed/mapping_review_candidates.csv` и `reports/mapping_review.html`.
- Для 1920 matched-продуктов выборочно извлечены reference images; все ссылки manifest на extracted files проверены по archive path и размеру.

## Подтверждено подготовкой manual mapping verification

- Создан отдельный слой ручных решений `data/manual/catalog_mapping_overrides.csv` с колонками `slug`, `selected_archive_path`, `decision`, `reason`, `note`; до применения решений файл содержал только заголовок.
- Builder читает overrides после deterministic mapping, проверяет slug и выбранный image path по media inventory и поддерживает `matched`, `missing`, `unresolved`. Ручные решения не зашиты в Python-код.
- Исторический итог пересборки без ручных решений: автоматически matched **1920**, manually matched **0**, missing **0**, unresolved **183** (44 ambiguous + 139 unmatched), usable coverage **91.30%**.
- `reports/mapping_review.html` разделён на группы A: 44 ambiguous, B: 61 Cyrillic/non-ASCII unmatched, C: 56 ASCII unmatched, D: 12 space-containing unmatched, E: 10 opaque/random unmatched, F: 13 shared photo names / 26 products. В HTML показаны slug, название, винодельня, категория, исходное `Название фото`, mapping status, candidates и доступные изображения.
- `data/processed/mapping_review_candidates.csv` расширен полем `category`; metadata/name candidates остаются только материалом для ручной проверки и не назначают mappings автоматически.
- Для missing asset workflow предписывает `decision=missing` без случайной замены. Shared-photo-name случаи вынесены отдельно и не считаются ошибкой автоматически.

Подробности и воспроизводимые артефакты: `reports/catalog_mapping_report.md`,
`data/processed/catalog_manifest.csv`, `data/processed/catalog_mapping_issues.csv`,
`data/processed/media_inventory.csv`, `data/processed/mapping_review_candidates.csv`.

## Подтверждено финальной ручной верификацией

- В `data/manual/catalog_mapping_overrides.csv` зафиксировано **122** явных `matched` решений: 44 бывших ambiguous и 78 бывших unmatched.
- После применения overrides builder пересобрал `catalog_manifest.csv`, `catalog_mapping_issues.csv`, `reference_images/`, `catalog_mapping_report.md`, `mapping_review_candidates.csv` и `mapping_review.html`.
- Финальный product-level итог: automatically matched **1920**, manually matched **122**, missing **0**, unresolved **61**; total usable catalog coverage **2042/2103 = 97.10%**.
- Оставшиеся 61 продукта не получили принудительных замен: 5 случаев с Cyrillic/non-ASCII именами, 39 ASCII без deterministic candidate, 7 с пробелами и 10 opaque/random.
- `missing=0` означает, что на этом проходе не было достаточно надёжного доказательства отсутствия конкретного image asset; unresolved cases оставлены для отдельной проверки.
- Визуальная проверка использовалась только вручную для candidate images; ML/CV/OCR, embeddings, fuzzy auto-mapping и изменение raw data не выполнялись.

## Подтверждено разделением evaluation datasets

- Создан `catalog-v1`: 2042 usable products из 2103 canonical products; manifest checksum сохранён в `data/processed/catalog_v1.json`.
- `synthetic_dev` зафиксирован как внутренний laboratory/regression benchmark: 4084 queries, по 2 reference-derived variants на product, seed `20260916`.
- Первый SigLIP2 zero-shot baseline на `synthetic_dev`: Top-1 **89.40%**, Recall@5 **99.83%**, MRR **0.9413**; это не real-world и не official/private eval.
- Создан `hard_near_duplicate_dev`: **498** disjoint hard families, **1722** уникальных products, **3444** queries. Семейства построены из canonical metadata и observed SigLIP2 confusion signals; query images переиспользованы из `synthetic_dev`.
- Построен `hard_near_duplicate_dev_v2`: **256** families, **575** products, **1150** queries (**28.16%** usable catalog). Pair selection требует минимум 2 из 4 сильных сигналов: frozen SigLIP2 confusion с target rank 2–5, Top1/Top2 margin ≤0.02, deterministic reference-image similarity или сильное normalized-name similarity. V2 — строгий diagnostic subset v1; query ids/bytes и predictions projection переиспользуют `synthetic_dev`, поэтому benchmark не является независимым real-world test.
- Подготовлена инфраструктура `generated_stress_dev`: deterministic stratified sample **150 products** из 2042 usable (category×region квоты, winery round-robin, seed `20260916`) и **600 planned queries** по 4 сценариям (`slight_angle`, `glare_bad_light`, `distance_crop`, `handheld`). Image API не вызывался, изображения не создавались. External outputs должны пройти importer и обязательную manual review; только accepted samples могут попасть в scored manifest. Benchmark производный от reference catalog и не является real-world independent dataset.
- Подготовлен изолированный `generated_stress_dev_pilot32`: **32 unique products** (**16 representative + 16 hard**), **8 hard families × 2 products** (4 vintage + 4 subtype) и **128 planned queries** по тем же четырём сценариям. Representative subset взят детерминированно из существующего 150-product плана с исключением всех v2 hard-family products; hard subset использует только frozen v2 evidence и явные family rules. После явного запуска AITUNNEL сгенерированы **128/128** изображений (`gpt-image-2.5-sunburst`, low), `failed=0`, стоимость **281.37 ₽**. Существующие benchmark-ы не изменялись; после accepted outputs общий evaluator дополнительно считает family-level metrics для hard-половины.
- Создан `web_extra_dev` из официального каталога `https://vino-svoe.ru/` и sitemap: **1981** exact-slug candidates, **5** accepted после консервативной dedup-проверки, **1976** rejected reference duplicates, **0** manual-review candidates. Scored manifest содержит только accepted rows; benchmark остаётся отдельным от synthetic/hard.
- Из-за только 5 accepted samples `web_extra_dev` сейчас experimental и слишком мал для quality conclusions; его нельзя использовать как основной quality signal.
- В `data/benchmarks/generated_stress_dev/` сохранены `selected_products.csv`, `scenarios.json`, `generation_manifest.csv`, `metadata.json` и workflow README; `generated_raw/` оставлен пустым до явного запуска генерации. Добавлен безопасный AITUNNEL pilot generator: по умолчанию 5 первых products × 4 существующих scenarios = 20 requests, сначала обязательный `--dry-run`; полный scope доступен только через явный `--all`. `AITUNNEL_API_KEY` читается только из env, runtime-статусы и reported usage/cost пишутся в `generation_runs/<run_id>/`, исходный generation plan не изменяется. Scored manifest появится только после importer и ручной верификации accepted outputs.
- Для benchmark-ов используется общий model-independent evaluator с Top-1, Recall@5, MRR, query count и unique target slugs; hard benchmark дополнительно имеет rank distribution, margins, family accuracy и confusion pairs. Для `generated_stress_dev_pilot32` добавлены `family_top1`, `family_recall_at_5`, `within_family_disambiguation_top1`, accuracy per family, `confusion_within_family` и breakdown vintage/subtype.

Ни один benchmark не следует выдавать за accuracy на private test организаторов и не следует объединять в одну headline-метрику.
