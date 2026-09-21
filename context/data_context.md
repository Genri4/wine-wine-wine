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

## Подтверждено diagnostic milestone по pilot32 (2026-09-19)

- Проведён diagnostic-only эксперимент на frozen SigLIP2: full-catalog target ranks для всех 128 queries, fixed center-crop ablation (85/70/55), fixed multi-crop (max/mean). Без обучения, OCR, reranker, detector; crop policy не использует target.
- Подтверждён текущий preprocessing SigLIP2: direct square stretch до 224x224 без center crop (probe-проверка на non-square изображении); pipeline не может отрезать часть бутылки, только искажает пропорции.
- Baseline full-rank distribution (128): rank 1 — 30, 2–5 — 31, 6–10 — 9, 11–25 — 20, 26–100 — 28, >100 — 10. Провал бимодальный: target почти никогда не бывает "чуть за Top-5".
- Representative и hard — разные режимы: representative Top-1 42.19% → 64.06% при center_crop_70 (R@5 85.94%); hard Top-1 4.69% → максимум 9.38%, family Top-1 ≤ 20.31%.
- center_crop_70 — лучший одиночный метод (Top-1 35.16%, R@5 69.53%); multicrop_mean — самый безопасный по transitions (11 wrong→correct, 0 correct→wrong Top-1). center_crop_55 помогает только distance_crop (R@5 75.00%) и разрушает glare_bad_light (R@5 40.62%, median 20) и hard.
- distance_crop: простой детерминированный zoom восстанавливает retrieval (baseline R@5 34.38% → 65.62–75.00% у crop-методов) — гипотеза "бутылка слишком мала в кадре" подтверждена для этого сценария.
- Family retrieval восстанавливается crop'ом (family R@5 до 59.38%, disambiguation до 66.67%), но точный выбор члена семьи (vintage/subtype) остаётся узким местом — это аргумент за будущий OCR/reranking, а не за замену encoder или fine-tuning.
- Baseline Top-5 alignment с frozen run: 125/128; 3 расхождения — near-tie перестановки позиций внутри Top-5 из-за cross-run float nondeterminism; overall Top-1/R@5 воспроизведены точно.
- Артефакты: `artifacts/experiments/siglip2_generated_stress_pilot32_crop_diagnostics_20260919T163711Z/`; отчёты: `reports/generated_stress_pilot32_diagnostic_report.md`, `reports/generated_stress_pilot32_error_analysis.html`. Подготовлен (но не выполнен) manual oracle crop workflow: `scripts/oracle_manual_crop.py` + `data/benchmarks/generated_stress_dev_pilot32/oracle_crop_bboxes.csv` (header-only).

## Подтверждено controlled сравнением preprocessing_v1 (2026-09-19)

- Реализована reusable query-view абстракция: `baseline_full` (1 view) и `preprocessing_v1` (full + center 85% + center 70%, один batched forward на query, mean similarity). Reference embeddings и benchmark datasets не менялись; стратегия фиксирована для всех queries, target/scenario/subset при inference не использовались.
- pilot32: Top-1 23.44% → 29.69%, R@5 47.66% → 62.50%, R@10 54.69% → 71.88%, median rank 6.5 → 3.0; transitions: 8 wrong→correct / 0 correct→wrong Top-1, 20 gained / 1 lost Top-5. Per scenario R@5: distance_crop 34.38→53.12, glare 56.25→65.62, handheld 50.00→65.62, slight_angle 50.00→65.62. Representative R@5 57.81→76.56; hard R@5 37.50→48.44 (exact Top-1 4.69→9.38, family Top-1 15.62→20.31, family R@5 46.88→54.69).
- synthetic_dev деградирует: Top-1 89.40% → 85.63% (-3.77pp), R@5 99.83% → 98.92%, 232 correct→wrong против 78 wrong→correct — mean с crop-view размывает exact-match margin на reference-derived queries.
- hard_v2 деградирует: Top-1 68.43% → 66.17% (-2.26pp), R@5 99.65% → 98.35% — тот же margin-compression эффект на clean hard queries.
- Latency: mean 12.2 → 25.3 ms (2.08×), p95 28.7 ms — значительно ниже SLA 3 s. Три views кодируются одним forward pass (batch=3 на query).
- Вердикт: preprocessing_v1 НЕ принят как безусловный default (провален заранее объявленный guard ≤1pp на synthetic/hard); данные мотивируют confidence-gated условную политику как отдельный следующий milestone. Артефакты: `artifacts/experiments/siglip2_preprocessing_v1_20260919T194507Z/`, отчёт `reports/preprocessing_v1_evaluation.md`.

## Подтверждено confidence_gated_v1 milestone (2026-09-19)

- Реализован confidence-gated fallback: full-image ranking первым, при `top1_score < 0.8824` — fallback на crop85+crop70 одним batch=2 forward (full embedding не пересчитывается, агрегация mean идентична preprocessing_v1). Gate использует только score-распределение full ranking; ground truth/labels/crops в решении не участвуют.
- Калибровка: product-level split pilot32 16/16 (стратификация representative/vintage/subtype, seed 20260919, fingerprint в selected_gate.json) + детерминированные clean-подвыборки (400 synthetic products, 120 hard_v2 products); 29 кандидатов из фиксированных quantile-сеток оценивались только на калибровочных подмножествах; guard ≤1pp Top-1 на обоих clean-наборах прошли 3 кандидата; margin-правила не прошли (clean margins малы в абсолюте).
- Frozen gate: `fallback if top1_score < 0.8824`. Воспроизводимость: 0 расхождений между калибровочным композитом и свежим прогоном.
- synthetic_dev: Top-1 89.40% → 88.10% (-1.30pp; v1 давал -3.77pp), R@5 99.83% → 99.17%, fallback 19.17%, mean latency 13.8 ms (1.38 views/query). hard_v2: Top-1 68.43% → 68.70% (+0.26pp), R@5 99.65% → 98.87%, fallback 20.52%.
- pilot32: Top-1 29.69% и R@5 62.50% — идентично unconditional preprocessing_v1 при fallback 93.75%; median rank 6.5 → 3.0. Held-out 16 продуктов (64 queries, не участвовавшие в калибровке): baseline R@5 45.31% → gated 62.50%, Top-1 23.44% → 28.13% — полный перенос gain без подгонки.
- Gate confusion (pilot32): 95 useful triggers (74.2%), 25 unnecessary, 3 missed; в fallback-сете pilot32 0 correct→wrong Top-1. synthetic fallback-сет нет-негативен по Top-1 (28 rescued / 81 broken) — честная цена глобального порога.
- Latency: clean ~13.8–14.0 ms mean (p95 26.3 ms); на generated с fallback 93.75% mean 35.3 ms — выше unconditional v1 (29.3 ms), т.к. два последовательных forward (1+2) вместо одного batch=3; все значения далеки от SLA 3 s.
- Вердикт: confidence_gated_v1 зафиксирован как retrieval default; финальный confidence остаётся доступным для будущего пользовательского Smart Retry (отдельный слой). Vintage/subtype дизамбигуация hard-подмножества не решена (exact Top-1 9.38%) — следующий milestone: OCR/reranking. Артефакты: `artifacts/experiments/siglip2_confidence_gated_v1_20260919T204926Z/`, отчёт `reports/confidence_gated_v1_evaluation.md`.

## Подтверждено OCR-reranking milestone на so400m backbone (2026-09-20)

- Encoder bake-off зафиксировал `siglip2_so400m_384` как новый retrieval backbone: synthetic_dev Top-1 95.10%, hard_v2 86.87%, pilot32 73.44% (R@5 100% на generated). Следом реализован OCR + text/metadata reranking строго поверх готового Top-5: OCR не участвует в candidate generation.
- Local OCR: PaddleOCR 3.7.0 + paddlepaddle-gpu 3.3.1 (cu126), PP-OCRv5_server_det + eslav_PP-OCRv5_mobile_rec, GPU; полностью локально, без LLM и облачных API. CPU-режим работал только с `enable_mkldnn=False`. Установка GPU-колеса перезаписала `nvidia/nccl/lib/libnccl.so.2` общего пути (конфликт cu12/cu13) — восстановлено переустановкой `nvidia-nccl-cu13==2.30.7`.
- OCR-кэши построены один раз: 2042 reference (1.81% пустых, медиана 7 токенов, 48.09% с детектированным годом) и 5362 уникальных query (98.3–99.2% с токенами, 42–48% с винтажом, 28–44% с именем — стилизованные шрифты названий читаются хуже всего). Query и reference кэши раздельны, OCR дедуплицируется по sha256 и не перезапускается экспериментами.
- Baseline reproduction (policy image_only) совпал с frozen bake-off точно: per-query agreement 1.0000 и те же 95.10% / 86.87% / 73.44%.
- Калибровка (seed 20260920, fingerprint): pilot32 по продуктам 16/16 (stratified), hard_v2 по цельным семьям 128/128, synthetic 400 продуктов; 32 конфига, guard ≤1pp synthetic, лексикографический выбор. Все конфиги совпали с image_only на pilot32 held-out (78.12%), выбор упал на hard_v2 calibration Top-1: frozen `reference_ocr_blend, alpha 0.30, vintage ±0.05, margin 0.05`.
- Полные множества: hard_v2 86.87% → 87.91% (+1.04pp, 16 rescued / 4 broken), synthetic 95.10% → 95.27% (+0.17pp, 22 / 15), pilot32 73.44% без изменений (0 / 0). R@5/R@10 инвариантны по построению (перестановка только внутри Top-5).
- Честные негативы: винтажные подмножества не сдвинулись ни при одной политике (pilot32 vintage 65.63%, hard_v2 vintage 76.47% — без изменений); весь прирост hard_v2 — subtype-дизамбигуация. Text-oracle на pilot32 72.7% (combined) / 75.0% (с винтажом) против 73.44% image baseline — OCR-сигнал на generated в лучшем случае на паритете, поэтому консервативный fusion корректно не переставляет. Ablation показал, что `combined_text_blend alpha 0.40` на полных множествах выше frozen-выбора (hard_v2 88.26%) — цена заморозки на calibration-половинах, зафиксирована как evidence.
- Latency: retrieval ~41 ms + OCR ~237 ms + rerank 0.2 ms ≈ 280 ms mean (p95 ~264 ms) — существенно ниже SLA 3 s. Peak VRAM: retrieval 2226.9 MB (torch), OCR ~2.1 GB device-wide.
- Вердикт: OCR-reranking принят как консервативный post-retrieval этап (net-positive на clean/hard, строго нейтрален на generated, SLA-safe). Vintage-дизамбигуация не решена; узкое место — дискриминативность OCR на стилизованных названиях/винтаже, а не веса fusion. Артефакты: `artifacts/experiments/so400m_ocr_reranker_20260920T193925Z/` (retrieval dumps `..._192004Z/`), отчёты `reports/so400m_ocr_reranker_report.md` и `reports/ocr_reranker_error_analysis.html`.
