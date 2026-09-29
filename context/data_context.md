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

## Подтверждено encoder bake-off (2026-09-20)

- Контролируемое сравнение 5 frozen encoders на неизменных benchmark-ах: current_siglip2 (base/224, reproduces frozen baseline exactly), dinov2_vitl14_reg, pe_core_l14_336, dfn5b_h14_378, siglip2_so400m_384. Протокол: per-model reference index (2042), cosine по L2-normalized global embeddings, single full image, официальный preprocessing каждой модели.
- siglip2_so400m_384 — лучший: synthetic 95.10%, hard_v2 86.87% (+18.4pp), generated 73.44% / R@5 100% (vs 23.44%/47.66% у base). PE-Core-L14-336 идентичен на generated (73.44%/100%) и чуть ниже на clean. DINOv2-reg-L collapses на generated (Top-1 0.78%, median rank 339) — нет contrastive-робастности к domain shift.
- Oracle-дополнение: so400m + current_siglip2 дают oracle R@5 100% на всех трёх (ошибки частично разные); но solo so400m сильнее любого ансамбля с base.
- Ресурсы: fp16, peak VRAM 2.5 GB (so400m), latency 93 ms mean — SLA соблюдён.
- Вердикт: siglip2_so400m_384 зафиксирован как новый retrieval backbone. Артефакты: `artifacts/experiments/encoder_bakeoff_20260920T143928Z/`, отчёт `reports/encoder_bakeoff_report.md`.

## Подтверждено so400m OCR reranker milestone (2026-09-20)

- PaddleOCR 3.7.0 GPU (PP-OCRv5_server_det + eslav_PP-OCRv5_mobile_rec), полностью локально; OCR-кэши построены один раз (2042 references: 1.81% empty, 48.1% с годом; 5362 queries: 98.3-99.2% с токенами).
- Conservative text reranking поверх frozen SO400M Top-5 (OCR никогда не участвует в candidate generation). 5 фиксированных политик, калибровка на product/family-level splits, frozen: `reference_ocr_blend, alpha 0.30, vintage ±0.05, margin 0.05`.
- Baseline воспроизведён точно (top1 agreement 1.0 на всех трёх). Итог: hard_v2 86.87% → 87.91% (+1.04pp, 16/4), synthetic 95.10% → 95.27% (22/15), generated 73.44% без изменений (0/0).
- Vintage slice не двинулся ни одной политикой; весь hard-прирост — subtype disambiguation. Oracle: перфектный текстовый судья достигает только 72.7-75.0% на pilot32 vs 73.44% image baseline — OCR-сигнал на generated примерно на паритете.
- Вердикт: OCR reranking зафиксирован как conservative post-retrieval stage; bottleneck — дискриминативное чтение стилизованной типографики, не fusion-веса.

## Подтверждено OCR + reranker bake-off (2026-09-21)

- OCR-конфигурации: A current_eslav (baseline), B cyrillic (тот же detector + cyrillic_PP-OCRv5_mobile_rec), C PaddleOCR-VL 0.9B (v1.6, локально). Реестр `ocr_engine.OCR_CONFIGS` с per-config cache keys (eslav key сохранён); инкрементальные crash-safe кэши (append+resume).
- VL throughput ~4.9 s/image на RTX 4060 (GPU util 10-15% — autoregressive decode); полный кэш был бы ~8.5 ч, поэтому VL построен на references + hard_v2 + generated, synthetic_dev пропущен (disclosed). VL wrapper читает parsing_res_list блоки (confidence 1.0 — VL не даёт per-line confidences).
- Дискриминация (target-vs-best-wrong text margin внутри SO400M Top-5): eslav 0.054 (59.9% target beats best wrong) на hard, cyrillic 57.6%, VL 41.1% с margin 0.003. VL читает меньше (75.8% non-empty на generated vs 99.2%) и хуже дискриминирует.
- Vintage аудит: 103/2042 (5.04%) продуктов имеют год в title; ref-OCR год у 48.1%; согласование title×OCR года 94.9% (n=78) — ref-OCR год = derived evidence с provenance.
- Structured reranker: 21 bounded feature (image margins, fuzzy/token/IDF-overlap, numeric overlap, vintage states, query quality) + standardized LogisticRegression; обучение только на calibration units (family-level hard 128, product-level synthetic 400, product-level pilot32 16; seed 20260920). heldout-половины + generated heldout — чистая оценка.
- BGE cross-encoder BAAI/bge-reranker-v2-m3 (apache-2.0, fp16, ~1.1 GB VRAM): sigmoid score (query OCR text × candidate doc без slug), fusion grid image/text 0.9/0.1-0.7/0.3 на hard+generated.
- Матрица (Top-1): hard_v2 — eslav+blend 87.91% (16/4), cyrillic+BGE0.3 88.17% (60/45), eslav+structured 87.30% (49/44), cyrillic+structured 87.48% (54/47), VL+blend 86.26% (4/11 net-negative); synthetic — structured до 95.54%, blend 95.27%; generated — 73.44% везде, кроме VL+structured 74.22% (generated-hard 39/64 → 40/64 единственный +1).
- Feature importance LR: image_margin_top1 (-9.09) доминирует, затем ref_ocr_year_match (+0.88), query_has_year (-0.66) — реранкер опирается на image margin и год-согласованность.
- Latency: eslav OCR 0.27 s mean (p95 0.23), cyrillic 0.17 s, VL 4.14 s, BGE +133 ms/query; полный пайплайн ~0.4 s mean — SLA 3 s соблюдён. Peak VRAM BGE 1108 MB.
- Вердикт: **KEEP CURRENT OCR + CURRENT RERANKER** (eslav + reference_ocr_blend alpha 0.30). OCR upgrade gain не даёт; structured LR не превосходит blend по rescued/broken; BGE +0.26pp hard ценой generated-деградации и латентности. Bottleneck — стилизованная vintage-типографика (OCR не читает лучше) — следующий уровень: VLM/fine-tuned recognition, вне этого milestone. 15 новых тестов (234 total pass).
- Артефакты: `artifacts/experiments/ocr_reranker_bakeoff_20260921T1/` (signal_cache, bge, structured, combos, summaries), диагностика `artifacts/experiments/ocr_reranker_bakeoff_diag/`; отчёт `reports/ocr_reranker_bakeoff_report.md`.

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

## Подтверждено vintage disambiguation milestone (2026-09-22)

- Создан diagnostic vintage challenge slice v1 (НЕ headline benchmark): hard_v2 46 queries / 17 vintage семей, generated 23 queries / 8 продуктов. Включение: target в vintage family AND в SO400M Top-5 AND same-family конкурент с другим известным годом тоже в Top-5. Артефакт: data/benchmarks/vintage_challenge_v1/ (перезаписывается из раннера), manifests в artifacts/experiments/vintage_disambiguation_20260922T1/.
- Candidate year evidence с provenance (catalog_metadata / product_title / reference_ocr / multiple_sources_agree; конфликт -> unknown_conflict): 934/2042 (45.7%) продуктов с годом; hard vintage-family 33/34; pilot32 vintage 8/8.
- Oracle ceiling (семантика без double counting): perfect query year спасает ВСЕ baseline-wrong vintage queries — 8/8 hard, 6/6 generated; потолок 100% на обоих слайсах (7/46 hard ambiguous: два члена семьи носят год target).
- Detector-box retry (повторное распознавание собственных det-боксов query при 1x/2x/4x): correct-year 56.5% -> 58.7% (hard 4x), 47.8% -> 52.2% (generated) — прирост маргинальный; bottleneck сам correct-year rate, не разрешение.
- Conservative family-scoped vintage rerank: +1 rescued / 0 broken на hard challenge, 0/0 на generated.
- Reference-guided year crop (SIFT + ratio-test + RANSAC с validity gates): alignment success 100% hard / 95.7% generated (median 227/29 good matches); projected-crop year читается в 53/61 aligned hard crops, 30 совпадают с годом кандидата. Conservative candidate-specific evidence дал 1 swap: спасён кейс fanagoriya-primum-alveus-brut-2014 (2016-продукт на Top-1, проекции читают 2014) — +1/-0.
- Latency: детекция 362 ms mean, SIFT 486 ms/кандидат; stage условный (same-family vintage ambiguity, ~10% hard queries) — SLA соблюдён.
- Вердикт: ADD REFERENCE-GUIDED VINTAGE STAGE как conditional conservative layer (C). Gain мал, но строго неотрицателен; масштабируется до +8/-0 hard при росте correct-year OCR. Не принят как безусловный default; production trigger в API не заведён. 23 новых теста (257 total pass).
- Артефакты: artifacts/experiments/vintage_disambiguation_20260922T1/; отчёт reports/vintage_disambiguation_report.md; модули src/recognition/vintage_disambiguation.py и src/recognition/reference_guided_year.py.

## Подтверждено candidate-constrained vintage recognition (2026-09-23)

- Восстановлены SIFT-aligned year crops детерминированно из frozen reference_guided_per_query.csv (69 queries: 61+21 query crops, 16+3 reference crops). Task framing: выбрать один allowed year из годов same-family Top-5 кандидатов; target identity на inference недоступна.
- Методы на decided queries: current eslav OCR 12/12+12/12; SO400M crop matching 25/25+12/12 (100%); PE-Core идентично; DINOv2 local-patch 64-67% с 11 broken (гипотеза local-matching отвергнута); digit-only CRNN (2988 synthetic crops, constrained CTC) 56-67% (synthetic-to-real gap); VLM ceiling Qwen2-VL-2B (apache-2.0, fp16) 24/25+12/12 = 96-100%.
- Решающая находка: bottleneck — availability, не recognition. 21/46 hard и 11/23 generated challenge queries не имеют year crop (eslav не прочитал reference год). 6/8 hard и 6/6 generated baseline-wrong — в этой no_crop группе, недостижимы ни одним crop-методом.
- Family-safe swap: hard rescued 1/0 broken (2014/2016 fanagoriya), generated 0/0. Ekstra-2017 swap корректно заблокирован (оба члена носят 2017 — subtype случай).
- Latency: crop matching 12-15 ms, digit ~2 ms, VLM 190-820 ms (ceiling only). Починено окружение: nvidia-cudnn-cu13 переустановлен (paddle installation подменил libcudnn.so.9 на stub, ломая torch LSTM).
- Вердикт: USE VISUAL REFERENCE-CROP MATCHING (SO400M/PE-Core chooser в family-scoped vintage stage) + offline VLM pass по vintage reference images для создания недостающих year boxes — это двигает 6/8 hard и 6/6 generated baseline-wrong из недостижимых в достижимые. VLM как production runtime отвергнут (latency), но его 96-100% подтверждает наличие сигнала в crops. 257 tests pass.
- Артефакты: artifacts/experiments/candidate_constrained_vintage_20260922T1/; отчёт reports/candidate_constrained_vintage_report.md; модули constrained_vintage.py, year_recognizer.py, vlm_year.py.

## Подтвержден final ML sanity check: error audit + Top-5 SIFT geometry (2026-09-23)

- Текущий production candidate воспроизведён в точности по сохранённым сигналам: hard_v2 1150/1150 и generated pilot32 128/128 совпадений полного Top-5 порядка. SIFT сравнивает только этот фиксированный Top-5; target не используется при inference.
- Из 139 hard_v2 ошибок target уже находится в Top-5 у 137 (98.6%), retrieval failures — 2. Из 34 generated ошибок target в Top-5 у всех 34. Проверка изображений и family evidence отметила near-identical packaging в 114 hard и 16 generated ошибках; сохранённый OCR evidence вводит в заблуждение в 4 hard ошибках. Generated domain-shift метка означает контекст принятого stress scenario, не доказанную причинность.
- SIFT descriptors offline закэшированы для всех 2042 usable references; fingerprint включает OpenCV version, параметры SIFT и SHA-256 каждого reference image. Посчитано ровно 5 пар на query: 5750 hard и 640 generated. Валидная homography: 85.1% hard и 54.8% generated.
- Фиксированный fusion `current production + 0.40 × normalized geometry` дал hard_v2 87.91% -> 91.91% (+4.00 pp) и generated pilot32 73.44% -> 80.47% (+7.03 pp), сохранив R@5 (99.83% / 100%). Current->fusion transitions: 51/5 hard и 12/3 generated rescued/broken; суммарно 63/8.
- Geometry-only Top-1: 89.74% hard, 76.56% generated. Среди текущих ошибок с target в Top-5 target имеет geo rank 1 у 45.3%/52.9%, rank ≤2 у 97.8%/88.2% (hard/generated). Selected fusion повышает hard vintage 76.5% -> 83.8%, subtype 90.7% -> 94.0%, other family 83.7% -> 88.7%; generated vintage 65.6% без изменений, subtype 56.2% -> 59.4%.
- Synthetic post-selection sanity check: Top-1 95.27% -> 97.14%, R@5 без изменений 99.95%. Дополнительная latency query SIFT + пять матчей: 64.7 ms mean (hard), 217.2 ms (generated); расчётный полный pipeline p95 431.5/558.2 ms, SLA 3 s соблюдён.
- Решение: **FIX GEOMETRIC RERANKER**, fixed weight 0.40; **CONTINUE ML только для подключения этого проверенного сигнала** к recognition path. Новые model/feature исследования отложить до появления реальных field queries. Production app path пока не подключён к выбранной политике. Полный suite: 274 passed.
- Отчёт и визуальный audit: `reports/final_ml_geometric_reranker_report.md`, `reports/final_ml_error_audit.md`, `reports/final_ml_error_audit.html`. Experiment: `artifacts/experiments/final_ml_geometric_reranker_20260923T082259Z/`.

## Подтвержден strong local visual Top-5 reranking milestone (2026-09-23)

- Frozen current baseline воспроизведён точно: 1150/1150 hard_v2 и 128/128 generated Top-5 порядков; SIFT 0.40 воспроизведён для всех 1278 queries. Candidate set оставался ровно Top-5; generated R@5 остался 100%.
- SIFT valid homography: 4891/5750 (85.06%) hard и 351/640 (54.84%) generated. Geometry-only Top-1: 89.74% / 76.56%; frozen current+SIFT: 91.91% / 80.47%.
- LightGlue официальный README/license проверены: документированы SIFT, ALIKED, DISK; код/matcher weights Apache-2.0, DISK Apache-2.0, ALIKED BSD-3-Clause. Официальные clone/archive downloads завершились timeout при недоступной сети; LightGlue SIFT/ALIKED/DISK фактически не запускались и их метрики не заявляются.
- Aligned SO400M и PE-Core по отдельности дали generated 80.47%, generated-hard 60.94%, по 1 rescued / 1 broken относительно SIFT. SO400M был быстрее по generated p95 (1077 ms против 1800 ms для PE-Core). Multiview mean/max дали 76.56% / 78.91%, с регрессией на generated-hard до 60.94%.
- Из 25 remaining generated errors aligned SO400M выигрывает у incumbent на 8, PE-Core на 7, multiview mean на 4 и max на 2; oracle union измеренных новых сигналов — 12/25. До 90% нужно 13 net corrections: ceiling измеренного union — 115/128 = 89.84%. LightGlue в ceiling не входит, так как не был измерен.
- Best fixed fusion grid candidate — `combined_local_signals_w0.10`: hard 92.00%, generated 80.47% (103/128), generated-hard 62.50%, 0 rescued / 0 broken; R@5 100%. Estimated total generated p95 — 2314 ms, ниже SLA 3 s. Метрика не достигает целевых 116/128.
- Production recommendation remains `current_plus_sift_w0.40`; its already completed post-selection synthetic sanity check was reused because this verdict leaves SIFT unchanged: synthetic_dev 97.14% Top-1 / 99.95% R@5. The synthetic split was not used for local-weight selection or retuning.
- Вердикт: **KEEP_SIFT**. Проверенные local alignment/multiview сигналы не добавили net Top-1 gains; production policy не менялась, fine-tuning не начинался. Для не протестированного LightGlue остаётся неизвестной потенциальная польза.
- Артефакты: `artifacts/experiments/strong_local_visual_reranker_final_20260923T1000Z/`; отчёт `reports/strong_local_visual_reranker_report.md`; gallery `reports/local_visual_reranker_errors.html`; runner `scripts/run_strong_local_visual_reranker.py`; local signal module `src/recognition/local_visual_reranker.py`; tests `tests/test_local_visual_reranker.py`. Полный suite: 286 passed (3 sklearn deprecation warnings).

## Получен новый набор без slug (2026-09-24)

- В `data/new_data/` найдено **100** верхнеуровневых WebP-фотографий; рядом лежат 100 AppleDouble sidecars в `__MACOSX/`, это метаданные macOS, не разметка. У изображений нет slug manifest, EXIF-разметки или других подтверждённых target labels; точных дубликатов по SHA-256 нет.
- Имена соответствуют шаблону `<числовой префикс>_<дата>_<время>.webp`; смысл префикса и происхождение/тип снимков пока **не подтверждены**. Префикс сохранён как исходный атрибут и не считается confidence или правильной меткой.
- Задача трактуется как open-set: у фото может быть slug из основного набора, товара может не быть в каталоге, либо решение может остаться неопределённым. Каждое фото ранжировалось против всех 2042 usable references; image-only frozen SO400M и выбранный R16 сохраняют Top-5 кандидатов, query batch size=1. Top-1 совпал у моделей на **51/100** изображениях — это agreement, не accuracy.
- Для новых фото нет калиброванного порога open-set отказа, поэтому по cosine сходству автоматически не выставляется «в каталоге нет». HTML review позволяет выбрать slug из полного каталога, отметить «slug в каталоге нет» или «пока неясно», сохраняет выборы в браузере и выгружает CSV. OCR/SIFT на новом наборе не запускались; raw-файлы не менялись, обучение и benchmark-selection не выполнялись.
- Review artifacts: `artifacts/experiments/new_data_slug_audit_20260924/` (`slug_candidates.csv`, `slug_candidate_review.html`, `audit_metadata.json`, resumable `progress.json`); генератор: `scripts/audit_unlabeled_new_data.py`.
- Из набора оформляется отдельный `new_data_open_set_v1`, не смешиваемый с прежними benchmark-ами. Новый threshold-free показатель `open_set_retrieval_auc` — площадь под кривой known exact classification rate vs unknown false-accept rate при пороге на max cosine similarity; диагностики: known Top-1 и Recall@5. Реализован evaluator `scripts/evaluate_new_data_open_set.py`; он сравнивает frozen SO400M и selected R16 image-only ранжирования и принимает human-reviewed CSV из HTML.
- Числовой score пока не заявлен: исходные 100 фото без labels. Полный score можно считать только после разрешения всех кадров в `catalog_match` с подтверждённым slug либо `no_catalog_match`; `uncertain`/`unreviewed` останутся видимыми и дают только provisional report. Протокол: `data/benchmarks/new_data_open_set_v1/README.md`. Метрика не заменяет полный OCR/SIFT production evaluation.
