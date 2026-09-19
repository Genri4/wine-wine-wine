# Canonical catalog mapping report

Построено read-only builder-ом из `raw_data/Датасет/`. Raw data не изменялись.
ML/CV/OCR не запускались. Fuzzy matching не использовался.

## Summary

- Canonical products: **2103**.
- Удалено только exact duplicate rows: **2044**.
- Automatically matched: **1920**.
- Manually matched: **122**.
- Matched total: **2042**.
- Missing assets по ручным решениям: **0**.
- Unresolved total: **61**.
- Ambiguous: **0**.
- Unmatched: **61**.
- Total usable catalog coverage: **2042/2103 = 97.10%**.
- По сравнению с предыдущим catalog pass: **+292 matched** и **+13.88 п.п.** coverage (77.41% → 91.30%).
- Извлечено reference images: **2042**.
- Media inventory: **15803** файлов, из них **15770** изображений.

## Mapping methods

| Метод | Количество |
|---|---:|
| `manual_override_matched` | 122 |
| `no_deterministic_media_candidate` | 61 |
| `strapi_normalized_crc_identical_original` | 6 |
| `strapi_normalized_original` | 1628 |
| `transliterated_normalized_crc_identical_original` | 4 |
| `transliterated_normalized_original` | 205 |
| `transliterated_punctuation_normalized_crc_identical_original` | 8 |
| `transliterated_punctuation_normalized_original` | 69 |

Для автоматически matched products выбран `original`; в 18 случаях несколько originals были приняты только при одинаковых size+CRC. 122 products дополнительно закрыты через manual override. Fallback с выбором максимальной площади resize-варианта не потребовался.

## Manual override status

Источник решений: `data/manual/catalog_mapping_overrides.csv`.
Применено строк: **122** (matched **122**, missing **0**, unresolved **0**).
До ручной верификации остаются **61** products: ambiguous **0**, unmatched **61**, manual unresolved **0**.
Workflow: открыть `reports/mapping_review.html` → выбрать candidate → записать одну строку в `data/manual/catalog_mapping_overrides.csv` → запустить builder с `--overwrite-generated` → проверить manifest и этот report.

## Manual review groups

- A. Ambiguous: **0** products.
- B. Cyrillic/non-ASCII unmatched: **5** products.
- C. ASCII unmatched: **39** products.
- D. Space-containing unmatched: **7** products.
- E. Opaque/random unmatched: **10** products.
- F. Shared-image risk: **13** photo_name groups / **26** products.

## Unique photo-name view

Предыдущая оценка **439** относилась к уникальным `photo_name`. В текущем проходе из **2090** уникальных имён: matched 2033, unmatched 55, mixed 2. Product-level counts выше из-за shared photo names.

## Transliteration review

Для ручной проверки отобраны первые **30** уникальных transliteration matches. Правило не изменило ни один из 1628 прежних matched mappings и не создало collision среди них.

Первые 20 репрезентативных пар:
- `2023г_Aya_Evolution_Pinot_noir.webp` → `2023g_Aya_Evolution_Pinot_noir_b850f06043.webp` (`evolution-reserve-pinot-noir`, AYA Organic Wine & Vineyards).
- `2ыфваываы.webp` → `2yfvayvay_83316bd121.webp` (`katharon-semi-sweet`, KATHARON).
- `AYA Мокап апассименто.webp` → `AYA_Mokap_apassimento_437849eda9.webp` (`appassimento-merlot`, AYA Organic Wine & Vineyards).
- `AYA Мокапы игристое белое.webp` → `AYA_Mokapy_igristoe_beloe_09976d020f.webp` (`evolution-pinot-noir-bryut-beloe`, AYA Organic Wine & Vineyards).
- `AYA Мокапы игристое розе.webp` → `AYA_Mokapy_igristoe_roze_e3c51b7973.webp` (`evolution-pinot-noir-bryut-rozovoe`, AYA Organic Wine & Vineyards).
- `CANTIANI_Igrist_Brut_vid5_042026 копия.webp` → `CANTIANI_Igrist_Brut_vid5_042026_kopiya_1b13527c60.webp` (`cantiani-brut`, Шато АЛВИСА).
- `CANTIANI_Igristoe Chardon_444х1200px.webp` → `CANTIANI_Igristoe_Chardon_444h1200px_fa4fe5363b.webp` (`cantiani-chardonnay`, Шато АЛВИСА).
- `CANTIANI_Igristoe Rkats_vid_032025 копия.webp` → `CANTIANI_Igristoe_Rkats_vid_032025_kopiya_29e44021af.webp` (`cantiani-rkatsiteli-1`, Шато АЛВИСА).
- `CANTIANI_Igristoe white semi_444х1200px.webp` → `CANTIANI_Igristoe_white_semi_444h1200px_2d2892b235.webp` (`cantiani-semisweet`, Шато АЛВИСА).
- `CANTIANI_Igristoe-Riesl_vid_032025 копия.webp` → `CANTIANI_Igristoe_Riesl_vid_032025_kopiya_8150a5e627.webp` (`cantiani-riesling-1`, Шато АЛВИСА).
- `Cantiani-Riesl_New vid_042025 копия-fotor-bg-remover-20260821201017.webp` → `Cantiani_Riesl_New_vid_042025_kopiya_fotor_bg_remover_20260821201017_c6605edd27.webp` (`cantiani-riesling`, Шато АЛВИСА).
- `Cantiani-Rkats_New vid_042025 копия-fotor-bg-remover-20260821203120.webp` → `Cantiani_Rkats_New_vid_042025_kopiya_fotor_bg_remover_20260821203120_9079733698.webp` (`cantiani-rkatsiteli`, Шато АЛВИСА).
- `Caucasian Джарагъ.webp` → `Caucasian_Dzharag_300681b16f.webp` (`caucasian-dzharag`, Дербент Вино).
- `Daniel 2022 Коффманн.webp` → `Daniel_2022_Koffmann_5bba62cc20.webp` (`daniel-22`, WINEMAFIA).
- `Gai Kodzor Русан.webp` → `Gai_Kodzor_Rusan_5381eb38cc.webp` (`gaj-kodzor-rusan`, Виноградники Гай-Кодзора).
- `Gai Kodzor Совиньон блан.webp` → `Gai_Kodzor_Sovinon_blan_813e69be18.webp` (`gaj-kodzor-sovinon-blan`, Виноградники Гай-Кодзора).
- `Inkerman Шато Блан белое полусухое.webp` → `Inkerman_Shato_Blan_beloe_polusuhoe_526e6530b9.webp` (`inkerman-shato-blan`, Инкерманский ЗМВ).
- `Inkerman мускатное белое полусладкое.webp` → `Inkerman_muskatnoe_beloe_polusladkoe_092002be0d.webp` (`inkerman-muskatnoe-beloe`, Инкерманский ЗМВ).
- `Katharon Каб Фран сухое.webp` → `Katharon_Kab_Fran_suhoe_0ed5f19ec1.webp` (`katharon-katharon-kaberne-fran`, KATHARON).
- `Katharon Мерло сухое.webp` → `Katharon_Merlo_suhoe_6947446370.webp` (`katharon-katharon-merlo`, KATHARON).

## Shared photo names

**13** разных `photo_name` используются **26** canonical products (13 пар slug). Shared photo name не объявлялся ошибкой автоматически.

Примеры:
- `03-Silvaner-2022-Barrel-Fermented.webp` → `vibes-silvaner-barrel-fermented-2022`, `vibes-vermentino-viognier-barrel-fermented-2022`.
- `4285_eqF1Fau-no-bg-preview (carve.photos).webp` → `aligote-avtorskoe`, `aligote-avtorskoe-vino`.
- `4300_tlKHpEg.webp` → `novyj-svet-polusladkoe`, `novyj-svet-vyderzhannoe-bryut`.
- `DSC00836.webp` → `method-classic-kokur`, `pino-nuar-2025`.
- `DSC00839-Photoroom.webp` → `kokur-2025`, `kokur-suhoe-2025`.

## Unresolved reasons

- `ASCII name without deterministic candidate`: 39 canonical products.
- `non-ASCII/Cyrillic name`: 5 canonical products.
- `opaque/random + timestamp-like name`: 10 canonical products.
- `space-containing name`: 7 canonical products.

Категории unmatched — диагностические признаки имени, а не доказанные причины отсутствия asset.

## Problem examples

### Ambiguous

### Unmatched
- `avtohtonnoe-vino-kryma-beloe-suhoe`; photo `PUSjv7dnMNilH25TO9BcbFN55lPrHsK4eSLja8tm9Ur62n50PyxhlRO1RU4s3eNZb1YqGT4HJPD_77QgWTf3Ng==.webp`; ASCII name without deterministic candidate.
- `aligote-barrel-2024`; photo `DSC09173.webp`; ASCII name without deterministic candidate.
- `agora-rosa-viva-cabernet-franc`; photo `fpZEJhYZPstBYLV_1775199670.webp`; opaque/random + timestamp-like name.
- `agora-rosa-viva-cabernet-sauvignon-shiraz`; photo `22RA11aBWFX8Yu9_1775199820.webp`; opaque/random + timestamp-like name.
- `agora-rosa-viva-muscat`; photo `kOtRttCJvP1R7jP_1775199321.webp`; opaque/random + timestamp-like name.
- `agora-rosa-viva-riesling`; photo `BYrAbT0xREdRg8j_1775199389.webp`; opaque/random + timestamp-like name.
- `agora-rosa-viva-beloe-bryut`; photo `O4nl0YWdgyo6HNd_1775200002.webp`; opaque/random + timestamp-like name.
- `agora-rosa-viva-beloe-polusladkoe`; photo `ugbHOGJxxJ9PvCj_1775200212.webp`; opaque/random + timestamp-like name.
- `agora-rosa-viva-rozovoe-polusladkoe`; photo `obmsa6YZNZPDCKY_1775200312.webp`; opaque/random + timestamp-like name.
- `aya-khrustaleva-76-merlot-organik`; photo `PUSjv7dnMNilH25TO9BcbBjoR8uZySbkV5WTEJjg3CKy_5U-vUeQvP6o-lXGG5qNEDMtdZtuUjEtjwaQrtTXAg==.webp`; ASCII name without deterministic candidate.

## Limitations

- Mapping использует только детерминированную filename normalization и не назначает fuzzy candidates.
- `candidate_paths` сохранены в manifest/issues для ручного разбора ambiguous случаев.
- `mapping_review.html` и `mapping_review_candidates.csv` содержат metadata candidates, score и ссылки на извлечённые ambiguous candidates; они не используются для auto-mapping.
- Manual overrides валидируются по slug и media inventory и применяются после deterministic mapping; решения не зашиты в Python-код.
- Query labels из `eval.zip` не использовались и не создавались.
- 7-Zip при listing RAR сообщил `There are data after the end of archive`; full integrity test не выполнялся.
