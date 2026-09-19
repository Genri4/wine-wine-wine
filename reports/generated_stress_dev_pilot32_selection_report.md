# generated_stress_dev_pilot32 selection report

Изолированный budgeted pilot plan. Скрипт только читает существующие артефакты; paid generation не запускалась, generated images не создавались, full `generated_stress_dev` не изменялся.

## Design

- Fixed seed: **20260916**.
- Products: **16 representative + 16 hard = 32 unique products**.
- Hard structure: **8 families × 2 products = 16 hard products**.
- Hard type split: **4 vintage families / 8 products** and **4 subtype families / 8 products**.
- Planned generations: **128 = 32 × 4**.
- Representative source: `existing generated_stress_dev 150-product selection, excluding all v2 hard-family products`.
- Hard source: `data/benchmarks/hard_near_duplicate_dev_v2/families.csv`; selection uses frozen v2 evidence, not a new model run.

Representative products cover the capture/domain-shift axis without deliberately concentrating the sample in v2 hard families. Hard products add a controlled family-aware diagnostic axis; keeping the halves separate allows both effects to be reported independently.

## Selection rules

- Representative: proportional `category × region` quotas, deterministic winery round-robin, fixed SHA-256 ordering inherited from the existing generated-stress selector; all v2 hard-family products are excluded.
- Vintage family: exactly two products, two distinct explicit years, identical normalized product line after removing the year, and at least two v2 strong signals.
- Subtype family: exactly two products, same winery/product-line tokens, at least two v2 strong signals, and a meaningful category/color/grape or name-variant difference.
- Winery diversity is maximized before evidence score; ties are resolved by source family id.
- Metadata-only equality is not sufficient to create a hard family.

## Catalog and subset distributions

Usable catalog: **2042** products. Existing full generated-stress selection: **150**. Representative subset: **16**. Combined pilot32: **32**.

### Category

| value | catalog | current 150 | representative | pilot32 |
|---|---:|---:|---:|---:|
| Белое | 963 (47.2%) | 71 (47.3%) | 9 (56.2%) | 13 (40.6%) |
| Красное | 786 (38.5%) | 57 (38.0%) | 5 (31.2%) | 13 (40.6%) |
| Оранжевое | 16 (0.8%) | 1 (0.7%) | 0 (0.0%) | 2 (6.2%) |
| Розовое | 277 (13.6%) | 21 (14.0%) | 2 (12.5%) | 4 (12.5%) |

### Region

| value | catalog | current 150 | representative | pilot32 |
|---|---:|---:|---:|---:|
| Дагестан | 94 (4.6%) | 7 (4.7%) | 1 (6.2%) | 3 (9.4%) |
| Дальневосточная зона | 1 (0.0%) | 0 (0.0%) | 0 (0.0%) | 1 (3.1%) |
| Долина Дона | 76 (3.7%) | 6 (4.0%) | 1 (6.2%) | 1 (3.1%) |
| Крым | 722 (35.4%) | 53 (35.3%) | 6 (37.5%) | 13 (40.6%) |
| Кубань | 1058 (51.8%) | 78 (52.0%) | 8 (50.0%) | 14 (43.8%) |
| Нижняя Волга | 22 (1.1%) | 1 (0.7%) | 0 (0.0%) | 0 (0.0%) |
| Самара | 21 (1.0%) | 2 (1.3%) | 0 (0.0%) | 0 (0.0%) |
| Северная Осетия — Алания | 5 (0.2%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| Ставрополье | 43 (2.1%) | 3 (2.0%) | 0 (0.0%) | 0 (0.0%) |

### Winery concentration

| metric | catalog | current 150 | representative | pilot32 |
|---|---:|---:|---:|---:|
| unique wineries | 135 | 74 | 10 | 18 |
| largest winery count | 98 | 4 | 2 | 2 |
| top-5 winery share | 20.4% | 11.3% | 62.5% | 31.2% |

### Representative quotas

| category × region stratum | quota |
|---|---:|
| `category=Белое|region=Дагестан` | 1 |
| `category=Белое|region=Долина Дона` | 1 |
| `category=Белое|region=Крым` | 3 |
| `category=Белое|region=Кубань` | 4 |
| `category=Белое|region=Нижняя Волга` | 0 |
| `category=Белое|region=Ставрополье` | 0 |
| `category=Красное|region=Дагестан` | 0 |
| `category=Красное|region=Долина Дона` | 0 |
| `category=Красное|region=Крым` | 2 |
| `category=Красное|region=Кубань` | 3 |
| `category=Красное|region=Ставрополье` | 0 |
| `category=Розовое|region=Дагестан` | 0 |
| `category=Розовое|region=Долина Дона` | 0 |
| `category=Розовое|region=Крым` | 1 |
| `category=Розовое|region=Кубань` | 1 |
| `category=Розовое|region=Самара` | 0 |

## Hard families

| pilot family | source family | type | products | source signals | confusion | rank 2–5 | margin count | visual pairs | name pairs |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| `pilot-family-vintage-01` | `family-040` | `vintage` | Cabernet Franc-Pinot Noir 2021, Cabernet Franc-Pinot Noir 2022 | `image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin` | 2 | 2 | 2 | 1 | 1 |
|  |  |  |  | same normalized product line after removing explicit year; distinct explicit years=2021,2022; source v2 evidence=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; margin=[0.0, 0.0]; target_ranks=[2] |  |  |  |  |  |
| `pilot-family-vintage-02` | `family-055` | `vintage` | David, 2020, David, 2021 | `image_similarity+normalized_name_similarity` | 0 | 0 | 0 | 1 | 1 |
|  |  |  |  | same normalized product line after removing explicit year; distinct explicit years=2020,2021; source v2 evidence=image_similarity+normalized_name_similarity; margin=[, ]; target_ranks=[] |  |  |  |  |  |
| `pilot-family-vintage-03` | `family-119` | `vintage` | Шато Тамань Терруар Красностоп-Саперави 2022, Шато Тамань Терруар Красностоп-Саперави 2023 | `image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin` | 2 | 2 | 2 | 1 | 1 |
|  |  |  |  | same normalized product line after removing explicit year; distinct explicit years=2022,2023; source v2 evidence=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; margin=[0.007545351982116699, 0.014817535877227783]; target_ranks=[2] |  |  |  |  |  |
| `pilot-family-vintage-04` | `family-154` | `vintage` | Ркацители 2023, Ркацители 2024 | `image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin` | 2 | 2 | 2 | 1 | 1 |
|  |  |  |  | same normalized product line after removing explicit year; distinct explicit years=2023,2024; source v2 evidence=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; margin=[0.0, 0.0]; target_ranks=[2] |  |  |  |  |  |
| `pilot-family-subtype-01` | `family-030` | `subtype` | Бельбек Каберне Фран, Каберне Фран | `image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin` | 2 | 2 | 2 | 1 | 1 |
|  |  |  |  | same winery/product-line tokens; line_similarity=1.000; metadata difference; source v2 evidence=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; margin=[0.0022734403610229492, 0.010294437408447266]; target_ranks=[2] |  |  |  |  |  |
| `pilot-family-subtype-02` | `family-065` | `subtype` | Эндемы Шардоне, Эндемы Шардоне | `image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin` | 2 | 2 | 2 | 1 | 1 |
|  |  |  |  | same winery/product-line tokens; line_similarity=1.000; metadata difference; source v2 evidence=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; margin=[0.0, 0.0]; target_ranks=[2] |  |  |  |  |  |
| `pilot-family-subtype-03` | `family-069` | `subtype` | White Blend Lipko, White Blend | `image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin` | 2 | 2 | 2 | 1 | 1 |
|  |  |  |  | same winery/product-line tokens; line_similarity=1.000; metadata difference; source v2 evidence=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; margin=[0.007490754127502441, 0.010960102081298828]; target_ranks=[2] |  |  |  |  |  |
| `pilot-family-subtype-04` | `family-083` | `subtype` | Alveus Ultra Cuvee. Брют розовое, Alveus Ultra Cuvee. Экстра брют розовое | `image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin` | 2 | 2 | 2 | 1 | 1 |
|  |  |  |  | same winery/product-line tokens; line_similarity=1.000; metadata difference; source v2 evidence=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; margin=[0.00040221214294433594, 0.0025044679641723633]; target_ranks=[3] |  |  |  |  |  |

## All 32 products

| role | slug | product | winery | category | region | family | type | selection reason |
|---|---|---|---|---|---|---|---|---|
| hard | `belbek-belbek-kaberne-fran-krasnoe-suhoe-13` | Бельбек Каберне Фран | Бельбек | Красное | Крым | pilot-family-subtype-01 | subtype | selected from hard_near_duplicate_dev_v2 family-030; family_type=subtype; same winery/product-line tokens; line_similarity=1.000; metadata difference; source v2 evidence=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; signals=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; confusion=2; rank_2_5=2; small_margin=2; target_ranks=[2]; margin=[0.0022734403610229492,0.010294437408447266]; visual_pairs=1; name_pairs=1 |
| hard | `belbek-kaberne-fran-krasnoe-suhoe-134` | Каберне Фран | Бельбек | Красное | Крым | pilot-family-subtype-01 | subtype | selected from hard_near_duplicate_dev_v2 family-030; family_type=subtype; same winery/product-line tokens; line_similarity=1.000; metadata difference; source v2 evidence=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; signals=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; confusion=2; rank_2_5=2; small_margin=2; target_ranks=[2]; margin=[0.0022734403610229492,0.010294437408447266]; visual_pairs=1; name_pairs=1 |
| hard | `cabernet-franc-pinot-noir-2021` | Cabernet Franc-Pinot Noir 2021 | Vibes | Красное | Крым | pilot-family-vintage-01 | vintage | selected from hard_near_duplicate_dev_v2 family-040; family_type=vintage; same normalized product line after removing explicit year; distinct explicit years=2021,2022; source v2 evidence=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; signals=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; confusion=2; rank_2_5=2; small_margin=2; target_ranks=[2]; margin=[0.0,0.0]; visual_pairs=1; name_pairs=1 |
| hard | `cabernet-franc-pinot-noir-2022` | Cabernet Franc-Pinot Noir 2022 | Vibes | Красное | Дальневосточная зона | pilot-family-vintage-01 | vintage | selected from hard_near_duplicate_dev_v2 family-040; family_type=vintage; same normalized product line after removing explicit year; distinct explicit years=2021,2022; source v2 evidence=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; signals=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; confusion=2; rank_2_5=2; small_margin=2; target_ranks=[2]; margin=[0.0,0.0]; visual_pairs=1; name_pairs=1 |
| hard | `david` | David, 2020 | WINEMAFIA | Красное | Кубань | pilot-family-vintage-02 | vintage | selected from hard_near_duplicate_dev_v2 family-055; family_type=vintage; same normalized product line after removing explicit year; distinct explicit years=2020,2021; source v2 evidence=image_similarity+normalized_name_similarity; signals=image_similarity+normalized_name_similarity; confusion=0; rank_2_5=0; small_margin=0; target_ranks=[]; margin=[,]; visual_pairs=1; name_pairs=1 |
| hard | `david-2021` | David, 2021 | WINEMAFIA | Красное | Кубань | pilot-family-vintage-02 | vintage | selected from hard_near_duplicate_dev_v2 family-055; family_type=vintage; same normalized product line after removing explicit year; distinct explicit years=2020,2021; source v2 evidence=image_similarity+normalized_name_similarity; signals=image_similarity+normalized_name_similarity; confusion=0; rank_2_5=0; small_margin=0; target_ranks=[]; margin=[,]; visual_pairs=1; name_pairs=1 |
| hard | `derbent-vino-endemy-shardone-beloe-bryut-105-125` | Эндемы Шардоне | Дербент Вино | Белое | Дагестан | pilot-family-subtype-02 | subtype | selected from hard_near_duplicate_dev_v2 family-065; family_type=subtype; same winery/product-line tokens; line_similarity=1.000; metadata difference; source v2 evidence=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; signals=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; confusion=2; rank_2_5=2; small_margin=2; target_ranks=[2]; margin=[0.0,0.0]; visual_pairs=1; name_pairs=1 |
| hard | `derbent-vino-endemy-shardone-beloe-suhoe-13` | Эндемы Шардоне | Дербент Вино | Белое | Дагестан | pilot-family-subtype-02 | subtype | selected from hard_near_duplicate_dev_v2 family-065; family_type=subtype; same winery/product-line tokens; line_similarity=1.000; metadata difference; source v2 evidence=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; signals=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; confusion=2; rank_2_5=2; small_margin=2; target_ranks=[2]; margin=[0.0,0.0]; visual_pairs=1; name_pairs=1 |
| hard | `domaine-lipko-white-blend-lipko-muskat-belyy-beloe-polusuhoe-115` | White Blend Lipko | Domaine Lipko | Белое | Крым | pilot-family-subtype-03 | subtype | selected from hard_near_duplicate_dev_v2 family-069; family_type=subtype; same winery/product-line tokens; line_similarity=1.000; metadata difference; source v2 evidence=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; signals=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; confusion=2; rank_2_5=2; small_margin=2; target_ranks=[2]; margin=[0.007490754127502441,0.010960102081298828]; visual_pairs=1; name_pairs=1 |
| hard | `domaine-lipko-white-blend-muskat-beloe-polusuhoe-12` | White Blend | Domaine Lipko | Белое | Крым | pilot-family-subtype-03 | subtype | selected from hard_near_duplicate_dev_v2 family-069; family_type=subtype; same winery/product-line tokens; line_similarity=1.000; metadata difference; source v2 evidence=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; signals=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; confusion=2; rank_2_5=2; small_margin=2; target_ranks=[2]; margin=[0.007490754127502441,0.010960102081298828]; visual_pairs=1; name_pairs=1 |
| hard | `fanagoriya-alveus-ultra-cuvee-bryut-rozovoe-merlo-igristoe-bryut-rozovoe-12` | Alveus Ultra Cuvee. Брют розовое | Фанагория | Розовое | Кубань | pilot-family-subtype-04 | subtype | selected from hard_near_duplicate_dev_v2 family-083; family_type=subtype; same winery/product-line tokens; line_similarity=1.000; metadata difference; source v2 evidence=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; signals=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; confusion=2; rank_2_5=2; small_margin=2; target_ranks=[3]; margin=[0.00040221214294433594,0.0025044679641723633]; visual_pairs=1; name_pairs=1 |
| hard | `fanagoriya-alveus-ultra-cuvee-ekstra-bryut-rozovoe-merlo-12` | Alveus Ultra Cuvee. Экстра брют розовое | Фанагория | Розовое | Кубань | pilot-family-subtype-04 | subtype | selected from hard_near_duplicate_dev_v2 family-083; family_type=subtype; same winery/product-line tokens; line_similarity=1.000; metadata difference; source v2 evidence=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; signals=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; confusion=2; rank_2_5=2; small_margin=2; target_ranks=[3]; margin=[0.00040221214294433594,0.0025044679641723633]; visual_pairs=1; name_pairs=1 |
| hard | `kuban-vino-shato-tamane-terruar-krasnostop-saperavi-2022-krasnoe-suhoe-125` | Шато Тамань Терруар Красностоп-Саперави 2022 | Кубань-Вино | Красное | Кубань | pilot-family-vintage-03 | vintage | selected from hard_near_duplicate_dev_v2 family-119; family_type=vintage; same normalized product line after removing explicit year; distinct explicit years=2022,2023; source v2 evidence=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; signals=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; confusion=2; rank_2_5=2; small_margin=2; target_ranks=[2]; margin=[0.007545351982116699,0.014817535877227783]; visual_pairs=1; name_pairs=1 |
| hard | `kuban-vino-shato-tamane-terruar-krasnostop-saperavi-2023-krasnoe-suhoe-125` | Шато Тамань Терруар Красностоп-Саперави 2023 | Кубань-Вино | Красное | Кубань | pilot-family-vintage-03 | vintage | selected from hard_near_duplicate_dev_v2 family-119; family_type=vintage; same normalized product line after removing explicit year; distinct explicit years=2022,2023; source v2 evidence=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; signals=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; confusion=2; rank_2_5=2; small_margin=2; target_ranks=[2]; margin=[0.007545351982116699,0.014817535877227783]; visual_pairs=1; name_pairs=1 |
| hard | `oxana-istratova-wine-rkatsiteli-2023-oranzhevoe-suhoe-113` | Ркацители 2023 | Oxana Istratova Wine | Оранжевое | Крым | pilot-family-vintage-04 | vintage | selected from hard_near_duplicate_dev_v2 family-154; family_type=vintage; same normalized product line after removing explicit year; distinct explicit years=2023,2024; source v2 evidence=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; signals=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; confusion=2; rank_2_5=2; small_margin=2; target_ranks=[2]; margin=[0.0,0.0]; visual_pairs=1; name_pairs=1 |
| hard | `oxana-istratova-wine-rkatsiteli-2024-oranzhevoe-suhoe-12` | Ркацители 2024 | Oxana Istratova Wine | Оранжевое | Крым | pilot-family-vintage-04 | vintage | selected from hard_near_duplicate_dev_v2 family-154; family_type=vintage; same normalized product line after removing explicit year; distinct explicit years=2023,2024; source v2 evidence=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; signals=image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin; confusion=2; rank_2_5=2; small_margin=2; target_ranks=[2]; margin=[0.0,0.0]; visual_pairs=1; name_pairs=1 |
| representative | `aromatnoe-malbek-rezerv-krasnoe-suhoe-125` | Мальбек резерв | Ароматное | Красное | Крым |  |  | deterministic proportional category×region quota with winery round-robin; existing generated_stress_dev 150-product selection, excluding all v2 hard-family products |
| representative | `aromatnoe-shenen-blan-beloe-suhoe-13` | Шенен Блан | Ароматное | Белое | Крым |  |  | deterministic proportional category×region quota with winery round-robin; existing generated_stress_dev 150-product selection, excluding all v2 hard-family products |
| representative | `cantiani-semisweet` | Cantiani Semisweet | Шато АЛВИСА | Белое | Дагестан |  |  | deterministic proportional category×region quota with winery round-robin; existing generated_stress_dev 150-product selection, excluding all v2 hard-family products |
| representative | `czimlyanskoe-beloe-polusladkoe` | Цимлянское белое полусладкое | Цимлянские вина | Белое | Долина Дона |  |  | deterministic proportional category×region quota with winery round-robin; existing generated_stress_dev 150-product selection, excluding all v2 hard-family products |
| representative | `golubitskoe-estate-noble-selection-white-blend-shardone-beloe-suhoe-135` | Noble Selection White Blend | Golubitskoe Estate | Белое | Кубань |  |  | deterministic proportional category×region quota with winery round-robin; existing generated_stress_dev 150-product selection, excluding all v2 hard-family products |
| representative | `golubitskoe-estate-winery-series-pino-nuar-rozovoe-suhoe-115` | Winery Series. Пино Нуар | Golubitskoe Estate | Розовое | Кубань |  |  | deterministic proportional category×region quota with winery round-robin; existing generated_stress_dev 150-product selection, excluding all v2 hard-family products |
| representative | `millstream-cellar-blanc-de-blanc-ekstra-bryut-beloe` | MILLSTREAM Cellar Blanc de Blanc игристое экстра брют белое | MILLSTREAM | Белое | Кубань |  |  | deterministic proportional category×region quota with winery round-robin; existing generated_stress_dev 150-product selection, excluding all v2 hard-family products |
| representative | `millstream-cellar-select-saperavi` | MILLSTREAM Cellar Select Саперави | MILLSTREAM | Красное | Кубань |  |  | deterministic proportional category×region quota with winery round-robin; existing generated_stress_dev 150-product selection, excluding all v2 hard-family products |
| representative | `orlov_merlo` | Мерло | Винодельня Орлова | Красное | Кубань |  |  | deterministic proportional category×region quota with winery round-robin; existing generated_stress_dev 150-product selection, excluding all v2 hard-family products |
| representative | `orlov_sauvignon_blan` | Совиньон Блан | Винодельня Орлова | Белое | Кубань |  |  | deterministic proportional category×region quota with winery round-robin; existing generated_stress_dev 150-product selection, excluding all v2 hard-family products |
| representative | `shato-ay-danil-kapriz-muskat-belyy-beloe-suhoe-135` | Каприз | Шато Ай-Даниль | Белое | Крым |  |  | deterministic proportional category×region quota with winery round-robin; existing generated_stress_dev 150-product selection, excluding all v2 hard-family products |
| representative | `uppa-winery-nebbiolo-krasnoe-suhoe-14` | Неббиоло | Uppa Winery | Красное | Крым |  |  | deterministic proportional category×region quota with winery round-robin; existing generated_stress_dev 150-product selection, excluding all v2 hard-family products |
| representative | `v2r-podnyat-parusa-kaberne-sovinon-krasnoe-suhoe-125` | Поднять паруса! | В2Р | Красное | Кубань |  |  | deterministic proportional category×region quota with winery round-robin; existing generated_stress_dev 150-product selection, excluding all v2 hard-family products |
| representative | `v2r-risling-risling-reynskiy-beloe-suhoe-122` | Рислинг | В2Р | Белое | Кубань |  |  | deterministic proportional category×region quota with winery round-robin; existing generated_stress_dev 150-product selection, excluding all v2 hard-family products |
| representative | `valerij-zaharin-kokur-i-ko-beloe-polusladkoe` | Валерий Захарьин Кокур и Ко белое полусладкое | Валерий Захарьин | Белое | Крым |  |  | deterministic proportional category×region quota with winery round-robin; existing generated_stress_dev 150-product selection, excluding all v2 hard-family products |
| representative | `valeriy-zaharin-aleatiko-kefesiya-avtohtonnoe-vino-kryma-ot-valeriya-zaharina-rozovoe-suhoe-125` | Алеатико - Кефесия. Автохтонное вино Крыма от Валерия Захарьина | Валерий Захарьин | Розовое | Крым |  |  | deterministic proportional category×region quota with winery round-robin; existing generated_stress_dev 150-product selection, excluding all v2 hard-family products |

## Planned generations

| group | products | scenarios per product | planned generations |
|---|---:|---:|---:|
| representative | 16 | 4 | 64 |
| hard vintage | 8 | 4 | 32 |
| hard subtype | 8 | 4 | 32 |
| total | 32 | 4 | **128** |

Every row starts with `generation_status=pending`; generation IDs are derived deterministically from slug and scenario. The plan is intentionally separate from the existing 150-product full plan, and no generated image is accepted or scored by this milestone.

## Why pilot32 fits the budget

It reduces the paid scope from 600 planned requests to 128 while retaining all four capture scenarios, a non-hard representative control half, and family-aware hard diagnostics. The 16+16 split makes it possible to compare capture robustness on ordinary products against capture robustness on already difficult near-duplicate families without collapsing the two axes into one score.

## Reproducibility

- Catalog source: `data/processed/catalog_manifest.csv`.
- Seed: `20260916`.
- Selection and generation IDs use stable lexical/SHA-256 ordering; Python hash/random state is not used.
- Re-running the builder with the same inputs and seed reproduces selected slugs, family IDs, family membership, and generation IDs.
