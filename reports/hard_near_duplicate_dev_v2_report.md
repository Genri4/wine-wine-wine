# hard_near_duplicate_dev v2 audit

Строгая v2 построена как воспроизводимое уточнение сохранённого v1 candidate universe. `synthetic_dev`, модель SigLIP2 и v1 не изменялись; query ids/bytes в v2 переиспользованы из synthetic_dev, а predictions.csv — точная проекция замороженного synthetic baseline.

## 1. Summary and v1 → v2

| benchmark | families | products | queries | catalog share | Top-1 | Recall@5 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1 | 498 | 1722 | 3444 | 84.33% | 87.78% | 99.80% | 93.27% |
| v2 | 256 | 575 | 1150 | 28.16% | 68.43% | 99.65% | 82.76% |

Визуальный разбор 20 самых сложных query: [hard_near_duplicate_dev_v2_cases.html](hard_near_duplicate_dev_v2_cases.html). В карточках показаны query, reference target и Top-5 reference images.

## 2. Почему v1 получился большим

В v1 было **498 families / 1722 products** и **3184** пар внутри уже отобранных семейств. Основное раздувание дала широкая metadata-ветка: одинаковая winery плюс слабое пересечение имени и один/несколько wine attributes. В результате **325 families / 974 products** попали без observed SigLIP confusion; только **173 / 748** имели model-confusion evidence в v1 reason.

В frozen synthetic baseline было 433 Top-1 confusion rows: 426 с target rank 2–5 и 7 с target вне Top-5. Поэтому часть v1 model evidence не соответствовала строгому strong-signal A.

Дополнительные слабости v1: image similarity не была критерием отбора; confusion допускался даже при target rank вне 2–5; greedy clique добавлял collateral products, которым не обязательно соответствовала сильная прямая связь; отдельные metadata-признаки могли поддерживать широкие producer-группы.

### Retrospective signal breakdown inside v1

Ниже — аудит всех v1 products/families по сильным сигналам v2. `at least` означает наличие сигнала хотя бы на одном прямом pair edge; `only` — единственный сигнал на уровне family/product. Это retrospective evidence, а не утверждение, что v1 использовал image/name/margin как критерии.

| signal | families at least | products at least | families only | products only |
|---|---:|---:|---:|---:|
| `real_confusion_rank_2_5` | 173 | 417 | 1 | 4 |
| `small_top1_top2_margin` | 162 | 386 | 0 | 0 |
| `image_similarity` | 316 | 964 | 83 | 359 |
| `normalized_name_similarity` | 204 | 669 | 35 | 217 |
| **several strong signals (≥2)** | **246** | **659** | — | — |

Важная интерпретация: `image_similarity` и `small_top1_top2_margin` в v1 не были самостоятельными правилами, поэтому их retrospective-only counts не являются причиной первоначального включения. В частности, v1 image-only selection = **0** по своей реализации; v2 требует комбинацию evidence.

## 3. Ужесточенные критерии v2

V2 рассматривает только пары из сохранённых v1 families и оставляет pair, если присутствуют минимум **2 из 4** сигналов:

- `real_confusion_rank_2_5`: замороженная ошибка SigLIP2, target rank 2–5;
- `small_top1_top2_margin`: margin ≤ `0.02`;
- `image_similarity`: deterministic perceptual/pixel fingerprint reference images с текущими conservative guards;
- `normalized_name_similarity`: token Jaccard ≥ `0.75` или sequence ratio ≥ `0.8`.

Winery/category/region/grape не являются selection signals и сами по себе family не создают. После фильтра пары образуют connected components; collateral member без qualifying edge больше не добавляется.

Результат: **256 families / 575 products / 1150 queries**, то есть 28.16% каталога с reference image.

### V2 selection reasons

| reason combination | families | products |
|---|---:|---:|
| `image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin` | 75 | 199 |
| `image_similarity+real_confusion_rank_2_5+small_top1_top2_margin` | 80 | 165 |
| `image_similarity+normalized_name_similarity` | 64 | 137 |
| `real_confusion_rank_2_5+small_top1_top2_margin` | 21 | 42 |
| `image_similarity+real_confusion_rank_2_5` | 6 | 12 |
| `normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin` | 5 | 10 |
| `image_similarity+normalized_name_similarity+real_confusion_rank_2_5` | 3 | 6 |
| `normalized_name_similarity+real_confusion_rank_2_5` | 2 | 4 |

Каждая строка `families.csv` содержит product-level `selection_reason`, family-level reason и `selection_evidence_json` с similarity, margins, confusion counts и target ranks.

## 4. Baseline overlap и метрики

- V2 Top-1 errors: **363 / 1150**; overlap with errors where predicted slug belongs to same v2 family: **342**.
- Для сравнения v1 same-family error count: **346** (по сохранённому hard v1 prediction artifact).
- Mean/p50/p95 latency в v2 не переоцениваются: это projection существующих predictions; исходный SigLIP2 latency остаётся тем же, что в synthetic baseline.

| metric | v1 | v2 |
|---|---:|---:|
| Top-1 | 87.78% | 68.43% |
| Recall@5 | 99.80% | 99.65% |
| MRR | 93.27% | 82.76% |

### Target rank distribution (v2)

| target rank | queries | share |
|---:|---:|---:|
| 1 | 787 | 68.43% |
| 2 | 285 | 24.78% |
| 3 | 50 | 4.35% |
| 4 | 16 | 1.39% |
| 5 | 8 | 0.70% |
| not-found | 4 | 0.35% |

### Top1–Top2 margin distributions

| subset | count | min | p10 | median | p90 | max |
|---|---:|---:|---:|---:|---:|---:|
| correct Top-1 | 787 | 0.00000 | 0.00089 | 0.02938 | 0.08449 | 0.17003 |
| Top-1 errors | 363 | 0.00000 | 0.00000 | 0.00554 | 0.02147 | 0.08342 |

All v2 margins: mean=0.02853, min=0.00000, max=0.17003.

### Most frequent confusion pairs (v2)

| count | target | predicted |
|---:|---|---|
| 2 | `abrau-dyurso-imperatorskoe-bryut-shardone-beloe-12` | `abrau-dyurso-imperatorskoe-polusuhoe-shardone-beloe-12` |
| 2 | `abrau-dyurso-imperatorskoe-polusladkoe-shardone-beloe-12` | `abrau-dyurso-imperatorskoe-polusuhoe-shardone-beloe-12` |
| 2 | `abrau-dyurso-russkoe-igristoe-polusladkoe-krasnoe-kaberne-sovinon-12` | `abrau-dyurso-russkoe-igristoe-polusuhoe-shardone-beloe-12` |
| 2 | `abrau-dyurso-russkoe-igristoe-polusladkoe-shardone-beloe-12` | `abrau-dyurso-russkoe-igristoe-polusuhoe-shardone-beloe-12` |
| 2 | `abrau-dyurso-victor-dravigny-extra-brut-shardone-beloe-bryut-125` | `abrau-dyurso-victor-dravigny-brut-shardone-beloe-bryut-12` |
| 2 | `agora-yachting-sauvignon` | `agora-yachting-cabernet-sauvignon` |
| 2 | `agrolayn-heritage-dg-skin-contact-rkatsiteli-rkatsiteli-krasnoe-suhoe-12` | `agrolayn-heritage-dg-skin-contact-red-saperavi-krasnoe-suhoe-13` |
| 2 | `agrolayn-mountain-eagle-traminer-traminer-beloe-suhoe-12` | `agrolayn-mountain-eagle-semillon-semilon-beloe-suhoe-11` |
| 2 | `alma-valley-solntse-vozduh-vinograd-sira-shiraz-krasnoe-polusuhoe-135` | `alma-valley-solntse-vozduh-vinograd-merlo-krasnoe-polusladkoe-14` |
| 2 | `andryus-yutsis-muscat-blanc-muskat-beloe-suhoe-112` | `andryus-yutsis-orange-vermentino-beloe-suhoe-112` |
| 2 | `andryus-yutsis-orange-vermentino-beloe-suhoe-112` | `andryus-yutsis-pinot-grigio-pino-gri-beloe-suhoe-125` |
| 2 | `aristov-anima-millesimato-beloe-bryut` | `aristov-anima-millesimato` |
| 2 | `aristov-anima-pinot-grigio-blush` | `kuban-vino-aristov-anima-pino-gridzhio-beloe-suhoe-12` |
| 2 | `aristov-kyuve-aleksandr-blan-de-blan` | `aristov-kyuve-aleksandr-blan-de-nuar` |
| 2 | `belbek-belbek-muskat-beloe-suhoe-128` | `belbek-muskat-muskat-belyy-beloe-suhoe-127` |

## 5. Самые сложные families

Сортировка: сначала Top-1 error count, затем error rate и размер family. Это диагностический порядок, не новая selection rule.

| family | products | queries | Top-1 | errors | reason |
|---|---:|---:|---:|---:|---|
| `family-129` | 3 | 6 | 16.67% | 5 | `image_similarity+real_confusion_rank_2_5+small_top1_top2_margin` |
| `family-072` | 4 | 8 | 37.50% | 5 | `image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin` |
| `family-091` | 4 | 8 | 37.50% | 5 | `image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin` |
| `family-225` | 6 | 12 | 58.33% | 5 | `image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin` |
| `family-017` | 2 | 4 | 0.00% | 4 | `image_similarity+real_confusion_rank_2_5+small_top1_top2_margin` |
| `family-003` | 3 | 6 | 33.33% | 4 | `image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin` |
| `family-005` | 3 | 6 | 33.33% | 4 | `image_similarity+normalized_name_similarity+real_confusion_rank_2_5+small_top1_top2_margin` |
| `family-076` | 3 | 6 | 33.33% | 4 | `image_similarity+real_confusion_rank_2_5+small_top1_top2_margin` |

### 20 hardest cases

Визуальные пары query/reference и Top-5 находятся в отдельном HTML, чтобы не встраивать изображения в Markdown и не менять data artifacts.

| # | query | target | rank | margin | Top-5 predictions |
|---:|---|---|---:|---:|---|
| 1 | `ferrum-winery-sabernet-sauvignon-cabernet-franc-kaberne-sovinon-krasnoe-suhoe-12__v02` | `ferrum-winery-sabernet-sauvignon-cabernet-franc-kaberne-sovinon-krasnoe-suhoe-12` | not-found | 0.00085 | `ferrum-winery-sabernet-sauvignon-kaberne-sovinon-krasnoe-suhoe-14, ferrum-winery-kaberne-fran-krasnoe-suhoe-12, bogovich-wine-vineyard-kaberne-sovinon-krasnoe-suhoe-125, bogovich-wine-vineyard-kaberne-fran-krasnoe-suhoe-135, leto-kaberne-sovinon-2023-polusuhoe-krasnoe` |
| 2 | `fanagoriya-dekanter-saperavi-2017-krasnoe-suhoe-135__v01` | `fanagoriya-dekanter-saperavi-2017-krasnoe-suhoe-135` | not-found | 0.01169 | `fanagoriya-dekanter-kaberne-sovinon-2018-krasnoe-suhoe-14, fanagoriya-dekanter-pino-nuar-2020-krasnoe-suhoe-135, fanagoriya-dekanter-riesling-2020-risling-reynskiy-beloe-suhoe-13, fanagoriya-dekanter-chardonnay-2020-shardone-beloe-suhoe-135, fanagoriya-dekanter-kaberne-fran-2019-krasnoe-suhoe-13` |
| 3 | `fanagoriya-dekanter-saperavi-2017-krasnoe-suhoe-135__v02` | `fanagoriya-dekanter-saperavi-2017-krasnoe-suhoe-135` | not-found | 0.01582 | `fanagoriya-dekanter-kaberne-sovinon-2018-krasnoe-suhoe-14, fanagoriya-dekanter-pino-nuar-2020-krasnoe-suhoe-135, fanagoriya-dekanter-riesling-2020-risling-reynskiy-beloe-suhoe-13, fanagoriya-dekanter-kaberne-fran-2019-krasnoe-suhoe-13, fanagoriya-dekanter-chardonnay-2020-shardone-beloe-suhoe-135` |
| 4 | `ferrum-winery-sabernet-sauvignon-cabernet-franc-kaberne-sovinon-krasnoe-suhoe-12__v01` | `ferrum-winery-sabernet-sauvignon-cabernet-franc-kaberne-sovinon-krasnoe-suhoe-12` | not-found | 0.03898 | `ferrum-winery-kaberne-fran-krasnoe-suhoe-12, bogovich-wine-vineyard-kaberne-sovinon-krasnoe-suhoe-125, ferrum-winery-sabernet-sauvignon-kaberne-sovinon-krasnoe-suhoe-14, bogovich-wine-vineyard-kaberne-fran-krasnoe-suhoe-135, usadba-divnomorskoe-pinot-noir-pino-nuar-krasnoe-suhoe-12` |
| 5 | `chteau-le-grand-vostock-krasnostop-reserve-krasnostop-krasnoe-suhoe-14__v01` | `chteau-le-grand-vostock-krasnostop-reserve-krasnostop-krasnoe-suhoe-14` | 5 | 0.00124 | `chteau-le-grand-vostock-cabernet-sauvignon-kaberne-sovinon-krasnoe-suhoe-14, chteau-le-grand-vostock-merlot-merlo-krasnoe-suhoe-14, chteau-le-grand-vostock-chardonnay-reserve-shardone-beloe-suhoe-14, chteau-le-grand-vostock-cabernet-sauvignon-reserve-kaberne-sovinon-krasnoe-suhoe-14, chteau-le-grand-vostock-krasnostop-reserve-krasnostop-krasnoe-suhoe-14` |
| 6 | `fanagoriya-primum-alveus-ekstra-bryut-2017-pino-nuar-beloe-11-13__v01` | `fanagoriya-primum-alveus-ekstra-bryut-2017-pino-nuar-beloe-11-13` | 5 | 0.00591 | `fanagoriya-primum-alveus-brut-2017-shardone-igristoe-bryut-beloe-12, fanagoriya-primum-alveus-extra-brut-2014-pino-nuar-beloe-bryut-12, fanagoriya-primum-alveus-extra-brut-2016-shardone-beloe-bryut-12, fanagoriya-primum-alveus-blanc-de-noirs-meunier-ekstra-bryut-2019-mene-beloe-11-13, fanagoriya-primum-alveus-ekstra-bryut-2017-pino-nuar-beloe-11-13` |
| 7 | `abrau-dyurso-imperatorskoe-bryut-shardone-beloe-12__v02` | `abrau-dyurso-imperatorskoe-bryut-shardone-beloe-12` | 5 | 0.00737 | `abrau-dyurso-imperatorskoe-polusuhoe-shardone-beloe-12, abrau-dyurso-udelnoe-vedomstvo-imperatorskoe-beloe-bryut, abrau-dyurso-udelnoe-vedomstvo-imperatorskoe-beloe-polusladkoe, abrau-dyurso-imperatorskoe-polusladkoe-shardone-beloe-12, abrau-dyurso-imperatorskoe-bryut-shardone-beloe-12` |
| 8 | `vinodelnya-batrak-perfekt-klassik-kaberne-sovinon-krasnoe-suhoe-135__v01` | `vinodelnya-batrak-perfekt-klassik-kaberne-sovinon-krasnoe-suhoe-135` | 5 | 0.01375 | `vinodelnya-batrak-kaberne-kortis-krasnoe-suhoe-12, vinodelnya-batrak-perfekt-klassik-saperavi-krasnoe-suhoe-135, vinodelnya-batrak-perfekt-klassik-kaberne-fran-krasnoe-suhoe-135, vinodelnya-batrak-perfekt-klassik-merlo-krasnoe-suhoe-135, vinodelnya-batrak-perfekt-klassik-kaberne-sovinon-krasnoe-suhoe-135` |
| 9 | `vinodelnya-batrak-perfekt-klassik-kaberne-sovinon-krasnoe-suhoe-135__v02` | `vinodelnya-batrak-perfekt-klassik-kaberne-sovinon-krasnoe-suhoe-135` | 5 | 0.01378 | `vinodelnya-batrak-kaberne-kortis-krasnoe-suhoe-12, vinodelnya-batrak-perfekt-klassik-saperavi-krasnoe-suhoe-135, vinodelnya-batrak-perfekt-klassik-kaberne-fran-krasnoe-suhoe-135, vinodelnya-batrak-kaberne-sovinon-krasnoe-suhoe-14, vinodelnya-batrak-perfekt-klassik-kaberne-sovinon-krasnoe-suhoe-135` |
| 10 | `golubitskoe-estate-kaberne-sovinon-noble-selection-krasnoe-suhoe-133__v02` | `golubitskoe-estate-kaberne-sovinon-noble-selection-krasnoe-suhoe-133` | 5 | 0.01452 | `golubitskoe-estate-noble-selection-red-blend-kaberne-sovinon-krasnoe-suhoe-136, golubitskoe-estate-pino-nuar-rezerv-krasnoe-suhoe-132, golubitskoe-estate-red-blend-kaberne-sovinon-krasnoe-suhoe-136, golubitskoe-estate-shardone-noble-selection-barrel-touch-beloe-suhoe-135, golubitskoe-estate-kaberne-sovinon-noble-selection-krasnoe-suhoe-133` |
| 11 | `fanagoriya-primum-alveus-extra-brut-2014-pino-nuar-beloe-bryut-12__v02` | `fanagoriya-primum-alveus-extra-brut-2014-pino-nuar-beloe-bryut-12` | 5 | 0.01682 | `fanagoriya-primum-alveus-brut-2017-shardone-igristoe-bryut-beloe-12, fanagoriya-primum-alveus-brut-2016-shardone-igristoe-bryut-beloe-12, fanagoriya-primum-alveus-extra-brut-2016-shardone-beloe-bryut-12, fanagoriya-primum-alveus-brut-2014-shardone-igristoe-bryut-beloe-12, fanagoriya-primum-alveus-extra-brut-2014-pino-nuar-beloe-bryut-12` |
| 12 | `belbek-belbek-risling-beloe-suhoe-128__v01` | `belbek-belbek-risling-beloe-suhoe-128` | 5 | 0.02536 | `belbek-muskat-muskat-belyy-beloe-suhoe-127, belbek-belbek-muskat-beloe-suhoe-128, belbek-risling-risling-reynskiy-beloe-suhoe-12, belbek-sira-rezerv-krasnoe-suhoe-132, belbek-belbek-risling-beloe-suhoe-128` |
| 13 | `abrau-dyurso-imperatorskoe-bryut-shardone-beloe-12__v01` | `abrau-dyurso-imperatorskoe-bryut-shardone-beloe-12` | 4 | 0.00040 | `abrau-dyurso-imperatorskoe-polusuhoe-shardone-beloe-12, abrau-dyurso-udelnoe-vedomstvo-imperatorskoe-beloe-bryut, abrau-dyurso-udelnoe-vedomstvo-imperatorskoe-beloe-polusladkoe, abrau-dyurso-imperatorskoe-bryut-shardone-beloe-12, abrau-dyurso-imperatorskoe-polusladkoe-shardone-beloe-12` |
| 14 | `fanagoriya-primum-alveus-ekstra-bryut-2017-pino-nuar-beloe-11-13__v02` | `fanagoriya-primum-alveus-ekstra-bryut-2017-pino-nuar-beloe-11-13` | 4 | 0.00064 | `fanagoriya-primum-alveus-extra-brut-2016-shardone-beloe-bryut-12, fanagoriya-primum-alveus-extra-brut-2014-pino-nuar-beloe-bryut-12, fanagoriya-primum-alveus-brut-2017-shardone-igristoe-bryut-beloe-12, fanagoriya-primum-alveus-ekstra-bryut-2017-pino-nuar-beloe-11-13, fanagoriya-primum-alveus-blanc-de-noirs-meunier-ekstra-bryut-2019-mene-beloe-11-13` |
| 15 | `toile-de-vin-cabernet-sauvignon__v02` | `toile-de-vin-cabernet-sauvignon` | 4 | 0.00229 | `toile-de-vin-chardonnay, toile-de-vin-pinot-blanc, toile-de-vin-merlot, toile-de-vin-cabernet-sauvignon, usadba-rodnoe-gnezdo-4-elements-sira-roze-rozovoe-polusuhoe-135` |
| 16 | `alma-valley-solntse-vozduh-vinograd-kokur-beloe-polusladkoe-13__v02` | `alma-valley-solntse-vozduh-vinograd-kokur-beloe-polusladkoe-13` | 4 | 0.00324 | `alma-valley-solntse-vozduh-vinograd-merlo-krasnoe-polusladkoe-14, alma-valley-solntse-vozduh-vinograd-shardone-beloe-polusuhoe-13, alma-valley-solntse-vozduh-vinograd-sira-shiraz-krasnoe-polusuhoe-135, alma-valley-solntse-vozduh-vinograd-kokur-beloe-polusladkoe-13, alma-valley-solntse-vozduh-vinograd-sira-shiraz-rozovoe-polusladkoe-12` |
| 17 | `esse-risling-beloe-suhoe-115__v02` | `esse-risling-beloe-suhoe-115` | 4 | 0.00400 | `esse-vione-beloe-suhoe-13, esse-sovinon-blan-fyume-beloe-suhoe-128, esse-esse-shardone-beloe-suhoe-125, esse-risling-beloe-suhoe-115, esse-gevyurtstraminer-beloe-suhoe-12` |
| 18 | `vibes-chardonnay-barrel-fermented-shardone-beloe-suhoe-125__v02` | `vibes-chardonnay-barrel-fermented-shardone-beloe-suhoe-125` | 4 | 0.00681 | `vermentino-viognier-2022, vibes-riesling-silvaner-risling-beloe-suhoe-115, vibes-silvaner-silvaner-beloe-suhoe-125, vibes-chardonnay-barrel-fermented-shardone-beloe-suhoe-125, vibes-chardonnay-barrel-fermented-2021` |
| 19 | `toile-de-vin-cabernet-sauvignon__v01` | `toile-de-vin-cabernet-sauvignon` | 4 | 0.00816 | `toile-de-vin-chardonnay, toile-de-vin-pinot-blanc, toile-de-vin-merlot, toile-de-vin-cabernet-sauvignon, aya-organic-wine-vineyards-evolution-pinot-noir-pino-nuar-krasnoe-suhoe-125` |
| 20 | `vinodelnya-myshako-marselan-krasnoe-suhoe-143__v01` | `vinodelnya-myshako-marselan-krasnoe-suhoe-143` | 4 | 0.00848 | `vinodelnya-myshako-marselan-krasnoe-suhoe-144, vinodelnya-myshako-marselan-avtorskaya-tehnologiya-krasnoe-suhoe-144, vinodelnya-myshako-shardone-beloe-suhoe-123, vinodelnya-myshako-marselan-krasnoe-suhoe-143, vinodelnya-myshako-shiraz-krasnoe-suhoe-146` |

## 6. Открытые ограничения

- V2 queries происходят из synthetic_dev и потому не являются независимым real-world test set.
- Image similarity здесь — детерминированный fingerprint/dedup-style сигнал, не CV/ML recognition.
- V2 не доказывает, что две бутылки genuinely near-duplicate без визуальной проверки; он формирует более узкий reproducible review set.
- Baseline errors не являются labels для обучения и не должны использоваться для tuning на том же benchmark без отдельного split.

## 7. Что делать дальше

1. Использовать v2 как диагностический hard subset и отдельно смотреть 20 HTML cases.
2. Зафиксировать эту selection policy и не смешивать v1/v2/synthetic_dev в одну headline metric.
3. Для следующего эксперимента держать одинаковый query split, модель и evaluator; изменения писать в experiment history.

## Files

- Benchmark: `data/benchmarks/hard_near_duplicate_dev_v2/` (`family_summary.csv` is one row per family; `families.csv` is one row per product).
- Visual cases: [hard_near_duplicate_dev_v2_cases.html](hard_near_duplicate_dev_v2_cases.html).
- Preserved v1: `data/benchmarks/hard_near_duplicate_dev/`.
