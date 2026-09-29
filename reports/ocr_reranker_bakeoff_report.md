# OCR + Reranker Bake-off (frozen SO400M retrieval)

Milestone: controlled comparison of 3 OCR configurations and 3 reranker
families over the frozen `siglip2_so400m_384` Top-5. Candidate generation is
untouched: OCR and rerankers only permute the Top-5, so R@5 is invariant by
construction and the primary metric is Top-1.

## TABLE 1 — OCR QUALITY

| OCR | Config | Ref empty% | Query non-empty% (gen) | Year% (gen) | Name% (gen) | Winery% (gen) | Hard text margin (mean / share target>best-wrong) | Latency mean |
|---|---|---|---|---|---|---|---|---|
| A current_eslav | PP-OCRv5_server_det + eslav_v5_mobile_rec | 1.81% | 99.2% | 48.4% | 48.4% | 44.5% | 0.054 / 59.9% | 0.27 s |
| B cyrillic | same det + cyrillic_v5_mobile_rec | 1.71% | 99.2% | 48.4% | 48.4% | 44.5% | 0.054 / 57.6% | 0.17 s |
| C paddleocr_vl | PaddleOCR-VL 0.9B (v1.6) | 17.8%* (363-ref partial) | 75.8% | 35.9% | 27.3% | 31.3% | 0.003 / 41.1% | 4.14 s |

*VL reference cache was rebuilt over hard_v2 + generated only; synthetic_dev
was skipped for VL (throughput: ~4.9 s/image on RTX 4060, ~8.5 h for a full
cache) — disclosed, not silently substituted.

## TABLE 2 — FINAL QUALITY (full sets)

| OCR | Reranker | Synthetic Top1 | Hard Top1 | Generated Top1 | Hard vintage | Hard subtype | Rescued/Broken (hard) |
|---|---|---|---|---|---|---|---|
| eslav | image_only | 95.10% | 86.87% | 73.44% | 76.47% | 87.52% | 0/0 |
| eslav | current blend | 95.27% | 87.91% | 73.44% | 76.47% | 88.63% | 16/4 |
| eslav | structured LR | 95.47% | 87.30% | 73.44% | 76.47% | 87.99% | 49/44 |
| eslav | BGE 0.2 | 95.27%* | 87.74% | 71.09% | 76.47% | 88.17%* | 59/49 |
| cyrillic | current blend | 95.10% | 87.48% | 73.44% | 76.47% | 88.17% | 9/2 |
| cyrillic | structured LR | 95.54% | 87.48% | 73.44% | 77.94% | 88.08% | 54/47 |
| cyrillic | BGE 0.2 | - | **88.00%** | 72.66% | 77.94% | 88.26%* | 56/43 |
| cyrillic | BGE 0.3 | - | **88.17%** | 71.88% | 77.94% | 88.26%* | 60/45 |
| VL | current blend | - | 86.26% | 73.44% | 76.47% | 87.43% | 4/11 |
| VL | structured LR | - | 86.87% | 74.22% | 76.47% | 87.52% | 40/40 |

*BGE values on synthetic/vintage/subtype partial where marked; the BGE fusion
grid ran on hard_v2 + generated only (query-side OCR text of each config).

## 1. Какой OCR лучший

**current_eslav** остаётся лучшим по discriminative качеству: text margin
target-vs-best-wrong на hard_v2 = 0.054 с 59.9% запросов «target читается
лучше ближайшего конкурента» (cyrillic: 57.6%; VL: 41.1% с margin 0.003).
cyrillic читает чуть иначе, но не лучше; VL читает меньше и хуже
дискриминирует при этом в 15-24 раза медленнее.

## 2-3. Name/year/winery и margin

VL хуже всех по покрытию: generated non-empty 75.8% против 99.2% у A/B,
name 27.3% против 48.4%. Margin VL на hard почти нулевой (0.003). Дискриминация
eslav и cyrillic практически идентична — оба Cyrillic-рекогнайзера PP-OCRv5
видят одно и то же; выбор между ними не влияет на качество.

## 4-6. Эффекты OCR × reranker

- **Новый OCR + старый reranker** (cyrillic + blend): hard 87.48% < 87.91%
  (eslav) — OCR B не даёт gain.
- **Тот же OCR + structured LR**: hard 87.30% (eslav) — ниже, чем current
  blend (87.91%), при 49/44 rescued/broken против 16/4. Structured LR не
  обгоняет hand-written blend на hard, но даёт лучший synthetic (95.47/95.54%)
  и лучший MRR.
- **Новый OCR + новый reranker**: cyrillic+BGE 0.3 даёт максимум hard
  Top-1 88.17% (+1.30pp к image-only), но с 60/45 rescued/broken и
  деградацией generated (71.88% < 73.44%).

## 7-11. Реранкеры и срезы

- Лучший hard Top-1: **cyrillic + BGE 0.3 = 88.17%**; но rescued/broken 60/45
  — «грязный» профиль.
- Лучший сбалансированный: **eslav + current blend = 87.91%** при 16/4.
- Vintage slice (hard, 68 queries): image-only 76.47% (52/68); улучшения нет
  ни у одного reranker, кроме cyrillic+structured/BGE (77.94%, 53/68 = +1
  запрос). Vintage остаётся нерешённым.
- Subtype slice (hard, 1082): 87.52% → 88.63% (eslav blend) — весь прирост
  hard от subtype-дизамбигуации.
- Generated hard (64): 39/64 → 40/64 только у VL+structured; остальные
  комбинации не двигают. VL единственный дал +1 (за счёт другого покрытия
  текста), но на overall generated это +0.78pp при VL-латентности 4.1s.

## 12. Rescued/Broken (hard_v2)

| Combo | rescued | broken |
|---|---|---|
| eslav + blend | 16 | 4 |
| cyrillic + blend | 9 | 2 |
| eslav + structured | 49 | 44 |
| cyrillic + structured | 54 | 47 |
| cyrillic + BGE 0.3 | 60 | 45 |
| VL + blend | 4 | 11 (net-negative!) |

## 13-14. Features и нужен ли BGE

Top features structured LR (|weight|): image_margin_top1 (−9.09),
ref_ocr_year_match (+0.88), query_has_year (−0.66), image_margin_next (+0.44),
ocr_token_set_ratio (−0.34), reference_ocr_score (+0.28). Модель в основном
полагается на image margin и year-согласованность.

**BGE добавляет немного на hard (+0.26pp к blend), но ломает generated
(−1.5pp) и стоит 133 ms/запрос.** Structured LR не превосходит current blend
по rescued/broken. BGE не нужен: существующий linear blend на
reference-OCR score остаётся лучшим по совокупности.

## 15-16. Latency / SLA

Пайплайн (GPU, RTX 4060): SO400M retrieval ~90-145 ms + OCR eslav
0.27 s mean (p95 0.23 s) + blend rerank <1 ms ≈ **~0.4 s mean** — SLA 3 s
соблюдается с 7× запасом. BGE добавил бы +133 ms; VL добавил бы +4.1 s
(нарушая SLA на одиночных изображениях).

## Диагностика OCR (detail)

| Скоуп | eslav margin mean | cyrillic margin mean | VL margin mean |
|---|---|---|---|
| synthetic (4082 in-top5) | 0.112 | 0.112 | — |
| hard_v2 (1148) | 0.054 | 0.054 | 0.003 |
| generated (128) | 0.120 | 0.126 | 0.046 |

## Vintage metadata audit

- 103/2042 (5.04%) продуктов имеют standalone год в title — источник
  metadata vintage.
- Reference-OCR содержит год у 48.1% references (eslav).
- Пересечение (title year AND ref-OCR year): 78 продуктов, согласование
  94.9% — ref-OCR год можно использовать как derived evidence с provenance.
- У hard vintage family queries: 68 — улучшений почти нет (+1 query
  cyrillic-based).

## Artifact/prune decisions

- VL × synthetic_dev пропущен (throughput), задокументировано; VL оценивался
  на hard_v2 + generated.
- OCR D (dual recognizer) не реализован: eslav и cyrillic дают почти
  идентичное покрытие текста, объединение не обещает нового сигнала.
- Полные комбинации 3×4 не запускались: VL+BGE исключён после того, как VL
  показал слабейшую дискриминацию (margin 0.003) и наихудший blend-результат
  (86.26%, net-negative 4/11).

## Verdict (Parts 39-40)

**A. KEEP CURRENT OCR + CURRENT RERANKER** (eslav + reference_ocr_blend
alpha 0.30).

Числа:
1. Лучший OCR — current_eslav (margin 0.054, 59.9% beats-best-wrong; cyrillic
   идентичен по пользе, VL хуже).
2. OCR upgrade не даёт gain: cyrillic+blend 87.48% < 87.91%.
3. Reranker upgrade: structured LR 87.30% с худшим rescued/broken; BGE до
   88.17% только с cyrillic и ценой generated-деградации и +133 ms.
4. Vintage не двигается ничем (bottleneck: стилизованная типографика года
   на этикетке — OCR не читает её надёжнее, чем уже читает eslav).
5. Generated: единственный +1 query (VL+structured) стоит 4.1 s/запрос и
   полного VL-кэша.
6. Latency/VRAM текущего пайплайна далеки от SLA.

Следующий резерв качества — не OCR/reranking, а данные/модель уровня
vintage-типографики (например, VLM-переосмысление этикетки или
fine-tuning рекогнайзера на доменных данных) — за пределами этого milestone.

## Artifacts

- Run: `artifacts/experiments/ocr_reranker_bakeoff_20260921T1/`
  (config, calibration_split, signal_cache, bge, structured models, combos,
  summaries)
- Diagnostics: `artifacts/experiments/ocr_reranker_bakeoff_diag/`
  (ocr_discrimination_summary.json, vintage_audit.json)
- OCR caches: `artifacts/ocr_cache/{eslav,cyrillic,paddleocr_vl}` (VL: refs +
  hard + generated; synthetic skipped — disclosed)
- Tests: 234 passed (15 new)
