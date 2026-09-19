# generated_stress_dev selection audit

Этот файл описывает только deterministic product selection и generation plan. Image API не вызывался, изображения не создавались, model predictions/errors не использовались.

## Input and method

- Source: `data/processed/catalog_manifest.csv`; usable products: **2042**.
- Selected products: **150**; fixed seed: **20260916**.
- Primary strata: `category × region` — оба поля заполнены, компактны и семантически устойчивы.
- `winery` используется для deterministic round-robin внутри каждой страты, чтобы крупные producers не заняли весь sample.
- `color` не использован как stratum: 822 unique values на 2042 products, поле слишком granular для устойчивых квот.
- `grape` не использован как stratum: 328 unique values и sparse multi-grape values; он полезен как описательная metadata, но noisy для квот.
- Selection order uses SHA-256(seed, value), not Python hash or model output; output rows are slug-sorted.

## Metadata profile

| field | unique non-empty values | empty |
|---|---:|---:|
| category | 4 | 0 |
| region | 9 | 0 |
| winery | 135 | 0 |
| color | 822 | 0 |
| grape | 329 | 1 |

## Category distribution

| value | catalog | catalog share | selected | selected share |
|---|---:|---:|---:|---:|
| Белое | 963 | 47.16% | 71 | 47.33% |
| Красное | 786 | 38.49% | 57 | 38.00% |
| Оранжевое | 16 | 0.78% | 1 | 0.67% |
| Розовое | 277 | 13.57% | 21 | 14.00% |

## Region distribution

| value | catalog | catalog share | selected | selected share |
|---|---:|---:|---:|---:|
| Дагестан | 94 | 4.60% | 7 | 4.67% |
| Дальневосточная зона | 1 | 0.05% | 0 | 0.00% |
| Долина Дона | 76 | 3.72% | 6 | 4.00% |
| Крым | 722 | 35.36% | 53 | 35.33% |
| Кубань | 1058 | 51.81% | 78 | 52.00% |
| Нижняя Волга | 22 | 1.08% | 1 | 0.67% |
| Самара | 21 | 1.03% | 2 | 1.33% |
| Северная Осетия — Алания | 5 | 0.24% | 0 | 0.00% |
| Ставрополье | 43 | 2.11% | 3 | 2.00% |

## Winery concentration

| metric | catalog | selected |
|---|---:|---:|
| unique wineries | 135 | 74 |
| largest winery count | 98 | 4 |
| top-10 winery share | 32.37% | 21.33% |

## Stratum quotas

| stratum | quota |
|---|---:|
| `category=Белое|region=Дагестан` | 4 |
| `category=Белое|region=Долина Дона` | 3 |
| `category=Белое|region=Крым` | 26 |
| `category=Белое|region=Кубань` | 35 |
| `category=Белое|region=Нижняя Волга` | 1 |
| `category=Белое|region=Самара` | 1 |
| `category=Белое|region=Северная Осетия — Алания` | 0 |
| `category=Белое|region=Ставрополье` | 1 |
| `category=Красное|region=Дагестан` | 2 |
| `category=Красное|region=Дальневосточная зона` | 0 |
| `category=Красное|region=Долина Дона` | 2 |
| `category=Красное|region=Крым` | 19 |
| `category=Красное|region=Кубань` | 32 |
| `category=Красное|region=Нижняя Волга` | 0 |
| `category=Красное|region=Северная Осетия — Алания` | 0 |
| `category=Красное|region=Ставрополье` | 2 |
| `category=Оранжевое|region=Крым` | 1 |
| `category=Оранжевое|region=Кубань` | 0 |
| `category=Оранжевое|region=Самара` | 0 |
| `category=Розовое|region=Дагестан` | 1 |
| `category=Розовое|region=Долина Дона` | 1 |
| `category=Розовое|region=Крым` | 7 |
| `category=Розовое|region=Кубань` | 11 |
| `category=Розовое|region=Нижняя Волга` | 0 |
| `category=Розовое|region=Самара` | 1 |
| `category=Розовое|region=Ставрополье` | 0 |

## Planned generation contract

- Four scenarios per product: `slight_angle, glare_bad_light, distance_crop, handheld`.
- Planned generations: **600**.
- All rows start with `generation_status=pending`.
- External generator must write exactly `generated_raw/<output_filename>`; see `generation_manifest.csv`.
