# web_extra_dev

Этот benchmark собирается с официального сайта `https://vino-svoe.ru/`.
В scored manifest попадают только image URLs из exact `/wines/{slug}` sitemap
entries, где slug совпал с canonical slug. Generated images не используются.

## Current collection

- Exact sitemap slug matches: 1981.
- Accepted: 5.
- Rejected as reference duplicates: 1976.
- Manual review: 0.

`manifest.csv` — единственный scored set. `rejected.csv` и
`review_candidates.csv` не могут попасть в него автоматически.
Каталог `images/` может содержать download-cache от предыдущих
консервативных recheck; файлы, не перечисленные в `manifest.csv`, не являются
частью scored dataset.

## Leakage protection

Each candidate is checked against every canonical reference using exact SHA-256
and conservative 32x32 grayscale perceptual hashes on the full image and its
center crop, plus aspect-ratio and 64x64 thumbnail difference thresholds. A
candidate is rejected if it is an exact or conservative perceptual duplicate,
including resized/recompressed reference assets and derivatives with a
decorative background. Rejected and review rows remain provenance-only.

Collection is rate-limited by a small worker pool and uses public sitemap/image
URLs only. Robots and anti-bot restrictions must be respected on future runs.
