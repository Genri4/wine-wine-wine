# hard_near_duplicate_dev report

Это диагностический benchmark для fine-grained confusion. Query images переиспользованы из `synthetic_dev`; независимость от catalog reference отсутствует по дизайну.

## Summary

- Hard families: **498**.
- Unique products: **1722**.
- Queries: **3444**.
- Family size distribution: `{2: 229, 3: 105, 4: 56, 5: 42, 6: 24, 7: 14, 8: 9, 9: 5, 10: 6, 12: 8}`.
- Current SigLIP2 errors where target and prediction are in the same family: **346**.

## Selection logic

Families use only canonical metadata plus observed SigLIP2 confusion pairs from the synthetic DEV run. Metadata-only edges require the same winery, meaningful name overlap, and at least one shared category/color/region/grape signal. A year is recorded only when a unique explicit 19xx/20xx token is present in title/slug.

## Strongest observed confusion pairs inside families

- `2`: `abrau-dyurso-imperatorskoe-bryut-shardone-beloe-12` → `abrau-dyurso-imperatorskoe-polusuhoe-shardone-beloe-12`
- `2`: `abrau-dyurso-imperatorskoe-polusladkoe-shardone-beloe-12` → `abrau-dyurso-imperatorskoe-polusuhoe-shardone-beloe-12`
- `2`: `abrau-dyurso-russkoe-igristoe-polusladkoe-krasnoe-kaberne-sovinon-12` → `abrau-dyurso-russkoe-igristoe-polusuhoe-shardone-beloe-12`
- `2`: `abrau-dyurso-russkoe-igristoe-polusladkoe-shardone-beloe-12` → `abrau-dyurso-russkoe-igristoe-polusuhoe-shardone-beloe-12`
- `2`: `abrau-dyurso-victor-dravigny-extra-brut-shardone-beloe-bryut-125` → `abrau-dyurso-victor-dravigny-brut-shardone-beloe-bryut-12`
- `2`: `agora-yachting-sauvignon` → `agora-yachting-cabernet-sauvignon`
- `2`: `agrolayn-heritage-dg-skin-contact-rkatsiteli-rkatsiteli-krasnoe-suhoe-12` → `agrolayn-heritage-dg-skin-contact-red-saperavi-krasnoe-suhoe-13`
- `2`: `agrolayn-mountain-eagle-traminer-traminer-beloe-suhoe-12` → `agrolayn-mountain-eagle-semillon-semilon-beloe-suhoe-11`
- `2`: `alma-valley-solntse-vozduh-vinograd-sira-shiraz-krasnoe-polusuhoe-135` → `alma-valley-solntse-vozduh-vinograd-merlo-krasnoe-polusladkoe-14`
- `2`: `andryus-yutsis-muscat-blanc-muskat-beloe-suhoe-112` → `andryus-yutsis-orange-vermentino-beloe-suhoe-112`

## Limitations

- This is not a real-world smartphone benchmark and must not be combined with `synthetic_dev` or `web_extra_dev` into a headline metric.
- Query images originate from catalog references through synthetic transforms, so this benchmark diagnoses fine-grained retrieval confusion rather than independent-image generalization.
- Family membership is a conservative reproducible diagnostic selection, not a claim that every member is visually identical.
