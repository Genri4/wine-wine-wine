# hard_near_duplicate_dev v2

Strict diagnostic benchmark derived from the preserved v1 family candidate
universe. A pair is retained only when it has at least two of the four strong
signals: real frozen SigLIP2 confusion with target rank 2--5, Top-1/Top-2
margin <= 0.02, deterministic reference-image similarity,
or strong normalized product-name similarity.

The `manifest.csv` query ids and query files are reused from `synthetic_dev`
exactly. `predictions.csv` is a projection of the frozen synthetic SigLIP2
predictions, so this artifact does not change the model or synthetic_dev.
Every family/product row carries `selection_reason` and JSON numerical evidence.
