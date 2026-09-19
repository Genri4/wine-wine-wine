# Experiment history

## 2026-09-13 — baseline implementation

- Added a pretrained visual-embedding retrieval baseline and offline catalog index builder.
- Added prediction and evaluation runners with configurable, uncalibrated cosine threshold.
- Quality metrics were not measured: the organizer dataset, target metric, split, and catalog format are not available yet.

## 2026-09-13 — baseline encoder comparison support

- Kept ResNet-18 and ResNet-50 and added DINOv2 (`facebook/dinov2-base`) and SigLIP (`google/siglip-base-patch16-224`).
- All four models are selected by one `model_name` value and produce L2-normalized embeddings with model-specific preprocessing.
- Migrated catalog embeddings from JSON to PyTorch `.pt` indexes with tensor embeddings and adjacent item metadata.
- Added run JSON fields for model name, accuracy, known top-1 accuracy, top-K recall, unknown detection accuracy, average latency, and sample count.
- No quality comparison was run: the organizer dataset, split, and target metric are still unavailable.

## 2026-09-14 — CUDA environment and experiment tooling

- Replaced the local CPU PyTorch pair with `torch==2.14.0+cu130` and `torchvision==0.29.0+cu130`; CUDA inference was verified on an NVIDIA GeForce RTX 4060.
- Added `scripts/cache_models.py` to initialize and cache all four pretrained baseline encoders sequentially.
- Added dataset diagnostics (`scripts/analyze_dataset.py`) and a sequential four-model benchmark (`scripts/benchmark.py`).
- Evaluation now keeps a separate error artifact with the query image, expected/predicted item, and top-5 candidates with scores.
- No real-dataset metrics were recorded: the organizer dataset format, split, target metric, and evaluation rules remain unknown.
