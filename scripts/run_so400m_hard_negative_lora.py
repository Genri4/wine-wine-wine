#!/usr/bin/env python3
"""Train one catalog-only last-four-block LoRA and evaluate after freezing it.

The training phase never reads benchmark manifests, queries, targets, OCR, or
benchmark-derived error pairs. Benchmark data is opened only after
``selected_model.json`` freezes the internally selected checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from recognition.geometric_reranker import (  # noqa: E402
    DEFAULT_CONFIG as SIFT_CONFIG,
    extract_sift,
    file_sha256,
    fuse_scores,
    match_sift_pair,
    read_rgb,
    valid_geometry_scores,
)
from recognition.ocr_reranker import FusionConfig, rerank_one_query  # noqa: E402
from recognition.so400m_lora import (  # noqa: E402
    BASE_MODEL_ID,
    BASE_REVISION,
    CAPTURE_BUCKETS,
    HIDDEN_SIZE,
    LORA_TARGETS,
    LoRALinear,
    adapted_reference_fingerprint,
    assert_same_lora_checkpoint,
    assert_catalog_only_manifest,
    build_capture_manifest,
    build_hard_negative_map,
    capture_rng_state,
    catalog_family_keys,
    derive_seed,
    infer_catalog_family_types,
    inject_last_vision_lora,
    lora_sha256,
    lora_state_dict,
    load_lora_state_dict,
    make_capture_view_v2,
    pooled_image_features,
    restore_rng_state,
    set_lora_enabled,
    sha256_file,
)
from recognition.encoder_followup_protocol import (  # noqa: E402
    CANONICAL_QUERY_BATCH_SIZE,
    require_canonical_query_batch_size,
    require_exact_baseline_reproduction,
    require_frozen_before_external,
    assert_catalog_only_capture_manifest,
    assert_rank_only_change,
    assert_same_manifests,
    assert_training_data_access_scope,
    make_encoder_fingerprint,
)
from recognition.text_signals import (  # noqa: E402
    build_candidate_text_index,
    build_query_text_evidence,
    build_reference_ocr_evidence,
    compute_text_signals,
)


BASE_CACHE = Path("artifacts/reference_embeddings/siglip2_so400m_384")
OCR_RUN = Path("artifacts/experiments/so400m_ocr_reranker_20260920T193925Z")
SIFT_RUN = Path("artifacts/experiments/final_ml_geometric_reranker_20260923T082259Z")
SOURCE_ADAPTER_RUN = Path("artifacts/experiments/hard_negative_metric_adapter_20260923T183443Z")
BENCHMARKS = ("hard_near_duplicate_dev_v2", "generated_stress_dev_pilot32", "synthetic_dev")
PRIMARY_BENCHMARKS = BENCHMARKS[:2]
SEED = 20260924
CONFIG = {
    "experiment": "PARTIAL SO400M HARD-NEGATIVE FINE-TUNING",
    "base_model_id": BASE_MODEL_ID,
    "base_revision": BASE_REVISION,
    "rank": 8,
    "alpha": 16,
    "dropout": 0.05,
    "last_vision_blocks": 4,
    "attention_targets": list(LORA_TARGETS),
    "train_views_per_sku": 8,
    "validation_views_per_sku": 2,
    "family_negative_count": 4,
    "visual_neighbor_negative_count": 6,
    "random_negative_count": 2,
    "neighbor_pool": 20,
    "seed": SEED,
    "temperature": 0.07,
    "preservation_loss_weight": 0.02,
    "optimizer": "AdamW",
    "learning_rate": 0.00005,
    "weight_decay": 0.0001,
    "max_epochs": 6,
    "early_stopping_patience": 3,
    "micro_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "gradient_clip_norm": 1.0,
    "reference_embedding_batch_size": 8,
    "query_embedding_batch_size": 4,
    "image_size": 384,
    "canonical_external_query_batch_size": 1,
    "mixed_precision": "fp16 frozen base with autocast; fp32 LoRA parameters/loss",
    "gradient_checkpointing_requested": True,
    "internal_full_catalog_guard_pp": 1.0,
    "checkpoint_each_epoch": True,
    "benchmark_selection_allowed": False,
    "synthetic_used_for_selection": False,
}
# The frozen bake-off encoded each benchmark query as a one-image batch. Keep
# that execution shape: FP16 kernels can drift slightly at different batch sizes.
EXTERNAL_QUERY_EMBEDDING_BATCH_SIZE = 1
EXTERNAL_QUERY_CHECKPOINT_INTERVAL = 64
CANONICAL_EXTERNAL_PROTOCOL = {
    "query_batch_size": CANONICAL_QUERY_BATCH_SIZE,
    "reference_embedding_batch_size": int(CONFIG["reference_embedding_batch_size"]),
    "baseline_artifact": "artifacts/experiments/frozen_baseline_forensics_20260924T044708Z/baseline_reproduction.json",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def write_state(path: Path, stage: str, **details: Any) -> None:
    write_json(path, {"stage": stage, "updated_at_utc": datetime.now(timezone.utc).isoformat(), **details})


def sha256_json(payload: Any) -> str:
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def load_catalog(root: Path) -> tuple[list[dict[str, str]], list[str], np.ndarray, dict[str, Any], dict[str, str]]:
    cache = root / BASE_CACHE
    metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
    payload = torch.load(cache / "embeddings.pt", map_location="cpu", weights_only=True)
    slugs = json.loads((cache / "slugs.json").read_text(encoding="utf-8"))
    embeddings = payload["embeddings"].float().numpy()
    if payload.get("fingerprint") != metadata.get("fingerprint"):
        raise RuntimeError("frozen SO400M reference embedding cache fingerprint mismatch")
    if metadata.get("checkpoint_revision") != BASE_REVISION or metadata.get("hf_model_id") != BASE_MODEL_ID:
        raise RuntimeError("frozen catalog embedding cache is not from the pinned SO400M checkpoint")
    if embeddings.shape != (len(slugs), HIDDEN_SIZE) or not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=2e-3):
        raise RuntimeError("frozen catalog embeddings are misaligned or not L2-normalized")
    all_products = read_csv(root / "data/processed/catalog_manifest.csv")
    by_slug = {row["slug"]: row for row in all_products if row.get("reference_image_path")}
    if set(slugs) != set(by_slug) or len(slugs) != int(metadata.get("reference_count", -1)):
        raise RuntimeError("frozen reference vectors and catalog manifest do not align exactly")
    catalog = [by_slug[slug] for slug in slugs]
    image_hashes = {}
    for row in catalog:
        path = root / row["reference_image_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        image_hashes[row["slug"]] = sha256_file(path)
    return catalog, slugs, embeddings, metadata, image_hashes


def load_model(device: torch.device):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(BASE_MODEL_ID, revision=BASE_REVISION, local_files_only=True)
    # Keep the unused, frozen text tower on CPU. Only the image tower is moved
    # to the 8 GB GPU; the checkpoint itself remains the original full model.
    model = AutoModel.from_pretrained(BASE_MODEL_ID, revision=BASE_REVISION,
                                      dtype=torch.float16 if device.type == "cuda" else torch.float32,
                                      local_files_only=True)
    model.requires_grad_(False)
    model.vision_model.to(device)
    lora_modules = inject_last_vision_lora(model, rank=CONFIG["rank"], alpha=CONFIG["alpha"],
                                           dropout=CONFIG["dropout"], last_blocks=CONFIG["last_vision_blocks"])
    checkpointing_enabled = False
    if CONFIG["gradient_checkpointing_requested"]:
        try:
            model.vision_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            checkpointing_enabled = bool(getattr(model.vision_model, "is_gradient_checkpointing", False))
        except (AttributeError, TypeError, ValueError):
            try:
                model.vision_model.gradient_checkpointing_enable()
                checkpointing_enabled = bool(getattr(model.vision_model, "is_gradient_checkpointing", False))
            except (AttributeError, TypeError, ValueError):
                checkpointing_enabled = False
    model.eval()
    return model, processor, lora_modules, checkpointing_enabled


def inspect_loaded_model(model: torch.nn.Module, processor: Any, lora_modules: Sequence[str], checkpointing_enabled: bool) -> dict[str, Any]:
    vision = model.vision_model
    layers = vision.encoder.layers
    linears = {}
    for index in range(len(layers) - 4, len(layers)):
        block = layers[index]
        linears[str(index)] = {name: {"type": type(module).__name__, "in_features": module.base.in_features if isinstance(module, LoRALinear) else module.in_features,
                                      "out_features": module.base.out_features if isinstance(module, LoRALinear) else module.out_features}
                               for name, module in block.named_modules()
                               if name.split(".")[-1] in LORA_TARGETS and isinstance(module, (torch.nn.Linear, LoRALinear))}
    ip = getattr(processor, "image_processor", processor)
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "model_id": BASE_MODEL_ID,
        "revision": BASE_REVISION,
        "implementation": f"{type(model).__module__}.{type(model).__name__}",
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()) - trainable_count,
        "loaded_parameters_including_lora": sum(parameter.numel() for parameter in model.parameters()),
        "vision_parameters": sum(parameter.numel() for parameter in vision.parameters()) - trainable_count,
        "text_parameters": sum(parameter.numel() for parameter in model.text_model.parameters()),
        "vision_block_count": len(layers),
        "hidden_size": int(model.config.vision_config.hidden_size),
        "intermediate_size": int(model.config.vision_config.intermediate_size),
        "attention_heads": int(model.config.vision_config.num_attention_heads),
        "image_size": int(model.config.vision_config.image_size),
        "patch_size": int(model.config.vision_config.patch_size),
        "vision_dtype": str(next(vision.parameters()).dtype),
        "text_dtype": str(next(model.text_model.parameters()).dtype),
        "processor": {"class": type(ip).__name__, "size": str(getattr(ip, "size", None)),
                      "image_mean": str(getattr(ip, "image_mean", None)), "image_std": str(getattr(ip, "image_std", None)),
                      "do_center_crop": str(getattr(ip, "do_center_crop", None))},
        "last_four_attention_linears": linears,
        "lora_modules": list(lora_modules),
        "trainable_parameter_count": trainable_count,
        "text_trainable_parameter_count": sum(parameter.numel() for parameter in model.text_model.parameters() if parameter.requires_grad),
        "gradient_checkpointing_enabled": checkpointing_enabled,
        "base_checkpoint_parameters_frozen": all(not parameter.requires_grad for name, parameter in model.named_parameters() if ".lora_A" not in name and ".lora_B" not in name),
    }


def _processor_pixels(processor: Any, images: Sequence[Image.Image], model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    pixel_values = processor(images=list(images), return_tensors="pt")["pixel_values"]
    dtype = next(model.vision_model.parameters()).dtype
    return pixel_values.to(device=device, dtype=dtype, non_blocking=True)


def _features(model: torch.nn.Module, pixels: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        vectors = pooled_image_features(model, pixels)
    return F.normalize(vectors.float(), p=2, dim=-1, eps=1e-12)


def make_view(row: Mapping[str, Any], root: Path) -> Image.Image:
    with Image.open(root / str(row["reference_image_path"])) as image:
        return make_capture_view_v2(image, int(row["augmentation_seed"]), str(row["split"]), str(row["capture_bucket"]))


def precompute_frozen_views(
    root: Path,
    run_dir: Path,
    split: str,
    rows: Sequence[Mapping[str, Any]],
    model: torch.nn.Module,
    processor: Any,
    device: torch.device,
    batch_size: int,
    manifest_fingerprint: str,
) -> Path:
    path = run_dir / f"frozen_{split}_view_embeddings.npy"
    progress_path = run_dir / f"frozen_{split}_view_embeddings_progress.json"
    expected = (len(rows), HIDDEN_SIZE)
    valid = False
    if path.is_file():
        try:
            existing = np.load(path, mmap_mode="r")
            valid = existing.shape == expected and existing.dtype == np.float16
            del existing
        except (OSError, ValueError):
            valid = False
    if not valid:
        np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=expected).flush()
        progress_path.unlink(missing_ok=True)
    done = 0
    if progress_path.is_file():
        saved = json.loads(progress_path.read_text(encoding="utf-8"))
        if saved.get("manifest_fingerprint") == manifest_fingerprint:
            done = int(saved.get("completed_rows", 0))
    target = np.lib.format.open_memmap(path, mode="r+")
    set_lora_enabled(model, False)
    model.vision_model.eval()
    for start in range(done, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        images = [make_view(row, root) for row in chunk]
        pixels = _processor_pixels(processor, images, model, device)
        with torch.inference_mode():
            vectors = _features(model, pixels, device).cpu().numpy().astype(np.float16)
        if vectors.shape != (len(chunk), HIDDEN_SIZE):
            raise RuntimeError("frozen capture-view embedding shape mismatch")
        target[start:start + len(chunk)] = vectors
        target.flush()
        done = start + len(chunk)
        write_json(progress_path, {"manifest_fingerprint": manifest_fingerprint, "completed_rows": done,
                                   "row_count": len(rows), "embedding_file": path.name,
                                   "lora_disabled": True})
        if done % 512 == 0 or done == len(rows):
            print(f"[frozen {split}] {done}/{len(rows)}", flush=True)
    del target
    set_lora_enabled(model, True)
    return path


def encode_views(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    model: torch.nn.Module,
    processor: Any,
    device: torch.device,
    batch_size: int,
    *,
    lora_enabled: bool = True,
) -> np.ndarray:
    set_lora_enabled(model, lora_enabled)
    model.vision_model.eval()
    output = np.empty((len(rows), HIDDEN_SIZE), dtype=np.float16)
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        images = [make_view(row, root) for row in chunk]
        pixels = _processor_pixels(processor, images, model, device)
        with torch.inference_mode():
            vectors = _features(model, pixels, device).cpu().numpy().astype(np.float16)
        output[start:start + len(chunk)] = vectors
        if (start + len(chunk)) % 512 == 0 or start + len(chunk) == len(rows):
            print(f"[encoded {rows[0]['split'] if rows else 'empty'}] {start + len(chunk)}/{len(rows)}", flush=True)
    return output


def encode_references(catalog: Sequence[Mapping[str, str]], root: Path, model: torch.nn.Module,
                      processor: Any, device: torch.device, batch_size: int) -> np.ndarray:
    model.vision_model.eval()
    output = np.empty((len(catalog), HIDDEN_SIZE), dtype=np.float16)
    for start in range(0, len(catalog), batch_size):
        chunk = catalog[start:start + batch_size]
        images = []
        for row in chunk:
            with Image.open(root / row["reference_image_path"]) as image:
                images.append(image.convert("RGB"))
        pixels = _processor_pixels(processor, images, model, device)
        with torch.inference_mode():
            vectors = _features(model, pixels, device).cpu().numpy().astype(np.float16)
        output[start:start + len(chunk)] = vectors
        if (start + len(chunk)) % 512 == 0 or start + len(chunk) == len(catalog):
            print(f"[adapted references] {start + len(chunk)}/{len(catalog)}", flush=True)
    return output


def write_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys = list(rows[0]) if rows else []
    write_csv(path, rows, keys)


def candidate_matrix(slugs: Sequence[str], negative_map: Mapping[str, Mapping[str, Sequence[int]]]) -> tuple[np.ndarray, np.ndarray]:
    idx = {slug: i for i, slug in enumerate(slugs)}
    candidates, same_family = [], []
    for slug in slugs:
        negative = list(negative_map[slug]["all"])
        if idx[slug] in negative or len(negative) != len(set(negative)):
            raise ValueError(f"bad negative candidate map for {slug}")
        candidates.append([idx[slug], *negative])
        same_family.append([idx[slug], *list(negative_map[slug]["same_family"])])
    return np.asarray(candidates, dtype=np.int64), np.asarray(same_family, dtype=object)


def internal_metrics(
    query_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    slugs: Sequence[str],
    negative_map: Mapping[str, Mapping[str, Sequence[int]]],
    family_types: Mapping[str, str],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    q = torch.as_tensor(np.asarray(query_embeddings, dtype=np.float32), device=device)
    r = torch.as_tensor(np.asarray(reference_embeddings, dtype=np.float32), device=device)
    with torch.inference_mode():
        similarities = (q @ r.T).cpu().numpy()
    full_top = np.argmax(similarities, axis=1)
    slug_idx = {slug: i for i, slug in enumerate(slugs)}
    counts = Counter()
    bucket_counts: dict[str, Counter] = defaultdict(Counter)
    margins: list[float] = []
    pos_sims: list[float] = []
    neg_sims: list[float] = []
    per_view = []
    for row_index, row in enumerate(rows):
        slug = str(row["slug"])
        target_index = slug_idx[slug]
        candidate_ids = [target_index, *negative_map[slug]["all"]]
        candidate_choice = candidate_ids[int(np.argmax(similarities[row_index, candidate_ids]))]
        family_ids = [target_index, *negative_map[slug]["same_family"]]
        family_choice = family_ids[int(np.argmax(similarities[row_index, family_ids]))]
        same_neg_ids = list(negative_map[slug]["same_family"])
        positive = float(similarities[row_index, target_index])
        hardest = max((float(similarities[row_index, i]) for i in same_neg_ids), default=float("nan"))
        margin = positive - hardest if same_neg_ids else float("nan")
        full_correct = int(full_top[row_index] == target_index)
        candidate_correct = int(candidate_choice == target_index)
        family_correct = int(family_choice == target_index)
        counts["n"] += 1
        counts["candidate_correct"] += candidate_correct
        counts["full_correct"] += full_correct
        counts["same_family_n"] += int(bool(same_neg_ids))
        counts["same_family_correct"] += family_correct * int(bool(same_neg_ids))
        family_type = family_types.get(slug, "other") if same_neg_ids else "no_family_negative"
        if same_neg_ids:
            counts[f"{family_type}_n"] += 1
            counts[f"{family_type}_correct"] += family_correct
            margins.append(margin)
            pos_sims.append(positive)
            neg_sims.append(hardest)
        bucket = str(row["capture_bucket"])
        bucket_counts[bucket]["n"] += 1
        bucket_counts[bucket]["candidate_correct"] += candidate_correct
        bucket_counts[bucket]["full_correct"] += full_correct
        bucket_counts[bucket]["family_correct"] += family_correct * int(bool(same_neg_ids))
        bucket_counts[bucket]["same_family_n"] += int(bool(same_neg_ids))
        per_view.append({
            "split": row["split"], "slug": slug, "view_index": row["view_index"], "capture_bucket": bucket,
            "capture_bucket_top1_correct": candidate_correct,
            "full_catalog_top1_correct": full_correct,
            "candidate_set_top1_correct": candidate_correct,
            "same_family_top1_correct": family_correct if same_neg_ids else "",
            "family_type": family_type,
            "positive_similarity": positive,
            "hardest_same_family_negative_similarity": hardest if same_neg_ids else "",
            "same_family_margin": margin if same_neg_ids else "",
        })
    n = max(1, counts["n"])
    family_n = max(1, counts["same_family_n"])
    metrics = {
        "query_views": counts["n"],
        "candidate_set_top1": counts["candidate_correct"] / n,
        "full_catalog_top1": counts["full_correct"] / n,
        "same_family_top1": counts["same_family_correct"] / family_n,
        "same_family_views": counts["same_family_n"],
        "same_family_margin_mean": statistics.fmean(margins) if margins else None,
        "positive_similarity_mean": statistics.fmean(pos_sims) if pos_sims else None,
        "hardest_same_family_negative_similarity_mean": statistics.fmean(neg_sims) if neg_sims else None,
        "family_type_metrics": {},
        "validation_buckets": {},
    }
    for family_type in ("vintage", "subtype", "other"):
        f_n = counts[f"{family_type}_n"]
        metrics["family_type_metrics"][family_type] = {
            "views": f_n,
            "top1": counts[f"{family_type}_correct"] / f_n if f_n else None,
        }
    for bucket in CAPTURE_BUCKETS:
        values = bucket_counts[bucket]
        metrics["validation_buckets"][bucket] = {
            "views": values["n"],
            "candidate_set_top1": values["candidate_correct"] / values["n"] if values["n"] else None,
            "full_catalog_top1": values["full_correct"] / values["n"] if values["n"] else None,
            "same_family_top1": values["family_correct"] / values["same_family_n"] if values["same_family_n"] else None,
        }
    return metrics, per_view


def _checkpoint_payload(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                        scheduler: torch.optim.lr_scheduler.LRScheduler, scaler: Any,
                        completed_epoch: int, config_fingerprint: str, metrics: Mapping[str, Any],
                        best_metrics: Mapping[str, Any] | None, best_epoch: int, stale_epochs: int) -> dict[str, Any]:
    return {
        "lora_state": lora_state_dict(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "completed_epoch": completed_epoch,
        "config_fingerprint": config_fingerprint,
        "rng_state": capture_rng_state(),
        "validation_metrics": dict(metrics),
        "best_validation_metrics": dict(best_metrics) if best_metrics is not None else None,
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _save_epoch_checkpoints(run_dir: Path, payload: Mapping[str, Any], is_best: bool) -> None:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(checkpoint_dir / f"epoch_{int(payload['completed_epoch']):03d}.pt", dict(payload))
    atomic_torch_save(checkpoint_dir / "latest.pt", dict(payload))
    if is_best:
        atomic_torch_save(checkpoint_dir / "best.pt", dict(payload))


def _metrics_better(metrics: Mapping[str, Any], best: Mapping[str, Any] | None,
                    frozen_metrics: Mapping[str, Any]) -> bool:
    full_guard = float(frozen_metrics["full_catalog_top1"]) - float(CONFIG["internal_full_catalog_guard_pp"]) / 100.0
    if float(metrics["full_catalog_top1"]) < full_guard:
        return False
    if best is None:
        return True
    primary = float(metrics["same_family_top1"])
    best_primary = float(best["same_family_top1"])
    return (primary, float(metrics["full_catalog_top1"]), float(metrics["candidate_set_top1"])) > (
        best_primary, float(best["full_catalog_top1"]), float(best["candidate_set_top1"])
    )


def build_candidate_id_matrix(
    rows: Sequence[Mapping[str, Any]],
    catalog_index: Mapping[str, int],
    negative_map: Mapping[str, Mapping[str, Sequence[int]]],
) -> tuple[np.ndarray, np.ndarray]:
    """Pad per-SKU candidate lists for batched CE and mark padding as invalid."""
    candidates: list[list[int]] = []
    for row in rows:
        slug = str(row["slug"])
        positive = int(catalog_index[slug])
        negatives = [int(index) for index in negative_map[slug]["all"]]
        if positive in negatives or len(negatives) != len(set(negatives)):
            raise ValueError(f"invalid candidate negatives for {slug}")
        candidates.append([positive, *negatives])
    if not candidates:
        return np.empty((0, 0), dtype=np.int64), np.empty((0, 0), dtype=np.bool_)
    width = max(len(candidate) for candidate in candidates)
    ids = np.zeros((len(candidates), width), dtype=np.int64)
    valid = np.zeros((len(candidates), width), dtype=np.bool_)
    for row_index, candidate in enumerate(candidates):
        ids[row_index, :len(candidate)] = candidate
        valid[row_index, :len(candidate)] = True
    return ids, valid


def train_one_config(
    root: Path,
    run_dir: Path,
    catalog: Sequence[Mapping[str, str]],
    slugs: Sequence[str],
    image_hashes: Mapping[str, str],
    base_reference_embeddings: np.ndarray,
    family_keys: Mapping[str, str | None],
    family_types: Mapping[str, str],
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    train_negative_map: Mapping[str, Mapping[str, Sequence[int]]],
    model: torch.nn.Module,
    processor: Any,
    device: torch.device,
    frozen_train_path: Path,
    frozen_validation_path: Path,
    config_fingerprint: str,
    max_epochs: int,
    resume: bool,
) -> dict[str, Any]:
    frozen_train = np.load(frozen_train_path, mmap_mode="r")
    frozen_validation = np.load(frozen_validation_path, mmap_mode="r")
    frozen_validation_metrics, _ = internal_metrics(frozen_validation, base_reference_embeddings, validation_rows,
                                                    slugs, train_negative_map, family_types, device)
    frozen_train_metrics, _ = internal_metrics(frozen_train, base_reference_embeddings, train_rows,
                                               slugs, train_negative_map, family_types, device)
    write_json(run_dir / "frozen_internal_metrics.json", {
        "validation": frozen_validation_metrics, "train": frozen_train_metrics,
        "selection_guard": f"full_catalog_top1 >= frozen - {CONFIG['internal_full_catalog_guard_pp']}pp",
    })
    cat_index = {slug: index for index, slug in enumerate(slugs)}
    candidate_ids, candidate_valid = build_candidate_id_matrix(train_rows, cat_index, train_negative_map)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    latest_path = run_dir / "checkpoints" / "latest.pt"
    completed_epoch, best_epoch, stale_epochs = 0, 0, 0
    best_metrics = None
    history = read_csv(run_dir / "training_history.csv") if (run_dir / "training_history.csv").exists() else []
    if resume and latest_path.is_file():
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        if checkpoint.get("config_fingerprint") != config_fingerprint:
            raise RuntimeError("resume checkpoint config/catalog fingerprint mismatch")
        load_lora_state_dict(model, checkpoint["lora_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint.get("scaler_state", {}))
        completed_epoch = int(checkpoint["completed_epoch"])
        best_epoch = int(checkpoint.get("best_epoch", 0))
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
        best_metrics = checkpoint.get("best_validation_metrics")
        restore_rng_state(checkpoint["rng_state"])
        print(f"[resume] restored completed epoch {completed_epoch}, best={best_epoch}", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started_all = time.perf_counter()
    train_count = len(train_rows)
    for epoch in range(completed_epoch + 1, max_epochs + 1):
        epoch_started = time.perf_counter()
        adapted_ref = encode_references(catalog, root, model, processor, device,
                                        int(CONFIG["reference_embedding_batch_size"])).astype(np.float32)
        adapted_ref_device = torch.as_tensor(adapted_ref, device=device, dtype=torch.float32)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.vision_model.train()
        optimizer.zero_grad(set_to_none=True)
        order = list(range(train_count))
        random.shuffle(order)
        batch_size = int(CONFIG["micro_batch_size"])
        accumulation = int(CONFIG["gradient_accumulation_steps"])
        window_width = batch_size * accumulation
        seen, stream_correct, loss_total, optimizer_steps = 0, 0, 0.0, 0
        for window_start in range(0, train_count, window_width):
            window = order[window_start:window_start + window_width]
            micro_batches = [window[i:i + batch_size] for i in range(0, len(window), batch_size)]
            optimizer.zero_grad(set_to_none=True)
            for micro in micro_batches:
                views = [make_view(train_rows[index], root) for index in micro]
                pixels = _processor_pixels(processor, views, model, device)
                target_slugs = [str(train_rows[index]["slug"]) for index in micro]
                candidate = candidate_ids[micro]
                candidate_mask = torch.as_tensor(candidate_valid[micro], device=device, dtype=torch.bool)
                candidate_tensor = torch.as_tensor(candidate, device=device, dtype=torch.long)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    query_features = pooled_image_features(model, pixels)
                query_features = F.normalize(query_features.float(), p=2, dim=-1, eps=1e-12)
                selected_ref = adapted_ref_device[candidate_tensor]
                scores = torch.einsum("bd,bcd->bc", query_features, selected_ref) / float(CONFIG["temperature"])
                scores = scores.masked_fill(~candidate_mask, float("-inf"))
                labels = torch.zeros(len(micro), device=device, dtype=torch.long)
                contrastive = F.cross_entropy(scores, labels)
                frozen_target = torch.as_tensor(np.asarray(frozen_train[micro], dtype=np.float32), device=device)
                preservation = (1.0 - F.cosine_similarity(query_features, frozen_target, dim=-1)).mean()
                loss = contrastive + float(CONFIG["preservation_loss_weight"]) * preservation
                scaler.scale(loss / len(micro_batches)).backward()
                stream_correct += int((scores.argmax(dim=1) == labels).sum().item())
                seen += len(micro)
                loss_total += float(loss.detach().item()) * len(micro)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, float(CONFIG["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += 1
            if optimizer_steps % 100 == 0 or window_start + len(window) == train_count:
                print(f"[epoch {epoch}] rows={min(window_start + len(window), train_count)}/{train_count} "
                      f"loss={loss_total/max(seen,1):.4f}", flush=True)
        scheduler.step()

        model.vision_model.eval()
        adapted_ref = encode_references(catalog, root, model, processor, device,
                                        int(CONFIG["reference_embedding_batch_size"])).astype(np.float32)
        val_features = encode_views(root, validation_rows, model, processor, device,
                                    int(CONFIG["query_embedding_batch_size"]), lora_enabled=True)
        val_metrics, _ = internal_metrics(val_features, adapted_ref, validation_rows, slugs,
                                          train_negative_map, family_types, device)
        is_best = _metrics_better(val_metrics, best_metrics, frozen_validation_metrics)
        if is_best:
            best_metrics = val_metrics
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        epoch_seconds = time.perf_counter() - epoch_started
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
        record = {
            "epoch": epoch,
            "train_loss": loss_total / max(seen, 1),
            "streaming_train_candidate_top1": stream_correct / max(seen, 1),
            "validation_candidate_top1": val_metrics["candidate_set_top1"],
            "validation_full_catalog_top1": val_metrics["full_catalog_top1"],
            "validation_same_family_top1": val_metrics["same_family_top1"],
            "validation_same_family_margin": val_metrics["same_family_margin_mean"],
            "validation_vintage_family_top1": val_metrics["family_type_metrics"]["vintage"]["top1"],
            "validation_subtype_family_top1": val_metrics["family_type_metrics"]["subtype"]["top1"],
            "validation_other_family_top1": val_metrics["family_type_metrics"]["other"]["top1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "optimizer_steps": optimizer_steps,
            "epoch_seconds": epoch_seconds,
            "peak_train_vram_mb": peak_vram,
            "is_selected_best": is_best,
        }
        history = [row for row in history if int(row.get("epoch", -1)) != epoch]
        history.append(record)
        write_csv(run_dir / "training_history.csv", history)
        payload = _checkpoint_payload(model, optimizer, scheduler, scaler, epoch, config_fingerprint,
                                      val_metrics, best_metrics, best_epoch, stale_epochs)
        _save_epoch_checkpoints(run_dir, payload, is_best)
        write_json(run_dir / "checkpoints" / "checkpoint_metadata.json", {
            "checkpoint_policy": "LoRA-only weights plus optimizer/scheduler/scaler/RNG/validation state after every epoch",
            "base_model_copied": False,
            "latest_epoch": epoch,
            "best_epoch": best_epoch,
            "latest_lora_sha256": lora_sha256(model),
            "best_validation_metrics": best_metrics,
        })
        write_json(run_dir / "checkpoint_metadata.json", {
            "checkpoint_policy": "LoRA-only weights plus optimizer/scheduler/scaler/RNG/validation state after every epoch",
            "base_model_copied": False,
            "latest_epoch": epoch,
            "best_epoch": best_epoch,
            "latest_lora_sha256": lora_sha256(model),
            "best_validation_metrics": best_metrics,
        })
        write_state(run_dir / "state.json", "training", completed_epoch=epoch, best_epoch=best_epoch,
                    stale_epochs=stale_epochs, elapsed_seconds=time.perf_counter() - started_all)
        print(f"[epoch {epoch}] val same-family={val_metrics['same_family_top1']:.4f}, "
              f"full={val_metrics['full_catalog_top1']:.4f}, best={best_epoch}, "
              f"duration={epoch_seconds/60:.1f}m, peak_vram={peak_vram:.0f}MB", flush=True)
        if stale_epochs >= int(CONFIG["early_stopping_patience"]):
            print(f"[early stop] no eligible internal improvement for {stale_epochs} epochs", flush=True)
            break
    if best_epoch == 0:
        # No adapted checkpoint passed the predeclared full-catalog guard.
        raise RuntimeError("no LoRA checkpoint passed the internal full-catalog guard")
    best = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    load_lora_state_dict(model, best["lora_state"])
    return {
        "frozen_validation_metrics": frozen_validation_metrics,
        "frozen_train_metrics": frozen_train_metrics,
        "best_validation_metrics": best["best_validation_metrics"],
        "best_epoch": int(best["best_epoch"]),
        "completed_epochs": int(best["completed_epoch"]),
        "lora_sha256": lora_sha256(model),
        "best_checkpoint": str(run_dir / "checkpoints" / "best.pt"),
        "peak_train_vram_mb": max((float(row.get("peak_train_vram_mb") or 0) for row in history), default=0),
        "history": history,
    }


def _as_json_list(value: str | Sequence[Any]) -> list[Any]:
    if isinstance(value, str):
        return json.loads(value) if value else []
    return list(value)


def load_external_inputs(root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Load evaluation labels only after the internal checkpoint is frozen."""
    import run_strong_local_visual_reranker as frozen_runner

    primary = frozen_runner.load_benchmark_inputs(root)
    records: dict[str, list[dict[str, Any]]] = {}
    for benchmark in PRIMARY_BENCHMARKS:
        records[benchmark] = list(primary[benchmark].values())
    synthetic_manifest = read_csv(root / "data/benchmarks/synthetic_dev/manifest.csv")
    synthetic_predictions = {
        row["query_id"]: row for row in read_csv(root / SIFT_RUN / "synthetic_dev/predictions.csv")
    }
    synthetic_rows = []
    for row in synthetic_manifest:
        prediction = synthetic_predictions.get(row["query_id"], {})
        synthetic_rows.append({
            "benchmark": "synthetic_dev",
            "query_id": row["query_id"],
            "target_slug": row.get("target_slug") or row.get("slug") or prediction.get("target_slug", ""),
            "query_path": row["query_path"],
            "scenario_id": row.get("scenario_id", ""),
            "subset_role": row.get("subset_role", ""),
            "family_type": row.get("family_type", ""),
            "sift_selected_top1": prediction.get("selected_policy_top1", prediction.get("selected_top1", "")),
            "sift_selected_ranking": _as_json_list(
                prediction.get("selected_policy_ranking", prediction.get("selected_ranking", ""))
            ),
            "old_geometry": {},
        })
    records["synthetic_dev"] = synthetic_rows
    return records, primary


def _load_jsonl(path: Path, key: str) -> dict[str, dict[str, Any]]:
    output = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                output[str(row[key])] = row
    return output


def load_ocr_evidence(root: Path, benchmarks: Sequence[str], catalog: Sequence[Mapping[str, str]]) -> tuple[dict, dict, dict]:
    ref_rows = _load_jsonl(root / OCR_RUN / "ocr/reference_ocr.jsonl", "slug")
    reference = {
        slug: build_reference_ocr_evidence([(line["text"], line["confidence"]) for line in row.get("lines", [])])
        for slug, row in ref_rows.items()
    }
    query: dict[str, dict[str, Any]] = {}
    for benchmark in benchmarks:
        rows = _load_jsonl(root / OCR_RUN / f"ocr/query_ocr_{benchmark}.jsonl", "query_id")
        query[benchmark] = {
            query_id: build_query_text_evidence([(line["text"], line["confidence"]) for line in row.get("lines", [])])
            for query_id, row in rows.items()
        }
    catalog_index = {row["slug"]: build_candidate_text_index(row) for row in catalog}
    return query, reference, catalog_index


def _production_rerank_one(
    image_top5: Sequence[str],
    image_scores: Sequence[float],
    query_evidence: Any,
    reference_evidence: Mapping[str, Any],
    catalog_index: Mapping[str, Any],
    fusion_config: FusionConfig,
    query_sift: Any,
    feature_reader: Any,
    sift_weight: float = 0.40,
) -> dict[str, Any]:
    if len(image_top5) != 5 or len(set(image_top5)) != 5 or len(image_scores) != 5:
        raise ValueError("adapted retrieval must provide exactly five unique candidates and aligned scores")
    signals = [compute_text_signals(query_evidence, catalog_index[slug], reference_evidence.get(slug))
               for slug in image_top5]
    ocr = rerank_one_query(image_scores, signals, fusion_config)
    ocr_indices = list(ocr.final_order)
    ocr_slugs = [image_top5[index] for index in ocr_indices]
    ocr_scores = [float(ocr.final_scores[index]) for index in ocr_indices]
    geometry = [match_sift_pair(query_sift, feature_reader.get(slug), SIFT_CONFIG) for slug in ocr_slugs]
    geometry_scores = [item.geometric_score if item.homography_valid else 0.0 for item in geometry]
    fused = fuse_scores(ocr_scores, geometry_scores, sift_weight)
    final_order = sorted(range(5), key=lambda index: (-float(fused[index]), index))
    return {
        "image_top5": list(image_top5),
        "ocr_top5": ocr_slugs,
        "final_top5": [ocr_slugs[index] for index in final_order],
        "ocr_reordered": ocr.reordered,
        "ocr_reason": ocr.reason,
        "ocr_scores": ocr_scores,
        "sift_scores": geometry_scores,
        "sift_valid_count": sum(item.homography_valid for item in geometry),
        "final_scores": [float(fused[index]) for index in final_order],
        "geometry": geometry,
    }


def _query_embedding_fingerprint(root: Path, benchmark: str,
                                 query_records: Sequence[Mapping[str, Any]],
                                 selected_lora_sha256: str,
                                 batch_size: int = 1) -> str:
    require_canonical_query_batch_size(batch_size)
    identity = {
        "benchmark": benchmark,
        "base_model_id": BASE_MODEL_ID,
        "base_revision": BASE_REVISION,
        "lora_sha256": selected_lora_sha256,
        "query_batch_size": int(batch_size),
        "query_encoder_code_sha256": sha256_file(Path(__file__).resolve()),
        "processor": {
            "class": "SiglipImageProcessor",
            "size": "384x384",
            "resample": "2",
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "center_crop": False,
        },
        "queries": [
            {
                "query_id": str(row["query_id"]),
                "query_path": str(row["query_path"]),
                "query_sha256": sha256_file(root / str(row["query_path"])),
            }
            for row in query_records
        ],
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _encode_query_pair(root: Path, run_dir: Path, benchmark: str,
                       query_records: Sequence[Mapping[str, Any]], model: torch.nn.Module,
                       processor: Any, device: torch.device, batch_size: int,
                       selected_lora_sha256: str) -> tuple[np.ndarray, np.ndarray]:
    require_canonical_query_batch_size(batch_size)
    # The frozen app encoder returns normalized float32 vectors from its FP16
    # model without an autocast context. Keep that precision for the exact
    # baseline reproduction guard; half-casting query scores can change the
    # order of near-tied Top-5 items in a 2,042-item catalog.
    # Preserve float32 query outputs for both paths, as the app adapter does.
    # Persist each benchmark's query features and progress so an interrupted
    # post-freeze evaluation can resume without repeating completed batches.
    cache_dir = run_dir / "benchmarks" / benchmark / "query_embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_path = cache_dir / "frozen_base.npy"
    adapted_path = cache_dir / "selected_lora.npy"
    progress_path = cache_dir / "progress.json"
    fingerprint = _query_embedding_fingerprint(root, benchmark, query_records, selected_lora_sha256, batch_size)
    expected_shape = (len(query_records), HIDDEN_SIZE)
    done = 0
    valid = False
    if base_path.is_file() and adapted_path.is_file() and progress_path.is_file():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            base_view = np.load(base_path, mmap_mode="r")
            adapted_view = np.load(adapted_path, mmap_mode="r")
            valid = (
                progress.get("fingerprint") == fingerprint
                and base_view.shape == expected_shape and base_view.dtype == np.dtype("float32")
                and adapted_view.shape == expected_shape and adapted_view.dtype == np.dtype("float32")
            )
            if valid:
                done = int(progress.get("completed_rows", 0))
                valid = 0 <= done <= len(query_records)
            del base_view, adapted_view
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            valid = False
    if not valid:
        np.lib.format.open_memmap(base_path, mode="w+", dtype=np.float32, shape=expected_shape).flush()
        np.lib.format.open_memmap(adapted_path, mode="w+", dtype=np.float32, shape=expected_shape).flush()
        done = 0
    base = np.load(base_path, mmap_mode="r+")
    adapted = np.load(adapted_path, mmap_mode="r+")
    model.vision_model.eval()
    checkpoint_interval = EXTERNAL_QUERY_CHECKPOINT_INTERVAL
    for start in range(done, len(query_records), batch_size):
        chunk = query_records[start:start + batch_size]
        images = []
        for row in chunk:
            with Image.open(root / str(row["query_path"])) as image:
                images.append(image.convert("RGB"))
        pixels = _processor_pixels(processor, images, model, device)
        set_lora_enabled(model, False)
        with torch.inference_mode():
            base_vecs = pooled_image_features(model, pixels).cpu().numpy()
        set_lora_enabled(model, True)
        with torch.inference_mode():
            adapted_vecs = _features(model, pixels, device).cpu().numpy().astype(np.float32)
        base[start:start + len(chunk)] = base_vecs
        adapted[start:start + len(chunk)] = adapted_vecs
        base.flush()
        adapted.flush()
        completed = start + len(chunk)
        if completed % checkpoint_interval < len(chunk) or completed == len(query_records):
            write_json(progress_path, {
                "benchmark": benchmark,
                "fingerprint": fingerprint,
                "completed_rows": completed,
                "row_count": len(query_records),
                "embedding_dim": HIDDEN_SIZE,
                "dtype": "float32",
                "lora_checkpoint_sha256": selected_lora_sha256,
                "base_model_revision": BASE_REVISION,
                "inference_batch_size": batch_size,
            })
        if (start + len(chunk)) % 256 == 0 or start + len(chunk) == len(query_records):
            print(f"[external {query_records[0].get('benchmark','')}] query embeddings {start + len(chunk)}/{len(query_records)}", flush=True)
    base.flush()
    adapted.flush()
    if not progress_path.is_file() or json.loads(progress_path.read_text(encoding="utf-8")).get("completed_rows") != len(query_records):
        write_json(progress_path, {
            "benchmark": benchmark,
            "fingerprint": fingerprint,
            "completed_rows": len(query_records),
            "row_count": len(query_records),
            "embedding_dim": HIDDEN_SIZE,
            "dtype": "float32",
            "lora_checkpoint_sha256": selected_lora_sha256,
            "base_model_revision": BASE_REVISION,
            "inference_batch_size": batch_size,
        })
    return np.load(base_path, mmap_mode="r"), np.load(adapted_path, mmap_mode="r")


def _full_rankings(query_embeddings: np.ndarray, reference_embeddings: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    q = torch.tensor(np.array(query_embeddings, dtype=np.float32, copy=True), device=device)
    r = torch.tensor(np.array(reference_embeddings, dtype=np.float32, copy=True), device=device)
    with torch.inference_mode():
        sims = (q @ r.T).cpu().numpy()
    ranks = np.argsort(-sims, axis=1, kind="stable").astype(np.int32)
    return sims, ranks


def _external_metric(rows: Sequence[Mapping[str, Any]], rankings: np.ndarray, slugs: Sequence[str],
                     subset: Sequence[int] | None = None) -> dict[str, Any]:
    chosen = list(range(len(rows))) if subset is None else list(subset)
    slug_index = {slug: index for index, slug in enumerate(slugs)}
    ranks, reciprocal, correct1, correct5, correct10 = [], [], 0, 0, 0
    for row_index in chosen:
        target = str(rows[row_index]["target_slug"])
        if target not in slug_index:
            raise KeyError(f"target outside usable catalog: {target}")
        order = rankings[row_index].tolist()
        rank = order.index(slug_index[target]) + 1
        ranks.append(rank)
        correct1 += rank == 1
        correct5 += rank <= 5
        correct10 += rank <= 10
        reciprocal.append(1.0 / rank)
    count = len(chosen)
    return {
        "queries": count,
        "top1_correct": correct1,
        "top1": correct1 / count if count else None,
        "recall_at_5_correct": correct5,
        "recall_at_5": correct5 / count if count else None,
        "recall_at_10_correct": correct10,
        "recall_at_10": correct10 / count if count else None,
        "mrr": statistics.fmean(reciprocal) if reciprocal else None,
    }


def measure_latency(
    root: Path,
    generated_rows: Sequence[Mapping[str, Any]],
    model: torch.nn.Module,
    processor: Any,
    device: torch.device,
    frozen_references: np.ndarray,
    adapted_references: np.ndarray,
    slugs: Sequence[str],
    query_evidence: Mapping[str, Any],
    reference_ocr: Mapping[str, Any],
    candidate_text: Mapping[str, Any],
    fusion_config: FusionConfig,
    feature_reader: Any,
) -> list[dict[str, Any]]:
    """Measure query-only and whole adapted/frozen OCR+SIFT inference."""
    rows = []
    ref_tensors = {
        "frozen": torch.as_tensor(frozen_references, dtype=torch.float32, device=device),
        "adapted": torch.as_tensor(adapted_references, dtype=torch.float32, device=device),
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for mode in ("frozen", "adapted"):
        enabled = mode == "adapted"
        set_lora_enabled(model, enabled)
        model.vision_model.eval()
        pipeline_times, encoder_times = [], []
        # Warm up several forward calls before taking the measured 128-query pass.
        for query in list(generated_rows)[:3]:
            with Image.open(root / str(query["query_path"])) as image:
                pixels = _processor_pixels(processor, [image.convert("RGB")], model, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            with torch.inference_mode():
                _features(model, pixels, device)
        for query in generated_rows:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            pipeline_started = time.perf_counter()
            with Image.open(root / str(query["query_path"])) as image:
                pixels = _processor_pixels(processor, [image.convert("RGB")], model, device)
            encoder_started = time.perf_counter()
            with torch.inference_mode():
                vector = _features(model, pixels, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            encoder_times.append((time.perf_counter() - encoder_started) * 1000.0)
            with torch.inference_mode():
                similarities = torch.matmul(vector, ref_tensors[mode].T).squeeze(0).cpu().numpy()
            image_order = np.argsort(-similarities, kind="stable")[:5]
            image_top5 = [slugs[int(index)] for index in image_order]
            image_scores = [float(similarities[int(index)]) for index in image_order]
            query_sift = extract_sift(read_rgb(root / str(query["query_path"])), SIFT_CONFIG)
            _production_rerank_one(
                image_top5, image_scores, query_evidence[str(query["query_id"])],
                reference_ocr, candidate_text, fusion_config, query_sift, feature_reader, 0.40,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            pipeline_times.append((time.perf_counter() - pipeline_started) * 1000.0)
        peak = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
        rows.append({
            "mode": mode,
            "queries": len(generated_rows),
            "encoder_mean_ms": statistics.fmean(encoder_times) if encoder_times else None,
            "encoder_p95_ms": float(np.percentile(encoder_times, 95)) if encoder_times else None,
            "full_pipeline_mean_ms": statistics.fmean(pipeline_times) if pipeline_times else None,
            "full_pipeline_p95_ms": float(np.percentile(pipeline_times, 95)) if pipeline_times else None,
            "full_pipeline_max_ms": max(pipeline_times) if pipeline_times else None,
            "peak_inference_vram_mb": peak,
            "sla_3s_passed": bool(pipeline_times and np.percentile(pipeline_times, 95) < 3000),
        })
    set_lora_enabled(model, True)
    return rows


def _subset_indices(rows: Sequence[Mapping[str, Any]], field: str, value: str) -> list[int]:
    return [i for i, row in enumerate(rows) if str(row.get(field, "")) == value]


def _load_family_labels(root: Path, benchmark: str) -> tuple[dict[str, str], dict[str, str]]:
    if benchmark == "hard_near_duplicate_dev_v2":
        family_file = root / "data/benchmarks/hard_near_duplicate_dev_v2/families.csv"
        family_id_field = "family_id"
        type_map = {}
        hard_families = defaultdict(list)
        for row in read_csv(family_file):
            hard_families[row[family_id_field]].append(row)
        year_rx = __import__("re").compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
        for family_id, members in hard_families.items():
            years = {year for row in members for year in year_rx.findall(" ".join((row.get("product_name", ""), row.get("year_if_known", ""))))}
            grapes = {row.get("grape", "").strip().casefold() for row in members if row.get("grape", "").strip()}
            family_type = "vintage" if len(years) >= 2 else "subtype" if len(grapes) >= 2 else "other"
            for row in members:
                type_map[row["slug"]] = family_type
        return {row["slug"]: row[family_id_field] for row in read_csv(family_file)}, type_map
    if benchmark == "generated_stress_dev_pilot32":
        family_file = root / "data/benchmarks/generated_stress_dev_pilot32/hard_families.csv"
        family_map, type_map = {}, {}
        for row in read_csv(family_file):
            for slug in _as_json_list(row["member_slugs"]):
                family_map[str(slug)] = row["pilot_family_id"]
                type_map[str(slug)] = row["family_type"]
        return family_map, type_map
    return {}, {}


def evaluate_internal_final(
    root: Path,
    run_dir: Path,
    model: torch.nn.Module,
    processor: Any,
    device: torch.device,
    catalog: Sequence[Mapping[str, str]],
    slugs: Sequence[str],
    image_hashes: Mapping[str, str],
    base_reference_embeddings: np.ndarray,
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    negative_map: Mapping[str, Mapping[str, Sequence[int]]],
    family_types: Mapping[str, str],
    selected: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    model.vision_model.eval()
    checkpoint_hash = lora_sha256(model)
    if checkpoint_hash != selected["lora_sha256"]:
        raise RuntimeError("loaded LoRA weights differ from the internally selected checkpoint")
    adapted_refs = encode_references(catalog, root, model, processor, device,
                                     int(CONFIG["reference_embedding_batch_size"])).astype(np.float32)
    if not np.allclose(np.linalg.norm(adapted_refs, axis=1), 1.0, atol=2e-3):
        raise RuntimeError("adapted reference embeddings are not normalized")
    fingerprint = adapted_reference_fingerprint(
        str(json.loads((root / BASE_CACHE / "metadata.json").read_text(encoding="utf-8"))["fingerprint"]),
        checkpoint_hash, slugs, image_hashes,
        query_batch_size=CANONICAL_EXTERNAL_PROTOCOL["query_batch_size"],
    )
    ref_dir = run_dir / "adapted_reference_cache"
    ref_dir.mkdir(parents=True, exist_ok=True)
    np.save(ref_dir / "embeddings.npy", adapted_refs.astype(np.float16))
    write_json(ref_dir / "metadata.json", {
        "base_model_id": BASE_MODEL_ID, "base_revision": BASE_REVISION,
        "lora_checkpoint_sha256": checkpoint_hash, "fingerprint": fingerprint,
        "query_batch_size": CANONICAL_EXTERNAL_PROTOCOL["query_batch_size"],
        "reference_count": len(slugs), "embedding_dim": HIDDEN_SIZE,
        "all_references_encoded_with_selected_checkpoint": True,
        "reference_slugs_file": "slugs.json",
    })
    write_json(ref_dir / "slugs.json", list(slugs))
    train_features = encode_views(root, train_rows, model, processor, device,
                                  int(CONFIG["query_embedding_batch_size"]), lora_enabled=True)
    validation_features = encode_views(root, validation_rows, model, processor, device,
                                       int(CONFIG["query_embedding_batch_size"]), lora_enabled=True)
    train_metrics, train_margins = internal_metrics(train_features, adapted_refs, train_rows, slugs,
                                                    negative_map, family_types, device)
    validation_metrics, validation_margins = internal_metrics(validation_features, adapted_refs, validation_rows,
                                                              slugs, negative_map, family_types, device)
    frozen_val = np.load(run_dir / "frozen_validation_view_embeddings.npy", mmap_mode="r")
    frozen_val_metrics, frozen_val_margins = internal_metrics(frozen_val, base_reference_embeddings, validation_rows,
                                                              slugs, negative_map, family_types, device)
    write_json(run_dir / "internal_validation_metrics.json", {
        "selected_checkpoint_sha256": checkpoint_hash,
        "selected_epoch": selected["best_epoch"],
        "train_frozen": json.loads((run_dir / "frozen_internal_metrics.json").read_text(encoding="utf-8"))["train"],
        "validation_frozen": frozen_val_metrics,
        "train_adapted": train_metrics,
        "validation_adapted": validation_metrics,
        "selected_validation_checkpoint_metrics": selected["best_validation_metrics"],
        "train_minus_validation_candidate_top1": train_metrics["candidate_set_top1"] - validation_metrics["candidate_set_top1"],
        "train_minus_validation_full_catalog_top1": train_metrics["full_catalog_top1"] - validation_metrics["full_catalog_top1"],
    })
    margin_rows = []
    for group, values in (("validation_frozen", frozen_val_margins), ("validation_adapted", validation_margins)):
        for row in values:
            margin_rows.append({"source": group, **row})
    write_csv(run_dir / "embedding_margin_analysis.csv", margin_rows)
    write_json(run_dir / "internal_validation_buckets.json", {
        "validation_frozen": frozen_val_metrics["validation_buckets"],
        "validation_adapted": validation_metrics["validation_buckets"],
    })
    return adapted_refs, {"fingerprint": fingerprint, "checkpoint_hash": checkpoint_hash,
                          "train": train_metrics, "validation": validation_metrics,
                          "validation_frozen": frozen_val_metrics}


def _baseline_top5_rows(root: Path, benchmark: str) -> dict[str, dict[str, str]]:
    return {row["query_id"]: row for row in read_csv(root / OCR_RUN / "benchmarks" / benchmark / "baseline_predictions.csv")}


def exact_top5_reproduction(actual: Sequence[str], expected: Sequence[str]) -> bool:
    return len(actual) == len(expected) == 5 and list(actual) == list(expected)


def sift_reference_cache_aligned(catalog_slugs: Sequence[str], cached_reference_paths: Mapping[str, Path]) -> bool:
    return len(catalog_slugs) == len(cached_reference_paths) and set(catalog_slugs) == set(cached_reference_paths)


def _sim_margin(scores: np.ndarray, target_slug: str, family_id: str | None,
                family_map: Mapping[str, str], slugs: Sequence[str]) -> tuple[float | None, float | None, int]:
    slug_index = {slug: index for index, slug in enumerate(slugs)}
    if target_slug not in slug_index or not family_id:
        return None, None, 0
    family_negatives = [slug_index[slug] for slug in slugs if slug != target_slug and family_map.get(slug) == family_id]
    if not family_negatives:
        return float(scores[slug_index[target_slug]]), None, 0
    positive = float(scores[slug_index[target_slug]])
    hardest = max(float(scores[index]) for index in family_negatives)
    return positive, hardest, len(family_negatives)


def _slice_keys(row: Mapping[str, Any]) -> list[tuple[str, str]]:
    keys = [("overall", "all")]
    for field in ("subset_role", "scenario_id", "family_type"):
        value = str(row.get(field, ""))
        if value and value.lower() not in {"none", "nan"}:
            keys.append((field, value))
    return keys


def _top1_transitions(rows: Sequence[Mapping[str, Any]], key: str = "adapted_production_top1",
                      baseline_key: str = "frozen_production_top1") -> dict[str, int]:
    result = Counter()
    for row in rows:
        old_correct = row[baseline_key] == row["target_slug"]
        new_correct = row[key] == row["target_slug"]
        if not old_correct and new_correct:
            result["rescued"] += 1
        elif old_correct and not new_correct:
            result["broken"] += 1
        elif old_correct:
            result["correct_to_correct"] += 1
        else:
            result["wrong_to_wrong"] += 1
    for category in ("rescued", "broken", "correct_to_correct", "wrong_to_wrong"):
        result.setdefault(category, 0)
    result["net_gain"] = result["rescued"] - result["broken"]
    return dict(result)


def run_external_benchmarks(
    root: Path,
    run_dir: Path,
    catalog: Sequence[Mapping[str, str]],
    slugs: Sequence[str],
    base_reference_embeddings: np.ndarray,
    adapted_reference_embeddings: np.ndarray,
    model: torch.nn.Module,
    processor: Any,
    device: torch.device,
    selected: Mapping[str, Any],
    selected_cache: Mapping[str, Any],
) -> dict[str, Any]:
    """Run frozen and adapted retrieval, then OCR+SIFT, after checkpoint freeze."""
    require_canonical_query_batch_size(CANONICAL_EXTERNAL_PROTOCOL["query_batch_size"])
    if EXTERNAL_QUERY_EMBEDDING_BATCH_SIZE != CANONICAL_EXTERNAL_PROTOCOL["query_batch_size"]:
        raise RuntimeError("external query encoder batch differs from the canonical benchmark protocol")
    baseline_artifact = root / CANONICAL_EXTERNAL_PROTOCOL["baseline_artifact"]
    baseline_payload = json.loads(baseline_artifact.read_text(encoding="utf-8"))
    baseline_state = json.loads((baseline_artifact.parent / "state.json").read_text(encoding="utf-8"))
    if baseline_state.get("baseline_valid") is not True:
        raise RuntimeError("canonical frozen baseline artifact is not marked valid; external evaluation is blocked")
    require_exact_baseline_reproduction(baseline_payload["after_single_query_fix"], {
        "hard_near_duplicate_dev_v2": 1150,
        "generated_stress_dev_pilot32": 128,
        "synthetic_dev": 4084,
    })
    import run_final_ml_sanity_check as final_runner
    import run_hard_negative_metric_adapter as adapter_runner
    import run_strong_local_visual_reranker as strong_runner

    if not (run_dir / "selected_model.json").is_file():
        raise RuntimeError("external benchmark evaluation is blocked until selected_model.json is frozen")
    selected_policy = json.loads((run_dir / "selected_policy.json").read_text(encoding="utf-8"))
    require_frozen_before_external({
        "stage": "frozen" if selected_policy.get("frozen") is True else "unfrozen",
        "checkpoint_sha256": selected_policy.get("checkpoint_sha256"),
    })
    require_canonical_query_batch_size(selected_policy.get("canonical_query_batch_size", -1))
    if selected_policy.get("checkpoint_sha256") != selected.get("lora_sha256"):
        raise RuntimeError("external benchmark policy checkpoint differs from the selected frozen model")
    if lora_sha256(model) != selected["lora_sha256"]:
        raise RuntimeError("external evaluation LoRA hash differs from the frozen model")
    assert_same_lora_checkpoint(str(selected["lora_sha256"]),
                                str(selected_cache.get("lora_checkpoint_sha256", "")))
    benchmark_records, primary_data = load_external_inputs(root)
    ocr_evidence, reference_ocr, candidate_text = load_ocr_evidence(root, BENCHMARKS, catalog)
    reranker_config = json.loads((root / OCR_RUN / "selected_reranker.json").read_text(encoding="utf-8"))["config"]
    fusion_config = FusionConfig(**reranker_config)
    reference_feature_reader, reference_paths, sift_index = strong_runner.load_reference_feature_reader(root)
    if not sift_reference_cache_aligned(slugs, reference_paths):
        raise RuntimeError("SIFT reference cache is not aligned with the unchanged 2042-image catalog")
    baseline_sift_reproduction = adapter_runner.verify_frozen_sift_baseline(root)
    sift_reproduction = strong_runner.load_benchmark_inputs(root)
    # The final frozen artifact includes exact current+SIFT Top-5 order checks.
    for benchmark in PRIMARY_BENCHMARKS:
        for row in benchmark_records[benchmark]:
            if not row.get("sift_selected_top1"):
                raise RuntimeError(f"frozen production prediction missing for {benchmark}/{row['query_id']}")

    query_metrics: dict[str, Any] = {}
    all_external_margin_rows = []
    all_production_predictions: dict[str, list[dict[str, Any]]] = {}
    all_summary_rows, all_transition_rows, all_scenario_rows, all_family_rows = [], [], [], []
    reference_slugs_path = run_dir / "reference_slugs.json"
    write_json(reference_slugs_path, list(slugs))
    external_margin_by_benchmark: dict[str, list[dict[str, Any]]] = {}
    for benchmark in BENCHMARKS:
        records = benchmark_records[benchmark]
        if not records:
            raise RuntimeError(f"empty external benchmark: {benchmark}")
        expected_ids = {str(row["query_id"]) for row in records}
        expected_ocr = set(ocr_evidence[benchmark])
        if expected_ids != expected_ocr:
            raise RuntimeError(f"OCR cache IDs do not align with frozen benchmark {benchmark}")
        base_q, adapted_q = _encode_query_pair(
            root, run_dir, benchmark, records, model, processor, device,
            EXTERNAL_QUERY_EMBEDDING_BATCH_SIZE, str(selected["lora_sha256"]),
        )
        base_sim, base_ranks = _full_rankings(base_q, base_reference_embeddings, device)
        adapted_sim, adapted_ranks = _full_rankings(adapted_q, adapted_reference_embeddings, device)
        benchmark_dir = run_dir / "benchmarks" / benchmark
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        np.save(benchmark_dir / "frozen_full_ranking_indices.npy", base_ranks)
        np.save(benchmark_dir / "adapted_full_ranking_indices.npy", adapted_ranks)
        write_json(benchmark_dir / "ranking_reference_slugs.json", list(slugs))
        # Baseline reproduction is against the stored frozen image Top-5; the
        # target is used only for the subsequent evaluation metrics.
        expected_top5 = _baseline_top5_rows(root, benchmark)
        exact_orders = 0
        mismatched_examples = []
        for index, row in enumerate(records):
            saved = expected_top5.get(str(row["query_id"]))
            if not saved:
                raise RuntimeError(f"frozen image predictions missing for {benchmark}/{row['query_id']}")
            expected = _as_json_list(saved.get("top5_slugs", ""))
            actual = [slugs[int(i)] for i in base_ranks[index, :5]]
            reproduced = exact_top5_reproduction(actual, expected)
            exact_orders += reproduced
            if not reproduced and len(mismatched_examples) < 20:
                mismatched_examples.append({
                    "query_id": str(row["query_id"]),
                    "expected_top5": expected,
                    "recomputed_top5": actual,
                })
        write_json(benchmark_dir / "base_top5_reproduction.json", {
            "exact_top5": exact_orders,
            "queries": len(records),
            "exact_reproduction": exact_orders == len(records),
            "mismatched_query_count": len(records) - exact_orders,
            "mismatched_examples_first_20": mismatched_examples,
        })
        if exact_orders != len(records):
            raise RuntimeError(
                f"STOP adapted external evaluation: canonical frozen Top-5 is {exact_orders}/{len(records)} "
                f"on {benchmark} with query_batch_size={CANONICAL_QUERY_BATCH_SIZE}"
            )

        baseline_image_metrics = _external_metric(records, base_ranks, slugs)
        adapted_image_metrics = _external_metric(records, adapted_ranks, slugs)
        family_map, family_type_map = _load_family_labels(root, benchmark)
        slug_index = {slug: index for index, slug in enumerate(slugs)}
        production_rows = []
        benchmark_margin_rows = []
        adapted_top1s = []
        frozen_final_top1s = []
        image_top1s = []
        for index, row in enumerate(records):
            target = str(row["target_slug"])
            image_candidates = [slugs[int(i)] for i in adapted_ranks[index, :5]]
            image_scores = [float(adapted_sim[index, int(i)]) for i in adapted_ranks[index, :5]]
            query_rgb = read_rgb(root / str(row["query_path"]))
            query_sift = extract_sift(query_rgb, SIFT_CONFIG)
            output = _production_rerank_one(
                image_candidates, image_scores, ocr_evidence[benchmark][str(row["query_id"])],
                reference_ocr, candidate_text, fusion_config, query_sift, reference_feature_reader, 0.40,
            )
            final = output["final_top5"]
            baseline_final = str(row.get("sift_selected_top1", ""))
            if not baseline_final and benchmark == "synthetic_dev":
                raise RuntimeError(f"synthetic frozen production prediction missing for {row['query_id']}")
            family_id = family_map.get(target)
            base_margin = _sim_margin(base_sim[index], target, family_id, family_map, slugs)
            adapted_margin = _sim_margin(adapted_sim[index], target, family_id, family_map, slugs)
            frozen_incumbent = str(row.get("sift_selected_top1", ""))
            frozen_incumbent_index = slug_index.get(frozen_incumbent)
            frozen_target_score = float(base_sim[index, slug_index[target]])
            adapted_target_score = float(adapted_sim[index, slug_index[target]])
            frozen_incumbent_score = float(base_sim[index, frozen_incumbent_index]) if frozen_incumbent_index is not None else None
            adapted_incumbent_score = float(adapted_sim[index, frozen_incumbent_index]) if frozen_incumbent_index is not None else None
            target_rank = int(np.flatnonzero(base_ranks[index] == slug_index[target])[0]) + 1
            adapted_target_rank = int(np.flatnonzero(adapted_ranks[index] == slug_index[target])[0]) + 1
            record = {
                "benchmark": benchmark,
                "query_id": str(row["query_id"]),
                "target_slug": target,
                "scenario_id": str(row.get("scenario_id", "")),
                "subset_role": str(row.get("subset_role", "")),
                "family_type": str(row.get("family_type") or family_type_map.get(target, "")),
                "frozen_image_top1": slugs[int(base_ranks[index, 0])],
                "adapted_image_top1": slugs[int(adapted_ranks[index, 0])],
                "frozen_image_target_rank": target_rank,
                "adapted_image_target_rank": adapted_target_rank,
                "adapted_image_top5": json.dumps(image_candidates, ensure_ascii=False),
                "adapted_ocr_top5": json.dumps(output["ocr_top5"], ensure_ascii=False),
                "adapted_production_top5": json.dumps(final, ensure_ascii=False),
                "frozen_production_top1": baseline_final,
                "adapted_production_top1": final[0],
                "target_in_adapted_top5": target in image_candidates,
                "target_in_adapted_top10": target in [slugs[int(i)] for i in adapted_ranks[index, :10]],
                "sift_valid_pairs": output["sift_valid_count"],
                "lora_checkpoint_sha256": selected["lora_sha256"],
            }
            production_rows.append(record)
            adapted_top1s.append(record["adapted_production_top1"])
            frozen_final_top1s.append(baseline_final)
            image_top1s.append(record["adapted_image_top1"])
            margin_row = {
                "benchmark": benchmark, "query_id": str(row["query_id"]), "target_slug": target,
                "family_id": family_id or "", "family_competitors": base_margin[2],
                "frozen_positive_similarity": base_margin[0] if base_margin[0] is not None else "",
                "frozen_hardest_family_negative_similarity": base_margin[1] if base_margin[1] is not None else "",
                "frozen_margin": base_margin[0] - base_margin[1] if base_margin[0] is not None and base_margin[1] is not None else "",
                "adapted_positive_similarity": adapted_margin[0] if adapted_margin[0] is not None else "",
                "adapted_hardest_family_negative_similarity": adapted_margin[1] if adapted_margin[1] is not None else "",
                "adapted_margin": adapted_margin[0] - adapted_margin[1] if adapted_margin[0] is not None and adapted_margin[1] is not None else "",
                "adapted_image_rank": adapted_target_rank,
                "frozen_production_incumbent": frozen_incumbent,
                "frozen_target_minus_incumbent_similarity": frozen_target_score - frozen_incumbent_score if frozen_incumbent_score is not None else "",
                "adapted_target_minus_frozen_incumbent_similarity": adapted_target_score - adapted_incumbent_score if adapted_incumbent_score is not None else "",
                "adapted_target_beats_frozen_incumbent": adapted_target_score > adapted_incumbent_score if adapted_incumbent_score is not None else "",
            }
            benchmark_margin_rows.append(margin_row)
            all_external_margin_rows.append(margin_row)
        write_csv(benchmark_dir / "production_predictions.csv", production_rows)
        write_csv(benchmark_dir / "embedding_margin_analysis.csv", benchmark_margin_rows)
        external_margin_by_benchmark[benchmark] = benchmark_margin_rows
        all_production_predictions[benchmark] = production_rows
        metrics = {
            "queries": len(records),
            "frozen_image_only": baseline_image_metrics,
            "adapted_image_only": adapted_image_metrics,
            "frozen_production_top1_correct": sum(row["frozen_production_top1"] == row["target_slug"] for row in production_rows),
            "adapted_production_top1_correct": sum(row["adapted_production_top1"] == row["target_slug"] for row in production_rows),
            "adapted_production_top1": sum(row["adapted_production_top1"] == row["target_slug"] for row in production_rows) / len(records),
            "adapted_production_recall_at_5": sum(row["target_in_adapted_top5"] for row in production_rows) / len(records),
            "adapted_production_recall_at_10": adapted_image_metrics["recall_at_10"],
            "exact_frozen_image_top5_reproduction": exact_orders,
            "frozen_reference_sift_cache_fingerprint": sift_index["cache_fingerprint"],
        }
        if benchmark == "generated_stress_dev_pilot32":
            for subset in ("representative", "hard"):
                subset_rows = [row for row in production_rows if row["subset_role"] == subset]
                metrics[f"generated_{subset}"] = {
                    "correct": sum(row["adapted_production_top1"] == row["target_slug"] for row in subset_rows),
                    "queries": len(subset_rows),
                    "top1": sum(row["adapted_production_top1"] == row["target_slug"] for row in subset_rows) / max(1, len(subset_rows)),
                }
            metrics["generated_transitions"] = _top1_transitions(production_rows)
        query_metrics[benchmark] = metrics

        baseline_correct = sum(row["frozen_production_top1"] == row["target_slug"] for row in production_rows)
        adapted_correct = sum(row["adapted_production_top1"] == row["target_slug"] for row in production_rows)
        all_summary_rows.extend([
            {"benchmark": benchmark, "method": "frozen_image_only", **baseline_image_metrics},
            {"benchmark": benchmark, "method": "adapted_image_only", **adapted_image_metrics},
            {"benchmark": benchmark, "method": "frozen_production", "queries": len(records),
             "top1_correct": baseline_correct, "top1": baseline_correct / len(records)},
            {"benchmark": benchmark, "method": "adapted_production", "queries": len(records),
             "top1_correct": adapted_correct, "top1": adapted_correct / len(records)},
        ])
        transition_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in production_rows:
            for group in _slice_keys(row):
                transition_groups[group].append(row)
        for (dimension, label), subset_rows in transition_groups.items():
            trans = _top1_transitions(subset_rows)
            all_transition_rows.append({"benchmark": benchmark, "slice": dimension, "value": label,
                                        "queries": len(subset_rows), **trans})
        for dimension in ("scenario_id", "subset_role", "family_type"):
            groups = defaultdict(list)
            for row in production_rows:
                if row.get(dimension):
                    groups[str(row[dimension])].append(row)
            for value, subset_rows in groups.items():
                all_scenario_rows.append({
                    "benchmark": benchmark, "dimension": dimension, "value": value,
                    "queries": len(subset_rows),
                    "frozen_production_correct": sum(row["frozen_production_top1"] == row["target_slug"] for row in subset_rows),
                    "adapted_image_only_correct": sum(row["adapted_image_top1"] == row["target_slug"] for row in subset_rows),
                    "adapted_production_correct": sum(row["adapted_production_top1"] == row["target_slug"] for row in subset_rows),
                    "adapted_production_top1": sum(row["adapted_production_top1"] == row["target_slug"] for row in subset_rows) / len(subset_rows),
                    **_top1_transitions(subset_rows),
                })
        query_metrics[benchmark].update({"frozen_production_correct": baseline_correct,
                                         "adapted_production_correct": adapted_correct,
                                         "production_net_gain": adapted_correct - baseline_correct})
        write_json(benchmark_dir / "metrics.json", metrics)
        print(f"[{benchmark}] adapted image={adapted_image_metrics['top1_correct']}/{len(records)} "
              f"production={adapted_correct}/{len(records)}", flush=True)

    generated_rows = benchmark_records["generated_stress_dev_pilot32"]
    latency_rows = measure_latency(
        root, generated_rows, model, processor, device, base_reference_embeddings,
        adapted_reference_embeddings, slugs, ocr_evidence["generated_stress_dev_pilot32"],
        reference_ocr, candidate_text, fusion_config, reference_feature_reader,
    )
    write_csv(run_dir / "latency_summary.csv", latency_rows)

    write_csv(run_dir / "benchmark_summary.csv", all_summary_rows)
    write_csv(run_dir / "transition_summary.csv", all_transition_rows)
    write_csv(run_dir / "scenario_metrics.csv", all_scenario_rows)
    write_csv(run_dir / "family_metrics.csv", [row for row in all_scenario_rows if row["dimension"] == "family_type"])
    old_margin_path = run_dir / "embedding_margin_analysis.csv"
    old_margin_rows = read_csv(old_margin_path) if old_margin_path.is_file() else []
    margin_fields = list(dict.fromkeys(key for row in [*old_margin_rows, *all_external_margin_rows] for key in row))
    write_csv(run_dir / "embedding_margin_analysis.csv", [*old_margin_rows, *all_external_margin_rows], margin_fields)
    write_json(run_dir / "baseline_reproduction.json", {
        "frozen_current_ocr_order_reproduced": baseline_sift_reproduction,
        "frozen_current_plus_sift_reproduced": baseline_sift_reproduction["current_plus_sift_w0.40"],
        "base_encoder_query_top5_reproduction": {name: query_metrics[name]["exact_frozen_image_top5_reproduction"] for name in BENCHMARKS},
    })
    if "generated_stress_dev_pilot32" in external_margin_by_benchmark:
        frozen_errors = [row for row in all_production_predictions["generated_stress_dev_pilot32"]
                         if row["frozen_production_top1"] != row["target_slug"]]
        error_ids = {row["query_id"] for row in frozen_errors}
        oracle_rows = [row for row in external_margin_by_benchmark["generated_stress_dev_pilot32"]
                       if row["query_id"] in error_ids]
        oracle_wins = sum(row["adapted_target_beats_frozen_incumbent"] is True or str(row["adapted_target_beats_frozen_incumbent"]).lower() == "true" for row in oracle_rows)
        write_json(run_dir / "remaining_generated_error_oracle.json", {
            "frozen_generated_errors": len(frozen_errors),
            "adapted_target_beats_frozen_production_incumbent": oracle_wins,
            "mean_adapted_target_minus_incumbent_similarity": statistics.fmean(
                float(row["adapted_target_minus_frozen_incumbent_similarity"]) for row in oracle_rows
                if row["adapted_target_minus_frozen_incumbent_similarity"] != ""),
            "note": "Diagnostic only. The target is used only after ranking for evaluation; it is never used in candidate selection or training.",
        })
    return {
        "metrics": query_metrics,
        "production_predictions": all_production_predictions,
        "sift_cache_fingerprint": sift_index["cache_fingerprint"],
        "reference_cache_path": str(sift_index.get("cache_dir", "")),
        "base_encoder_query_top5_reproduced": all(row["exact_frozen_image_top5_reproduction"] == row["queries"] for row in query_metrics.values()),
        "latency": latency_rows,
        "benchmark_count": len(BENCHMARKS),
    }


def make_final_verdict(internal: Mapping[str, Any], external: Mapping[str, Any],
                      production_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[str, str, dict[str, Any]]:
    generated = external["metrics"]["generated_stress_dev_pilot32"]
    hard = external["metrics"]["hard_near_duplicate_dev_v2"]
    generated_rows = production_rows["generated_stress_dev_pilot32"]
    hard_rows = production_rows["hard_near_duplicate_dev_v2"]
    gen_hard = [row for row in generated_rows if row["subset_role"] == "hard"]
    gen_rep = [row for row in generated_rows if row["subset_role"] == "representative"]
    gen_hard_correct = sum(row["adapted_production_top1"] == row["target_slug"] for row in gen_hard)
    gen_rep_correct = sum(row["adapted_production_top1"] == row["target_slug"] for row in gen_rep)
    hard_correct = int(hard["adapted_production_correct"])
    hard_guard = hard_correct / max(1, len(hard_rows)) >= 0.909
    generated_correct = int(generated["adapted_production_correct"])
    baseline_generated_correct = int(generated["frozen_production_correct"])
    net_gain = generated_correct - baseline_generated_correct
    internal_gain = float(internal["validation"]["same_family_top1"]) - float(internal["validation_frozen"]["same_family_top1"])
    full_gain = float(internal["validation"]["full_catalog_top1"]) - float(internal["validation_frozen"]["full_catalog_top1"])
    if generated_correct >= 116 and hard_guard:
        verdict = "A. FIX_LORA"
        reason = "Generated production reached at least 116/128 and hard_v2 stayed at or above the 90.9% guard."
    elif net_gain >= 5:
        verdict = "B. LORA_HELPS_BUT_BELOW_TARGET"
        reason = f"LoRA produced a net external gain of {net_gain} generated queries, but the required 116/128 target was not reached."
    elif internal_gain >= 0.05 or full_gain >= 0.05:
        verdict = "C. TRAIN_DOMAIN_MISMATCH"
        reason = (f"Internal validation improved (same-family {internal_gain:+.1%}, full-catalog {full_gain:+.1%}) "
                  f"while generated production gained only {net_gain} net queries; this points to a capture-domain mismatch.")
    else:
        verdict = "D. LORA_CAPACITY_INSUFFICIENT"
        reason = f"Internal improvement was weak (same-family {internal_gain:+.1%}, full-catalog {full_gain:+.1%}) and generated production gained {net_gain} net queries."
    decision = {
        "generated_production_correct": generated_correct, "generated_production_queries": len(generated_rows),
        "generated_hard_correct": gen_hard_correct, "generated_hard_queries": len(gen_hard),
        "generated_representative_correct": gen_rep_correct, "generated_representative_queries": len(gen_rep),
        "hard_v2_correct": hard_correct, "hard_v2_queries": len(hard_rows), "hard_guard_passed": hard_guard,
        "generated_net_gain": net_gain, "internal_same_family_gain": internal_gain,
        "internal_full_catalog_gain": full_gain, "target_90_percent_reached": generated_correct >= 116,
        "verdict": verdict, "reason": reason,
    }
    return verdict, reason, decision


def write_final_report(root: Path, run_dir: Path, selected: Mapping[str, Any], internal: Mapping[str, Any],
                       external: Mapping[str, Any], production_rows: Mapping[str, Sequence[Mapping[str, Any]]],
                       decision: Mapping[str, Any], inspection: Mapping[str, Any]) -> Path:
    metrics = external["metrics"]
    hard_name, gen_name, synth_name = "hard_near_duplicate_dev_v2", "generated_stress_dev_pilot32", "synthetic_dev"
    hard_n, gen_n = metrics[hard_name]["queries"], metrics[gen_name]["queries"]
    hard_rows = production_rows[hard_name]
    gen_rows = production_rows[gen_name]
    gen_hard_rows = [row for row in gen_rows if row["subset_role"] == "hard"]
    gen_rep_rows = [row for row in gen_rows if row["subset_role"] == "representative"]
    gen_hard_baseline = sum(row["frozen_production_top1"] == row["target_slug"] for row in gen_hard_rows)
    gen_rep_baseline = sum(row["frozen_production_top1"] == row["target_slug"] for row in gen_rep_rows)
    gen_hard_image = sum(row["adapted_image_top1"] == row["target_slug"] for row in gen_hard_rows)
    gen_rep_image = sum(row["adapted_image_top1"] == row["target_slug"] for row in gen_rep_rows)
    image_gen_trans = _top1_transitions(gen_rows, "adapted_image_top1")
    prod_gen_trans = _top1_transitions(gen_rows)
    gen_hard_trans = _top1_transitions(gen_hard_rows)
    gen_rep_trans = _top1_transitions(gen_rep_rows)
    image_hard_trans = _top1_transitions(hard_rows, "adapted_image_top1")
    prod_hard_trans = _top1_transitions(hard_rows)
    baseline_gen = int(metrics[gen_name]["frozen_production_correct"])
    baseline_hard = int(metrics[hard_name]["frozen_production_correct"])
    adapted_gen = int(metrics[gen_name]["adapted_production_correct"])
    adapted_hard = int(metrics[hard_name]["adapted_production_correct"])
    adapted_image_gen = int(metrics[gen_name]["adapted_image_only"]["top1_correct"])
    adapted_image_hard = int(metrics[hard_name]["adapted_image_only"]["top1_correct"])
    margin_rows = read_csv(run_dir / "embedding_margin_analysis.csv")
    ext_margin = {}
    for benchmark in (hard_name, gen_name):
        frozen = [float(row["frozen_margin"]) for row in margin_rows if row.get("benchmark") == benchmark and row.get("frozen_margin") not in (None, "")]
        adapted = [float(row["adapted_margin"]) for row in margin_rows if row.get("benchmark") == benchmark and row.get("adapted_margin") not in (None, "")]
        ext_margin[benchmark] = (statistics.fmean(frozen) if frozen else float("nan"), statistics.fmean(adapted) if adapted else float("nan"))
    oracle = json.loads((run_dir / "remaining_generated_error_oracle.json").read_text(encoding="utf-8"))
    internal_frozen, internal_adapted = internal["validation_frozen"], internal["validation"]
    epoch_minutes = sum(float(row["epoch_seconds"]) for row in selected["history"]) / 60.0
    latency_frozen, latency_adapted = external["latency"]
    synthetic = metrics[synth_name]
    train_count = len(read_csv(run_dir / "train_manifest.csv"))
    val_count = len(read_csv(run_dir / "validation_manifest.csv"))
    table = [
        "| Method | Hard Top-1 | Generated Top-1 | Generated correct /128 | Generated hard /64 | Representative /64 | R@5 (hard / generated) | Rescued / broken (generated) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Frozen production (SO400M + OCR + SIFT 0.40) | {baseline_hard}/{hard_n} ({baseline_hard/hard_n:.2%}) | {baseline_gen}/{gen_n} ({baseline_gen/gen_n:.2%}) | {baseline_gen}/128 | {gen_hard_baseline}/64 | {gen_rep_baseline}/64 | {metrics[hard_name]['frozen_image_only']['recall_at_5_correct']}/{hard_n} / {metrics[gen_name]['frozen_image_only']['recall_at_5_correct']}/{gen_n} | — |",
        f"| Adapted SO400M image-only | {adapted_image_hard}/{hard_n} ({adapted_image_hard/hard_n:.2%}) | {adapted_image_gen}/{gen_n} ({adapted_image_gen/gen_n:.2%}) | {adapted_image_gen}/128 | {gen_hard_image}/64 | {gen_rep_image}/64 | {metrics[hard_name]['adapted_image_only']['recall_at_5_correct']}/{hard_n} / {metrics[gen_name]['adapted_image_only']['recall_at_5_correct']}/{gen_n} | {image_gen_trans['rescued']} / {image_gen_trans['broken']} |",
        f"| Adapted SO400M + existing OCR + SIFT 0.40 | {adapted_hard}/{hard_n} ({adapted_hard/hard_n:.2%}) | {adapted_gen}/{gen_n} ({adapted_gen/gen_n:.2%}) | {adapted_gen}/128 | {decision['generated_hard_correct']}/64 | {decision['generated_representative_correct']}/64 | {metrics[hard_name]['adapted_image_only']['recall_at_5_correct']}/{hard_n} / {metrics[gen_name]['adapted_image_only']['recall_at_5_correct']}/{gen_n} | {prod_gen_trans['rescued']} / {prod_gen_trans['broken']} |",
    ]
    image_metrics_table = [
        "| Split | Frozen image Top-1 | Adapted image Top-1 | Adapted R@5 | Adapted R@10 | Adapted MRR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for benchmark in BENCHMARKS:
        before = metrics[benchmark]["frozen_image_only"]
        after = metrics[benchmark]["adapted_image_only"]
        image_metrics_table.append(
            f"| {benchmark} | {before['top1_correct']}/{before['queries']} ({before['top1']:.2%}) | "
            f"{after['top1_correct']}/{after['queries']} ({after['top1']:.2%}) | "
            f"{after['recall_at_5_correct']}/{after['queries']} | {after['recall_at_10_correct']}/{after['queries']} | {after['mrr']:.4f} |"
        )
    hard_family_table = [
        "| Hard-v2 family type | Frozen production | Adapted production | Net gain |",
        "|---|---:|---:|---:|",
    ]
    for family_type in sorted({str(row.get("family_type", "other")) for row in hard_rows}):
        family_rows = [row for row in hard_rows if str(row.get("family_type", "other")) == family_type]
        frozen_correct = sum(row["frozen_production_top1"] == row["target_slug"] for row in family_rows)
        adapted_correct = sum(row["adapted_production_top1"] == row["target_slug"] for row in family_rows)
        hard_family_table.append(
            f"| {family_type} | {frozen_correct}/{len(family_rows)} ({frozen_correct / len(family_rows):.2%}) | "
            f"{adapted_correct}/{len(family_rows)} ({adapted_correct / len(family_rows):.2%}) | "
            f"{adapted_correct - frozen_correct:+d} |"
        )
    second_table = [
        "| Model | Internal full catalog | Internal family accuracy | Internal family margin | Generated hard | External family margin (hard_v2 / generated) |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Frozen SO400M | {internal_frozen['full_catalog_top1']:.2%} | {internal_frozen['same_family_top1']:.2%} | {internal_frozen['same_family_margin_mean']:.4f} | {gen_hard_baseline}/64 | {ext_margin[hard_name][0]:.4f} / {ext_margin[gen_name][0]:.4f} |",
        f"| Selected LoRA SO400M | {internal_adapted['full_catalog_top1']:.2%} | {internal_adapted['same_family_top1']:.2%} | {internal_adapted['same_family_margin_mean']:.4f} | {decision['generated_hard_correct']}/64 | {ext_margin[hard_name][1]:.4f} / {ext_margin[gen_name][1]:.4f} |",
    ]
    exact_reproduction = ", ".join(
        f"{name} {metrics[name]['exact_frozen_image_top5_reproduction']}/{metrics[name]['queries']}"
        for name in BENCHMARKS
    )
    lines = [
        "# Partial SO400M hard-negative LoRA milestone", "",
        f"Run: `{run_dir.relative_to(root)}`  ",
        f"Selected checkpoint: `{selected['best_checkpoint']}`  ",
        f"Selected checkpoint SHA-256: `{selected['lora_sha256']}`  ",
        f"Internal-selection epoch: {selected['best_epoch']}",
        "",
        "## Benchmark results", "", *table, "",
        "Encoder-only retrieval (full-catalog ranking):", "", *image_metrics_table, "",
        "Hard-v2 production by family type:", "", *hard_family_table, "",
        "## Internal and external family margins", "", *second_table, "",
        "## Model, training data, and integrity", "",
        f"- Checkpoint `{BASE_MODEL_ID}` at revision `{BASE_REVISION}`; Transformers {inspection['transformers_version']}; vision blocks {inspection['vision_block_count']}, hidden size {inspection['hidden_size']}, 16 heads.",
        f"- Canonical external benchmark protocol: query_batch_size={CANONICAL_QUERY_BATCH_SIZE}; reference precompute batch_size={CONFIG['reference_embedding_batch_size']}; internal training-view batch_size={CONFIG['query_embedding_batch_size']}.",
        f"- LoRA: rank {CONFIG['rank']}, alpha {CONFIG['alpha']}, dropout {CONFIG['dropout']}; blocks 23–26, targets `{', '.join(LORA_TARGETS)}`; {inspection['trainable_parameter_count']:,} trainable parameters. Text tower trainable parameters: {inspection['text_trainable_parameter_count']}; all other checkpoint parameters frozen.",
        f"- Train/validation split: {CONFIG['train_views_per_sku']} / {CONFIG['validation_views_per_sku']} views per SKU; {train_count:,} / {val_count:,} rows.",
        "- Capture-v2 uses random placement and scale, perspective warps, procedural subdued backgrounds, exposure and color-temperature shifts, uneven lighting, mild glare and shadows, defocus/motion blur, down/up-sampling, sensor noise, and JPEG artifacts. Train/validation transform ranges differ. Local images outside benchmark/reference paths were catalog-mapping review candidates, not usable scene backgrounds. With no reliable bottle masks, the whole reference photo is placed as a card; the code does not claim bottle cutout compositing.",
        "- Negatives come only from same-family catalog metadata, frozen SO400M reference Top-20 neighbors, and seeded random catalog SKUs. Training never reads benchmark images, query IDs, target ranks, error pairs, or screenshots.",
        f"- Re-embedded frozen image Top-5 agreement: {exact_reproduction}. The exact baseline guard failed, so adapted external metrics are diagnostic and the milestone is marked INVALID. The SIFT reference cache fingerprint `{external['sift_cache_fingerprint']}` matched the unchanged catalog images.",
        f"- Adapted reference-cache fingerprint includes LoRA checkpoint SHA-256: `{selected['lora_sha256']}` and canonical query_batch_size={CANONICAL_QUERY_BATCH_SIZE}.",
        "- Post-freeze query embeddings and progress checkpoints are stored under `benchmarks/<split>/query_embeddings/`; fingerprints cover benchmark IDs, source image hashes, base revision, and selected LoRA SHA-256.",
        "",
        "## Internal generalization and cost", "",
        f"- Frozen → LoRA validation: candidate-set {internal_frozen['candidate_set_top1']:.2%} → {internal_adapted['candidate_set_top1']:.2%}; full catalog {internal_frozen['full_catalog_top1']:.2%} → {internal_adapted['full_catalog_top1']:.2%}; same-family {internal_frozen['same_family_top1']:.2%} → {internal_adapted['same_family_top1']:.2%}; family margin {internal_frozen['same_family_margin_mean']:.4f} → {internal_adapted['same_family_margin_mean']:.4f}.",
        f"- Selected train → validation candidate Top-1: {internal['train']['candidate_set_top1']:.2%} → {internal_adapted['candidate_set_top1']:.2%}; full-catalog Top-1: {internal['train']['full_catalog_top1']:.2%} → {internal_adapted['full_catalog_top1']:.2%}.",
        "- Per-bucket validation results (`clean-ish`, `perspective`, `background_scale`, `blur`, `glare`) are in `internal_validation_buckets.json`.",
        f"- Training: {selected['completed_epochs']} epochs, best epoch {selected['best_epoch']}; summed epoch duration {epoch_minutes:.1f} minutes; peak VRAM {selected['peak_train_vram_mb']:.0f} MB; microbatch {CONFIG['micro_batch_size']}, accumulation {CONFIG['gradient_accumulation_steps']}.",
        f"- Generated query latency: frozen encoder p95 {latency_frozen['encoder_p95_ms']:.1f} ms; adapted encoder p95 {latency_adapted['encoder_p95_ms']:.1f} ms; adapted OCR+SIFT full pipeline p95 {latency_adapted['full_pipeline_p95_ms']:.1f} ms; SLA <3 s: {latency_adapted['sla_3s_passed']}; peak inference VRAM {latency_adapted['peak_inference_vram_mb']:.0f} MB.",
        f"- Frozen generated error oracle: {oracle['frozen_generated_errors']} errors; adapted target similarity beats the frozen production incumbent in {oracle['adapted_target_beats_frozen_production_incumbent']} cases (Phase A metric-adapter result was 5/25).",
        f"- Synthetic post-selection sanity: frozen production {synthetic['frozen_production_correct']}/{synthetic['queries']} ({synthetic['frozen_production_correct']/synthetic['queries']:.2%}); adapted production {synthetic['adapted_production_correct']}/{synthetic['queries']} ({synthetic['adapted_production_top1']:.2%}); adapted image-only R@5 {synthetic['adapted_image_only']['recall_at_5_correct']}/{synthetic['queries']}.",
        "",
        "## Transitions and verdict", "",
        f"- Hard_v2 rescued/broken: adapted image-only {image_hard_trans['rescued']}/{image_hard_trans['broken']}; adapted production {prod_hard_trans['rescued']}/{prod_hard_trans['broken']}.",
        f"- Generated hard rescued/broken: {gen_hard_trans['rescued']}/{gen_hard_trans['broken']}; representative rescued/broken: {gen_rep_trans['rescued']}/{gen_rep_trans['broken']}.",
        f"- Generated production {decision['generated_production_correct']}/128; target ≥116/128: **{'YES' if decision['target_90_percent_reached'] else 'NO'}**. Hard_v2 guard ≥90.9%: **{'PASS' if decision['hard_guard_passed'] else 'FAIL'}**.",
        f"- **{decision['verdict']}** — {decision['reason']}",
        "- Scenario, representative/hard, vintage/subtype/other, and transition tables: `scenario_metrics.csv`, `family_metrics.csv`, `transition_summary.csv`.",
        "- Frozen and adapted full-catalog rank-index matrices are under `benchmarks/<split>/`; the corresponding `ranking_reference_slugs.json` maps each index to a catalog SKU.",
        f"- Resume from the latest completed epoch with `python scripts/run_so400m_hard_negative_lora.py --max-epochs {selected['planned_max_epochs']} --resume {run_dir.relative_to(root)}`.",
        "",
    ]
    path = root / "reports/so400m_hard_negative_lora_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    write_json(run_dir / "decision.json", dict(decision))
    return path


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
    integer_fields = {"view_index", "augmentation_seed"}
    for row in rows:
        for field in integer_fields:
            row[field] = int(row[field])
    return rows


def _load_saved_internal(run_dir: Path) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]] | None:
    metadata_path = run_dir / "adapted_reference_cache/metadata.json"
    embeddings_path = run_dir / "adapted_reference_cache/embeddings.npy"
    internal_path = run_dir / "internal_validation_metrics.json"
    if not (metadata_path.is_file() and embeddings_path.is_file() and internal_path.is_file()):
        return None
    embeddings = np.load(embeddings_path, mmap_mode="r")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    values = json.loads(internal_path.read_text(encoding="utf-8"))
    internal = {
        "train": values["train_adapted"],
        "validation": values["validation_adapted"],
        "validation_frozen": values["validation_frozen"],
    }
    return embeddings, internal, metadata


def run(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    effective_max_epochs = int(args.max_epochs)
    if args.resume:
        run_dir = (ROOT / args.resume).resolve() if not Path(args.resume).is_absolute() else Path(args.resume).resolve()
        if not run_dir.is_dir() or not (run_dir / "config.json").is_file():
            raise FileNotFoundError(f"not a resumable experiment directory: {run_dir}")
        saved_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if saved_config.get("training_config") != CONFIG:
            raise RuntimeError("resume run configuration differs from this runner")
        if int(saved_config.get("effective_max_epochs", CONFIG["max_epochs"])) != effective_max_epochs:
            raise RuntimeError("resume with the same --max-epochs value saved for this run")
        run_id = run_dir.name
    else:
        run_id = args.run_id or datetime.now(timezone.utc).strftime("so400m_hard_negative_lora_%Y%m%dT%H%M%SZ")
        run_relative = Path(run_id)
        if run_relative.is_absolute() or ".." in run_relative.parts or not run_relative.parts:
            raise ValueError("--run-id must be a safe relative path under artifacts/experiments")
        experiments_root = (ROOT / "artifacts/experiments").resolve()
        run_dir = (experiments_root / run_relative).resolve()
        if experiments_root not in run_dir.parents:
            raise ValueError("--run-id must stay inside artifacts/experiments")
        run_dir.mkdir(parents=True, exist_ok=False)
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"
    if args.resume and state_path.is_file():
        saved_state = json.loads(state_path.read_text(encoding="utf-8"))
        if saved_state.get("stage") == "complete" and (ROOT / "reports/so400m_hard_negative_lora_report.md").is_file():
            print(f"Run already complete: {ROOT / 'reports/so400m_hard_negative_lora_report.md'}", flush=True)
            return ROOT / "reports/so400m_hard_negative_lora_report.md", json.loads((run_dir / "milestone_summary.json").read_text(encoding="utf-8"))
        if saved_state.get("stage") == "internal_complete" and args.internal_only:
            selected = json.loads((run_dir / "selected_model.json").read_text(encoding="utf-8"))
            print(f"Internal-only run already frozen at {selected['best_checkpoint']}", flush=True)
            return run_dir / "selected_model.json", selected

    seed = int(CONFIG["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    catalog, slugs, base_references, base_meta, image_hashes = load_catalog(ROOT)
    family_keys = catalog_family_keys(catalog)
    family_types = infer_catalog_family_types(catalog, family_keys)
    if args.resume:
        train_rows = _load_manifest(run_dir / "train_manifest.csv")
        validation_rows = _load_manifest(run_dir / "validation_manifest.csv")
        saved_negative_rows = read_csv(run_dir / "hard_negative_map.csv")
        negative_map, expected_negative_rows = build_hard_negative_map(slugs, base_references, family_keys, seed,
            family_count=int(CONFIG["family_negative_count"]), visual_count=int(CONFIG["visual_neighbor_negative_count"]),
            random_count=int(CONFIG["random_negative_count"]), neighbor_pool=int(CONFIG["neighbor_pool"]))
        if len(saved_negative_rows) != len(expected_negative_rows):
            raise RuntimeError("resume negative map differs from the catalog-only deterministic map")
        for saved, expected in zip(saved_negative_rows, expected_negative_rows):
            if any(str(saved.get(key, "")) != str(expected.get(key, ""))
                   for key in ("anchor_slug", "negative_slug", "negative_rank", "negative_source", "catalog_family_key")):
                raise RuntimeError("resume negative map row differs from the deterministic catalog-only map")
            if abs(float(saved["frozen_reference_cosine"]) - float(expected["frozen_reference_cosine"])) > 1e-7:
                raise RuntimeError("resume negative-map cosine differs from the frozen catalog reference cache")
        negative_rows = expected_negative_rows
    elif args.reuse_manifest_from:
        source = Path(args.reuse_manifest_from)
        source = (ROOT / source).resolve() if not source.is_absolute() else source.resolve()
        source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        source_training_config = source_config.get("training_config", {})
        assert_rank_only_change(source_training_config, CONFIG)
        for filename in ("train_manifest.csv", "validation_manifest.csv", "hard_negative_map.csv"):
            if not (source / filename).is_file():
                raise FileNotFoundError(source / filename)
            shutil.copy2(source / filename, run_dir / filename)
        train_rows = _load_manifest(run_dir / "train_manifest.csv")
        validation_rows = _load_manifest(run_dir / "validation_manifest.csv")
        assert_same_manifests(
            _load_manifest(source / "train_manifest.csv"), train_rows,
            _load_manifest(source / "validation_manifest.csv"), validation_rows,
        )
        assert_catalog_only_manifest(train_rows)
        assert_catalog_only_manifest(validation_rows)
        assert_catalog_only_capture_manifest(train_rows)
        assert_catalog_only_capture_manifest(validation_rows)
        if any(row.get("augmentation_version") != "catalog_capture_v2" for row in train_rows + validation_rows):
            raise RuntimeError("capacity comparison requires the frozen r8 Capture-v2 manifests")
        saved_negative_rows = read_csv(run_dir / "hard_negative_map.csv")
        negative_map, negative_rows = build_hard_negative_map(slugs, base_references, family_keys, seed,
            family_count=int(CONFIG["family_negative_count"]), visual_count=int(CONFIG["visual_neighbor_negative_count"]),
            random_count=int(CONFIG["random_negative_count"]), neighbor_pool=int(CONFIG["neighbor_pool"]))
        if saved_negative_rows != negative_rows:
            if len(saved_negative_rows) != len(negative_rows):
                raise RuntimeError("reused hard-negative map has a different catalog-only candidate set")
            for saved, expected in zip(saved_negative_rows, negative_rows):
                if any(str(saved.get(key, "")) != str(expected.get(key, ""))
                       for key in ("anchor_slug", "negative_slug", "negative_rank", "negative_source", "catalog_family_key")):
                    raise RuntimeError("reused hard-negative map differs from the deterministic R8 control map")
                if abs(float(saved["frozen_reference_cosine"]) - float(expected["frozen_reference_cosine"])) > 1e-7:
                    raise RuntimeError("reused hard-negative map cosine differs from frozen SO400M")
    else:
        train_rows, validation_rows = build_capture_manifest(
            catalog, image_hashes, seed, int(CONFIG["train_views_per_sku"]), int(CONFIG["validation_views_per_sku"])
        )
        assert_catalog_only_manifest(train_rows)
        assert_catalog_only_manifest(validation_rows)
        negative_map, negative_rows = build_hard_negative_map(
            slugs, base_references, family_keys, seed,
            family_count=int(CONFIG["family_negative_count"]), visual_count=int(CONFIG["visual_neighbor_negative_count"]),
            random_count=int(CONFIG["random_negative_count"]), neighbor_pool=int(CONFIG["neighbor_pool"]),
        )
        write_manifest(run_dir / "train_manifest.csv", train_rows)
        write_manifest(run_dir / "validation_manifest.csv", validation_rows)
        write_csv(run_dir / "hard_negative_map.csv", negative_rows)

    manifest_fingerprint = sha256_json({"train": train_rows, "validation": validation_rows})
    negative_fingerprint = sha256_json(negative_rows)
    config_fingerprint = sha256_json({
        "training_config": CONFIG,
        "catalog_reference_fingerprint": base_meta["fingerprint"],
        "catalog_image_hashes": sorted(image_hashes.items()),
        "view_manifest_fingerprint": manifest_fingerprint,
        "hard_negative_map_fingerprint": negative_fingerprint,
        "effective_max_epochs": effective_max_epochs,
    })
    if args.resume:
        previous_dataset = json.loads((run_dir / "dataset_identity.json").read_text(encoding="utf-8"))
        if previous_dataset.get("config_fingerprint") != config_fingerprint:
            raise RuntimeError("resume dataset/reference/negative-map fingerprint mismatch")
    else:
        write_json(run_dir / "config.json", {
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_config": CONFIG,
            "canonical_benchmark_protocol": CANONICAL_EXTERNAL_PROTOCOL,
            "effective_max_epochs": effective_max_epochs,
            "scope": f"one rank-{CONFIG['rank']} LoRA config in vision blocks 23-26 only",
            "candidate_generation": "adapted SO400M full-catalog retrieval; after freeze only",
            "production_policy": "adapted SO400M -> Top-5 -> unchanged existing OCR -> unchanged SIFT w=0.40",
            "benchmark_selection_allowed": False,
            "benchmark_error_mining": False,
            "text_encoder_adapted": False,
            "raw_data_modified": False,
        })
        write_json(run_dir / "augmentation_config.json", {
            "version": "catalog_capture_v2",
            "background_source": "procedural backgrounds; project images outside benchmark/reference paths were catalog-mapping review candidates rather than scene backgrounds",
            "object_mask_available": False,
            "compositing_policy": "place the whole catalog reference photo as a perspective-warped inset card; no fake bottle segmentation",
            "train_ranges": {"clean-ish": [0.82, 1.00], "perspective_scale": [0.66, 0.92], "background_scale": [0.50, 0.76], "blur_scale": [0.63, 0.88], "glare_scale": [0.69, 0.94]},
            "validation_ranges": {"clean-ish": [0.78, 0.98], "perspective_scale": [0.61, 0.88], "background_scale": [0.46, 0.71], "blur_scale": [0.59, 0.84], "glare_scale": [0.65, 0.91]},
            "effects": ["random placement", "increased background framing", "partial edge crop", "mild-to-strong perspective", "brightness/contrast/color temperature", "uneven illumination", "soft local highlight/glare", "card shadow", "defocus/motion blur", "down-up sampling", "sensor-like noise", "JPEG compression"],
            "bucket_distribution": list(CAPTURE_BUCKETS),
        })
        write_json(run_dir / "dataset_identity.json", {
            "usable_catalog_count": len(catalog),
            "reference_slugs_sha256": sha256_json(slugs),
            "catalog_manifest_sha256": sha256_file(ROOT / "data/processed/catalog_manifest.csv"),
            "catalog_reference_sha256": image_hashes,
            "base_reference_cache_fingerprint": base_meta["fingerprint"],
            "view_manifest_fingerprint": manifest_fingerprint,
            "hard_negative_map_fingerprint": negative_fingerprint,
            "config_fingerprint": config_fingerprint,
            "training_paths_catalog_only": all(str(row["reference_image_path"]).startswith("data/processed/reference_images/") for row in [*train_rows, *validation_rows]),
            "benchmark_query_count_in_training": 0,
        })
        write_json(run_dir / "reference_slugs.json", list(slugs))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("controlled SO400M LoRA milestone requires the configured CUDA GPU")
    model, processor, lora_modules, checkpointing_enabled = load_model(device)
    inspection = inspect_loaded_model(model, processor, lora_modules, checkpointing_enabled)
    write_json(run_dir / "model_inspection.json", inspection)
    if inspection["total_parameters"] != 1_136_008_498 or inspection["vision_parameters"] != 428_225_600:
        raise RuntimeError("loaded checkpoint parameter count differs from the inspected base model")
    expected_trainable = 294_912 * int(CONFIG["rank"]) // 8
    if inspection["trainable_parameter_count"] != expected_trainable or len(lora_modules) != 16:
        raise RuntimeError(f"LoRA parameter/module count differs from rank-{CONFIG['rank']} 4-block Q/K/V/out scope")
    if inspection["text_trainable_parameter_count"] != 0 or not inspection["base_checkpoint_parameters_frozen"]:
        raise RuntimeError("a non-LoRA parameter became trainable")
    if not args.resume:
        gpu_total = torch.cuda.get_device_properties(device).total_memory
        write_json(run_dir / "environment.json", {
            "python": sys.version,
            "platform": __import__("platform").platform(),
            "torch": torch.__version__,
            "transformers": inspection["transformers_version"],
            "numpy": np.__version__,
            "opencv": __import__("cv2").__version__,
            "device": torch.cuda.get_device_name(device),
            "device_memory_bytes": int(gpu_total),
            "device_total_memory_mb": gpu_total / (1024**2),
            "base_weight_dtype": inspection["vision_dtype"],
            "revision": BASE_REVISION,
            "gradient_checkpointing_enabled": checkpointing_enabled,
            "canonical_external_query_batch_size": CANONICAL_QUERY_BATCH_SIZE,
            "training_view_batch_size": int(CONFIG["query_embedding_batch_size"]),
            "reference_precompute_batch_size": int(CONFIG["reference_embedding_batch_size"]),
        })

    pretrain_path = run_dir / "baseline_reproduction_pretraining.json"
    if not pretrain_path.is_file():
        baseline_artifact = ROOT / CANONICAL_EXTERNAL_PROTOCOL["baseline_artifact"]
        baseline_payload = json.loads(baseline_artifact.read_text(encoding="utf-8"))
        baseline_state = json.loads((baseline_artifact.parent / "state.json").read_text(encoding="utf-8"))
        if baseline_state.get("baseline_valid") is not True:
            raise RuntimeError("STOP: exact canonical query baseline is not recorded as valid")
        baseline_rows = baseline_payload["after_single_query_fix"]
        require_exact_baseline_reproduction(baseline_rows, {
            "hard_near_duplicate_dev_v2": 1150,
            "generated_stress_dev_pilot32": 128,
            "synthetic_dev": 4084,
        })
        # Only aggregate Top-5 reproduction counts are admitted into a training
        # run. Benchmark images, IDs, targets, and per-query labels stay closed.
        baseline_check = {
            "source": CANONICAL_EXTERNAL_PROTOCOL["baseline_artifact"],
            "query_batch_size": CANONICAL_QUERY_BATCH_SIZE,
            "exact_frozen_top5_only": {
                name: {
                    "queries": int(row["queries"]),
                    "top5_exact_order": int(row["top5_exact_order"]),
                    "max_score_abs_diff_on_historical_top5": float(row["max_score_abs_diff_on_historical_top5"]),
                }
                for name, row in baseline_rows.items()
            },
            "benchmark_images_or_labels_opened_by_training": False,
            "benchmark_based_model_selection": False,
        }
        write_json(pretrain_path, baseline_check)
        training_access = {
            "catalog_references_only": True,
            "benchmark_images_opened": False,
            "benchmark_labels_opened": False,
            "benchmark_error_mining": False,
            "selection_source": "internal validation manifests only",
            "baseline_guard_source": CANONICAL_EXTERNAL_PROTOCOL["baseline_artifact"],
        }
        assert_training_data_access_scope(training_access)
        write_json(run_dir / "training_data_access.json", training_access)
        print("[baseline] loaded exact batch-1 reproduction summary; benchmark labels remain closed", flush=True)

    selected_path = run_dir / "selected_model.json"
    if selected_path.is_file():
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        best_checkpoint = torch.load(run_dir / "checkpoints/best.pt", map_location="cpu", weights_only=False)
        load_lora_state_dict(model, best_checkpoint["lora_state"])
        if lora_sha256(model) != selected["lora_sha256"]:
            raise RuntimeError("frozen selected checkpoint SHA-256 mismatch")
        training = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
        # These report fields are runtime metadata and are intentionally kept
        # in training_summary.json rather than the immutable model-selection file.
        selected["history"] = training["history"]
        selected["peak_train_vram_mb"] = training["peak_train_vram_mb"]
        write_state(state_path, "checkpoint_frozen", selected_epoch=selected["best_epoch"],
                    lora_sha256=selected["lora_sha256"], resumed_after_freeze=True)
    else:
        frozen_train_path = precompute_frozen_views(
            ROOT, run_dir, "train", train_rows, model, processor, device,
            int(CONFIG["query_embedding_batch_size"]), manifest_fingerprint,
        )
        frozen_validation_path = precompute_frozen_views(
            ROOT, run_dir, "validation", validation_rows, model, processor, device,
            int(CONFIG["query_embedding_batch_size"]), manifest_fingerprint,
        )
        training = train_one_config(
            ROOT, run_dir, catalog, slugs, image_hashes, base_references, family_keys,
            family_types, train_rows, validation_rows, negative_map, model, processor,
            device, frozen_train_path, frozen_validation_path, config_fingerprint,
            effective_max_epochs, bool(args.resume),
        )
        selected = {
            "selected_by": "internal validation same-family Top-1; full-catalog >= frozen - 1pp guard",
            "benchmark_data_used_for_selection": False,
            "synthetic_used_for_selection": False,
            "best_epoch": training["best_epoch"],
            "completed_epochs": len(training["history"]),
            "planned_max_epochs": effective_max_epochs,
            "best_checkpoint": str((run_dir / "checkpoints/best.pt").relative_to(ROOT)),
            "lora_sha256": training["lora_sha256"],
            "best_validation_metrics": training["best_validation_metrics"],
            "frozen_validation_metrics": training["frozen_validation_metrics"],
            "config_fingerprint": config_fingerprint,
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(run_dir / "training_summary.json", training)
        write_json(selected_path, selected)
        write_state(state_path, "checkpoint_frozen", selected_epoch=selected["best_epoch"],
                    lora_sha256=selected["lora_sha256"], benchmark_evaluation_started=False)
        print(f"[freeze] selected epoch {selected['best_epoch']} SHA-256 {selected['lora_sha256']}", flush=True)

    selected["canonical_query_batch_size"] = CANONICAL_QUERY_BATCH_SIZE
    selected["canonical_model_identity"] = make_encoder_fingerprint(
        base_model_id=BASE_MODEL_ID,
        base_revision=BASE_REVISION,
        checkpoint_sha256=selected["lora_sha256"],
        preprocessing=inspection["processor"],
        dtype=inspection["vision_dtype"],
        code_sha256=sha256_file(Path(__file__).resolve()),
        lora_rank=int(CONFIG["rank"]),
        lora_alpha=int(CONFIG["alpha"]),
        target_modules=LORA_TARGETS,
        last_vision_blocks=int(CONFIG["last_vision_blocks"]),
        query_batch_size=CANONICAL_QUERY_BATCH_SIZE,
    )
    write_json(selected_path, selected)
    write_json(run_dir / "selected_policy.json", {
        "checkpoint_sha256": selected["lora_sha256"],
        "canonical_query_batch_size": CANONICAL_QUERY_BATCH_SIZE,
        "canonical_benchmark_protocol": CANONICAL_EXTERNAL_PROTOCOL,
        "model_fingerprint": selected["canonical_model_identity"],
        "reference_precompute_batch_size": int(CONFIG["reference_embedding_batch_size"]),
        "training_view_batch_size": int(CONFIG["query_embedding_batch_size"]),
        "selected_by": selected.get("selected_by", "internal validation same-family Top-1"),
        "benchmark_data_used_for_selection": False,
        "frozen": True,
    })

    saved_internal = _load_saved_internal(run_dir)
    if saved_internal is None:
        adapted_references, internal = evaluate_internal_final(
            ROOT, run_dir, model, processor, device, catalog, slugs, image_hashes,
            base_references, train_rows, validation_rows, negative_map, family_types, selected,
        )
        selected_cache = json.loads((run_dir / "adapted_reference_cache/metadata.json").read_text(encoding="utf-8"))
    else:
        adapted_references, internal, selected_cache = saved_internal
        if selected_cache.get("lora_checkpoint_sha256") != selected["lora_sha256"]:
            raise RuntimeError("saved adapted reference cache has a different LoRA checkpoint fingerprint")
    if args.internal_only:
        internal_summary = {
            "run_id": run_dir.name,
            "run_dir": str(run_dir.relative_to(ROOT)),
            "selected_model": selected,
            "internal": internal,
            "external_evaluation_started": False,
        }
        write_json(run_dir / "internal_only_summary.json", internal_summary)
        write_state(state_path, "internal_complete", selected_epoch=selected["best_epoch"],
                    lora_sha256=selected["lora_sha256"], benchmark_evaluation_started=False)
        print(json.dumps(internal_summary, ensure_ascii=False, indent=2), flush=True)
        return run_dir / "selected_model.json", internal_summary
    write_state(state_path, "external_evaluation_started", selected_epoch=selected["best_epoch"],
                lora_sha256=selected["lora_sha256"])
    external = run_external_benchmarks(
        ROOT, run_dir, catalog, slugs, base_references, adapted_references,
        model, processor, device, selected, selected_cache,
    )
    verdict, reason, decision = make_final_verdict(internal, external, external["production_predictions"])
    if not external["base_encoder_query_top5_reproduced"]:
        reproduction = {
            name: {
                "exact_top5": external["metrics"][name]["exact_frozen_image_top5_reproduction"],
                "queries": external["metrics"][name]["queries"],
            }
            for name in BENCHMARKS
        }
        reason = (
            "Frozen SO400M query embeddings did not exactly reproduce the stored Top-5 baseline: "
            + ", ".join(f"{name} {row['exact_top5']}/{row['queries']}" for name, row in reproduction.items())
            + ". Adapted metrics are diagnostic and do not qualify as a valid milestone result."
        )
        verdict = "E. INVALID / LEAKAGE / TRAINING FAILURE"
        decision.update({
            "verdict": verdict,
            "reason": reason,
            "baseline_guard_passed": False,
            "baseline_query_top5_reproduction": reproduction,
        })
    else:
        decision.update({"baseline_guard_passed": True})
    report_path = write_final_report(ROOT, run_dir, selected, internal, external,
                                     external["production_predictions"], decision, inspection)
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "selected_model": selected,
        "internal": internal,
        "external": external["metrics"],
        "latency": external["latency"],
        "decision": decision,
        "report": str(report_path.relative_to(ROOT)),
    }
    write_json(run_dir / "milestone_summary.json", summary)
    write_state(state_path, "complete", verdict=verdict, report=str(report_path.relative_to(ROOT)))
    print(json.dumps({"report": str(report_path), "verdict": verdict, "decision": decision}, ensure_ascii=False, indent=2), flush=True)
    return report_path, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", metavar="RUN_DIR", help="resume a run from its latest completed epoch/checkpoint")
    parser.add_argument("--max-epochs", type=int, default=int(CONFIG["max_epochs"]),
                        help=f"cap epochs (predeclared config maximum is {CONFIG['max_epochs']})")
    parser.add_argument("--rank", type=int, choices=(8, 16), default=8,
                        help="capacity variant; all other training settings remain fixed")
    parser.add_argument("--alpha", type=int, help="LoRA alpha (must equal 2*rank for this controlled comparison)")
    parser.add_argument("--run-id", help="new experiment subdirectory path under artifacts/experiments")
    parser.add_argument("--reuse-manifest-from", help="copy exact train/validation/negative manifests from a frozen control run")
    parser.add_argument("--internal-only", action="store_true",
                        help="freeze and report internal validation without opening external benchmark inputs")
    args = parser.parse_args()
    CONFIG["rank"] = int(args.rank)
    CONFIG["alpha"] = int(args.alpha if args.alpha is not None else args.rank * 2)
    if CONFIG["alpha"] != 2 * CONFIG["rank"]:
        parser.error("controlled rank comparison requires alpha=2*rank (r8/16, r16/32)")
    if not 1 <= args.max_epochs <= int(CONFIG["max_epochs"]):
        parser.error(f"--max-epochs must be between 1 and {CONFIG['max_epochs']}")
    run(args)


if __name__ == "__main__":
    main()
