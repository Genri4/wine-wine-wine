#!/usr/bin/env python3
"""Train and evaluate a small catalog-only residual metric adapter.

Training inputs are exclusively catalog reference images and deterministic
views generated from them. The benchmark loaders are called only after the
internal-validation checkpoint has been frozen and its metadata is written.
The runner checkpoints every view-embedding batch and every training epoch.
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
sys.path.insert(0, str(ROOT / "scripts"))

from recognition.encoder_adapters import SigLIP2So400m384Adapter  # noqa: E402
from recognition.geometric_reranker import fuse_scores, rank_candidates, validate_top5_candidates  # noqa: E402
from recognition.local_visual_reranker import (  # noqa: E402
    fuse_local_signals,
    validate_normalized_embeddings,
)
from recognition.metric_adapter import (  # noqa: E402
    ResidualMetricAdapter,
    adapter_state_fingerprint,
    assert_training_manifest_catalog_only,
    build_hard_negative_map,
    build_view_manifest,
    candidate_index_matrix,
    catalog_family_keys,
    make_catalog_view,
    rank_fixed_top5,
    reference_cache_fingerprint,
    transition_name,
)

BASE_CACHE = Path("artifacts/reference_embeddings/siglip2_so400m_384")
SOURCE_RUN = Path("artifacts/experiments/so400m_ocr_reranker_20260920T193925Z")
SIFT_RUN = Path("artifacts/experiments/final_ml_geometric_reranker_20260923T082259Z")
STRONG_RUN = Path("artifacts/experiments/strong_local_visual_reranker_final_20260923T1000Z")
PRIMARY_BENCHMARKS = ("hard_near_duplicate_dev_v2", "generated_stress_dev_pilot32")
SYNTHETIC_BENCHMARK = "synthetic_dev"
FUSION_WEIGHTS = (0.10, 0.20, 0.30, 0.40)
TEMPERATURE = 0.07
BASE_REVISION = "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"
PREPROCESSING = {
    "class": "SiglipImageProcessor",
    "do_center_crop": "None",
    "image_mean": "(0.5, 0.5, 0.5)",
    "image_std": "(0.5, 0.5, 0.5)",
    "resample": "2",
    "size": "SizeDict(height=384, width=384, longest_edge=None, shortest_edge=None, max_height=None, max_width=None, min_pixels=None, max_pixels=None)",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    if rows:
        fields = list(rows[0])
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    else:
        tmp.write_text("", encoding="utf-8")
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


def npy_has_shape(path: Path, shape: tuple[int, ...], dtype: np.dtype) -> bool:
    if not path.exists():
        return False
    try:
        array = np.load(path, mmap_mode="r")
        valid = array.shape == shape and array.dtype == np.dtype(dtype)
        del array
        return valid
    except (OSError, ValueError):
        return False


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_state(path: Path, stage: str, **details: Any) -> None:
    write_json(path, {"stage": stage, "updated_at_utc": datetime.now(timezone.utc).isoformat(), **details})


def load_catalog_and_reference_cache(root: Path) -> tuple[list[dict[str, str]], list[str], np.ndarray, dict, dict[str, str]]:
    cache_dir = root / BASE_CACHE
    meta = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    payload = torch.load(cache_dir / "embeddings.pt", map_location="cpu", weights_only=True)
    slugs = json.loads((cache_dir / "slugs.json").read_text(encoding="utf-8"))
    embeddings = payload["embeddings"].float().numpy()
    if payload.get("fingerprint") != meta.get("fingerprint"):
        raise RuntimeError("Frozen SO400M reference cache fingerprint mismatch")
    if meta.get("checkpoint_revision") != BASE_REVISION or meta.get("preprocessing_config") != PREPROCESSING:
        raise RuntimeError("Frozen SO400M revision or preprocessing differs from the recorded baseline")
    validate_normalized_embeddings(embeddings)
    manifest_rows = read_csv(root / "data/processed/catalog_manifest.csv")
    by_slug = {row["slug"]: row for row in manifest_rows if row.get("reference_image_path")}
    if set(slugs) != set(by_slug) or len(slugs) != int(meta["reference_count"]):
        raise RuntimeError("Reference cache and usable catalog images do not align exactly")
    catalog = [by_slug[slug] for slug in slugs]
    image_hashes = {}
    for row in catalog:
        image_path = root / row["reference_image_path"]
        if not image_path.is_file():
            raise FileNotFoundError(f"catalog reference missing: {row['reference_image_path']}")
        image_hashes[row["slug"]] = sha256_file(image_path)
    return catalog, slugs, embeddings, meta, image_hashes


def verify_frozen_sift_baseline(root: Path) -> dict[str, Any]:
    """Reconstruct the selected OCR+SIFT Top-5 orders before training starts."""
    import run_final_ml_sanity_check as frozen_runner
    import run_strong_local_visual_reranker as strong_runner

    current_result = frozen_runner.verify_frozen_current_pipeline(root)
    benchmark_data = strong_runner.load_benchmark_inputs(root)
    results = {}
    for benchmark, queries in benchmark_data.items():
        exact = 0
        top1_exact = 0
        for query_id, query in queries.items():
            geometry = []
            for slug in query["candidate_slugs"]:
                row = query["old_geometry"].get((query_id, slug))
                if row is None:
                    raise RuntimeError(f"missing frozen SIFT pair {benchmark}/{query_id}/{slug}")
                valid = str(row["homography_valid"]).lower() == "true"
                geometry.append(float(row["geometric_score"]) if valid else 0.0)
            scores = fuse_scores(query["current_scores"], geometry, 0.40)
            ranking = rank_candidates(query["candidate_slugs"], scores)
            expected = query["sift_selected_ranking"]
            exact += ranking == expected
            top1_exact += ranking[0] == query["sift_selected_top1"]
        results[benchmark] = {
            "queries": len(queries),
            "exact_top5_order": exact,
            "exact_top5_order_agreement": exact / max(1, len(queries)),
            "exact_top1": top1_exact,
            "reproduced": exact == len(queries) and top1_exact == len(queries),
        }
        if exact != len(queries) or top1_exact != len(queries):
            raise RuntimeError(f"STOP: current+SIFT baseline mismatch on {benchmark}: {exact}/{len(queries)}")
    return {
        "current_pipeline": current_result,
        "current_plus_sift_w0.40": results,
        "training_started_after_exact_reproduction": True,
    }


def make_run_fingerprint(config: dict[str, Any], manifest_sha: str, image_hashes: Mapping[str, str]) -> str:
    payload = {
        "config": config,
        "catalog_manifest_sha256": manifest_sha,
        "catalog_reference_sha256": sorted(image_hashes.items()),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def precompute_catalog_view_split(
    root: Path,
    run_dir: Path,
    split: str,
    rows: Sequence[Mapping[str, Any]],
    encoder: SigLIP2So400m384Adapter,
    run_fingerprint: str,
    *,
    batch_size: int,
) -> Path:
    embedding_path = run_dir / f"{split}_view_embeddings.npy"
    progress_path = run_dir / f"{split}_view_precompute_progress.json"
    expected_shape = (len(rows), 1152)
    if not npy_has_shape(embedding_path, expected_shape, np.float16):
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        np.lib.format.open_memmap(embedding_path, mode="w+", dtype=np.float16, shape=expected_shape).flush()
        progress_path.unlink(missing_ok=True)
    done = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("run_fingerprint") == run_fingerprint and progress.get("row_count") == len(rows):
            done = int(progress.get("completed_rows", 0))
            if not 0 <= done <= len(rows):
                raise RuntimeError(f"invalid {split} embedding progress checkpoint")
    matrix = np.lib.format.open_memmap(embedding_path, mode="r+")
    for start in range(done, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        views = []
        for row in batch_rows:
            with Image.open(root / str(row["reference_image_path"])) as image:
                views.append(make_catalog_view(image, int(row["augmentation_seed"])))
        vectors = np.asarray(encoder.encode_images(views), dtype=np.float32)
        validate_normalized_embeddings(vectors)
        if vectors.shape != (len(batch_rows), 1152):
            raise RuntimeError(f"unexpected {split} embedding batch shape {vectors.shape}")
        matrix[start:start + len(batch_rows)] = vectors.astype(np.float16)
        matrix.flush()
        completed = start + len(batch_rows)
        write_json(progress_path, {
            "run_fingerprint": run_fingerprint,
            "split": split,
            "row_count": len(rows),
            "completed_rows": completed,
            "last_completed_batch_start": start,
            "last_completed_batch_size": len(batch_rows),
            "embedding_file": embedding_path.name,
        })
        if completed % 512 < batch_size or completed == len(rows):
            print(f"[{split} views] {completed}/{len(rows)}", flush=True)
    del matrix
    return embedding_path


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _encode_adapter_batch(adapter, query_np: np.ndarray, refs: torch.Tensor, candidate_ids: np.ndarray, device: torch.device):
    query = torch.as_tensor(np.asarray(query_np, dtype=np.float32), device=device)
    query = adapter(query)
    candidate_tensor = torch.as_tensor(candidate_ids, device=device, dtype=torch.long)
    candidates = adapter(refs[candidate_tensor.reshape(-1)]).reshape(candidate_tensor.shape[0], candidate_tensor.shape[1], -1)
    return query, candidates


@torch.inference_mode()
def evaluate_view_split(
    adapter: ResidualMetricAdapter,
    embedding_path: Path,
    anchors: int,
    views_per_anchor: int,
    reference_embeddings: torch.Tensor,
    candidate_ids: np.ndarray,
    family_mask: np.ndarray,
    nearest20: np.ndarray,
    *,
    temperature: float,
    batch_size: int = 256,
) -> dict[str, float | int | None]:
    device = reference_embeddings.device
    adapter.eval()
    views = np.load(embedding_path, mmap_mode="r")
    adapted_refs = adapter(reference_embeddings)
    total = len(views)
    candidate_correct = full_correct = nn_correct = 0
    candidate_loss = 0.0
    family_correct = family_count = 0
    for start in range(0, total, batch_size):
        stop = min(total, start + batch_size)
        anchor_ids = np.arange(start, stop, dtype=np.int64) // views_per_anchor
        q, candidates = _encode_adapter_batch(
            adapter, views[start:stop], reference_embeddings, candidate_ids[anchor_ids], device
        )
        logits = (q[:, None, :] * candidates).sum(dim=-1) / temperature
        labels = torch.zeros(stop - start, dtype=torch.long, device=device)
        candidate_loss += float(F.cross_entropy(logits.float(), labels, reduction="sum").item())
        candidate_correct += int((logits.argmax(dim=1) == 0).sum().item())
        family = torch.as_tensor(family_mask[anchor_ids], dtype=torch.bool, device=device)
        family_logits = logits.masked_fill(~family, -torch.inf)
        family_rows = family.any(dim=1) & family[:, 1:].any(dim=1)
        if bool(family_rows.any()):
            family_correct += int((family_logits.argmax(dim=1)[family_rows] == 0).sum().item())
            family_count += int(family_rows.sum().item())
        full_logits = q @ adapted_refs.T
        full_correct += int((full_logits.argmax(dim=1).cpu().numpy() == anchor_ids).sum())
        nn_ids = np.column_stack((anchor_ids, nearest20[anchor_ids]))
        nn_tensor = torch.as_tensor(nn_ids, dtype=torch.long, device=device)
        nn_logits = (q[:, None, :] * adapted_refs[nn_tensor]).sum(dim=-1)
        nn_correct += int((nn_logits.argmax(dim=1) == 0).sum().item())
    del views
    return {
        "views": total,
        "candidate_top1_correct": candidate_correct,
        "candidate_top1_accuracy": candidate_correct / max(1, total),
        "candidate_cross_entropy": candidate_loss / max(1, total),
        "full_catalog_top1_correct": full_correct,
        "full_catalog_top1_accuracy": full_correct / max(1, total),
        "same_family_top1_correct": family_correct,
        "same_family_views": family_count,
        "same_family_top1_accuracy": family_correct / family_count if family_count else None,
        "reference_nearest20_top1_correct": nn_correct,
        "reference_nearest20_top1_accuracy": nn_correct / max(1, total),
    }


@torch.inference_mode()
def evaluate_candidate_accuracy(
    adapter: ResidualMetricAdapter,
    embedding_path: Path,
    anchors: int,
    views_per_anchor: int,
    reference_embeddings: torch.Tensor,
    candidate_ids: np.ndarray,
    *,
    temperature: float,
    batch_size: int = 256,
) -> dict[str, float | int]:
    device = reference_embeddings.device
    adapter.eval()
    views = np.load(embedding_path, mmap_mode="r")
    total, correct, loss_sum = len(views), 0, 0.0
    for start in range(0, total, batch_size):
        stop = min(total, start + batch_size)
        anchor_ids = np.arange(start, stop, dtype=np.int64) // views_per_anchor
        q, candidates = _encode_adapter_batch(adapter, views[start:stop], reference_embeddings,
                                               candidate_ids[anchor_ids], device)
        logits = (q[:, None, :] * candidates).sum(dim=-1) / temperature
        labels = torch.zeros(stop - start, dtype=torch.long, device=device)
        correct += int((logits.argmax(dim=1) == 0).sum().item())
        loss_sum += float(F.cross_entropy(logits.float(), labels, reduction="sum").item())
    del views
    return {"views": total, "candidate_top1_correct": correct,
            "candidate_top1_accuracy": correct / max(1, total),
            "candidate_cross_entropy": loss_sum / max(1, total)}


def _save_epoch_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    payload = {**payload, "rng_state": _rng_state()}
    atomic_torch_save(path, payload)


def train_adapter(
    run_dir: Path,
    device: torch.device,
    ref_np: np.ndarray,
    train_embedding_path: Path,
    validation_embedding_path: Path,
    candidate_ids: np.ndarray,
    family_mask: np.ndarray,
    nearest20: np.ndarray,
    *,
    train_views_per_sku: int,
    validation_views_per_sku: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    seed: int,
) -> tuple[ResidualMetricAdapter, dict[str, Any]]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    ref = torch.as_tensor(ref_np, dtype=torch.float32, device=device)
    adapter = ResidualMetricAdapter(input_dim=ref.shape[1], hidden_dim=256, residual_scale=0.1).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path, best_path = checkpoint_dir / "latest.pt", checkpoint_dir / "best.pt"
    train_views = np.load(train_embedding_path, mmap_mode="r")
    train_count = len(train_views)
    latest = None
    history = read_csv(run_dir / "training_history.csv") if (run_dir / "training_history.csv").exists() else []
    if latest_path.exists():
        latest = torch.load(latest_path, map_location=device, weights_only=False)
        adapter.load_state_dict(latest["adapter_state"])
        optimizer.load_state_dict(latest["optimizer_state"])
        scaler.load_state_dict(latest.get("scaler_state", {}))
        if latest.get("rng_state"):
            _restore_rng_state(latest["rng_state"])
        start_epoch = int(latest["epoch"]) + 1
        best = torch.load(best_path, map_location="cpu", weights_only=False) if best_path.exists() else None
        if not any(int(row.get("epoch", -1)) == int(latest["epoch"]) for row in history):
            train_metrics = latest["train_metrics"]
            val_metrics = latest["validation_metrics"]
            history.append({"epoch": latest["epoch"],
                            **{f"train_{key}": value for key, value in train_metrics.items()},
                            "training_loss_online": latest.get("training_loss_online"),
                            **{f"validation_{key}": value for key, value in val_metrics.items()},
                            "selected_best": int(latest.get("best_epoch", -1)) == int(latest["epoch"])})
            history.sort(key=lambda row: int(row["epoch"]))
            write_csv(run_dir / "training_history.csv", history)
        print(f"[resume] latest completed epoch={latest['epoch']}; next={start_epoch}", flush=True)
    else:
        start_epoch = 1
        train0 = evaluate_candidate_accuracy(adapter, train_embedding_path, len(ref), train_views_per_sku,
                                             ref, candidate_ids, temperature=temperature)
        val0 = evaluate_view_split(adapter, validation_embedding_path, len(ref), validation_views_per_sku,
                                   ref, candidate_ids, family_mask, nearest20, temperature=temperature)
        initial = {"epoch": 0, "train_metrics": train0, "validation_metrics": val0,
                   "adapter_state": adapter.state_dict(), "optimizer_state": optimizer.state_dict(),
                   "scaler_state": scaler.state_dict(), "best_epoch": 0, "epochs_without_improvement": 0}
        _save_epoch_checkpoint(checkpoint_dir / "epoch_000.pt", initial)
        _save_epoch_checkpoint(best_path, initial)
        _save_epoch_checkpoint(latest_path, initial)
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        history = [{"epoch": 0, **{f"train_{k}": v for k, v in train0.items()},
                    **{f"validation_{k}": v for k, v in val0.items()}, "selected_best": True}]
        write_csv(run_dir / "training_history.csv", history)
        write_json(run_dir / "validation_metrics.json", {"baseline_identity_adapter": val0})
    best_accuracy = float(best["validation_metrics"]["candidate_top1_accuracy"]) if best else -1.0
    best_loss = float(best["validation_metrics"]["candidate_cross_entropy"]) if best else float("inf")
    best_epoch = int(best["epoch"]) if best else 0
    epochs_without_improvement = int(latest.get("epochs_without_improvement", 0)) if latest else 0
    use_amp = device.type == "cuda"

    for epoch in range(start_epoch, max_epochs + 1):
        adapter.train()
        permutation = torch.randperm(train_count).numpy()
        train_loss_sum = 0.0
        seen = 0
        for offset in range(0, train_count, batch_size):
            ids = permutation[offset:offset + batch_size]
            anchors = ids // train_views_per_sku
            x_np = np.asarray(train_views[ids], dtype=np.float32)
            q0 = torch.as_tensor(x_np, device=device)
            ids_t = torch.as_tensor(candidate_ids[anchors], device=device, dtype=torch.long)
            ref_candidates = ref[ids_t.reshape(-1)].reshape(len(ids), ids_t.shape[1], -1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                q = adapter(q0)
                r = adapter(ref_candidates.reshape(-1, ref.shape[1])).reshape(len(ids), ids_t.shape[1], -1)
                logits = (q[:, None, :] * r).sum(dim=-1) / temperature
                labels = torch.zeros(len(ids), device=device, dtype=torch.long)
                loss = F.cross_entropy(logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.detach().item()) * len(ids)
            seen += len(ids)
        train_metrics = evaluate_candidate_accuracy(adapter, train_embedding_path, len(ref), train_views_per_sku,
                                                    ref, candidate_ids, temperature=temperature)
        validation_metrics = evaluate_view_split(adapter, validation_embedding_path, len(ref), validation_views_per_sku,
                                                 ref, candidate_ids, family_mask, nearest20,
                                                 temperature=temperature)
        improved = (
            validation_metrics["candidate_top1_accuracy"] > best_accuracy + 1e-12
            or (abs(validation_metrics["candidate_top1_accuracy"] - best_accuracy) <= 1e-12
                and validation_metrics["candidate_cross_entropy"] < best_loss - 1e-9)
        )
        if improved:
            best_accuracy = float(validation_metrics["candidate_top1_accuracy"])
            best_loss = float(validation_metrics["candidate_cross_entropy"])
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        payload = {
            "epoch": epoch,
            "adapter_state": adapter.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "train_metrics": train_metrics,
            "training_loss_online": train_loss_sum / max(1, seen),
            "validation_metrics": validation_metrics,
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_accuracy,
            "epochs_without_improvement": epochs_without_improvement,
        }
        epoch_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        _save_epoch_checkpoint(epoch_path, payload)
        if improved:
            _save_epoch_checkpoint(best_path, payload)
            best = payload
        _save_epoch_checkpoint(latest_path, payload)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
               "training_loss_online": train_loss_sum / max(1, seen),
               **{f"validation_{k}": v for k, v in validation_metrics.items()}, "selected_best": improved}
        history.append(row)
        write_csv(run_dir / "training_history.csv", history)
        print(f"[epoch {epoch:02d}] loss={train_loss_sum / max(1, seen):.5f} "
              f"train_top1={train_metrics['candidate_top1_accuracy']:.4f} "
              f"val_top1={validation_metrics['candidate_top1_accuracy']:.4f} "
              f"val_full={validation_metrics['full_catalog_top1_accuracy']:.4f} "
              f"best={best_epoch} patience={epochs_without_improvement}/{patience}", flush=True)
        if epochs_without_improvement >= patience:
            break

    del train_views

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    adapter.load_state_dict(best["adapter_state"])
    adapter.eval()
    baseline_metrics = next((row for row in history if int(row["epoch"]) == 0), None)
    if baseline_metrics is None:
        raise RuntimeError("training history is missing the identity checkpoint")
    selected_validation = best["validation_metrics"]
    selected_train = best["train_metrics"]
    initial_val_acc = float(baseline_metrics["validation_candidate_top1_accuracy"])
    train_acc, val_acc = float(selected_train["candidate_top1_accuracy"]), float(selected_validation["candidate_top1_accuracy"])
    overfit_gap = train_acc - val_acc
    overfit = bool(train_acc >= 0.98 and val_acc <= initial_val_acc + 0.005 and overfit_gap >= 0.10)
    result = {
        "baseline_identity_adapter": {
            "candidate_top1_accuracy": float(baseline_metrics["validation_candidate_top1_accuracy"]),
            "candidate_cross_entropy": float(baseline_metrics["validation_candidate_cross_entropy"]),
            "full_catalog_top1_accuracy": float(baseline_metrics["validation_full_catalog_top1_accuracy"]),
            "same_family_top1_accuracy": baseline_metrics["validation_same_family_top1_accuracy"],
            "reference_nearest20_top1_accuracy": float(baseline_metrics["validation_reference_nearest20_top1_accuracy"]),
        },
        "selected_checkpoint": {
            "epoch": int(best["epoch"]),
            "candidate_top1_accuracy": float(selected_validation["candidate_top1_accuracy"]),
            "candidate_cross_entropy": float(selected_validation["candidate_cross_entropy"]),
            "full_catalog_top1_accuracy": float(selected_validation["full_catalog_top1_accuracy"]),
            "same_family_top1_accuracy": selected_validation["same_family_top1_accuracy"],
            "reference_nearest20_top1_accuracy": float(selected_validation["reference_nearest20_top1_accuracy"]),
            "train_candidate_top1_accuracy": train_acc,
            "train_validation_accuracy_gap": overfit_gap,
        },
        "validation_candidate_top1_delta": float(selected_validation["candidate_top1_accuracy"]) - initial_val_acc,
        "overfit_flag": overfit,
        "overfit_rule": "train>=0.98 and val<=identity+0.005 and train-val gap>=0.10",
        "selected_by": "maximum internal validation candidate-set Top-1; ties broken by lower candidate CE",
        "phase_b_started": False,
    }
    write_json(run_dir / "validation_metrics.json", result)
    return adapter, {"best_checkpoint": best, "validation_report": result}


def build_family_and_neighbor_masks(
    slugs: Sequence[str], ref_np: np.ndarray, family_keys: Mapping[str, str | None],
    negative_indices: Mapping[str, Sequence[int]], negative_rows: Sequence[Mapping[str, Any]],
    candidate_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[int]], dict[str, list[int]]]:
    index = {slug: i for i, slug in enumerate(slugs)}
    by_anchor: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in negative_rows:
        by_anchor[str(row["anchor_slug"])].append((index[str(row["negative_slug"])], str(row["negative_source"])))
    family_indices: dict[str, list[int]] = {}
    random_indices: dict[str, list[int]] = {}
    for slug in slugs:
        family_indices[slug] = [i for i, source in by_anchor[slug] if source == "catalog_metadata_same_family"]
        random_indices[slug] = [i for i, source in by_anchor[slug] if source == "uniform_catalog_random"]
    family_mask = np.zeros_like(candidate_ids, dtype=bool)
    family_mask[:, 0] = True
    for anchor, slug in enumerate(slugs):
        family_set = set(family_indices[slug])
        for col, candidate in enumerate(candidate_ids[anchor, 1:], start=1):
            family_mask[anchor, col] = int(candidate) in family_set
    vectors = np.asarray(ref_np, dtype=np.float32)
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, -np.inf)
    nearest20 = np.argsort(-similarity, axis=1, kind="stable")[:, :20].astype(np.int64)
    if nearest20.shape != (len(slugs), 20):
        raise RuntimeError("catalog must contain at least 21 usable references for nearest-20 validation")
    return family_mask, nearest20, family_indices, random_indices


def adapter_metadata(adapter: ResidualMetricAdapter, base_meta: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "architecture": "ResidualMetricAdapter",
        "input_dim": adapter.input_dim,
        "hidden_dim": adapter.hidden_dim,
        "residual_scale": adapter.residual_scale,
        "parameter_count": sum(parameter.numel() for parameter in adapter.parameters()),
        "normalization": "L2 normalize(z + 0.1 * Linear(GELU(Linear(z))))",
        "base_encoder": base_meta["hf_model_id"],
        "base_revision": base_meta["checkpoint_revision"],
        "base_preprocessing": base_meta["preprocessing_config"],
        "temperature": TEMPERATURE,
    }


def freeze_adapter_and_references(
    run_dir: Path, adapter: ResidualMetricAdapter, ref_np: np.ndarray,
    slugs: Sequence[str], image_hashes: Mapping[str, str], base_meta: Mapping[str, Any],
    validation_report: Mapping[str, Any], device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    adapter.eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    architecture = adapter_metadata(adapter, base_meta)
    write_json(run_dir / "adapter_architecture.json", architecture)
    state_hash = adapter_state_fingerprint(adapter)
    atomic_torch_save(run_dir / "frozen_adapter.pt", {
        "state_dict": {key: value.detach().cpu() for key, value in adapter.state_dict().items()},
        "architecture": architecture,
        "adapter_sha256": state_hash,
    })
    reference_fp = reference_cache_fingerprint(base_meta["fingerprint"], state_hash, slugs, image_hashes)
    reference_path = run_dir / "adapted_reference_embeddings.npy"
    matrix = np.lib.format.open_memmap(reference_path, mode="w+", dtype=np.float16, shape=(len(slugs), 1152))
    with torch.inference_mode():
        for start in range(0, len(slugs), 256):
            refs = torch.as_tensor(ref_np[start:start + 256], dtype=torch.float32, device=device)
            adapted = adapter(refs).cpu().numpy()
            matrix[start:start + len(adapted)] = adapted.astype(np.float16)
    matrix.flush()
    adapted_refs = np.asarray(matrix, dtype=np.float32)
    validate_normalized_embeddings(adapted_refs)
    del matrix
    cache_meta = {
        "cache_fingerprint": reference_fp,
        "adapter_sha256": state_hash,
        "base_reference_fingerprint": base_meta["fingerprint"],
        "reference_count": len(slugs),
        "embedding_dim": 1152,
        "dtype": "float16",
        "slugs_file": str(run_dir / "reference_slugs.json"),
        "embeddings_file": str(reference_path),
    }
    write_json(run_dir / "reference_cache_metadata.json", cache_meta)
    write_json(run_dir / "reference_slugs.json", list(slugs))
    checkpoint_meta = {
        "frozen": True,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": "checkpoints/best.pt",
        "checkpoint_epoch": int(validation_report["selected_checkpoint"]["epoch"]),
        "adapter_sha256": state_hash,
        "architecture": architecture,
        "validation_metrics": validation_report,
        "reference_cache_fingerprint": reference_fp,
        "no_benchmark_data_used_for_checkpoint_selection": True,
        "model_selection_source": "catalog-derived internal validation only",
    }
    write_json(run_dir / "checkpoint_metadata.json", checkpoint_meta)
    atomic_state(run_dir / "state.json", "adapter_frozen", checkpoint_metadata="checkpoint_metadata.json")
    return adapted_refs, checkpoint_meta


def _parse_list(value: str) -> list:
    return json.loads(value) if value else []


def load_primary_benchmark_inputs(root: Path) -> dict[str, dict]:
    import run_strong_local_visual_reranker as strong_runner

    return strong_runner.load_benchmark_inputs(root)


def load_postfreeze_family_slices(root: Path) -> dict[str, dict[str, dict[str, str]]]:
    """Read evaluation-only family labels after the adapter freeze barrier."""
    hard_rows = read_csv(root / "data/benchmarks/hard_near_duplicate_dev_v2/families.csv")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in hard_rows:
        groups[row["family_id"]].append(row)
    hard_map: dict[str, dict[str, str]] = {}
    import re
    for family_id, members in groups.items():
        years = {year for member in members for year in re.findall(
            r"(?<!\d)(?:19|20)\d{2}(?!\d)", " ".join((member.get("product_name", ""), member.get("year_if_known", ""))))}
        grapes = {member.get("grape", "").strip().casefold() for member in members if member.get("grape", "").strip()}
        family_type = "vintage" if len(years) >= 2 else "subtype" if len(grapes) >= 2 else "other"
        for member in members:
            hard_map[member["slug"]] = {"family_id": family_id, "family_type": family_type}

    generated_map: dict[str, dict[str, str]] = {}
    for family in read_csv(root / "data/benchmarks/generated_stress_dev_pilot32/hard_families.csv"):
        members = _parse_list(family.get("member_slugs", ""))
        for slug in members:
            generated_map[slug] = {"family_id": family.get("pilot_family_id", ""),
                                   "family_type": family.get("family_type", "")}
    return {"hard_near_duplicate_dev_v2": hard_map,
            "generated_stress_dev_pilot32": generated_map}


def load_synthetic_inputs(root: Path) -> dict[str, dict]:
    source = root / SOURCE_RUN / "benchmarks" / SYNTHETIC_BENCHMARK
    sift_dir = root / SIFT_RUN / SYNTHETIC_BENCHMARK
    current_rows = {row["query_id"]: row for row in read_csv(source / "reranked_predictions.csv")}
    selected_rows = {row["query_id"]: row for row in read_csv(sift_dir / "predictions.csv")}
    manifests = {row["query_id"]: row for row in read_csv(root / "data/benchmarks" / SYNTHETIC_BENCHMARK / "manifest.csv")}
    pair_rows: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in read_csv(sift_dir / "geometry_pair_scores.csv"):
        pair_rows[row["query_id"]][row["candidate_slug"]] = row
    result = {}
    for query_id, current in current_rows.items():
        selected = selected_rows[query_id]
        manifest = manifests[query_id]
        candidates = _parse_list(current["top5_slugs"])
        validate_top5_candidates(candidates)
        selected_order = _parse_list(selected["selected_ranking"])
        if set(candidates) != set(selected_order) or len(pair_rows[query_id]) != 5:
            raise RuntimeError(f"synthetic frozen candidate set or SIFT records differ for {query_id}")
        result[query_id] = {
            "benchmark": SYNTHETIC_BENCHMARK,
            "query_id": query_id,
            "target_slug": current["target_slug"],
            "scenario_id": manifest.get("transform_type", ""),
            "subset_role": "synthetic",
            "family_type": selected.get("target_family_type", ""),
            "query_path": manifest["query_path"],
            "candidate_slugs": candidates,
            "current_scores": [float(value) for value in _parse_list(current["top5_scores"])],
            "current_top1": candidates[0],
            "sift_selected_top1": selected["selected_top1"],
            "sift_selected_ranking": selected_order,
            "final_row": selected,
            "old_geometry": {
                (query_id, slug): pair_rows[query_id][slug] for slug in candidates
            },
        }
    return {SYNTHETIC_BENCHMARK: result}


def precompute_query_embeddings(
    root: Path, run_dir: Path, benchmark: str, queries: Mapping[str, Mapping[str, Any]],
    encoder: SigLIP2So400m384Adapter, checkpoint_hash: str, batch_size: int,
) -> np.ndarray:
    eval_dir = run_dir / "evaluation_queries"
    eval_dir.mkdir(parents=True, exist_ok=True)
    path = eval_dir / f"{benchmark}_embeddings.npy"
    progress_path = eval_dir / f"{benchmark}_progress.json"
    query_items = list(queries.items())
    count = len(query_items)
    if not npy_has_shape(path, (count, 1152), np.float16):
        np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(count, 1152)).flush()
        progress_path.unlink(missing_ok=True)
    done = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("checkpoint_hash") == checkpoint_hash and progress.get("query_count") == count:
            done = int(progress.get("completed_queries", 0))
            if not 0 <= done <= count:
                raise RuntimeError(f"invalid {benchmark} query embedding progress checkpoint")
    matrix = np.lib.format.open_memmap(path, mode="r+")
    for start in range(done, count, batch_size):
        batch = query_items[start:start + batch_size]
        images = []
        for _query_id, query in batch:
            with Image.open(root / str(query["query_path"])) as image:
                images.append(image.convert("RGB"))
        vectors = np.asarray(encoder.encode_images(images), dtype=np.float32)
        validate_normalized_embeddings(vectors)
        matrix[start:start + len(batch)] = vectors.astype(np.float16)
        matrix.flush()
        done = start + len(batch)
        write_json(progress_path, {"checkpoint_hash": checkpoint_hash, "query_count": count,
                                   "completed_queries": done, "last_completed_batch_start": start,
                                   "last_completed_batch_size": len(batch)})
        if done % 512 < batch_size or done == count:
            print(f"[{benchmark} queries] {done}/{count}", flush=True)
    del matrix
    return np.load(path, mmap_mode="r")


def score_query_without_target(
    query_embedding: np.ndarray, candidate_slugs: Sequence[str], candidate_ids: Sequence[int],
    adapted_reference_embeddings: np.ndarray,
) -> list[float]:
    """Target-blind inference function: it accepts only the frozen Top-5."""
    validate_top5_candidates(candidate_slugs)
    if len(candidate_ids) != 5:
        raise ValueError("frozen Top-5 must contain five IDs")
    return [float(np.dot(query_embedding, adapted_reference_embeddings[index])) for index in candidate_ids]


def summarize_one_method(rows: Sequence[Mapping[str, Any]], top1_field: str, benchmark: str, method: str) -> dict[str, Any]:
    correct = sum(row["target_slug"] == row[top1_field] for row in rows)
    recall = sum(row["target_slug"] in row["candidate_slugs"] for row in rows)
    return {"benchmark": benchmark, "method": method, "queries": len(rows), "top1_correct": correct,
            "top1_accuracy": correct / max(1, len(rows)), "recall_at_5_correct": recall,
            "recall_at_5": recall / max(1, len(rows))}


def _groups_for_query(query: Mapping[str, Any], benchmark: str) -> list[str]:
    groups = ["overall"]
    if benchmark == "generated_stress_dev_pilot32":
        role = str(query.get("subset_role", "")).casefold()
        if role:
            groups.append(role)
        family_type = str(query.get("family_type", "")).casefold()
        if family_type in {"vintage", "subtype"}:
            groups.append(family_type)
            if role == "hard":
                groups.append(f"hard_{family_type}")
    elif benchmark == "hard_near_duplicate_dev_v2":
        family_type = str(query.get("family_type", "")).casefold()
        if family_type in {"vintage", "subtype", "other"}:
            groups.append(family_type)
        if query.get("family_id"):
            groups.append("within_family")
    return groups


def _write_eval_tables(run_dir: Path, all_rows: list[dict[str, Any]], selected_weight: float) -> None:
    benchmark_summary = []
    transition_summary = []
    family_metrics = []
    methods = [
        ("baseline_current_plus_sift_w0.40", "baseline_top1"),
        ("metric_adapter_only", "adapter_only_top1"),
    ] + [(f"current_plus_sift_plus_adapter_w{weight:.2f}", f"fused_top1_w{weight:.2f}") for weight in FUSION_WEIGHTS]
    grouped = defaultdict(list)
    for row in all_rows:
        grouped[row["benchmark"]].append(row)
    for benchmark, rows in grouped.items():
        for method, field in methods:
            metrics = summarize_one_method(rows, field, benchmark, method)
            metrics["selected_policy"] = method == f"current_plus_sift_plus_adapter_w{selected_weight:.2f}"
            benchmark_summary.append(metrics)
            group_names = sorted({group for row in rows for group in _groups_for_query(row, benchmark)})
            for group_name in group_names:
                subset = [row for row in rows if group_name in _groups_for_query(row, benchmark)]
                if not subset:
                    continue
                transitions = Counter(transition_name(row["target_slug"] == row["baseline_top1"],
                                                      row["target_slug"] == row[field]) for row in subset)
                transition_summary.append({"benchmark": benchmark, "slice": group_name, "method": method,
                                          "queries": len(subset), **{key: transitions[key] for key in (
                                              "correct_to_correct", "correct_to_wrong", "wrong_to_correct", "wrong_to_wrong")},
                                          "rescued": transitions["wrong_to_correct"],
                                          "broken": transitions["correct_to_wrong"]})
                if group_name in {"vintage", "subtype", "other", "within_family", "hard_vintage", "hard_subtype"}:
                    correct = sum(row["target_slug"] == row[field] for row in subset)
                    family_metrics.append({"benchmark": benchmark, "family_slice": group_name, "method": method,
                                           "queries": len(subset), "top1_correct": correct,
                                           "top1_accuracy": correct / len(subset)})
    write_csv(run_dir / "benchmark_summary.csv", benchmark_summary)
    write_csv(run_dir / "transition_summary.csv", transition_summary)
    write_csv(run_dir / "family_metrics.csv", family_metrics)


def _embedding_diagnostics(
    validation_path: Path, val_views_per_sku: int, slugs: Sequence[str], ref_base: np.ndarray,
    ref_adapted: np.ndarray, family_indices: Mapping[str, Sequence[int]], random_indices: Mapping[str, Sequence[int]],
    adapter: ResidualMetricAdapter, device: torch.device,
) -> list[dict[str, Any]]:
    views = np.load(validation_path, mmap_mode="r")
    queries_after = np.empty((len(views), 1152), dtype=np.float32)
    adapter.eval()
    with torch.inference_mode():
        for start in range(0, len(views), 256):
            q = torch.as_tensor(np.asarray(views[start:start + 256], dtype=np.float32), device=device)
            queries_after[start:start + len(q)] = adapter(q).cpu().numpy()
    queries_before = np.asarray(views, dtype=np.float32)
    anchors = np.arange(len(views)) // val_views_per_sku
    values = {"before": defaultdict(list), "after": defaultdict(list)}
    for mode, qmatrix, refs in (("before", queries_before, ref_base), ("after", queries_after, ref_adapted)):
        for row, anchor_id in enumerate(anchors):
            slug = slugs[int(anchor_id)]
            positive = float(np.dot(qmatrix[row], refs[anchor_id]))
            values[mode]["positive_similarity"].append(positive)
            families = family_indices.get(slug, [])
            if families:
                hardest = max(float(np.dot(qmatrix[row], refs[index])) for index in families)
                values[mode]["hardest_family_negative_similarity"].append(hardest)
                values[mode]["positive_minus_hardest_family_margin"].append(positive - hardest)
            randoms = random_indices.get(slug, [])
            if randoms:
                values[mode]["random_negative_similarity"].append(float(np.dot(qmatrix[row], refs[randoms[0]])))
    rows = []
    for metric in ("positive_similarity", "hardest_family_negative_similarity", "random_negative_similarity",
                   "positive_minus_hardest_family_margin"):
        for mode in ("before", "after"):
            data = values[mode][metric]
            rows.append({"metric": metric, "state": mode, "views": len(data),
                         "mean": statistics.fmean(data) if data else None,
                         "median": statistics.median(data) if data else None})
    del views
    return rows


def _measure_latency(adapter: ResidualMetricAdapter, ref_adapted: np.ndarray, sample_queries: np.ndarray,
                     candidate_ids: Sequence[int], base_scores: Sequence[float], adapter_scores: Sequence[float],
                     device: torch.device) -> list[dict[str, Any]]:
    query_ms: list[float] = []
    similarity_ms: list[float] = []
    fusion_ms: list[float] = []
    adapter.eval()
    refs_t = torch.as_tensor(np.array(ref_adapted, dtype=np.float32, copy=True), device=device)
    sample = torch.as_tensor(sample_queries[0], dtype=torch.float32, device=device)
    ids_t = torch.as_tensor(candidate_ids, dtype=torch.long, device=device)
    with torch.inference_mode():
        for _ in range(20):
            adapter(sample)
            adapter(sample) @ refs_t[ids_t].T
    for _ in range(200):
        start = time.perf_counter()
        with torch.inference_mode():
            q = adapter(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        query_ms.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        with torch.inference_mode():
            q @ refs_t[ids_t].T
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        similarity_ms.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        fuse_local_signals(base_scores, [adapter_scores], 0.20)
        fusion_ms.append((time.perf_counter() - start) * 1000)
    def row(name: str, data: Sequence[float]) -> dict[str, Any]:
        return {"component": name, "mean_ms": statistics.fmean(data), "p50_ms": statistics.median(data),
                "p95_ms": float(np.percentile(data, 95)), "samples": len(data)}
    rows = [row("adapter_query_inference", query_ms), row("top5_cosine_similarity", similarity_ms), row("fusion", fusion_ms)]
    added_p95 = sum(float(item["p95_ms"]) for item in rows)
    rows.append({"component": "adapter_added_total", "mean_ms": sum(float(item["mean_ms"]) for item in rows),
                 "p50_ms": sum(float(item["p50_ms"]) for item in rows), "p95_ms": added_p95, "samples": 200})
    baseline_p95 = 514.0964479994791
    rows.append({"component": "estimated_current_plus_sift_plus_adapter_pipeline", "mean_ms": None,
                 "p50_ms": None, "p95_ms": baseline_p95 + added_p95, "samples": 200,
                 "baseline_current_plus_sift_p95_ms": baseline_p95, "sla_3s_ok": baseline_p95 + added_p95 < 3000})
    return rows


def evaluate_frozen_adapter(
    root: Path, run_dir: Path, adapter: ResidualMetricAdapter, ref_adapted: np.ndarray,
    slugs: Sequence[str], checkpoint_meta: Mapping[str, Any], *, batch_size: int,
) -> None:
    if not checkpoint_meta.get("frozen") or not (run_dir / "checkpoint_metadata.json").is_file():
        raise RuntimeError("benchmark access is blocked until frozen checkpoint metadata exists")
    # Benchmark data is first loaded after the frozen barrier above.
    primary = load_primary_benchmark_inputs(root)
    synthetic = load_synthetic_inputs(root)
    all_data = {**primary, **synthetic}
    postfreeze_families = load_postfreeze_family_slices(root)
    slug_to_index = {slug: index for index, slug in enumerate(slugs)}
    evaluation_rows: list[dict[str, Any]] = []
    base_rows: dict[str, list[np.ndarray]] = {}
    query_embed_dir = run_dir / "evaluation_queries"
    query_embed_dir.mkdir(parents=True, exist_ok=True)
    adapter_hash = str(checkpoint_meta["adapter_sha256"])
    adapter.eval()
    for benchmark, queries in all_data.items():
        if "query_encoder" not in locals():
            query_encoder = _get_encoder(batch_size)
        query_matrix = precompute_query_embeddings(root, run_dir, benchmark, queries,
                                                   query_encoder, adapter_hash, batch_size)
        # The SO400M encoder is released immediately after each resumable query cache is complete.
        positions = {query_id: index for index, (query_id, _query) in enumerate(queries.items())}
        current_rows = []
        with torch.inference_mode():
            for query_id, query in queries.items():
                candidates = query["candidate_slugs"]
                ids = [slug_to_index[slug] for slug in candidates]
                geo = []
                for slug in candidates:
                    old = query["old_geometry"].get((query_id, slug))
                    if old is None:
                        raise RuntimeError(f"missing frozen SIFT pair {benchmark}/{query_id}/{slug}")
                    valid = str(old["homography_valid"]).lower() == "true"
                    geo.append(float(old["geometric_score"]) if valid else 0.0)
                sift_scores = fuse_scores(query["current_scores"], geo, 0.40)
                baseline_order = rank_candidates(candidates, sift_scores)
                if baseline_order != query["sift_selected_ranking"]:
                    raise RuntimeError(f"frozen current+SIFT ranking changed after freeze: {benchmark}/{query_id}")
                q = np.asarray(query_matrix[positions[query_id]], dtype=np.float32)
                q_tensor = torch.as_tensor(q, dtype=torch.float32, device=next(adapter.parameters()).device)
                q_adapted = adapter(q_tensor).cpu().numpy()
                adapter_scores = score_query_without_target(q_adapted, candidates, ids, ref_adapted)
                adapter_only_order = rank_fixed_top5(candidates, adapter_scores)
                fused_scores = {weight: fuse_local_signals(sift_scores, [adapter_scores], weight)
                                for weight in FUSION_WEIGHTS}
                fused_orders = {weight: rank_fixed_top5(candidates, fused_scores[weight]) for weight in FUSION_WEIGHTS}
                # Target is first read here, after every score vector and ranking has been fixed.
                target = query["target_slug"]
                family_info = postfreeze_families.get(benchmark, {}).get(target, {})
                family_type = family_info.get("family_type", query.get("family_type", ""))
                row: dict[str, Any] = {
                    "benchmark": benchmark, "query_id": query_id, "target_slug": target,
                    "candidate_slugs": candidates, "candidate_set_unchanged": set(candidates) == set(query["sift_selected_ranking"]),
                    "baseline_top1": baseline_order[0], "baseline_ranking": baseline_order,
                    "adapter_only_top1": adapter_only_order[0], "adapter_only_ranking": adapter_only_order,
                    "target_family_type": family_type,
                    "family_type": family_type,
                    "family_id": family_info.get("family_id", ""),
                    "subset_role": query.get("subset_role", ""),
                    "adapter_scores": adapter_scores, "sift_scores": sift_scores,
                    "baseline_correct": baseline_order[0] == target,
                    "adapter_only_correct": adapter_only_order[0] == target,
                }
                for weight in FUSION_WEIGHTS:
                    key = f"w{weight:.2f}"
                    row[f"fused_top1_{key}"] = fused_orders[weight][0]
                    row[f"fused_ranking_{key}"] = fused_orders[weight]
                    row[f"fused_correct_{key}"] = fused_orders[weight][0] == target
                if set(row["candidate_slugs"]) != set(row["baseline_ranking"]):
                    raise AssertionError("a metric adapter ranking changed the candidate set")
                current_rows.append(row)
        base_rows[benchmark] = current_rows
        print(f"[{benchmark}] scored {len(current_rows)} frozen Top-5 queries", flush=True)
        # Keep base SO400M embeddings on disk; only adapted reference vectors stay resident.
        del query_matrix
        gc.collect()
    if "query_encoder" in locals():
        query_encoder.release()
        del query_encoder
        gc.collect()

    primary_gen = base_rows["generated_stress_dev_pilot32"]
    hard_rows = base_rows["hard_near_duplicate_dev_v2"]
    baseline_gen = sum(row["baseline_correct"] for row in primary_gen)
    baseline_rep = sum(row["baseline_correct"] for row in primary_gen if str(row["subset_role"]).casefold() == "representative")
    baseline_hard = sum(row["baseline_correct"] for row in primary_gen if str(row["subset_role"]).casefold() == "hard")
    hard_guard_min = math.ceil((sum(row["baseline_correct"] for row in hard_rows) / len(hard_rows) - 0.01) * len(hard_rows))
    generated_rep_rows = [row for row in primary_gen if str(row["subset_role"]).casefold() == "representative"]
    generated_hard_rows = [row for row in primary_gen if str(row["subset_role"]).casefold() == "hard"]
    representative_guard_min = baseline_rep
    if len(primary_gen) != 128 or len(generated_rep_rows) != 64 or len(generated_hard_rows) != 64:
        raise RuntimeError("generated pilot32 must preserve the frozen 128/64/64 query slices")
    policy_results = []
    for weight in FUSION_WEIGHTS:
        gen_count = sum(bool(row[f"fused_correct_w{weight:.2f}"]) for row in primary_gen)
        hard_count = sum(row[f"fused_top1_w{weight:.2f}"] == row["target_slug"] for row in hard_rows)
        rep_count = sum(bool(row[f"fused_correct_w{weight:.2f}"]) for row in generated_rep_rows)
        subset_hard_count = sum(bool(row[f"fused_correct_w{weight:.2f}"]) for row in generated_hard_rows)
        policy_results.append({"weight": weight, "generated_correct": gen_count, "hard_v2_correct": hard_count,
                               "generated_hard_correct": subset_hard_count, "generated_representative_correct": rep_count,
                               "guards_pass": hard_count >= hard_guard_min and rep_count >= representative_guard_min})
    eligible = [row for row in policy_results if row["guards_pass"]]
    pool = eligible or policy_results
    selected = sorted(pool, key=lambda row: (-row["generated_correct"], row["weight"]))[0]
    selected_weight = float(selected["weight"])
    selected_policy = {
        "method": "current_plus_sift_plus_metric_adapter",
        "adapter_weight": selected_weight,
        "fixed_grid": list(FUSION_WEIGHTS),
        "selection_rule": "highest generated pilot32 Top-1 on fixed grid among policies preserving the hard_v2 <=1pp and representative no-regression guards; ties choose lower weight; if none preserve guards, choose highest generated Top-1 and mark guards failed",
        "grid_results": policy_results,
        "hard_v2_guard_min_correct": hard_guard_min,
        "generated_representative_guard_min_correct": representative_guard_min,
        "selected_policy_guards_pass": bool(selected["guards_pass"]),
        "selected_after_adapter_freeze": True,
        "synthetic_used_for_selection": False,
    }
    write_json(run_dir / "selected_policy.json", selected_policy)
    for benchmark, rows in base_rows.items():
        evaluation_rows.extend(rows)
    write_json(run_dir / "evaluation_rows.json", evaluation_rows)
    write_csv(run_dir / "per_query_scores.csv", [{
        "benchmark": row["benchmark"], "query_id": row["query_id"], "target_slug": row["target_slug"],
        "candidate_slugs": json.dumps(row["candidate_slugs"], ensure_ascii=False),
        "baseline_top1": row["baseline_top1"], "adapter_only_top1": row["adapter_only_top1"],
        **{f"fused_top1_w{weight:.2f}": row[f"fused_top1_w{weight:.2f}"] for weight in FUSION_WEIGHTS},
        "adapter_scores": json.dumps(row["adapter_scores"]), "sift_scores": json.dumps(row["sift_scores"]),
    } for row in evaluation_rows])
    _write_eval_tables(run_dir, evaluation_rows, selected_weight)

    gen_remaining = [row for row in primary_gen if not row["baseline_correct"]]
    oracle_margins = []
    oracle_wins = oracle_losses = oracle_ties = 0
    for row in gen_remaining:
        target_i = row["candidate_slugs"].index(row["target_slug"]) if row["target_slug"] in row["candidate_slugs"] else None
        incumbent_i = row["candidate_slugs"].index(row["baseline_top1"])
        if target_i is None:
            continue
        margin = float(row["adapter_scores"][target_i] - row["adapter_scores"][incumbent_i])
        oracle_margins.append(margin)
        oracle_wins += margin > 1e-12
        oracle_losses += margin < -1e-12
        oracle_ties += abs(margin) <= 1e-12
    write_csv(run_dir / "oracle_summary.csv", [{
        "benchmark": "generated_stress_dev_pilot32", "baseline_remaining_errors": len(gen_remaining),
        "target_in_top5_errors": len(oracle_margins), "adapter_target_wins": oracle_wins,
        "adapter_target_losses": oracle_losses, "adapter_ties": oracle_ties,
        "mean_target_minus_incumbent_adapter_similarity": statistics.fmean(oracle_margins) if oracle_margins else None,
    }])

    # Internal validation diagnostics are computed against catalog-derived families only.
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    catalog, current_slugs, ref_base, base_meta, image_hashes = load_catalog_and_reference_cache(root)
    family_keys = catalog_family_keys(catalog)
    negative_map_payload = read_csv(run_dir / "hard_negative_map.csv")
    negative_indices, family_indices, negative_rows = build_hard_negative_map(
        current_slugs, ref_base, family_keys, int(config["seed"]),
        family_count=4, hard_count=8, random_count=2, neighbor_pool=20,
    )
    _, _, family_indices, random_indices = build_family_and_neighbor_masks(
        current_slugs, ref_base, family_keys, negative_indices, negative_rows,
        candidate_index_matrix(current_slugs, negative_indices),
    )
    val_manifest = read_csv(run_dir / "validation_manifest.csv")
    diagnostics = _embedding_diagnostics(
        run_dir / "validation_view_embeddings.npy", int(config["validation_views_per_sku"]), current_slugs,
        ref_base, ref_adapted, family_indices, random_indices, adapter, next(adapter.parameters()).device,
    )
    write_csv(run_dir / "embedding_diagnostics.csv", diagnostics)

    # Query-stage timing excludes SO400M, which remains frozen and already runs upstream.
    sample_query = np.asarray(np.load(run_dir / "evaluation_queries/generated_stress_dev_pilot32_embeddings.npy", mmap_mode="r")[:1], dtype=np.float32)
    first = primary_gen[0]
    latency = _measure_latency(adapter, ref_adapted, sample_query,
                               [slug_to_index[slug] for slug in first["candidate_slugs"]], first["sift_scores"],
                               first["adapter_scores"], next(adapter.parameters()).device)
    write_csv(run_dir / "latency_summary.csv", latency)
    adapter_only_hard = sum(row["adapter_only_top1"] == row["target_slug"] for row in hard_rows)
    adapter_only_gen = sum(row["adapter_only_top1"] == row["target_slug"] for row in primary_gen)
    chosen_gen = int(selected["generated_correct"])
    chosen_hard_gen = int(selected["generated_hard_correct"])
    chosen_rep = int(selected["generated_representative_correct"])
    chosen_hard_v2 = int(selected["hard_v2_correct"])
    baseline_gen_remaining = len(gen_remaining)
    generated_fused_rows = [row for row in primary_gen if row["target_slug"] == row[f"fused_top1_w{selected_weight:.2f}"]]
    rescued = sum(not row["baseline_correct"] and row["target_slug"] == row[f"fused_top1_w{selected_weight:.2f}"] for row in primary_gen)
    broken = sum(row["baseline_correct"] and row["target_slug"] != row[f"fused_top1_w{selected_weight:.2f}"] for row in primary_gen)
    hard_rescued = sum(not row["baseline_correct"] and row["target_slug"] == row[f"fused_top1_w{selected_weight:.2f}"] for row in generated_hard_rows)
    hard_broken = sum(row["baseline_correct"] and row["target_slug"] != row[f"fused_top1_w{selected_weight:.2f}"] for row in generated_hard_rows)
    rep_rescued = sum(not row["baseline_correct"] and row["target_slug"] == row[f"fused_top1_w{selected_weight:.2f}"] for row in generated_rep_rows)
    rep_broken = sum(row["baseline_correct"] and row["target_slug"] != row[f"fused_top1_w{selected_weight:.2f}"] for row in generated_rep_rows)
    validation_report = json.loads((run_dir / "validation_metrics.json").read_text(encoding="utf-8"))
    gain = chosen_gen - baseline_gen
    if validation_report["overfit_flag"]:
        verdict = "D. OVERFIT / INVALID"
    elif chosen_gen >= 116 and selected_policy["selected_policy_guards_pass"]:
        verdict = "A. FIX_METRIC_ADAPTER"
    elif gain >= 5:
        verdict = "B. METRIC_ADAPTER_HELPS_BUT_BELOW_TARGET"
    else:
        verdict = "C. FROZEN_FEATURES_INSUFFICIENT"
    if baseline_gen_remaining != 25:
        remaining_error_note = f"The frozen generated baseline has {baseline_gen_remaining} errors, not 25; oracle uses all frozen errors."
    else:
        remaining_error_note = "Oracle covers the 25 frozen generated baseline errors."
    report_payload = {
        "baseline_generated_correct": baseline_gen,
        "adapter_only_hard_correct": adapter_only_hard,
        "adapter_only_hard_queries": len(hard_rows),
        "adapter_only_generated_correct": adapter_only_gen,
        "adapter_only_generated_hard_correct": sum(row["adapter_only_top1"] == row["target_slug"] for row in generated_hard_rows),
        "adapter_only_generated_queries": len(primary_gen),
        "selected_weight": selected_weight,
        "selected_generated_correct": chosen_gen,
        "selected_generated_hard_correct": chosen_hard_gen,
        "selected_generated_representative_correct": chosen_rep,
        "selected_hard_v2_correct": chosen_hard_v2,
        "hard_v2_queries": len(hard_rows),
        "generated_queries": len(primary_gen),
        "generated_hard_queries": len(generated_hard_rows),
        "generated_representative_queries": len(generated_rep_rows),
        "generated_rescued": rescued,
        "generated_broken": broken,
        "generated_hard_rescued": hard_rescued,
        "generated_hard_broken": hard_broken,
        "generated_representative_rescued": rep_rescued,
        "generated_representative_broken": rep_broken,
        "oracle_wins": oracle_wins,
        "oracle_errors": len(oracle_margins),
        "oracle_note": remaining_error_note,
        "primary_goal_reached": chosen_gen >= 116 and selected_policy["selected_policy_guards_pass"],
        "validation_report": validation_report,
        "selected_policy": selected_policy,
        "synthetic_dev": summarize_one_method(base_rows[SYNTHETIC_BENCHMARK], f"fused_top1_w{selected_weight:.2f}",
                                               SYNTHETIC_BENCHMARK, "selected_policy"),
        "synthetic_baseline": summarize_one_method(base_rows[SYNTHETIC_BENCHMARK], "baseline_top1",
                                                    SYNTHETIC_BENCHMARK, "baseline_current_plus_sift_w0.40"),
        "verdict": verdict,
        "phase_b_started": False,
    }
    write_json(run_dir / "milestone_summary.json", report_payload)
    write_report(run_dir, report_payload, diagnostics, latency, baseline_gen, baseline_hard,
                 baseline_rep, hard_guard_min)
    atomic_state(run_dir / "state.json", "evaluation_complete", verdict=verdict,
                 generated_correct=chosen_gen, adapter_sha256=adapter_hash)


def _get_encoder(batch_size: int) -> SigLIP2So400m384Adapter:
    os.environ["HF_HUB_OFFLINE"] = "1"
    encoder = SigLIP2So400m384Adapter(batch_size=batch_size)
    if encoder.checkpoint_revision != BASE_REVISION or encoder.preprocessing_config != PREPROCESSING:
        encoder.release()
        raise RuntimeError("SO400M encoder does not match the frozen reference cache")
    return encoder


def write_report(run_dir: Path, summary: Mapping[str, Any], diagnostics: Sequence[Mapping[str, Any]],
                 latency: Sequence[Mapping[str, Any]], baseline_gen: int, baseline_hard: int,
                 baseline_rep: int, hard_guard_min: int) -> None:
    selected = summary["selected_weight"]
    benchmark_rows = read_csv(run_dir / "benchmark_summary.csv")
    by_key = {(row["benchmark"], row["method"]): row for row in benchmark_rows}
    chosen_method = f"current_plus_sift_plus_adapter_w{selected:.2f}"
    hard_result = by_key[("hard_near_duplicate_dev_v2", chosen_method)]
    gen_result = by_key[("generated_stress_dev_pilot32", chosen_method)]
    baseline_hard_metric = by_key[("hard_near_duplicate_dev_v2", "baseline_current_plus_sift_w0.40")]
    base_gen_metric = by_key[("generated_stress_dev_pilot32", "baseline_current_plus_sift_w0.40")]
    gen_hard_result = next(row for row in summary["selected_policy"]["grid_results"] if row["weight"] == selected)
    diag = {(row["metric"], row["state"]): row["mean"] for row in diagnostics}
    def fmt(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.4f}"
    latency_added = next(row for row in latency if row["component"] == "adapter_added_total")
    latency_total = next(row for row in latency if row["component"] == "estimated_current_plus_sift_plus_adapter_pipeline")
    family_rows = [row for row in read_csv(run_dir / "family_metrics.csv") if row["method"] == chosen_method]
    family_table = ["| Benchmark slice | Top1 | Correct / queries |", "|---|---:|---:|"]
    family_table.extend(
        f"| {row['benchmark']} · {row['family_slice']} | {float(row['top1_accuracy']):.2%} | {row['top1_correct']}/{row['queries']} |"
        for row in family_rows
    )
    text = [
        "# Hard-negative metric adapter milestone",
        "",
        f"Run: `{run_dir.name}`. Verdict: **{summary['verdict']}**.",
        "",
        "## Primary results",
        "",
        "| Method | Hard Top1 | Generated Top1 | Generated correct/128 | Generated-hard correct/64 | Rescued | Broken |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| current + SIFT w=0.40 | {float(baseline_hard_metric['top1_accuracy']):.2%} | {float(base_gen_metric['top1_accuracy']):.2%} | {int(base_gen_metric['top1_correct'])}/128 | {baseline_hard}/64 | — | — |",
        f"| current + SIFT + adapter w={selected:.2f} | {float(hard_result['top1_accuracy']):.2%} ({int(hard_result['top1_correct'])}/{int(hard_result['queries'])}) | {float(gen_result['top1_accuracy']):.2%} | {int(gen_result['top1_correct'])}/128 | {gen_hard_result['generated_hard_correct']}/64 | {summary['generated_rescued']} | {summary['generated_broken']} |",
        f"| adapter only | {summary['adapter_only_hard_correct']}/{summary['adapter_only_hard_queries']} | {summary['adapter_only_generated_correct'] / 128:.2%} | {summary['adapter_only_generated_correct']}/128 | {summary['adapter_only_generated_hard_correct']}/64 | — | — |",
        "",
        "## Embedding diagnostics",
        "",
        "| State | Positive sim | Hard-negative sim | Margin | Random-negative sim |",
        "|---|---:|---:|---:|---:|",
        f"| Before | {fmt(diag.get(('positive_similarity','before')))} | {fmt(diag.get(('hardest_family_negative_similarity','before')))} | {fmt(diag.get(('positive_minus_hardest_family_margin','before')))} | {fmt(diag.get(('random_negative_similarity','before')))} |",
        f"| After | {fmt(diag.get(('positive_similarity','after')))} | {fmt(diag.get(('hardest_family_negative_similarity','after')))} | {fmt(diag.get(('positive_minus_hardest_family_margin','after')))} | {fmt(diag.get(('random_negative_similarity','after')))} |",
        "",
        "## Training and checks",
        "",
        f"- Training views: {len(read_csv(run_dir / 'train_manifest.csv')):,}; validation views: {len(read_csv(run_dir / 'validation_manifest.csv')):,}.",
        "- Hard negatives: catalog metadata same-family candidates first, frozen SO400M reference Top-20 neighbors next, plus two deterministic random catalog negatives; no benchmark error mining.",
        "- Adapter: 1152 → 256 → 1152 residual MLP, identity initialized; 591,232 parameters; SO400M remains frozen.",
        f"- Internal validation candidate Top-1: {summary['validation_report']['baseline_identity_adapter']['candidate_top1_accuracy']:.2%} → {summary['validation_report']['selected_checkpoint']['candidate_top1_accuracy']:.2%}; selected epoch {summary['validation_report']['selected_checkpoint']['epoch']}.",
        f"- Family guard: hard_v2 minimum {hard_guard_min}/{summary['hard_v2_queries']}; representative minimum {baseline_rep}/64. Selected policy guards: {summary['selected_policy']['selected_policy_guards_pass']}.",
        f"- Hard_v2 guard result: {hard_result['top1_correct']}/{hard_result['queries']}; generated representative: {summary['selected_generated_representative_correct']}/64 (rescued {summary['generated_representative_rescued']}, broken {summary['generated_representative_broken']}).",
        f"- Generated-hard: baseline {baseline_hard}/64 → {summary['selected_generated_hard_correct']}/64 (rescued {summary['generated_hard_rescued']}, broken {summary['generated_hard_broken']}).",
        f"- Adapter oracle on frozen generated errors: {summary['oracle_wins']}/{summary['oracle_errors']} target scores beat the incumbent. {summary['oracle_note']}",
        f"- Synthetic sanity (post-selection): {summary['synthetic_dev']['top1_correct']}/{summary['synthetic_dev']['queries']} = {summary['synthetic_dev']['top1_accuracy']:.2%}; baseline current+SIFT is {summary['synthetic_baseline']['top1_accuracy']:.2%}.",
        f"- Added adapter stage p95: {latency_added['p95_ms']:.3f} ms; estimated total pipeline p95: {latency_total['p95_ms']:.3f} ms (SLA <3000 ms: {latency_total['sla_3s_ok']}).",
        f"- Overfit flag: {summary['validation_report']['overfit_flag']}; phase B started: no.",
        f"- Reached at least 116/128 with guards: {'YES' if summary['primary_goal_reached'] else 'NO'}.",
        "",
        "## Family slices",
        "",
        *family_table,
        "",
        "Transition details by overall, representative, hard, vintage, subtype, and within-family slices are in `transition_summary.csv`.",
        "",
        "## Checkpoint and resume",
        "",
        "`checkpoints/latest.pt` is written after each epoch; `checkpoints/best.pt` is selected only on internal validation. Train and validation view embedding memmaps flush after each encoded batch with adjacent progress JSON, so restart with the same `--run-dir` resumes unfinished preprocessing or the next whole epoch. Benchmark query embeddings are also resumable and were created only after the frozen checkpoint metadata.",
        "",
        f"Artifacts: `{run_dir}`. Frozen checkpoint metadata: `checkpoint_metadata.json`.",
    ]
    report_path = ROOT / "reports/hard_negative_metric_adapter_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(text) + "\n", encoding="utf-8")


def read_eval_rows(run_dir: Path, benchmark: str, subset: str | None = None) -> list[dict[str, Any]]:
    # The detailed rows remain in memory during report construction in normal runs;
    # this fallback reconstructs only the report slice from the saved audit CSV.
    rows = read_csv(run_dir / "per_query_scores.csv")
    metrics = read_csv(run_dir / "benchmark_summary.csv")
    del metrics
    if subset:
        # Persist subset/family labels in a separate compact report-side field when available.
        in_memory = run_dir / "evaluation_rows.json"
        if in_memory.exists():
            raw = json.loads(in_memory.read_text(encoding="utf-8"))
            return [row for row in raw if row["benchmark"] == benchmark and str(row.get("subset_role", "")).casefold() == subset]
    return [row for row in rows if row["benchmark"] == benchmark]


def run(args: argparse.Namespace) -> None:
    root = ROOT
    run_dir = (root / args.run_dir).resolve() if args.run_dir else root / "artifacts/experiments" / (
        "hard_negative_metric_adapter_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    config = {
        "experiment": "HARD-NEGATIVE METRIC FINE-TUNING FOR TOP-5 SKU DISAMBIGUATION",
        "seed": args.seed,
        "base_encoder": "google/siglip2-so400m-patch14-384",
        "base_revision": BASE_REVISION,
        "base_preprocessing": PREPROCESSING,
        "input_dim": 1152,
        "hidden_dim": 256,
        "residual_scale": 0.1,
        "train_views_per_sku": args.train_views,
        "validation_views_per_sku": args.validation_views,
        "augmentation_version": "catalog_capture_v1",
        "hard_negatives": {"same_family_max": 4, "reference_neighbor_count": 20,
                           "selected_hard_count": 8, "random_count": 2},
        "candidate_count": 11,
        "temperature": TEMPERATURE,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_epochs": args.max_epochs,
        "early_stopping_patience": args.patience,
        "training_batch_size": args.training_batch_size,
        "precompute_batch_size": args.precompute_batch_size,
        "mixed_precision": "CUDA autocast float16 + GradScaler when CUDA is available",
        "benchmark_policy_grid": list(FUSION_WEIGHTS),
        "benchmark_candidate_set": "frozen Top-5 only",
        "benchmark_error_mining": False,
        "phase_b_started": False,
    }
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise RuntimeError("run-dir config differs from this invocation; resume with the original config")
    else:
        write_json(config_path, config)

    # Gate 1: exact frozen baseline. No training begins unless this passes.
    baseline = verify_frozen_sift_baseline(root)
    write_json(run_dir / "baseline_reproduction.json", baseline)
    atomic_state(run_dir / "state.json", "baseline_verified", baseline_reproduction="baseline_reproduction.json")
    print("[gate] current and current+SIFT Top-5 orders reproduce exactly", flush=True)

    catalog, slugs, ref_np, base_meta, image_hashes = load_catalog_and_reference_cache(root)
    manifest_sha = sha256_file(root / "data/processed/catalog_manifest.csv")
    run_fingerprint = make_run_fingerprint(config, manifest_sha, image_hashes)
    dataset_identity_path = run_dir / "dataset_identity.json"
    if dataset_identity_path.exists():
        dataset_identity = json.loads(dataset_identity_path.read_text(encoding="utf-8"))
        if dataset_identity.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError("catalog or reference images changed since this run started; refusing an unsafe resume")
    else:
        write_json(dataset_identity_path, {"run_fingerprint": run_fingerprint,
                                           "catalog_manifest_sha256": manifest_sha,
                                           "catalog_reference_count": len(slugs),
                                           "catalog_reference_hashes": image_hashes})
    write_json(run_dir / "environment.json", {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "numpy": np.__version__,
        "pillow": Image.__version__ if hasattr(Image, "__version__") else "unknown",
        "base_reference_cache_fingerprint": base_meta["fingerprint"],
        "base_encoder_metadata": base_meta,
        "catalog_references": len(slugs),
        "run_fingerprint": run_fingerprint,
    })

    frozen_meta_path = run_dir / "checkpoint_metadata.json"
    if frozen_meta_path.exists() and json.loads(frozen_meta_path.read_text(encoding="utf-8")).get("frozen"):
        checkpoint_meta = json.loads(frozen_meta_path.read_text(encoding="utf-8"))
        saved = torch.load(run_dir / "frozen_adapter.pt", map_location="cpu", weights_only=True)
        adapter = ResidualMetricAdapter(input_dim=1152, hidden_dim=256, residual_scale=0.1)
        adapter.load_state_dict(saved["state_dict"])
        if adapter_state_fingerprint(adapter) != checkpoint_meta["adapter_sha256"]:
            raise RuntimeError("frozen adapter checkpoint hash differs from metadata")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adapter.to(device).eval()
        ref_adapted = np.load(run_dir / "adapted_reference_embeddings.npy", mmap_mode="r")
        validate_normalized_embeddings(np.asarray(ref_adapted, dtype=np.float32))
        evaluate_frozen_adapter(root, run_dir, adapter, ref_adapted, slugs, checkpoint_meta,
                                batch_size=args.precompute_batch_size)
        return

    # Training inputs are derived from catalog rows/cache only; benchmark structs never enter this section.
    family_keys = catalog_family_keys(catalog)
    family_counts = Counter(key for key in family_keys.values() if key)
    train_manifest, validation_manifest = build_view_manifest(
        catalog, image_hashes, args.seed, args.train_views, args.validation_views)
    assert_training_manifest_catalog_only(train_manifest)
    assert_training_manifest_catalog_only(validation_manifest)
    write_csv(run_dir / "train_manifest.csv", train_manifest)
    write_csv(run_dir / "validation_manifest.csv", validation_manifest)
    negative_map, family_candidate_indices, negative_rows = build_hard_negative_map(
        slugs, ref_np, family_keys, args.seed, family_count=4, hard_count=8, random_count=2, neighbor_pool=20)
    allowed_sources = {"catalog_metadata_same_family", "frozen_so400m_reference_top20",
                       "uniform_catalog_random", "frozen_so400m_reference_top20_fallback"}
    if any(row["negative_source"] not in allowed_sources or row["anchor_slug"] == row["negative_slug"]
           for row in negative_rows):
        raise RuntimeError("invalid catalog-only hard-negative map")
    write_csv(run_dir / "hard_negative_map.csv", negative_rows)
    write_json(run_dir / "catalog_family_metadata.json", {
        "source": "catalog_manifest.csv only",
        "explicit_family_column_available": False,
        "conservative_proxy": "same winery + at least two shared distinctive title tokens after generic/grape/year removal",
        "sku_with_family_proxy": sum(value is not None for value in family_keys.values()),
        "family_groups": len(family_counts),
        "largest_family_size": max(family_counts.values(), default=0),
    })
    write_json(run_dir / "negative_source_summary.json", dict(Counter(str(row["negative_source"]) for row in negative_rows)))
    candidate_ids = candidate_index_matrix(slugs, negative_map)
    family_mask, nearest20, family_indices, random_indices = build_family_and_neighbor_masks(
        slugs, ref_np, family_keys, negative_map, negative_rows, candidate_ids)
    write_json(run_dir / "candidate_training_protocol.json", {
        "positive_column": 0,
        "negative_count": candidate_ids.shape[1] - 1,
        "candidate_count": candidate_ids.shape[1],
        "family_positive_mask_count": int(family_mask[:, 1:].sum()),
        "fixed_per_anchor": True,
    })
    if candidate_ids.shape[1] != 11:
        raise RuntimeError("expected exactly positive + 8 hard + 2 random candidates")

    train_progress_path = run_dir / "train_view_precompute_progress.json"
    validation_progress_path = run_dir / "validation_view_precompute_progress.json"
    train_expected = len(train_manifest)
    validation_expected = len(validation_manifest)
    def complete_view_split(progress_path: Path, expected_count: int) -> bool:
        if not progress_path.exists():
            return False
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        split = progress.get("split", "")
        array_path = run_dir / f"{split}_view_embeddings.npy"
        return (progress.get("run_fingerprint") == run_fingerprint
                and int(progress.get("completed_rows", -1)) == expected_count
                and npy_has_shape(array_path, (expected_count, 1152), np.float16))
    views_complete = (complete_view_split(train_progress_path, train_expected)
                      and complete_view_split(validation_progress_path, validation_expected)
                      and (run_dir / "train_view_embeddings.npy").is_file()
                      and (run_dir / "validation_view_embeddings.npy").is_file())
    if not views_complete:
        encoder = _get_encoder(args.precompute_batch_size)
        try:
            if encoder.checkpoint_revision != base_meta["checkpoint_revision"]:
                raise RuntimeError("precompute encoder revision differs from frozen catalog embedding cache")
            precompute_catalog_view_split(root, run_dir, "train", train_manifest, encoder, run_fingerprint,
                                          batch_size=args.precompute_batch_size)
            precompute_catalog_view_split(root, run_dir, "validation", validation_manifest, encoder, run_fingerprint,
                                          batch_size=args.precompute_batch_size)
        finally:
            encoder.release()
            del encoder
            gc.collect()
    train_path = run_dir / "train_view_embeddings.npy"
    val_path = run_dir / "validation_view_embeddings.npy"
    if not npy_has_shape(train_path, (len(train_manifest), 1152), np.float16) or not npy_has_shape(
            val_path, (len(validation_manifest), 1152), np.float16):
        raise RuntimeError("view embedding precompute is incomplete; re-run with the same --run-dir to resume")
    atomic_state(run_dir / "state.json", "catalog_views_precomputed", run_fingerprint=run_fingerprint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter, trained = train_adapter(
        run_dir, device, ref_np, train_path, val_path, candidate_ids, family_mask, nearest20,
        train_views_per_sku=args.train_views, validation_views_per_sku=args.validation_views,
        max_epochs=args.max_epochs, patience=args.patience, batch_size=args.training_batch_size,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay, temperature=TEMPERATURE, seed=args.seed)
    validation_report = trained["validation_report"]
    atomic_state(run_dir / "state.json", "training_complete", best_epoch=trained["best_checkpoint"]["epoch"])
    if validation_report["overfit_flag"]:
        # The checkpoint stays frozen for a diagnostic evaluation; verdict D prevents adoption.
        print("[validation] overfit diagnostic flagged; checkpoint remains frozen for reporting", flush=True)
    ref_adapted, checkpoint_meta = freeze_adapter_and_references(
        run_dir, adapter, ref_np, slugs, image_hashes, base_meta, validation_report, device)
    evaluate_frozen_adapter(root, run_dir, adapter, ref_adapted, slugs, checkpoint_meta,
                            batch_size=args.precompute_batch_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", help="existing artifact directory to resume, or a new explicit run directory")
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--train-views", type=int, default=8)
    parser.add_argument("--validation-views", type=int, default=2)
    parser.add_argument("--precompute-batch-size", type=int, default=8)
    parser.add_argument("--training-batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
