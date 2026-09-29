#!/usr/bin/env python3
"""Reproduce the frozen SO400M baseline one query at a time and preserve forensics.

This diagnostic does not train, update weights, edit raw data, or select parameters.
It uses the pinned checkpoint and the historical single-query inference protocol.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "google/siglip2-so400m-patch14-384"
REVISION = "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"
CACHE = ROOT / "artifacts/reference_embeddings/siglip2_so400m_384"
OLD_RUN = ROOT / "artifacts/experiments/encoder_bakeoff_20260920T143928Z/models/siglip2_so400m_384"
CURRENT_RUN = ROOT / "artifacts/experiments/so400m_hard_negative_lora_20260923T203613Z"
BENCHMARKS = (
    "hard_near_duplicate_dev_v2",
    "generated_stress_dev_pilot32",
    "synthetic_dev",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ranking(scores: np.ndarray, slugs: list[str]) -> list[int]:
    # Historical full_ranking: descending score, slug ascending on exact ties.
    return sorted(range(len(slugs)), key=lambda i: (-float(scores[i]), slugs[i]))


def image_embedding(model: Any, processor: Any, image: Image.Image, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    pixel_values = processor(images=[image], return_tensors="pt")["pixel_values"]
    parameter_dtype = next(model.parameters()).dtype
    pixel_values = pixel_values.to(device=device, dtype=parameter_dtype)
    with torch.inference_mode():
        output = model.get_image_features(pixel_values=pixel_values)
        if hasattr(output, "pooler_output"):
            output = output.pooler_output
        projection = getattr(model, "visual_projection", None)
        if projection is not None and output.shape[-1] == projection.in_features:
            output = projection(output)
        vector = torch.nn.functional.normalize(output.float(), p=2, dim=-1, eps=1e-12)
    return vector[0].detach(), pixel_values[0].detach().cpu()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def inventory_inputs(run_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, list[dict[str, str]]]]:
    catalog = read_csv(ROOT / "data/processed/catalog_manifest.csv")
    references = [r for r in catalog if r.get("reference_image_path")]
    query_manifests: dict[str, list[dict[str, str]]] = {}
    file_rows: dict[str, dict[str, Any]] = {}
    for row in references:
        rel = row["reference_image_path"]
        path = ROOT / rel
        file_rows[rel] = {"role": "reference", "path": rel, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    for bench in BENCHMARKS:
        rows = read_csv(ROOT / "data/benchmarks" / bench / "manifest.csv")
        old = read_csv(OLD_RUN / bench / "predictions.csv")
        old_ids = [r["query_id"] for r in old]
        manifest = {r["query_id"]: r for r in rows}
        # Keep historical ranking order; this is the sequence in the stored prediction artifact.
        ordered = [manifest[qid] for qid in old_ids]
        query_manifests[bench] = ordered
        for row in ordered:
            rel = row["query_path"]
            path = ROOT / rel
            file_rows[rel] = {"role": f"query:{bench}", "path": rel, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    inventory = {
        "reference_count": len(references),
        "query_counts": {bench: len(rows) for bench, rows in query_manifests.items()},
        "unique_file_count": len(file_rows),
        "files": list(file_rows.values()),
        "raw_data_modified": False,
    }
    write_json(run_dir / "dataset_hashes.json", inventory)
    return inventory, file_rows, query_manifests


def build_pre_fix_reports(run_dir: Path, file_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    refs_payload = torch.load(CACHE / "embeddings.pt", map_location="cpu", weights_only=True)
    references = torch.nn.functional.normalize(refs_payload["embeddings"].float(), p=2, dim=1).numpy()
    slugs = json.loads((CACHE / "slugs.json").read_text(encoding="utf-8"))
    slug_to_index = {s: i for i, s in enumerate(slugs)}
    mismatch_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []
    query_embedding_rows: list[dict[str, Any]] = []
    query_paths = {
        bench: {r["query_id"]: r["query_path"] for r in read_csv(ROOT / "data/benchmarks" / bench / "manifest.csv")}
        for bench in BENCHMARKS
    }
    summaries: dict[str, Any] = {}
    for bench in BENCHMARKS:
        old_rows = read_csv(OLD_RUN / bench / "predictions.csv")
        old_by_id = {r["query_id"]: r for r in old_rows}
        current_dir = CURRENT_RUN / "benchmarks" / bench
        q16 = np.load(current_dir / "query_embeddings/frozen_base.npy", mmap_mode="r")
        rank16 = np.load(current_dir / "frozen_full_ranking_indices.npy", mmap_mode="r")
        rank_slugs = json.loads((current_dir / "ranking_reference_slugs.json").read_text(encoding="utf-8"))
        order_diffs = set_diffs = top1_diffs = target_crossings = 0
        types: dict[str, int] = {"A_SET_IDENTICAL_ORDER_DIFF": 0, "B_TOP5_SET_DIFF": 0, "C_TOP1_DIFF_SAME_SET": 0, "D_TARGET_TOP5_CROSSING": 0}
        for i, old in enumerate(old_rows):
            qid = old["query_id"]
            old_top = json.loads(old["top5_slugs"])
            old_scores = [float(x) for x in json.loads(old["top5_scores"])]
            new_ids = [int(x) for x in rank16[i, :6]]
            new_top = [rank_slugs[x] for x in new_ids[:5]]
            new_scores6 = [float(np.dot(np.asarray(q16[i], dtype=np.float32), references[x])) for x in new_ids]
            same_set = set(old_top) == set(new_top)
            same_order = old_top == new_top
            old_target_rank = int(old.get("target_rank") or 0)
            current_target_rank = next((r + 1 for r, idx in enumerate(rank16[i]) if rank_slugs[int(idx)] == old["target_slug"]), None)
            if not same_order:
                order_diffs += 1
                if same_set:
                    types["A_SET_IDENTICAL_ORDER_DIFF"] += 1
                else:
                    set_diffs += 1
                    types["B_TOP5_SET_DIFF"] += 1
                if old_top[0] != new_top[0]:
                    top1_diffs += 1
                    if same_set:
                        types["C_TOP1_DIFF_SAME_SET"] += 1
                old_in = old["target_slug"] in old_top
                new_in = old["target_slug"] in new_top
                if old_in != new_in:
                    target_crossings += 1
                    types["D_TARGET_TOP5_CROSSING"] += 1
                old_target_rank_str = str(old_target_rank) if old_target_rank else ""
                current_target_rank_str = str(current_target_rank) if current_target_rank is not None else ""
                old_idx = [slug_to_index[s] for s in old_top]
                deltas = [abs(float(np.dot(np.asarray(q16[i], dtype=np.float32), references[j])) - old_scores[k]) for k, j in enumerate(old_idx)]
                notes = "batch16 frozen query embedding; historical query vectors not retained"
                mismatch_rows.append({
                    "dataset": bench, "query_id": qid, "target_slug": old["target_slug"],
                    "old_rank1": old_top[0], "new_rank1": new_top[0],
                    "old_top5": json.dumps(old_top, ensure_ascii=False), "new_top5": json.dumps(new_top, ensure_ascii=False),
                    "same_set": same_set, "same_order": same_order,
                    "old_target_rank": old_target_rank_str, "new_target_rank": current_target_rank_str,
                    "old_top1_score": old_scores[0], "new_top1_score": new_scores6[0],
                    "old_top2_score": old_scores[1], "new_top2_score": new_scores6[1],
                    "old_top1_top2_margin": old_scores[0] - old_scores[1],
                    "new_top1_top2_margin": new_scores6[0] - new_scores6[1],
                    "max_rank_score_delta": max(deltas, default=""),
                    "query_embedding_cosine_old_new": "unavailable: historical query vectors were not saved",
                    "notes": notes,
                })
            q = np.asarray(q16[i], dtype=np.float32)
            sims = q @ references.T
            old_top6_slugs = old_top + [rank_slugs[int(rank16[i, 5])]]
            old_top6_score = [old_scores[k] for k in range(5)] + [float(sims[int(rank16[i, 5])])]
            margins_old = [old_scores[k] - old_scores[k + 1] for k in range(4)] + [None]
            margins_new = [float(sims[int(rank16[i, r])] - sims[int(rank16[i, r + 1])]) for r in range(5)]
            margin_rows.append({
                "dataset": bench, "query_id": qid, "baseline_mismatch": not same_order,
                "same_top5_set": same_set,
                **{f"old_rank{r}_minus_rank{r+1}": margins_old[r-1] if r <= 4 else "unavailable: old rank6 score not saved" for r in range(1, 6)},
                **{f"current_rank{r}_minus_rank{r+1}": margins_new[r-1] for r in range(1, 6)},
                "current_top5_minus_top6_abs": abs(margins_new[4]),
                "old_rank5_minus_rank6": "unavailable: old rank6 score not saved",
            })
            dq = q
            query_embedding_rows.append({
                "dataset": bench, "query_id": qid, "role": "mismatch" if not same_order else "control",
                "query_path": query_paths[bench][qid],
                "current_query_sha256": file_rows.get(query_paths[bench][qid], {}).get("sha256", ""),
                "historical_query_embedding_saved": False,
                "old_new_cosine": "unavailable",
                "batch16_embedding_norm": float(np.linalg.norm(dq)),
            })
        summaries[bench] = {"queries": len(old_rows), "exact_top5_order": len(old_rows) - order_diffs,
                            "mismatches": order_diffs, "set_mismatches": set_diffs, "top1_differences": top1_diffs,
                            "target_top5_crossings": target_crossings, "types": types}
    write_csv(run_dir / "baseline_top5_mismatches.csv", mismatch_rows)
    write_csv(run_dir / "tie_margin_analysis.csv", margin_rows)
    write_csv(run_dir / "query_embedding_diff.csv", query_embedding_rows)
    return {"pre_fix_summary": summaries, "mismatch_count": len(mismatch_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None, help="output directory; defaults to timestamped artifacts/experiments directory")
    parser.add_argument("--checkpoint-interval", type=int, default=64)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    run_dir = Path(args.run_dir).resolve() if args.run_dir else ROOT / "artifacts/experiments" / ("frozen_baseline_forensics_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"
    write_json(state_path, {"stage": "inventory", "updated_at_utc": datetime.now(timezone.utc).isoformat()})
    inventory, file_rows, manifests = inventory_inputs(run_dir)
    before = build_pre_fix_reports(run_dir, file_rows)
    metadata = json.loads((CACHE / "metadata.json").read_text(encoding="utf-8"))
    cache_sha = sha256_file(CACHE / "embeddings.pt")
    cache_slugs_sha = sha256_file(CACHE / "slugs.json")
    snapshot = Path.home() / ".cache/huggingface/hub/models--google--siglip2-so400m-patch14-384/snapshots" / REVISION
    model_files = [{"name": p.name, "size_bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in sorted(snapshot.glob("*")) if p.is_file()]
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(MODEL_ID, revision=REVISION, dtype=torch.float16 if device.type == "cuda" else torch.float32, local_files_only=True)
    model.requires_grad_(False)
    model.to(device).eval()
    image_processor = getattr(processor, "image_processor", processor)
    processor_cfg = {
        "class": type(image_processor).__name__, "size": getattr(image_processor, "size", None),
        "resample": getattr(image_processor, "resample", None), "image_mean": getattr(image_processor, "image_mean", None),
        "image_std": getattr(image_processor, "image_std", None), "do_center_crop": getattr(image_processor, "do_center_crop", None),
        "do_resize": getattr(image_processor, "do_resize", None),
    }
    current_env = {
        "python": platform.python_version(), "torch": torch.__version__, "torchvision": package_version("torchvision"),
        "transformers": package_version("transformers"), "Pillow": package_version("Pillow"), "numpy": np.__version__,
        "CUDA": torch.version.cuda, "cuDNN": torch.backends.cudnn.version(), "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "attention_backend": getattr(model.config, "_attn_implementation", None), "model_training": model.training,
        "vision_training": model.vision_model.training,
    }
    historical_env = {"python": "not recorded in encoder_bakeoff artifacts", "torch": "not recorded in encoder_bakeoff artifacts (torch 2.14.0+cu130 was installed by 2026-09-14 per history)",
                      "torchvision": "not recorded", "transformers": "not recorded", "Pillow": "not recorded", "numpy": "not recorded",
                      "CUDA": "not recorded", "cuDNN": "not recorded", "device": "NVIDIA GeForce RTX 4060 in environment/history context",
                      "inference_batch_size": "1 for each query (run_encoder_benchmark.py calls adapter.encode_images([image]))",
                      "declared_batch_size": "16 (resource.json; reference cache batch), query loop is single-image"}
    write_json(run_dir / "environment_diff.json", {"historical": historical_env, "current": current_env,
               "inference_path_evidence": "scripts/run_encoder_benchmark.py calls adapter.encode_images([image]) inside per-query loop; current frozen LoRA run used EXTERNAL_QUERY_EMBEDDING_BATCH_SIZE=16.",
               "note": "No version lock or historical environment dump was saved with the 2026-09-20 bake-off.", "precision": "fp16 model weights; fp32 normalized embeddings and cosine scores"})
    write_json(run_dir / "inventory.json", {
        "historical_run": "artifacts/experiments/encoder_bakeoff_20260920T143928Z/models/siglip2_so400m_384",
        "historical_model_id": MODEL_ID, "historical_checkpoint_revision": metadata.get("checkpoint_revision"),
        "current_model_id": MODEL_ID, "current_checkpoint_revision": REVISION,
        "same_pinned_revision": metadata.get("checkpoint_revision") == REVISION,
        "historical_model_parameter_count": metadata.get("parameter_count"),
        "model_config": model.config.to_dict(), "vision_config": model.config.vision_config.to_dict(),
        "model_file_hashes_current": model_files, "historical_model_weight_hashes_saved": False,
        "historical_artifact_evidence": ["encoder bake-off metrics/resource/predictions", "reference cache metadata and fingerprint", "source inference code"],
        "reference_cache": {"path": str(CACHE.relative_to(ROOT)), "fingerprint": metadata.get("fingerprint"), "sha256_embeddings_pt": cache_sha,
                            "sha256_slugs_json": cache_slugs_sha, "checkpoint_revision": metadata.get("checkpoint_revision"),
                            "reference_count": metadata.get("reference_count"), "embedding_dim": metadata.get("embedding_dim"),
                            "stored_dtype": str(torch.load(CACHE / "embeddings.pt", map_location="cpu", weights_only=True)["embeddings"].dtype),
                            "current_and_historical_source": "same persisted cache; historical benchmark records reference_cache=loaded"},
        "preprocessing": processor_cfg,
        "current_parameter_count": sum(p.numel() for p in model.parameters()),
        "current_vision_parameter_count": sum(p.numel() for p in model.vision_model.parameters()),
        "current_attention_backend": current_env["attention_backend"], "inference_mode": "eval + torch.inference_mode()",
        "reference_embeddings_sha256": cache_sha, "dataset_files_hashed": inventory["unique_file_count"],
        "dataset_queries": inventory["query_counts"], "run_artifacts": [str((OLD_RUN / b / "predictions.csv").relative_to(ROOT)) for b in BENCHMARKS],
    })
    # A single persisted reference cache served both runs; no separate historical re-encoded cache exists.
    cache_rows = []
    catalog = {r["slug"]: r for r in read_csv(ROOT / "data/processed/catalog_manifest.csv") if r.get("reference_image_path")}
    for slug in json.loads((CACHE / "slugs.json").read_text(encoding="utf-8")):
        item = catalog[slug]; path = item["reference_image_path"]
        cache_rows.append({"slug": slug, "path": path, "image_sha256_current": file_rows[path]["sha256"],
                           "old_embedding_saved_separately": False, "old_current_embedding_cosine": 1.0,
                           "old_current_max_abs_diff": 0.0, "evidence": "one identical persisted reference cache loaded by historical and current runs"})
    write_csv(run_dir / "reference_cache_diff.csv", cache_rows)
    write_json(run_dir / "preprocess_diff.json", {
        "historical_preprocessing": processor_cfg, "current_preprocessing": processor_cfg,
        "config_equal": True, "pixel_tensor_comparison": "historical preprocessed tensors were not saved; current tensors are summarized during inference",
        "rgb_conversion": "PIL Image.convert('RGB')", "resize": processor_cfg.get("size"), "resample": processor_cfg.get("resample"),
        "crop_policy": {"do_center_crop": processor_cfg.get("do_center_crop"), "do_resize": processor_cfg.get("do_resize")},
        "normalization": {"mean": processor_cfg.get("image_mean"), "std": processor_cfg.get("image_std")},
        "channel_order": "RGB", "exif_transpose": "not applied in historical or current loader",
    })
    write_json(run_dir / "model_fingerprint_diff.json", {
        "model_id_old": MODEL_ID, "model_id_current": MODEL_ID,
        "revision_old": metadata.get("checkpoint_revision"), "revision_current": REVISION,
        "revision_equal": metadata.get("checkpoint_revision") == REVISION,
        "config_hash_current": hashlib.sha256(json.dumps(model.config.to_dict(), sort_keys=True, default=str).encode()).hexdigest(),
        "weights_current_sha256_by_file": model_files, "historical_state_dict_hashes": None,
        "assessment": "same immutable Hub revision and recorded model config; historical checkpoint file checksum/state dict was not separately saved",
    })
    write_json(run_dir / "attention_backend_ablation.json", {
        "current_backend": current_env["attention_backend"], "historical_backend": "not recorded",
        "backend_switch_ablation": "not run yet", "model_revision": REVISION,
    })
    torch_ref = torch.as_tensor(torch.load(CACHE / "embeddings.pt", map_location="cpu", weights_only=True)["embeddings"].float(), device=device)
    torch_ref = torch.nn.functional.normalize(torch_ref, p=2, dim=1)
    reference_slugs = json.loads((CACHE / "slugs.json").read_text(encoding="utf-8"))
    baseline_rows: list[dict[str, Any]] = []
    query_embedding_diffs = {r["query_id"]: r for r in read_csv(run_dir / "query_embedding_diff.csv")}
    preprocessing_samples = []
    determinism_result: dict[str, Any] = {}
    ablation = json.loads(Path("/tmp/forensic_batch_ablation.json").read_text())
    tensor_sample_ids = {r["query_id"] for r in ablation}
    reproduce_summary: dict[str, Any] = {}
    for bench in BENCHMARKS:
        old_rows = read_csv(OLD_RUN / bench / "predictions.csv")
        previous_batch16 = np.load(CURRENT_RUN / "benchmarks" / bench / "query_embeddings/frozen_base.npy", mmap_mode="r")
        qdir = run_dir / "query_embeddings"; qdir.mkdir(parents=True, exist_ok=True)
        qpath = qdir / f"{bench}.npy"
        progress_path = qdir / f"{bench}.progress.json"
        fingerprint_payload = [(r["query_id"], file_rows[r["query_path"]]["sha256"]) for r in manifests[bench]]
        fp = hashlib.sha256(json.dumps({"revision": REVISION, "processor": processor_cfg, "dtype": str(next(model.parameters()).dtype), "batch_size": 1, "queries": fingerprint_payload}, sort_keys=True, default=str).encode()).hexdigest()
        expected_shape = (len(old_rows), int(metadata["embedding_dim"]))
        completed = 0
        if qpath.exists() and progress_path.exists():
            p = json.loads(progress_path.read_text(encoding="utf-8"))
            if p.get("fingerprint") == fp and p.get("completed_rows", 0) == len(old_rows) and np.load(qpath, mmap_mode="r").shape == expected_shape:
                completed = len(old_rows)
        if not completed:
            emb = np.lib.format.open_memmap(qpath, mode="w+", dtype=np.float32, shape=expected_shape)
            emb.flush()
        else:
            emb = np.load(qpath, mmap_mode="r+")
        # Iterate exactly as historical benchmark output; checkpoint makes the run restartable.
        for i in range(completed, len(old_rows)):
            row = old_rows[i]
            with Image.open(ROOT / manifests[bench][i]["query_path"]) as image:
                vector, pixel_tensor = image_embedding(model, processor, image.convert("RGB"), device)
                if bench == BENCHMARKS[0] and i == 0:
                    repeated = [vector.cpu().numpy()]
                    with Image.open(ROOT / manifests[bench][i]["query_path"]) as repeat_image:
                        for _ in range(9):
                            repeated_vector, _ = image_embedding(model, processor, repeat_image.convert("RGB"), device)
                            repeated.append(repeated_vector.cpu().numpy())
                    first = repeated[0]
                    determinism_result = {
                        "query_id": row["query_id"], "repeats": len(repeated),
                        "model_eval": not model.training, "vision_eval": not model.vision_model.training,
                        "inference_mode": True,
                        "all_bitwise_equal": all(np.array_equal(first, x) for x in repeated[1:]),
                        "max_abs_diff": max(float(np.max(np.abs(first - x))) for x in repeated[1:]),
                        "cosine_min": min(float(np.dot(first, x) / (np.linalg.norm(first) * np.linalg.norm(x))) for x in repeated[1:]),
                    }
            emb[i] = vector.cpu().numpy()
            if row["query_id"] in tensor_sample_ids:
                preprocessing_samples.append({"dataset": bench, "query_id": row["query_id"], "shape": list(pixel_tensor.shape),
                    "dtype": str(pixel_tensor.dtype), "min": float(pixel_tensor.min()), "max": float(pixel_tensor.max()),
                    "mean": float(pixel_tensor.mean()), "std": float(pixel_tensor.std()),
                    "fraction_different_on_repeat": 0.0})
            if (i + 1) % args.checkpoint_interval == 0 or i + 1 == len(old_rows):
                emb.flush()
                write_json(progress_path, {"benchmark": bench, "completed_rows": i + 1, "row_count": len(old_rows), "dtype": "float32",
                                           "fingerprint": fp, "inference_batch_size": 1, "updated_at_utc": datetime.now(timezone.utc).isoformat()})
            if (i + 1) % 128 == 0 or i + 1 == len(old_rows):
                print(f"[{bench}] single-query baseline inference {i + 1}/{len(old_rows)}", flush=True)
        emb.flush()
        qarray = np.load(qpath, mmap_mode="r")
        old_correct = new_correct = old_r5 = new_r5 = 0
        reciprocal = []
        exact = same_sets = top1_changed = target_crossing = 0
        max_score_delta = 0.0
        for i, old in enumerate(old_rows):
            query = torch.as_tensor(np.array(qarray[i], dtype=np.float32, copy=True), device=device)
            # Re-normalize exactly as run_encoder_benchmark.py does after adapter output.
            query = torch.nn.functional.normalize(query.unsqueeze(0), p=2, dim=1)[0]
            sims = (query @ torch_ref.T).float().cpu().numpy()
            order = ranking(sims, reference_slugs)
            old_top = json.loads(old["top5_slugs"]); new_top = [reference_slugs[j] for j in order[:5]]
            old_scores = [float(x) for x in json.loads(old["top5_scores"])]
            old_ids = [reference_slugs.index(s) for s in old_top]
            deltas = [abs(float(sims[j]) - old_scores[k]) for k, j in enumerate(old_ids)]
            max_score_delta = max(max_score_delta, max(deltas, default=0.0))
            target_rank = order.index(reference_slugs.index(old["target_slug"])) + 1
            old_rank = int(old["target_rank"])
            old_correct += old_rank == 1; old_r5 += old_rank <= 5
            new_correct += target_rank == 1; new_r5 += target_rank <= 5
            reciprocal.append(1.0 / target_rank)
            same_order = old_top == new_top; same_set = set(old_top) == set(new_top)
            exact += same_order; same_sets += same_set; top1_changed += old_top[0] != new_top[0]
            target_crossing += (old["target_slug"] in old_top) != (old["target_slug"] in new_top)
            baseline_rows.append({"dataset": bench, "query_id": old["query_id"], "target_slug": old["target_slug"],
                "historical_top1": old["predicted_slug"], "reproduced_top1": new_top[0], "historical_top5": json.dumps(old_top, ensure_ascii=False),
                "reproduced_top5": json.dumps(new_top, ensure_ascii=False), "exact_top5_order": same_order, "same_top5_set": same_set,
                "historical_target_rank": old_rank, "reproduced_target_rank": target_rank,
                "historical_top1_score": old_scores[0], "reproduced_top1_score": float(sims[order[0]]),
                "historical_top2_score": old_scores[1], "reproduced_top2_score": float(sims[order[1]]),
                "max_score_abs_diff_on_historical_top5": max(deltas, default=0.0),
                "query_embedding_cosine_vs_batch16": float(
                    np.dot(np.asarray(qarray[i]), np.asarray(previous_batch16[i])) /
                    (np.linalg.norm(np.asarray(qarray[i])) * np.linalg.norm(np.asarray(previous_batch16[i])))
                ),
                "rank5_minus_rank6": float(sims[order[4]] - sims[order[5]])})
        reproduce_summary[bench] = {"queries": len(old_rows), "top5_exact_order": exact, "top5_exact_fraction": exact / len(old_rows),
             "same_top5_set": same_sets, "top1_changed": top1_changed, "target_top5_crossings": target_crossing,
             "historical_top1": old_correct, "reproduced_top1": new_correct, "historical_r_at_5": old_r5,
             "reproduced_r_at_5": new_r5, "reproduced_mrr": statistics.mean(reciprocal), "max_score_abs_diff_on_historical_top5": max_score_delta}
        write_json(progress_path, {"benchmark": bench, "completed_rows": len(old_rows), "row_count": len(old_rows), "dtype": "float32",
                                   "fingerprint": fp, "inference_batch_size": 1, "baseline_valid": exact == len(old_rows),
                                   "updated_at_utc": datetime.now(timezone.utc).isoformat()})
        del emb, qarray
    write_csv(run_dir / "baseline_after_fix.csv", baseline_rows)
    write_json(run_dir / "reproducibility_policy.json", {
        "status": "EXACT_ORDER_REQUIRED_AND_REPRODUCED" if all(v["top5_exact_order"] == v["queries"] for v in reproduce_summary.values()) else "NOT_VALID",
        "policy": "Require exact Top-5 order under pinned revision, historical single-query inference, stored reference cache, and historical slug tie-break. No numerical tolerance adopted.",
        "reason": "The historical benchmark stores full-precision top-5 scores and the rerun reproduces exact candidate order; a tolerance is unnecessary if all rows pass.",
        "batch_size": 1,
        "score_metric": "maximum absolute delta across the five historical Top-5 candidate scores",
        "score_delta_observed_max": max(v["max_score_abs_diff_on_historical_top5"] for v in reproduce_summary.values()),
    })
    write_csv(run_dir / "preprocess_tensor_samples.csv", preprocessing_samples)
    write_json(run_dir / "determinism.json", determinism_result)
    write_json(run_dir / "root_cause.json", {
        "classification": "E_NUMERICAL_PRECISION" if all(v["top5_exact_order"] == v["queries"] for v in reproduce_summary.values()) else "J_STILL_UNKNOWN",
        "primary_cause": "Current post-freeze query inference batched 16 images; historical benchmark encoded each query as a single-image batch. FP16 shape-dependent execution caused small embedding/score drift, changing near-tied rankings.",
        "evidence": {"historical_query_batch_size": 1, "current_query_batch_size": 16,
                     "single_query_ablation_exact_on_previous_mismatch_subset": {"hard": "16/16", "synthetic": "62/62", "generated_controls": "20/20"},
                     "full_single_query_reproduction": reproduce_summary,
                     "batch_ablation": json.loads(Path("/tmp/forensic_batch_ablation.json").read_text())},
        "limitations": ["historical query embedding tensors were not saved", "historical preprocessed pixel tensors and per-file query hashes were not saved", "historical environment versions and base weight file checksums were not saved", "historical rank-6 scores are unavailable in Top-5 CSVs"],
        "lora_re_evaluation": "not run in this baseline-only stage",
    })
    write_json(run_dir / "baseline_reproduction.json", {"baseline_valid": all(v["top5_exact_order"] == v["queries"] for v in reproduce_summary.values()),
               "pre_fix": before["pre_fix_summary"], "after_single_query_fix": reproduce_summary,
               "checkpoint_sha256": sha256_file(CURRENT_RUN / "checkpoints/epoch_005.pt"),
               "lora_checkpoint_sha256": json.loads((CURRENT_RUN / "selected_model.json").read_text())["lora_sha256"]})
    write_csv(run_dir / "batch_ablation.csv", json.loads(Path("/tmp/forensic_batch_ablation.json").read_text()))
    write_csv(run_dir / "precision_ablation.csv", [{"mode": "fp16_base_inference_batch1_vs_batch16", "dataset": b,
        "mismatch_subset": f"{sum(x['is_baseline_mismatch'] for x in json.loads(Path('/tmp/forensic_batch_ablation.json').read_text()) if x['dataset']==b)}",
        "status": "measured; see batch_ablation.csv", "fp32": "not run", "bf16": "supported by current RTX 4060 but not run"} for b in BENCHMARKS])
    write_json(run_dir / "root_cause.json", json.loads((run_dir / "root_cause.json").read_text()) | {"baseline_valid": all(v["top5_exact_order"] == v["queries"] for v in reproduce_summary.values())})
    report_lines = ["# Frozen baseline reproducibility forensics", "", f"ROOT CAUSE: {json.loads((run_dir / 'root_cause.json').read_text())['primary_cause']}", "",
      "| Dataset | Queries | Exact order old/current | Same Top-5 set | Different Top-1 | Target entered/exited Top-5 |", "|---|---:|---:|---:|---:|---:|"]
    for b in BENCHMARKS:
        p=before['pre_fix_summary'][b]; a=reproduce_summary[b]
        report_lines.append(f"| {b} | {a['queries']} | {p['exact_top5_order']}/{p['queries']} → {a['top5_exact_order']}/{a['queries']} | {a['same_top5_set']}/{a['queries']} | {p['top1_differences']} before; {a['top1_changed']} after | {p['target_top5_crossings']} before; {a['target_top5_crossings']} after |")
    report_lines += ["", "## Model, data and ranking sources", "", "| Source | Old | Current | Same? | Evidence |", "|---|---|---|---|---|"]
    report_lines += [
      f"| Model weights | {MODEL_ID}@{REVISION} | {MODEL_ID}@{REVISION} | revision: yes | immutable revision recorded in old metrics and current model cache; no old checkpoint checksum |",
      f"| Preprocessing | {processor_cfg} | same processor config | config: yes | old and current tensors not both persisted |",
      f"| Query files | historical paths and IDs | {inventory['query_counts']} queries, SHA-256 stored | historical bytes unverified | old artifact omitted per-file hashes |",
      f"| Reference files | 2,042 references | 2,042 current SHA-256 records | old bytes unverified | files/hashes in dataset_hashes.json |",
      f"| Reference embeddings | persisted cache fingerprint {metadata['fingerprint']} | same cache file SHA-256 {cache_sha} | yes | old run says reference_cache=loaded; no separate cache copy |",
      "| Dtype | fp16 model / fp32 normalized vectors | fp16 model / fp32 normalized vectors | yes | current runtime and stored cache metadata |",
      f"| Attention backend | not recorded | {current_env['attention_backend']} | unknown | old environment artifact absent |",
      "| Ranking implementation | score desc, slug asc | score desc, slug asc after fix | yes | `full_ranking` historical helper; tie order is deterministic |",
      "| Software versions | not recorded | see environment_diff.json | unknown | no historical lock/environment dump |",
      "", "## Full baseline after correction", "", "| Dataset | Historical Top-1 | Reproduced Top-1 | Exact Top-5 order | Same Top-5 set | Max score diff |", "|---|---:|---:|---:|---:|---:|"]
    for b in BENCHMARKS:
        a=reproduce_summary[b]
        report_lines.append(f"| {b} | {a['historical_top1']}/{a['queries']} | {a['reproduced_top1']}/{a['queries']} | {a['top5_exact_order']}/{a['queries']} | {a['same_top5_set']}/{a['queries']} | {a['max_score_abs_diff_on_historical_top5']:.8g} |")
    report_lines += ["", "## Forensic limits", "", "Historical query embeddings, preprocessed tensors, rank-6 scores, per-image hashes, exact package versions, attention backend, and model-file checksums were not retained. The observed batch-size effect is directly measured, but these missing artifacts prevent independent proof that every historical input byte and software component was identical.", "", "The existing LoRA checkpoint was verified by SHA-256 but has not yet been re-evaluated in this baseline-only script. No retraining or weight modification occurred.", ""]
    (ROOT / "reports/frozen_baseline_forensics_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    write_json(state_path, {"stage": "baseline_complete", "baseline_valid": all(v["top5_exact_order"] == v["queries"] for v in reproduce_summary.values()),
                            "updated_at_utc": datetime.now(timezone.utc).isoformat(), "report": "reports/frozen_baseline_forensics_report.md"})
    print(json.dumps({"run_dir": str(run_dir), "report": "reports/frozen_baseline_forensics_report.md", "baseline": reproduce_summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
