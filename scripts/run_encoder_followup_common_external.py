#!/usr/bin/env python3
"""Run the frozen, common external pass for the encoder follow-up milestone.

The script refuses to start unless every candidate has an internal-only frozen
selection and the canonical batch-1 frozen encoder reproduces all three stored
Top-5 baselines exactly. Query embedding files are resumable through the
existing per-split progress checkpoints.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_so400m_hard_negative_lora as runner  # noqa: E402
from recognition.encoder_followup_protocol import (  # noqa: E402
    CANONICAL_QUERY_BATCH_SIZE,
    assert_immutable_query_ids,
    fixed_half_fusion,
    original_error_id_fingerprint,
    require_canonical_query_batch_size,
    require_exact_baseline_reproduction,
    require_frozen_before_external,
    require_model_identity,
    transition_counts,
)
from recognition.so400m_lora import (  # noqa: E402
    BASE_MODEL_ID,
    BASE_REVISION,
    HIDDEN_SIZE,
    LORA_TARGETS,
    lora_sha256,
    load_lora_state_dict,
)

RUN_ROOT = ROOT / "artifacts/experiments/encoder_followup_r16_capturev2_last2_20260924T083837Z"
R8_SOURCE = ROOT / "artifacts/experiments/so400m_hard_negative_lora_20260923T203613Z"
EXTERNAL_ROOT = RUN_ROOT / "external_eval"
SPLITS = runner.BENCHMARKS
EXPECTED = {
    "hard_near_duplicate_dev_v2": 1150,
    "generated_stress_dev_pilot32": 128,
    "synthetic_dev": 4084,
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{__import__('os').getpid()}")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temp = path.with_name(path.name + f".tmp.{__import__('os').getpid()}")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _lora_state(checkpoint_path: Path) -> tuple[dict[str, torch.Tensor], str]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("lora_state")
    if not isinstance(state, dict):
        raise RuntimeError(f"checkpoint has no LoRA-only state: {checkpoint_path}")
    return state, lora_sha256(state)


def _frozen_preflight(
    catalog: list[dict[str, str]], slugs: list[str], base_refs: np.ndarray,
    model: torch.nn.Module, processor: Any, device: torch.device,
) -> dict[str, Any]:
    """Check all canonical Top-5 lists before adapted scores are evaluated."""
    records, _ = runner.load_external_inputs(ROOT)
    results: dict[str, Any] = {}
    preflight_dir = EXTERNAL_ROOT / "canonical_preflight"
    selected = {"lora_sha256": read_json(R8_SOURCE / "selected_model.json")["lora_sha256"]}
    for split in SPLITS:
        base_q, _adapted_q = runner._encode_query_pair(
            ROOT, preflight_dir, split, records[split], model, processor, device,
            CANONICAL_QUERY_BATCH_SIZE, selected["lora_sha256"],
        )
        scores, ranks = runner._full_rankings(base_q, base_refs, device)
        saved = runner._baseline_top5_rows(ROOT, split)
        exact = 0
        same_set = 0
        max_score_delta = 0.0
        examples = []
        slug_to_index = {slug: index for index, slug in enumerate(slugs)}
        for index, row in enumerate(records[split]):
            old = saved.get(str(row["query_id"]))
            if old is None:
                raise RuntimeError(f"canonical frozen predictions missing for {split}/{row['query_id']}")
            expected = runner._as_json_list(old.get("top5_slugs", ""))
            actual = [slugs[int(item)] for item in ranks[index, :5]]
            if runner.exact_top5_reproduction(actual, expected):
                exact += 1
            elif len(examples) < 20:
                examples.append({"query_id": str(row["query_id"]), "expected": expected, "actual": actual})
            if len(expected) == 5 and set(actual) == set(expected):
                same_set += 1
            historical_scores = runner._as_json_list(old.get("top5_scores", ""))
            if len(historical_scores) == 5 and len(expected) == 5:
                max_score_delta = max(max_score_delta, *(
                    abs(float(scores[index, slug_to_index[slug]]) - float(value))
                    for slug, value in zip(expected, historical_scores)
                ))
            else:
                max_score_delta = float("inf")
        results[split] = {
            "queries": len(records[split]),
            "top5_exact_order": exact,
            "same_top5_set": same_set,
            "max_score_abs_diff_on_historical_top5": max_score_delta,
            "examples_first_20": examples,
        }
        write_json(preflight_dir / f"{split}_baseline_reproduction.json", results[split])
        print(f"[canonical preflight] {split}: {exact}/{len(records[split])} exact Top-5", flush=True)
    require_exact_baseline_reproduction(results, EXPECTED, max_score_abs_diff_tolerance=3e-6)
    # The preflight also encoded the frozen R8+Capture-v2 candidate. Reuse
    # those exact per-query tensors in its later full evaluation instead of
    # paying for another batch-1 encoder pass.
    r8_eval_dir = EXTERNAL_ROOT / "r8_validated"
    for split in SPLITS:
        source_dir = preflight_dir / "benchmarks" / split / "query_embeddings"
        target_dir = r8_eval_dir / "benchmarks" / split / "query_embeddings"
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("frozen_base.npy", "selected_lora.npy", "progress.json"):
            source = source_dir / filename
            target = target_dir / filename
            if source.is_file() and not target.exists():
                shutil.copy2(source, target)
    write_json(EXTERNAL_ROOT / "canonical_dynamic_preflight.json", {
        "query_batch_size": CANONICAL_QUERY_BATCH_SIZE,
        "reference_precompute_batch_size": int(runner.CONFIG["reference_embedding_batch_size"]),
        "reproduction": results,
        "passed": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    return {"records": records, "reproduction": results}


def _prepare_candidate(name: str, rank: int, alpha: int, checkpoint: Path,
                       selected_path: Path, catalog: list[dict[str, str]],
                       slugs: list[str], image_hashes: dict[str, str],
                       base_meta: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    runner.CONFIG["rank"] = rank
    runner.CONFIG["alpha"] = alpha
    model, processor, _modules, _checkpointing = runner.load_model(device)
    state, expected_hash = _lora_state(checkpoint)
    selected = read_json(selected_path)
    if selected.get("lora_sha256") != expected_hash:
        raise RuntimeError(f"frozen {name} checkpoint hash changed: {expected_hash}")
    load_lora_state_dict(model, state)
    if lora_sha256(model) != expected_hash:
        raise RuntimeError(f"loaded {name} weights do not match the frozen checkpoint hash")
    model.requires_grad_(False)
    model.eval()
    eval_dir = EXTERNAL_ROOT / name
    eval_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = runner.adapted_reference_fingerprint(
        str(base_meta["fingerprint"]), expected_hash, slugs, image_hashes,
        query_batch_size=CANONICAL_QUERY_BATCH_SIZE,
    )
    cache_dir = eval_dir / "adapted_reference_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "embeddings.npy"
    cache_meta_path = cache_dir / "metadata.json"
    cached_refs = None
    if cache_path.is_file() and cache_meta_path.is_file() and (cache_dir / "slugs.json").is_file():
        cache_meta = read_json(cache_meta_path)
        cached_slugs = json.loads((cache_dir / "slugs.json").read_text(encoding="utf-8"))
        if (
            cache_meta.get("fingerprint") == fingerprint
            and cache_meta.get("lora_checkpoint_sha256") == expected_hash
            and int(cache_meta.get("query_batch_size", -1)) == CANONICAL_QUERY_BATCH_SIZE
            and int(cache_meta.get("reference_count", -1)) == len(slugs)
            and int(cache_meta.get("embedding_dim", -1)) == HIDDEN_SIZE
            and cached_slugs == slugs
        ):
            cached_refs = np.load(cache_path).astype(np.float32)
            print(f"[external] reusing fingerprint-matched adapted reference cache for {name}", flush=True)
    if cached_refs is None:
        adapted_refs = runner.encode_references(
            catalog, ROOT, model, processor, device,
            int(runner.CONFIG["reference_embedding_batch_size"]),
        ).astype(np.float32)
    else:
        adapted_refs = cached_refs
    if adapted_refs.shape != (len(slugs), HIDDEN_SIZE):
        raise RuntimeError(f"adapted reference dimensions do not match for {name}")
    if not np.allclose(np.linalg.norm(adapted_refs, axis=1), 1.0, atol=2e-3):
        raise RuntimeError(f"adapted reference embeddings are not normalized for {name}")

    frozen_selected = dict(selected)
    frozen_selected.update({"stage": "frozen", "lora_sha256": expected_hash,
                            "canonical_query_batch_size": CANONICAL_QUERY_BATCH_SIZE})
    policy = {
        "checkpoint_sha256": expected_hash,
        "canonical_query_batch_size": CANONICAL_QUERY_BATCH_SIZE,
        "canonical_benchmark_protocol": runner.CANONICAL_EXTERNAL_PROTOCOL,
        "selected_by": "internal validation only",
        "benchmark_data_used_for_selection": False,
        "frozen": True,
    }
    write_json(eval_dir / "selected_model.json", frozen_selected)
    write_json(eval_dir / "selected_policy.json", policy)

    if cached_refs is None:
        np.save(cache_dir / "embeddings.npy", adapted_refs.astype(np.float16))
    write_json(cache_dir / "slugs.json", slugs)
    cache = {
        "lora_checkpoint_sha256": expected_hash,
        "fingerprint": fingerprint,
        "query_batch_size": CANONICAL_QUERY_BATCH_SIZE,
        "reference_count": len(slugs),
        "embedding_dim": HIDDEN_SIZE,
        "reference_precompute_batch_size": int(runner.CONFIG["reference_embedding_batch_size"]),
    }
    write_json(cache_dir / "metadata.json", cache)
    return {
        "name": name, "rank": rank, "alpha": alpha, "model": model,
        "processor": processor, "selected": frozen_selected,
        "adapted_refs": adapted_refs, "cache": cache, "eval_dir": eval_dir,
        "checkpoint_sha256": expected_hash,
    }


def _run_candidate(candidate: Mapping[str, Any], catalog: list[dict[str, str]],
                   slugs: list[str], base_refs: np.ndarray,
                   device: torch.device) -> dict[str, Any]:
    name = str(candidate["name"])
    summary_path = Path(candidate["eval_dir"]) / "external_summary.json"
    if summary_path.is_file():
        print(f"[external] reusing completed candidate {name}", flush=True)
        cached = read_json(summary_path)
        records, _ = runner.load_external_inputs(ROOT)
        cached["benchmark_records"] = records
        cached["run_dir"] = str(candidate["eval_dir"])
        return cached
    require_frozen_before_external({
        "stage": "frozen",
        "checkpoint_sha256": candidate["checkpoint_sha256"],
    })
    require_model_identity(candidate["checkpoint_sha256"], candidate["cache"]["lora_checkpoint_sha256"])
    eval_dir = Path(candidate["eval_dir"])
    completed_files = all(
        (eval_dir / "benchmarks" / split / filename).is_file()
        for split in SPLITS
        for filename in ("metrics.json", "production_predictions.csv", "embedding_margin_analysis.csv", "base_top5_reproduction.json")
    ) and (eval_dir / "latency_summary.csv").is_file()
    if completed_files:
        print(f"[external] reconstructing completed per-split outputs for {name}", flush=True)
        metrics = {split: read_json(eval_dir / "benchmarks" / split / "metrics.json") for split in SPLITS}
        predictions = {split: read_csv(eval_dir / "benchmarks" / split / "production_predictions.csv") for split in SPLITS}
        latency = read_csv(eval_dir / "latency_summary.csv")
        reproduction = {split: read_json(eval_dir / "benchmarks" / split / "base_top5_reproduction.json") for split in SPLITS}
        write_json(eval_dir / "baseline_reproduction.json", {
            "base_encoder_query_top5_reproduction": {split: row["exact_top5"] for split, row in reproduction.items()},
            "exact_top5_orders": all(row["exact_reproduction"] for row in reproduction.values()),
        })
        margin_rows = [
            {"model": name, **row}
            for split in SPLITS
            for row in read_csv(eval_dir / "benchmarks" / split / "embedding_margin_analysis.csv")
        ]
        write_csv(eval_dir / "embedding_margin_analysis.csv", margin_rows)
        results = {
            "metrics": metrics,
            "production_predictions": predictions,
            "sift_cache_fingerprint": metrics[SPLITS[0]]["frozen_reference_sift_cache_fingerprint"],
            "reference_cache_path": "",
            "base_encoder_query_top5_reproduced": all(row["exact_reproduction"] for row in reproduction.values()),
            "latency": latency,
            "benchmark_count": len(SPLITS),
        }
    else:
        results = runner.run_external_benchmarks(
            ROOT, eval_dir, catalog, slugs, base_refs,
            candidate["adapted_refs"], candidate["model"], candidate["processor"],
            device, candidate["selected"], candidate["cache"],
        )
    records, _ = runner.load_external_inputs(ROOT)
    results["benchmark_records"] = records
    results["run_dir"] = str(candidate["eval_dir"])
    persisted = {key: value for key, value in results.items() if key not in {"benchmark_records", "run_dir"}}
    write_json(summary_path, persisted)
    return results


def _select_best_lora(r8_metrics: Mapping[str, Any], r16_metrics: Mapping[str, Any]) -> str:
    r8 = r8_metrics["validation_adapted"]
    r16 = r16_metrics["validation_adapted"]
    frozen = r8_metrics["validation_frozen"]
    eligible = {
        "r8": float(r8["full_catalog_top1"]) >= float(frozen["full_catalog_top1"]) - 0.01,
        "r16": float(r16["full_catalog_top1"]) >= float(frozen["full_catalog_top1"]) - 0.01,
    }
    if not any(eligible.values()):
        raise RuntimeError("neither R8 nor R16 passes the internal full-catalog guard")
    if not eligible["r8"]:
        return "r16"
    if not eligible["r16"]:
        return "r8"
    # Treat changes smaller than 0.2 percentage points (roughly one view in
    # the 658 same-family validation views) as practically tied; keep R8 then.
    if float(r16["same_family_top1"]) <= float(r8["same_family_top1"]) + 0.002:
        return "r8"
    return "r16"


def _complementarity_and_oracle(results: Mapping[str, Mapping[str, Any]],
                                baseline_error_ids: list[str],
                                frozen_generated: list[Mapping[str, Any]]) -> None:
    comp_rows = []
    oracle_rows = []
    target_by_id = {str(row["query_id"]): str(row["target_slug"]) for row in frozen_generated}
    for model_name, result in results.items():
        for split, predictions in result["production_predictions"].items():
            if split not in SPLITS:
                continue
            prod = [
                {
                    "base": str(row["frozen_production_top1"]),
                    "adapted": str(row["adapted_production_top1"]),
                    "target": str(row["target_slug"]),
                }
                for row in predictions
            ]
            baseline_wrong_adapted_right = sum(row["base"] != row["target"] and row["adapted"] == row["target"] for row in prod)
            baseline_right_adapted_wrong = sum(row["base"] == row["target"] and row["adapted"] != row["target"] for row in prod)
            comp_rows.append({
                "model": model_name, "benchmark": split, "view": "production",
                "queries": len(prod),
                "baseline_wrong_adapted_right": baseline_wrong_adapted_right,
                "baseline_right_adapted_wrong": baseline_right_adapted_wrong,
                "oracle_either_correct": sum(row["base"] == row["target"] or row["adapted"] == row["target"] for row in prod),
            })
            bench_dir = Path(result["run_dir"]) / "benchmarks" / split
            frozen_ranks = np.load(bench_dir / "frozen_full_ranking_indices.npy", mmap_mode="r")
            adapted_ranks = np.load(bench_dir / "adapted_full_ranking_indices.npy", mmap_mode="r")
            ref_slugs = json.loads((bench_dir / "ranking_reference_slugs.json").read_text(encoding="utf-8"))
            target_rows = result["benchmark_records"][split]
            image_base = [ref_slugs[int(ranking[0])] for ranking in frozen_ranks]
            image_adapted = [ref_slugs[int(ranking[0])] for ranking in adapted_ranks]
            image_targets = [str(row["target_slug"]) for row in target_rows]
            image_rescued = sum(base != target and adapted == target for base, adapted, target in zip(image_base, image_adapted, image_targets))
            image_broken = sum(base == target and adapted != target for base, adapted, target in zip(image_base, image_adapted, image_targets))
            comp_rows.append({
                "model": model_name, "benchmark": split, "view": "image_only",
                "queries": len(target_rows), "baseline_wrong_adapted_right": image_rescued,
                "baseline_right_adapted_wrong": image_broken,
                "oracle_either_correct": sum(base == target or adapted == target for base, adapted, target in zip(image_base, image_adapted, image_targets)),
            })
        generated = result["production_predictions"]["generated_stress_dev_pilot32"]
        generated_map = {str(row["query_id"]): row for row in generated}
        margins_path = Path(result["run_dir"]) / "benchmarks/generated_stress_dev_pilot32/embedding_margin_analysis.csv"
        margin_map = {str(row["query_id"]): row for row in read_csv(margins_path)}
        for query_id in baseline_error_ids:
            row = generated_map[query_id]
            target = target_by_id[query_id]
            incumbent = str(row["frozen_production_top1"])
            margin = margin_map[query_id]
            delta = float(margin["adapted_target_minus_frozen_incumbent_similarity"])
            oracle_rows.append({
                "model": model_name, "query_id": query_id,
                "target_slug": target, "frozen_production_incumbent": incumbent,
                "adapted_target_minus_incumbent_similarity": delta,
                "target_score_relation": "win" if delta > 0 else "loss" if delta < 0 else "tie",
                "adapted_production_correct": row["adapted_production_top1"] == target,
            })
    write_csv(EXTERNAL_ROOT / "complementarity.csv", comp_rows)
    write_csv(EXTERNAL_ROOT / "oracle_25_errors.csv", oracle_rows)


def _run_fixed_fusion(best_name: str, best: Mapping[str, Any],
                      records: Mapping[str, list[dict[str, Any]]],
                      base_refs: np.ndarray, slugs: list[str], device: torch.device) -> list[dict[str, Any]]:
    index = {slug: position for position, slug in enumerate(slugs)}
    rows = []
    candidate_dir = Path(best["eval_dir"])
    for split in SPLITS:
        query_dir = candidate_dir / "benchmarks" / split / "query_embeddings"
        base_q = np.load(query_dir / "frozen_base.npy", mmap_mode="r")
        adapted_q = np.load(query_dir / "selected_lora.npy", mmap_mode="r")
        adapted_refs = np.load(candidate_dir / "adapted_reference_cache/embeddings.npy", mmap_mode="r").astype(np.float32)
        base_sims = np.asarray(base_q, dtype=np.float32) @ np.asarray(base_refs, dtype=np.float32).T
        adapted_sims = np.asarray(adapted_q, dtype=np.float32) @ adapted_refs.T
        for row_index, record in enumerate(records[split]):
            fused = fixed_half_fusion(base_sims[row_index].tolist(), adapted_sims[row_index].tolist())
            order = np.argsort(-np.asarray(fused), kind="stable")
            target_index = index[str(record["target_slug"])]
            target_rank = int(np.flatnonzero(order == target_index)[0]) + 1
            rows.append({
                "model": best_name,
                "benchmark": split,
                "query_id": str(record["query_id"]),
                "query_batch_size": CANONICAL_QUERY_BATCH_SIZE,
                "fusion_weights": "0.5_frozen+0.5_adapted",
                "normalization": "per-query min-max over full catalog for each encoder independently",
                "target_rank": target_rank,
                "top1_correct": target_rank == 1,
                "top5_correct": target_rank <= 5,
            })
        del base_sims, adapted_sims
    write_csv(EXTERNAL_ROOT / "fusion_50_50.csv", rows)
    return rows


def _summary_table(r8: Mapping[str, Any], r16: Mapping[str, Any], best_name: str,
                   results: Mapping[str, Mapping[str, Any]], fusion_rows: list[Mapping[str, Any]],
                   r8_internal: Mapping[str, Any], r16_internal: Mapping[str, Any]) -> list[dict[str, Any]]:
    frozen_result = results["r8_validated"]
    all_methods = [
        ("Frozen", None, None),
        ("Validated LoRA r8", "r8_validated", r8_internal),
        ("LoRA r16", "r16", r16_internal),
        ("Best LoRA + Capture-v2", best_name, r16_internal if best_name == "r16" else r8_internal),
    ]
    rows = []
    for label, key, internal in all_methods:
        result = frozen_result if key is None else results[key]
        hard = result["metrics"]["hard_near_duplicate_dev_v2"]
        generated = result["metrics"]["generated_stress_dev_pilot32"]
        generated_rows = result["production_predictions"]["generated_stress_dev_pilot32"]
        if key is None:
            prod_gen = generated["frozen_production_top1_correct"]
            generated_hard = sum(row["frozen_production_top1"] == row["target_slug"] for row in generated_rows if row["subset_role"] == "hard")
            representative = sum(row["frozen_production_top1"] == row["target_slug"] for row in generated_rows if row["subset_role"] == "representative")
            image_r5 = generated["frozen_image_only"]["recall_at_5_correct"]
            transitions = {"rescued": 0, "broken": 0}
        else:
            prod_gen = generated["adapted_production_top1_correct"]
            generated_hard = generated["generated_hard"]["correct"]
            representative = generated["generated_representative"]["correct"]
            image_r5 = generated["adapted_image_only"]["recall_at_5_correct"]
            transitions = generated["generated_transitions"]
        rows.append({
            "Method": label,
            "Internal family": (f"{100 * internal['validation_adapted']['same_family_top1']:.2f}%" if internal else ""),
            "Internal full": (f"{100 * internal['validation_adapted']['full_catalog_top1']:.2f}%" if internal else ""),
            "Hard": f"{hard['adapted_production_top1_correct'] if key else hard['frozen_production_top1_correct']}/1150",
            "Generated": f"{prod_gen}/128",
            "Correct/128": prod_gen,
            "Gen-hard/64": f"{generated_hard}/64",
            "Rep/64": f"{representative}/64",
            "R@5": f"{image_r5}/128",
            "Rescue/Break": f"{transitions.get('rescued', 0)}/{transitions.get('broken', 0)}",
        })
    for split in SPLITS:
        subset = [row for row in fusion_rows if row["benchmark"] == split]
        if split != "generated_stress_dev_pilot32":
            continue
        generated = results[best_name]["metrics"][split]
        rows.append({
            "Method": "50/50 diagnostic",
            "Internal family": "",
            "Internal full": "",
            "Hard": "",
            "Generated": f"{sum(bool(row['top1_correct']) for row in subset)}/128",
            "Correct/128": sum(bool(row["top1_correct"]) for row in subset),
            "Gen-hard/64": "diagnostic only",
            "Rep/64": "diagnostic only",
            "R@5": f"{sum(bool(row['top5_correct']) for row in subset)}/128",
            "Rescue/Break": "diagnostic only",
        })
    return rows


def _write_aggregated_artifacts(results: Mapping[str, Mapping[str, Any]],
                                r8_internal: Mapping[str, Any], r16_internal: Mapping[str, Any],
                                best_name: str, table: list[dict[str, Any]],
                                fusion_rows: list[Mapping[str, Any]],
                                preflight: Mapping[str, Any], baseline_error_ids: list[str]) -> dict[str, Any]:
    image_rows: list[dict[str, Any]] = []
    production_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    for model_name, result in results.items():
        for split, metrics in result["metrics"].items():
            for kind in ("frozen_image_only", "adapted_image_only"):
                image_rows.append({"model": model_name, "benchmark": split, "method": kind, **metrics[kind]})
            for kind, correct_key in (("frozen_production", "frozen_production_top1_correct"),
                                      ("adapted_production", "adapted_production_top1_correct")):
                production_rows.append({
                    "model": model_name, "benchmark": split, "method": kind,
                    "queries": metrics["queries"], "top1_correct": metrics[correct_key],
                    "top1": metrics[correct_key] / metrics["queries"],
                    "r_at_5": metrics["adapted_production_recall_at_5"] if kind == "adapted_production" else "",
                })
        for source_name, target_rows, filename in (
            ("transition_summary.csv", transition_rows, "transition_summary.csv"),
            ("family_metrics.csv", family_rows, "family_metrics.csv"),
            ("scenario_metrics.csv", scenario_rows, "scenario_metrics.csv"),
        ):
            path = Path(result["run_dir"]) / filename
            for row in read_csv(path):
                target_rows.append({"model": model_name, **row})
        for split in SPLITS:
            path = Path(result["run_dir"]) / "benchmarks" / split / "embedding_margin_analysis.csv"
            margin_rows.extend({"model": model_name, **row} for row in read_csv(path))
        for row in result["latency"]:
            latency_rows.append({"model": model_name, **row})

    write_csv(RUN_ROOT / "external_image_only.csv", image_rows)
    write_csv(RUN_ROOT / "external_production.csv", production_rows)
    write_csv(RUN_ROOT / "transition_summary.csv", transition_rows)
    write_csv(RUN_ROOT / "family_metrics.csv", family_rows)
    write_csv(RUN_ROOT / "scenario_metrics.csv", scenario_rows)
    write_csv(RUN_ROOT / "embedding_margin.csv", margin_rows)
    write_csv(RUN_ROOT / "latency_summary.csv", latency_rows)
    write_csv(RUN_ROOT / "fusion_diagnostic/fusion_50_50.csv", fusion_rows)
    write_csv(RUN_ROOT / "internal_validation.csv", [
        {"model": model_name, "split": view, "same_family_top1": metrics.get("same_family_top1"),
         "full_catalog_top1": metrics.get("full_catalog_top1"), "candidate_set_top1": metrics.get("candidate_set_top1"),
         "same_family_margin_mean": metrics.get("same_family_margin_mean")}
        for model_name, payload in (("r8_control", r8_internal), ("r16", r16_internal))
        for view, metrics in (("validation_frozen", payload["validation_frozen"]),
                              ("validation_adapted", payload["validation_adapted"]))
    ])
    bucket_rows = []
    for model_name, payload in (("r8_control", r8_internal), ("r16", r16_internal)):
        for mode, metrics in (("frozen", payload["validation_frozen"]), ("adapted", payload["validation_adapted"])):
            for bucket, bucket_metrics in metrics["validation_buckets"].items():
                bucket_rows.append({"model": model_name, "mode": mode, "bucket": bucket, **bucket_metrics})
    write_csv(RUN_ROOT / "bucket_validation.csv", bucket_rows)
    write_csv(RUN_ROOT / "external_eval/summary_table.csv", table)

    vram_rows = []
    for model_name, run_dir in (("r8_validated", R8_SOURCE), ("r16", RUN_ROOT / "r16")):
        summary_path = run_dir / "training_summary.json"
        if summary_path.is_file():
            data = read_json(summary_path)
            vram_rows.append({"model": model_name, "peak_training_vram_mb": data.get("peak_train_vram_mb", ""),
                              "peak_inference_vram_mb": ""})
    write_csv(RUN_ROOT / "vram_summary.csv", vram_rows)

    error_rows = read_csv(EXTERNAL_ROOT / "oracle_25_errors.csv")
    oracle_counts = {}
    for model_name in results:
        subset = [row for row in error_rows if row["model"] == model_name]
        oracle_counts[model_name] = {
            "wins": sum(row["target_score_relation"] == "win" for row in subset),
            "losses": sum(row["target_score_relation"] == "loss" for row in subset),
            "ties": sum(row["target_score_relation"] == "tie" for row in subset),
        }
    selected_internal = r16_internal if best_name == "r16" else r8_internal
    report_data = {
        "table": table,
        "best_model": best_name,
        "methods": results,
        "r8_internal": r8_internal,
        "r16_internal": r16_internal,
        "preflight": preflight,
        "oracle_25_error_counts": oracle_counts,
        "baseline_error_ids": baseline_error_ids,
        "selected_lora_internal_gain_pp": 100 * (selected_internal["validation_adapted"]["same_family_top1"] - selected_internal["validation_frozen"]["same_family_top1"]),
        "fusion_rows": fusion_rows,
        "latency_rows": latency_rows,
        "vram_rows": vram_rows,
    }
    return report_data


def _write_final_report(data: Mapping[str, Any]) -> Path:
    rows = data["table"]
    columns = list(rows[0])
    md_table = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        md_table.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    methods = data["methods"]
    r8m = methods["r8_validated"]["metrics"]
    r16m = methods["r16"]["metrics"]
    gen_name, hard_name, synthetic_name = "generated_stress_dev_pilot32", "hard_near_duplicate_dev_v2", "synthetic_dev"
    best_name = data["best_model"]
    best_result = methods[best_name]
    best_gen = best_result["metrics"][gen_name]
    best_hard = best_result["metrics"][hard_name]
    best_gen_correct = int(best_gen["adapted_production_top1_correct"])
    hard_correct = int(best_hard["adapted_production_top1_correct"])
    target_reached = best_gen_correct >= 116
    hard_guard = hard_correct >= 1045
    r8_gen = int(r8m[gen_name]["adapted_production_top1_correct"])
    r16_gen = int(r16m[gen_name]["adapted_production_top1_correct"])
    internal_winner = data["best_model"]
    if target_reached and hard_guard:
        verdict = "A. TARGET_REACHED_FREEZE_MODEL"
        reason = f"The internally selected frozen encoder reaches {best_gen_correct}/128 and passes the hard-v2 guard."
    elif internal_winner == "r16" and r16_gen >= r8_gen + 3:
        verdict = "C. LORA_CAPACITY_IS_MAIN_GAIN"
        reason = f"R16 wins internal selection and adds {r16_gen-r8_gen} generated correct queries over R8."
    elif best_gen_correct <= 109:
        verdict = "G. INVALID_EXPERIMENT"
        reason = ("The benchmark pass and R8/R16 capacity comparison are valid, but the milestone cannot estimate the requested Capture-v1→v2 effect: "+
                  "the frozen R8 control already used catalog_capture_v2, and retraining it was prohibited.")
    else:
        verdict = "E. REALISTIC_TRAIN_DATA_IS_NOW_BOTTLENECK"
        reason = "The frozen encoder adapts strongly on internal capture views but external generated transfer remains below target."
    baseline = data["preflight"]["reproduction"]
    oracle = data["oracle_25_error_counts"]
    r8_train = read_json(R8_SOURCE / "training_summary.json")
    r16_train = read_json(RUN_ROOT / "r16/training_summary.json")
    run_d = read_json(RUN_ROOT / "last2/state.json")
    selected_internal = data["r16_internal"] if best_name == "r16" else data["r8_internal"]
    selected_lora_gain_pp = data["selected_lora_internal_gain_pp"]
    best_latency = [row for row in data["latency_rows"] if row["model"] == best_name]
    best_inference = next((row for row in best_latency if row["mode"] == "adapted"), {})
    best_frozen = next((row for row in best_latency if row["mode"] == "frozen"), {})
    best_synthetic = best_result["metrics"][synthetic_name]
    test_result_path = ROOT / "artifacts/experiments/encoder_followup_r16_capturev2_last2_20260924T083837Z/verification_results.json"
    if test_result_path.is_file():
        test_result = read_json(test_result_path)
        all_tests = (
            f"full suite {test_result.get('passed', '?')} passed/{test_result.get('failed', '?')} failed "
            f"({test_result.get('warnings', 0)} warnings); final focused rerun "
            f"{test_result.get('final_targeted_passed', test_result.get('post_full_targeted_passed', '?'))} passed/"
            f"{test_result.get('final_targeted_failed', test_result.get('post_full_targeted_failed', '?'))} failed"
        )
    else:
        all_tests = "not yet run"
    content = [
        "# Encoder follow-up milestone: R16 capacity + Capture-v2 + conditional last2",
        "",
        "## Main comparison",
        "",
        *md_table,
        "",
        "## Protocol and interpretation",
        "",
        f"- Canonical external query batch size: **1** for hard-v2, generated pilot32, and synthetic; catalog reference precompute batch size: **{int(runner.CONFIG['reference_embedding_batch_size'])}**.",
        f"- Dynamic frozen baseline Top-5 reproduction before adapted ranking: hard **{baseline[hard_name]['top5_exact_order']}/{EXPECTED[hard_name]}**, generated **{baseline[gen_name]['top5_exact_order']}/{EXPECTED[gen_name]}**, synthetic **{baseline[synthetic_name]['top5_exact_order']}/{EXPECTED[synthetic_name]}**.",
        f"- Historical Top-5 score floats differ by at most {max(float(baseline[split]['max_score_abs_diff_on_historical_top5']) for split in (hard_name, gen_name, synthetic_name)):.3g}; accepted numerical tolerance was 3e-6. Top-5 membership and order matched exactly on every query.",
        f"- R8 source LoRA-state SHA-256 remains `{read_json(RUN_ROOT / 'r8_control/selected_model.json')['lora_sha256']}`. R8 and R16 use the exact same train/validation manifests and hard-negative map; only rank/alpha differ (8/16 vs 16/32). Both use Capture-v2 because that is the already frozen R8 domain.",
        f"- Selected LoRA + Capture-v2 internal same-family gain over frozen views is **{selected_lora_gain_pp:.2f} percentage points**; full-catalog is {100*selected_internal['validation_adapted']['full_catalog_top1']:.2f}% and family margin {selected_internal['validation_adapted']['same_family_margin_mean']:.4f}. This is an adapted-vs-frozen gain, not an isolated Capture-v2 gain.",
        "- **Causal limitation:** R8 already used catalog_capture_v2. R16 therefore used the exact same v2 train/validation manifests and hard-negative map; rank/alpha alone changed in the matched R8/R16 pair. R16 is the selected LoRA + Capture-v2 candidate, but no independent v1→v2 effect can be estimated while keeping R8 frozen. The available capacity comparison is matched under v2, not under the originally requested v1 domain.",
        f"- RUN D last2 status: not run. {run_d.get('reason', '')}",
        "- Training data access records show catalog references only; benchmark query images, labels, and benchmark-derived hard negatives were not used in training or internal checkpoint selection.",
        "",
        "## Generated and benchmark detail",
        "",
        f"- Frozen generated production: {r8m[gen_name]['frozen_production_top1_correct']}/128; validated R8: {r8_gen}/128; R16: {r16_gen}/128. Project target: **116/128**; reached: **{'YES' if target_reached else 'NO'}**.",
        f"- Best generated-hard: {best_gen['generated_hard']['correct']}/64; representative: {best_gen['generated_representative']['correct']}/64; production R@5: {best_gen['adapted_production_recall_at_5']*128:.0f}/128.",
        f"- Best hard-v2: {hard_correct}/1150 ({100*best_hard['adapted_production_top1']:.2f}%); guard >=1045/1150: **{'PASS' if hard_guard else 'FAIL'}**.",
        f"- Best synthetic post-selection: {best_synthetic['adapted_production_top1_correct']}/4084; image-only R@5 {best_synthetic['adapted_image_only']['recall_at_5_correct']}/4084.",
        f"- Generated rescued/broken relative to frozen: {best_gen['generated_transitions']['rescued']}/{best_gen['generated_transitions']['broken']}; hard-v2 transitions: {best_hard.get('production_net_gain', '')} net.",
        f"- Original frozen 25 generated errors remain immutable (fingerprint `{original_error_id_fingerprint(data['baseline_error_ids'])}`). Target-score wins/losses/ties vs frozen incumbent: R8 {oracle['r8_validated']['wins']}/{oracle['r8_validated']['losses']}/{oracle['r8_validated']['ties']}; R16 {oracle['r16']['wins']}/{oracle['r16']['losses']}/{oracle['r16']['ties']}.",
        f"- Best encoder latency: mean {float(best_inference.get('encoder_mean_ms', 0)):.2f} ms, p95 {float(best_inference.get('encoder_p95_ms', 0)):.2f} ms; full pipeline p95 {float(best_inference.get('full_pipeline_p95_ms', 0)):.2f} ms; frozen encoder p95 {float(best_frozen.get('encoder_p95_ms', 0)):.2f} ms; overhead p95 {float(best_inference.get('encoder_p95_ms', 0))-float(best_frozen.get('encoder_p95_ms', 0)):+.2f} ms; SLA <3 sec: {best_inference.get('sla_3s_passed', 'n/a')}.",
        f"- Training cost R16: {len(r16_train.get('history', []))} completed epochs, best epoch {r16_train.get('best_epoch', '')}, summed epoch time {sum(float(row['epoch_seconds']) for row in r16_train.get('history', []))/60:.1f} min, peak {r16_train.get('peak_train_vram_mb', ''):.0f} MB. R8 reference run was {len(r8_train.get('history', []))} epochs and {r8_train.get('peak_train_vram_mb', ''):.0f} MB peak.",
        "- Recovery: epoch checkpoints under `r16/checkpoints/` include model, optimizer, scheduler, scaler, RNG, validation metrics, and config fingerprint. Resume with `.venv/bin/python scripts/run_so400m_hard_negative_lora.py --rank 16 --alpha 32 --max-epochs 6 --resume artifacts/experiments/encoder_followup_r16_capturev2_last2_20260924T083837Z/r16 --internal-only`.",
        "",
        "## Required answers",
        "",
        f"1. Canonical batch=1 guard added: **YES**. 2. Baseline exact on all 3 sets: **YES** ({baseline[hard_name]['top5_exact_order']}/{EXPECTED[hard_name]}, {baseline[gen_name]['top5_exact_order']}/{EXPECTED[gen_name]}, {baseline[synthetic_name]['top5_exact_order']}/{EXPECTED[synthetic_name]}).",
        f"3. R8 vs R16 internal winner: **{internal_winner}**. 4. R16 generated production: **{r16_gen}/128**. 5. Selected LoRA + Capture-v2 internal gain over frozen views: **{selected_lora_gain_pp:.2f}pp** (not an isolated v2 effect). 6. Incremental Capture-v2 generated gain vs v1: **not separately identifiable**; best R16+v2 is {r16_gen}/128.",
        f"7. Last2 launched: **NO**. 8. Last2 generated result: **N/A**. 9. Best generated: **{best_gen_correct}/128**. 10. Best generated-hard: **{best_gen['generated_hard']['correct']}/64**. 11. Representative: **{best_gen['generated_representative']['correct']}/64**.",
        f"12. Hard-v2: **{hard_correct}/1150**. 13. R@5: **{best_gen['adapted_image_only']['recall_at_5_correct']}/128 image-only**. 14. Rescued/broken: **{best_gen['generated_transitions']['rescued']}/{best_gen['generated_transitions']['broken']}**. 15. Original 25 wins/losses/ties: **R8 {oracle['r8_validated']['wins']}/{oracle['r8_validated']['losses']}/{oracle['r8_validated']['ties']}; R16 {oracle['r16']['wins']}/{oracle['r16']['losses']}/{oracle['r16']['ties']}**.",
        f"16. Fixed 50/50 fusion diagnostic: generated **{sum(bool(row['top1_correct']) for row in data['fusion_rows'] if row['benchmark']==gen_name)}/128**, R@5 **{sum(bool(row['top5_correct']) for row in data['fusion_rows'] if row['benchmark']==gen_name)}/128**; no weight search. 17. Main gain attribution: **not causally separable for Capture-v2**; R16 is isolated capacity comparison.",
        f"18. Peak VRAM: R16 training **{r16_train.get('peak_train_vram_mb', 0):.0f} MB**, selected inference **{best_inference.get('peak_inference_vram_mb', 'n/a')} MB**. 19. R16 training time: **{sum(float(row['epoch_seconds']) for row in r16_train.get('history', []))/60:.1f} min**. 20. Test suite: **{all_tests}**. 21. >=116/128 reached: **{'YES' if target_reached else 'NO'}**.",
        "",
        "## Final verdict",
        "",
        f"**{verdict}** — {reason}",
        "",
        "No next encoder milestone was started.",
    ]
    path = ROOT / "reports/encoder_followup_r16_capturev2_last2_report.md"
    path.write_text("\n".join(content) + "\n", encoding="utf-8")
    write_json(RUN_ROOT / "selected_model.json", {
        "rank": 8 if best_name == "r8_validated" else 16, "model": best_name, "selected_by": "internal validation same-family Top-1 with full-catalog guard",
        "checkpoint_sha256": read_json(EXTERNAL_ROOT / best_name / "selected_model.json")["lora_sha256"],
        "canonical_query_batch_size": CANONICAL_QUERY_BATCH_SIZE,
        "external_evaluation_after_freeze": True,
        "benchmark_selection_allowed": False,
    })
    write_json(RUN_ROOT / "decision.json", {"verdict": verdict, "reason": reason, "target_reached": target_reached, "hard_guard_passed": hard_guard})
    return path


def main() -> None:
    require_canonical_query_batch_size(CANONICAL_QUERY_BATCH_SIZE)
    if not (RUN_ROOT / "r16/selected_model.json").is_file():
        raise RuntimeError("R16 is not internally selected and frozen; common external evaluation is blocked")
    baseline_forensics = ROOT / runner.CANONICAL_EXTERNAL_PROTOCOL["baseline_artifact"]
    baseline_payload = read_json(baseline_forensics)
    require_exact_baseline_reproduction(baseline_payload["after_single_query_fix"], EXPECTED)

    # Confirm the frozen R8 was never overwritten before using it as control.
    source_state, source_hash = _lora_state(R8_SOURCE / "checkpoints/best.pt")
    if source_hash != "b3dfce35160cc4c745090f7ec03dd783f86a4c0a82d7566521e7aa21e6a04a04":
        raise RuntimeError("validated R8 checkpoint hash changed")
    if source_hash != read_json(RUN_ROOT / "r8_control/selected_model.json")["lora_sha256"]:
        raise RuntimeError("copied R8 frozen control differs from the original checkpoint")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("common external SO400M evaluation requires CUDA")
    catalog, slugs, base_refs, base_meta, image_hashes = runner.load_catalog(ROOT)
    base_refs = base_refs.astype(np.float32)

    # Internal selection is completed without external labels. The frozen R8
    # already used v2, so R16 reuses those exact manifests; this supports a
    # matched capacity comparison under v2 but cannot isolate a v1-to-v2 effect.
    r8_internal = read_json(RUN_ROOT / "r8_control/internal_validation_metrics.json")
    r16_internal = read_json(RUN_ROOT / "r16/internal_validation_metrics.json")
    best_rank = _select_best_lora(r8_internal, r16_internal)
    best_model_name = "r8_validated" if best_rank == "r8" else "r16"
    r8_val = r8_internal["validation_adapted"]
    selected_internal = r16_internal if best_model_name == "r16" else r8_internal
    selected_val = selected_internal["validation_adapted"]
    frozen_val = r8_internal["validation_frozen"]
    selected_lora_gain_pp = 100.0 * (float(selected_val["same_family_top1"]) - float(frozen_val["same_family_top1"]))
    # RUN D is conditional. The selected LoRA + v2 candidate has a large
    # internal gain and margin and passes the full-catalog guard, so last2 is skipped.
    run_c_internal_final = (
        selected_lora_gain_pp >= 5.0
        and float(selected_val["full_catalog_top1"]) >= float(frozen_val["full_catalog_top1"]) - 0.01
        and float(selected_val["same_family_margin_mean"]) > 0.0
    )
    run_d = {
        "started": False,
        "internal_finality_guard_passed": run_c_internal_final,
        "reason": "Selected LoRA + Capture-v2 is internally conclusive: +%.2fpp same-family Top-1, full-catalog %.2f%% (frozen %.2f%%), margin %.4f; no deeper unfreeze started." % (
            selected_lora_gain_pp, 100.0 * float(selected_val["full_catalog_top1"]),
            100.0 * float(frozen_val["full_catalog_top1"]), float(selected_val["same_family_margin_mean"]))
            if run_c_internal_final else "Selected LoRA + Capture-v2 fails the internal finality guard; the conditional last2 experiment is required.",
    }
    if not run_c_internal_final:
        raise RuntimeError("Selected LoRA + Capture-v2 failed the predeclared internal finality guard; last2 must run before any external evaluation")
    write_json(RUN_ROOT / "last2/state.json", {"stage": "skipped_by_internal_stop_condition", **run_d, "updated_at_utc": datetime.now(timezone.utc).isoformat()})
    write_json(RUN_ROOT / "best_lora_internal_selection.json", {
        "best_rank": int(best_rank[1:]),
        "best_model": best_model_name,
        "selection_source": "internal validation only",
        "selection_metrics": {"r8": r8_val, "r16": r16_internal["validation_adapted"], "frozen": frozen_val},
        "run_c_capture_v2_is_selected_lora_alias": best_model_name,
        "independent_capture_v1_vs_v2_effect_estimable": False,
        "last2": run_d,
    })
    run_config = read_json(RUN_ROOT / "config.json")
    run_config["run_c"] = (
        "Selected LoRA + Capture-v2 is the internally selected candidate (%s). The frozen R8 already used catalog_capture_v2, "
        "so R16 used the exact same v2 manifests; independent v1-to-v2 effect is not estimable without changing the frozen control."
    ) % best_model_name
    write_json(RUN_ROOT / "config.json", run_config)

    runner.CONFIG["rank"] = 8
    runner.CONFIG["alpha"] = 16
    r8_model, r8_processor, _r8_modules, _ = runner.load_model(device)
    load_lora_state_dict(r8_model, source_state)
    if lora_sha256(r8_model) != source_hash:
        raise RuntimeError("R8 model did not load the unchanged frozen state")
    r8_model.requires_grad_(False)
    r8_model.eval()
    write_json(EXTERNAL_ROOT / "common_state.json", {
        "stage": "canonical_baseline_dynamic_preflight",
        "all_candidate_models_frozen": True,
        "r8_sha256": source_hash,
        "r16_sha256": read_json(RUN_ROOT / "r16/selected_model.json")["lora_sha256"],
        "query_batch_size": CANONICAL_QUERY_BATCH_SIZE,
    })
    root_state = read_json(RUN_ROOT / "state.json")
    root_state.update({
        "stage": "all_candidates_frozen_baseline_preflight",
        "r16_training_completed": True,
        "internal_best_rank": int(best_rank[1:]),
        "last2_started": False,
        "external_evaluation_started": False,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    write_json(RUN_ROOT / "state.json", root_state)
    preflight = _frozen_preflight(catalog, slugs, base_refs, r8_model, r8_processor, device)
    del r8_model
    torch.cuda.empty_cache()

    write_json(EXTERNAL_ROOT / "common_state.json", {
        "stage": "common_external_evaluation_started",
        "all_candidate_models_frozen": True,
        "baseline_preflight_passed_all_splits": True,
        "query_batch_size": CANONICAL_QUERY_BATCH_SIZE,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    root_state.update({"stage": "common_external_evaluation_started", "external_evaluation_started": True,
                       "updated_at_utc": datetime.now(timezone.utc).isoformat()})
    write_json(RUN_ROOT / "state.json", root_state)
    r8_candidate = _prepare_candidate(
        "r8_validated", 8, 16, R8_SOURCE / "checkpoints/best.pt",
        RUN_ROOT / "r8_control/selected_model.json", catalog, slugs, image_hashes, base_meta, device,
    )
    r8_result = _run_candidate(r8_candidate, catalog, slugs, base_refs, device)
    del r8_candidate["model"]
    torch.cuda.empty_cache()
    r16_checkpoint = RUN_ROOT / "r16/checkpoints/best.pt"
    r16_candidate = _prepare_candidate(
        "r16", 16, 32, r16_checkpoint, RUN_ROOT / "r16/selected_model.json",
        catalog, slugs, image_hashes, base_meta, device,
    )
    r16_result = _run_candidate(r16_candidate, catalog, slugs, base_refs, device)
    del r16_candidate["model"]
    torch.cuda.empty_cache()

    results = {"r8_validated": r8_result, "r16": r16_result}
    # Record RUN C's alias to the selected frozen candidate without duplicating
    # metrics payloads (which contain tuple-keyed internal maps).
    capture_alias = {
        "alias": "best_lora_capture_v2",
        "identity_alias_of": best_model_name,
        "checkpoint_sha256": read_json(EXTERNAL_ROOT / best_model_name / "selected_model.json")["lora_sha256"],
        "rank": int(best_rank[1:]),
        "internal_selection_only": True,
        "augmentation_version": "catalog_capture_v2",
        "independent_v1_to_v2_effect_estimable": False,
        "note": "Selected LoRA + Capture-v2 alias; the frozen R8 control was already trained with Capture-v2.",
    }
    write_json(EXTERNAL_ROOT / "capture_v2_alias.json", capture_alias)

    records = preflight["records"]
    generated_records = records["generated_stress_dev_pilot32"]
    frozen_errors = [row for row in generated_records if str(row.get("sift_selected_top1", "")) != str(row["target_slug"])]
    baseline_error_ids = sorted(str(row["query_id"]) for row in frozen_errors)
    immutable_path = RUN_ROOT / "original_25_error_ids.json"
    if immutable_path.is_file():
        existing = read_json(immutable_path)
        assert_immutable_query_ids(baseline_error_ids, existing["query_ids"])
    else:
        write_json(immutable_path, {
            "query_ids": baseline_error_ids,
            "count": len(baseline_error_ids),
            "fingerprint": original_error_id_fingerprint(baseline_error_ids),
            "source": "canonical frozen OCR+SIFT generated baseline predictions",
            "immutable": True,
        })
    if len(baseline_error_ids) != 25:
        raise RuntimeError(f"expected the immutable canonical 25-error set; found {len(baseline_error_ids)}")
    _complementarity_and_oracle(results, baseline_error_ids, generated_records)

    fusion_rows = _run_fixed_fusion(best_model_name, r8_candidate if best_model_name == "r8_validated" else r16_candidate,
                                    records, base_refs, slugs, device)
    table = _summary_table(r8_result, r16_result, best_model_name, results,
                           fusion_rows, r8_internal, r16_internal)
    write_csv(RUN_ROOT / "external_eval/summary_table.csv", table)

    method_summary = {}
    for name, result in results.items():
        method_summary[name] = {
            "checkpoint_sha256": read_json(EXTERNAL_ROOT / name / "selected_model.json")["lora_sha256"],
            "metrics": result["metrics"],
            "latency": result["latency"],
            "external_top5_reproduced": result["base_encoder_query_top5_reproduced"],
        }
    summary = {
        "stage": "complete",
        "query_batch_size": CANONICAL_QUERY_BATCH_SIZE,
        "dynamic_baseline_reproduction": preflight["reproduction"],
        "r8_unchanged_sha256": source_hash,
        "internal_best_rank": int(best_rank[1:]),
        "selected_lora_internal_gain_same_family_pp": selected_lora_gain_pp,
        "capture_v2_alias_of_best_lora": best_model_name,
        "independent_capture_v1_vs_v2_effect_estimable": False,
        "last2": run_d,
        "methods": method_summary,
        "original_25_error_ids_fingerprint": original_error_id_fingerprint(baseline_error_ids),
        "fixed_fusion_method": "per-query min-max normalize each full-catalog score vector; fixed 0.5/0.5",
        "table": table,
    }
    write_json(RUN_ROOT / "external_eval/common_summary.json", summary)
    report_data = _write_aggregated_artifacts(
        results, r8_internal, r16_internal, best_model_name, table, fusion_rows,
        preflight, baseline_error_ids,
    )
    report_path = _write_final_report(report_data)
    write_json(EXTERNAL_ROOT / "common_state.json", {
        "stage": "complete",
        "query_batch_size": CANONICAL_QUERY_BATCH_SIZE,
        "baseline_preflight_passed_all_splits": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    root_state.update({"stage": "complete", "external_evaluation_started": True,
                       "report": str(report_path.relative_to(ROOT)),
                       "verdict": read_json(RUN_ROOT / "decision.json")["verdict"],
                       "updated_at_utc": datetime.now(timezone.utc).isoformat()})
    write_json(RUN_ROOT / "state.json", root_state)
    print(json.dumps({"stage": "complete", "best_rank": best_rank, "report": str(report_path), "table": table}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
