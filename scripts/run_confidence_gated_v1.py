#!/usr/bin/env python3
"""Confidence-gated multi-crop fallback: calibration, freeze, evaluation.

Milestone flow:
1. calibration inference: baseline_full + preprocessing_v1 rankings on
   deterministic product-level calibration subsets only (synthetic_dev,
   hard_v2, generated pilot32);
2. signal analysis (Part B) from the frozen full-benchmark baseline run;
3. bounded gate candidate grid (margin/score/margin_or_score) evaluated on
   calibration subsets only, with a lexicographic objective
   (clean guard -> generated gain -> fallback rate);
4. gate freeze into selected_gate.json;
5. full-benchmark fresh runs of baseline_full, preprocessing_v1 and
   confidence_gated_v1 with production-like timing (full embedding is never
   computed twice inside the gated path);
6. artifacts + report.

The gate uses only the full-image score distribution. No ground truth,
scenario, subset, family, filename, or crop information enters the gate.
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from recognition.cache import (  # noqa: E402
    EmbeddingCacheMismatch,
    load_embedding_cache,
    save_embedding_cache,
    sha256_file,
)
from recognition.crop_diagnostics import (  # noqa: E402
    full_ranking,
    rank_metrics,
    target_rank_full,
)
from recognition.encoder import VisualEncoder  # noqa: E402
from recognition.family_metrics import family_diagnostics  # noqa: E402
from recognition.gated_strategy import (  # noqa: E402
    FALLBACK_CROP_VIEWS,
    GateRule,
    confidence_features,
    distribution_summary,
    quantile,
    retrieve_with_confidence_gate,
    roc_auc,
    seeded_product_sample,
    stratified_product_split,
)
from recognition.preprocessing import load_rgb_image  # noqa: E402
from recognition.reporting import write_csv  # noqa: E402
from recognition.view_strategy import (  # noqa: E402
    BASELINE_FULL,
    PREPROCESSING_V1,
    aggregate_strategy_scores,
    rank_change_counts,
    strategy_views,
    top1_transition_matrix,
    top5_transition_matrix,
)
from scripts.run_siglip2_baseline import (  # noqa: E402
    _catalog_items,
    _default_benchmark_manifest,
    _path,
    _read_csv,
    _validate_benchmark,
)

BENCHMARKS = ("synthetic_dev", "hard_near_duplicate_dev_v2", "generated_stress_dev_pilot32")
SCENARIOS = ("distance_crop", "glare_bad_light", "handheld", "slight_angle")
SUBSETS = ("representative", "hard")
CALIBRATION_SEED = 20260919
SYNTHETIC_CALIBRATION_PRODUCTS = 400
HARD_CALIBRATION_PRODUCTS = 120
MARGIN_QUANTILES = (0.5, 0.6, 0.7, 0.8, 0.9)
SCORE_QUANTILES = (0.05, 0.10, 0.20, 0.30)
CLEAN_TOP1_GUARD = 0.01


def main() -> None:
    parser = argparse.ArgumentParser(description="Run confidence_gated_v1 milestone")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--catalog-version", default="data/processed/catalog_v1.json")
    parser.add_argument("--catalog-manifest", default="data/processed/catalog_manifest.csv")
    parser.add_argument("--cache", default="artifacts/catalog_embeddings/siglip2_catalog.pt")
    parser.add_argument("--artifacts-root", default="artifacts/experiments")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--report", default="reports/confidence_gated_v1_evaluation.md")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    catalog_meta = json.loads(_path(project_root, args.catalog_version).read_text(encoding="utf-8"))
    catalog_items, catalog_by_slug = _catalog_items(
        project_root, _read_csv(_path(project_root, args.catalog_manifest)), catalog_meta
    )
    catalog_slugs = [item.item_id for item in catalog_items]

    encoder = VisualEncoder(model_name="siglip", device=args.device, batch_size=args.batch_size)
    expected_cache_metadata = {
        "model_id": encoder.model_id,
        "model_version": encoder.model_version,
        "catalog_version": catalog_meta["catalog_version"],
        "catalog_manifest_sha256": catalog_meta["manifest_sha256"],
        "preprocessing_config": encoder.preprocessing_config,
        "catalog_count": len(catalog_items),
        "embedding_dim": encoder.embedding_dim,
        "catalog_item_ids": catalog_slugs,
    }
    try:
        catalog_embeddings = load_embedding_cache(_path(project_root, args.cache), expected_cache_metadata)
        cache_status = "loaded"
    except (FileNotFoundError, EmbeddingCacheMismatch) as exc:
        print(f"Catalog embedding cache not reused: {exc}; computing it now.")
        catalog_embeddings = encoder.encode([item.image_path for item in catalog_items])
        save_embedding_cache(_path(project_root, args.cache), catalog_embeddings, expected_cache_metadata)
        cache_status = "recomputed"

    import torch

    device = encoder.device
    reference_matrix = torch.nn.functional.normalize(
        torch.tensor(catalog_embeddings, dtype=torch.float32, device=device), p=2, dim=1
    )

    run_id = datetime.now(timezone.utc).strftime("siglip2_confidence_gated_v1_%Y%m%dT%H%M%SZ")
    run_dir = _path(project_root, args.artifacts_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    manifests: dict[str, list[dict[str, str]]] = {}
    for benchmark in BENCHMARKS:
        manifest_path = _default_benchmark_manifest(project_root, benchmark, None)
        rows = _read_csv(manifest_path)
        _validate_benchmark(project_root, rows, catalog_items, benchmark)
        manifests[benchmark] = rows

    # ------------------------------------------------------------------
    # Phase 1: calibration splits (product-level, deterministic).
    # ------------------------------------------------------------------
    pilot_split = stratified_product_split(manifests["generated_stress_dev_pilot32"], CALIBRATION_SEED)
    synthetic_products = sorted({row["target_slug"] for row in manifests["synthetic_dev"]})
    hard_products = sorted({row["target_slug"] for row in manifests["hard_near_duplicate_dev_v2"]})
    synthetic_cal_products = seeded_product_sample(
        synthetic_products, SYNTHETIC_CALIBRATION_PRODUCTS, CALIBRATION_SEED
    )
    hard_cal_products = seeded_product_sample(
        hard_products, HARD_CALIBRATION_PRODUCTS, CALIBRATION_SEED
    )
    calibration_rows = []
    for slug in sorted(pilot_split):
        pilot_row = next(row for row in manifests["generated_stress_dev_pilot32"] if row["target_slug"] == slug)
        calibration_rows.append(
            {
                "benchmark": "generated_stress_dev_pilot32",
                "slug": slug,
                "split": pilot_split[slug],
                "subset_role": pilot_row.get("subset_role", ""),
                "family_type": pilot_row.get("family_type", ""),
                "seed": CALIBRATION_SEED,
            }
        )
    for benchmark, products, role in (
        ("synthetic_dev", synthetic_cal_products, "calibration"),
        ("hard_near_duplicate_dev_v2", hard_cal_products, "calibration"),
    ):
        for slug in products:
            calibration_rows.append(
                {
                    "benchmark": benchmark,
                    "slug": slug,
                    "split": role,
                    "subset_role": "",
                    "family_type": "",
                    "seed": CALIBRATION_SEED,
                }
            )
    write_csv(
        run_dir / "calibration_split.csv",
        ["benchmark", "slug", "split", "subset_role", "family_type", "seed"],
        calibration_rows,
    )
    split_csv_bytes = (run_dir / "calibration_split.csv").read_bytes()
    calibration_fingerprint = hashlib.sha256(split_csv_bytes).hexdigest()
    cal_query_ids: dict[str, set[str]] = {
        "synthetic_dev": {
            row["query_id"]
            for row in manifests["synthetic_dev"]
            if row["target_slug"] in set(synthetic_cal_products)
        },
        "hard_near_duplicate_dev_v2": {
            row["query_id"]
            for row in manifests["hard_near_duplicate_dev_v2"]
            if row["target_slug"] in set(hard_cal_products)
        },
        "generated_stress_dev_pilot32": {
            row["query_id"]
            for row in manifests["generated_stress_dev_pilot32"]
            if pilot_split[row["target_slug"]] == "calibration"
        },
    }
    print(
        "Calibration query counts: "
        + ", ".join(f"{name}={len(ids)}" for name, ids in cal_query_ids.items())
    )

    # ------------------------------------------------------------------
    # Phase 2: calibration inference (baseline_full and preprocessing_v1).
    # ------------------------------------------------------------------
    cal_results: dict[str, dict[str, dict[str, Any]]] = {}
    for benchmark in BENCHMARKS:
        cal_rows = [row for row in manifests[benchmark] if row["query_id"] in cal_query_ids[benchmark]]
        cal_results[benchmark] = {
            BASELINE_FULL.name: _run_fullview(encoder, cal_rows, reference_matrix, catalog_slugs, project_root, device, benchmark),
            PREPROCESSING_V1.name: _run_multicrop(encoder, cal_rows, reference_matrix, catalog_slugs, project_root, device, benchmark),
        }

    # ------------------------------------------------------------------
    # Phase 3: bounded gate candidate grid on calibration subsets only.
    # ------------------------------------------------------------------
    pooled_margins: list[float] = []
    pooled_scores: list[float] = []
    for benchmark in BENCHMARKS:
        for row in cal_results[benchmark][BASELINE_FULL.name]["rows"]:
            features = confidence_features(row["top5_scores"])
            pooled_margins.append(features["top1_top2_margin"])
            pooled_scores.append(features["top1_score"])
    margin_candidates = [quantile(pooled_margins, q) for q in MARGIN_QUANTILES]
    score_candidates = [quantile(pooled_scores, q) for q in SCORE_QUANTILES]

    candidate_rules: list[GateRule] = []
    for value in margin_candidates:
        candidate_rules.append(GateRule("margin", margin_threshold=round(value, 4)))
    for value in score_candidates:
        candidate_rules.append(GateRule("score", score_threshold=round(value, 4)))
    for margin in margin_candidates:
        for score in score_candidates:
            candidate_rules.append(
                GateRule("margin_or_score", margin_threshold=round(margin, 4), score_threshold=round(score, 4))
            )

    candidate_rows = []
    evaluated_candidates: list[dict[str, Any]] = []
    for rule in candidate_rules:
        evaluation = _evaluate_candidate(rule, cal_results, cal_query_ids, manifests)
        evaluated_candidates.append({"rule": rule, **evaluation})
        candidate_rows.append(
            {
                "rule_type": rule.rule_type,
                "margin_threshold": rule.margin_threshold if rule.margin_threshold is not None else "",
                "score_threshold": rule.score_threshold if rule.score_threshold is not None else "",
                "syn_cal_top1_delta": _round4(evaluation["syn"]["top1_delta"]),
                "syn_cal_r5_delta": _round4(evaluation["syn"]["r5_delta"]),
                "syn_cal_fallback_rate": _round4(evaluation["syn"]["fallback_rate"]),
                "hard_cal_top1_delta": _round4(evaluation["hard"]["top1_delta"]),
                "hard_cal_r5_delta": _round4(evaluation["hard"]["r5_delta"]),
                "hard_cal_fallback_rate": _round4(evaluation["hard"]["fallback_rate"]),
                "gen_cal_top1": _round4(evaluation["gen"]["top1"]),
                "gen_cal_r5": _round4(evaluation["gen"]["r5"]),
                "gen_cal_r10": _round4(evaluation["gen"]["r10"]),
                "gen_cal_fallback_rate": _round4(evaluation["gen"]["fallback_rate"]),
                "passes_clean_guard": evaluation["passes_guard"],
            }
        )
    write_csv(
        run_dir / "gate_candidates.csv",
        [
            "rule_type", "margin_threshold", "score_threshold",
            "syn_cal_top1_delta", "syn_cal_r5_delta", "syn_cal_fallback_rate",
            "hard_cal_top1_delta", "hard_cal_r5_delta", "hard_cal_fallback_rate",
            "gen_cal_top1", "gen_cal_r5", "gen_cal_r10", "gen_cal_fallback_rate",
            "passes_clean_guard",
        ],
        candidate_rows,
    )

    survivors = [candidate for candidate in evaluated_candidates if candidate["passes_guard"]]
    pool = survivors if survivors else evaluated_candidates
    selected = sorted(
        pool,
        key=lambda candidate: (
            -candidate["gen"]["r5"],
            -candidate["gen"]["top1"],
            (candidate["syn"]["fallback_rate"] + candidate["hard"]["fallback_rate"]) / 2,
            candidate["rule"].rule_type,
            candidate["rule"].margin_threshold or 0.0,
            candidate["rule"].score_threshold or 0.0,
        ),
    )[0]
    selected_rule = selected["rule"]
    selected_gate = {
        "name": "confidence_gated_v1",
        "gate_type": selected_rule.rule_type,
        "margin_threshold": selected_rule.margin_threshold,
        "score_threshold": selected_rule.score_threshold,
        "rule_description": selected_rule.describe(),
        "calibration_seed": CALIBRATION_SEED,
        "calibration_fingerprint": calibration_fingerprint,
        "calibration_sizes": {name: len(ids) for name, ids in cal_query_ids.items()},
        "threshold_sources": {
            "margin_quantiles": list(MARGIN_QUANTILES),
            "score_quantiles": list(SCORE_QUANTILES),
        },
        "selection_objective": (
            "lexicographic: (1) clean calibration Top-1 regression <= 1pp on both clean sets, "
            "(2) maximize generated calibration R@5, (3) generated calibration Top-1, "
            "(4) lower mean clean fallback rate"
        ),
        "survivors_of_clean_guard": len(survivors),
        "frozen": True,
    }
    (run_dir / "selected_gate.json").write_text(
        json.dumps(selected_gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Gate frozen: {selected_rule.describe()}")

    # ------------------------------------------------------------------
    # Phase 4: full-benchmark fresh runs (baseline, v1, gated).
    # ------------------------------------------------------------------
    full_runs: dict[str, dict[str, dict[str, Any]]] = {}
    for benchmark in BENCHMARKS:
        rows = manifests[benchmark]
        print(f"[{benchmark}] full run: baseline_full")
        full_runs[benchmark] = {
            BASELINE_FULL.name: _run_fullview(encoder, rows, reference_matrix, catalog_slugs, project_root, device, benchmark),
        }
        print(f"[{benchmark}] full run: preprocessing_v1")
        full_runs[benchmark][PREPROCESSING_V1.name] = _run_multicrop(
            encoder, rows, reference_matrix, catalog_slugs, project_root, device, benchmark
        )
        print(f"[{benchmark}] full run: confidence_gated_v1")
        full_runs[benchmark]["confidence_gated_v1"] = _run_gated(
            encoder, rows, reference_matrix, catalog_slugs, project_root, device, selected_rule, benchmark
        )

    # Consistency: the frozen-gate fresh predictions must reproduce the
    # candidate-evaluation composite on calibration queries.
    mismatches_total = 0
    for benchmark in BENCHMARKS:
        gated_rows = {row["query_id"]: row for row in full_runs[benchmark]["confidence_gated_v1"]["rows"]}
        for row in cal_results[benchmark][BASELINE_FULL.name]["rows"]:
            query_id = row["query_id"]
            v1_row = next(r for r in cal_results[benchmark][PREPROCESSING_V1.name]["rows"] if r["query_id"] == query_id)
            features = confidence_features(row["top5_scores"])
            fallback = selected_rule.should_fallback(
                features["top1_score"], features["top2_score"], features["top5_score"]
            )
            expected_rank = v1_row["target_rank"] if fallback else row["target_rank"]
            actual_rank = gated_rows[query_id]["target_rank"]
            if expected_rank != actual_rank:
                mismatches_total += 1
    print(f"Gate reproducibility mismatches on calibration queries: {mismatches_total}")

    # ------------------------------------------------------------------
    # Phase 5: artifacts.
    # ------------------------------------------------------------------
    method_names = (BASELINE_FULL.name, PREPROCESSING_V1.name, "confidence_gated_v1")
    for benchmark in BENCHMARKS:
        for method in method_names:
            run = full_runs[benchmark][method]
            metrics = _metrics_payload(benchmark, method, run, len(manifests[benchmark]))
            (run_dir / f"metrics_{benchmark}_{method}.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            prediction_columns = [
                "query_id", "target_slug", "predicted_slug", "top1_score",
                "top5_slugs", "top5_scores", "correct_top1", "correct_top5",
                "target_rank", "embedding_latency_ms", "retrieval_latency_ms", "latency_ms",
            ]
            if method == "confidence_gated_v1":
                prediction_columns += ["used_fallback", "gate_reason", "views_used"]
            write_csv(run_dir / f"predictions_{benchmark}_{method}.csv", prediction_columns, run["rows"])
        gated_rows = full_runs[benchmark]["confidence_gated_v1"]["rows"]
        baseline_by_query = {row["query_id"]: row for row in full_runs[benchmark][BASELINE_FULL.name]["rows"]}
        for row in gated_rows:
            baseline_row = baseline_by_query[str(row["query_id"])]
            row["baseline_top1_correct"] = bool(baseline_row["correct_top1"])
            row["baseline_target_in_top5"] = bool(baseline_row["correct_top5"])
            row["baseline_target_rank"] = baseline_row["target_rank"]
            row["final_target_rank"] = row["target_rank"]
        write_csv(
            run_dir / f"per_query_gate_diagnostics_{benchmark}.csv",
            [
                "query_id", "used_fallback", "gate_reason", "confidence_score",
                "top1_top2_margin", "views_used", "path",
                "embedding_latency_ms", "retrieval_latency_ms", "latency_ms",
                "baseline_top1_correct", "baseline_target_in_top5", "baseline_target_rank", "final_target_rank",
            ],
            gated_rows,
        )

    summary_rows = []
    latency_rows = []
    for benchmark in BENCHMARKS:
        for method in method_names:
            run = full_runs[benchmark][method]
            metrics = _metric_cells(run["ranks"])
            fallback_rate = (
                _fallback_rate(run["rows"]) if method == "confidence_gated_v1" else 0.0
            )
            summary_rows.append(
                {
                    "benchmark": benchmark,
                    "method": method,
                    "queries": len(manifests[benchmark]),
                    **metrics,
                    "fallback_rate": _round4(fallback_rate),
                    **_latency_cells(run["latencies_ms"]),
                }
            )
            latency_cells = _latency_cells(run["latencies_ms"])
            views = (
                _effective_views(run["rows"]) if method == "confidence_gated_v1" else (1 if method == BASELINE_FULL.name else 3)
            )
            latency_rows.append(
                {
                    "benchmark": benchmark,
                    "method": method,
                    **latency_cells,
                    "fallback_rate": _round4(fallback_rate),
                    "effective_views_per_query": _round2(views),
                }
            )
    write_csv(
        run_dir / "benchmark_summary.csv",
        ["benchmark", "method", "queries", "top1_accuracy", "recall_at_5", "recall_at_10", "mrr",
         "median_target_rank_found", "missing_count", "fallback_rate",
         "mean_latency_ms", "p50_latency_ms", "p95_latency_ms"],
        summary_rows,
    )
    write_csv(
        run_dir / "latency_summary.csv",
        ["benchmark", "method", "mean_latency_ms", "p50_latency_ms", "p95_latency_ms",
         "fallback_rate", "effective_views_per_query"],
        latency_rows,
    )

    fallback_rows = []
    for benchmark in BENCHMARKS:
        baseline_rows = full_runs[benchmark][BASELINE_FULL.name]["rows"]
        gated_rows = full_runs[benchmark]["confidence_gated_v1"]["rows"]
        fallback_rows.append(
            {"benchmark": benchmark, **_fallback_confusion(baseline_rows, gated_rows)}
        )
    write_csv(
        run_dir / "fallback_summary.csv",
        ["benchmark", "trusted_correct", "unnecessary_fallback", "useful_trigger", "missed_opportunity",
         "trusted_correct_share", "unnecessary_fallback_share", "useful_trigger_share", "missed_opportunity_share"],
        fallback_rows,
    )

    effectiveness_rows = []
    for benchmark in BENCHMARKS:
        baseline_rows = full_runs[benchmark][BASELINE_FULL.name]["rows"]
        gated_rows = full_runs[benchmark]["confidence_gated_v1"]["rows"]
        effectiveness_rows.append(
            {"benchmark": benchmark, **_fallback_effectiveness(baseline_rows, gated_rows)}
        )
    write_csv(
        run_dir / "fallback_transitions.csv",
        ["benchmark", "fallback_queries", "top1_wrong_to_correct", "top1_correct_to_wrong",
         "top5_outside_to_inside", "top5_inside_to_outside"],
        effectiveness_rows,
    )

    signal_rows = _signal_analysis(full_runs, manifests)
    write_csv(
        run_dir / "signals_analysis.csv",
        ["benchmark", "scope", "signal", "queries", "auc_top1_correct", "auc_target_in_top5",
         "median_correct", "median_incorrect", "q10_correct", "q90_correct", "q10_incorrect", "q90_incorrect"],
        signal_rows,
    )

    pilot_payload = _pilot32_breakdown(
        run_dir, manifests["generated_stress_dev_pilot32"], full_runs
    )

    config = {
        "run_id": run_id,
        "purpose": "confidence-gated multi-crop fallback milestone (confidence_gated_v1)",
        "model_id": encoder.model_id,
        "model_version": encoder.model_version,
        "device": device,
        "catalog_version": catalog_meta["catalog_version"],
        "catalog_embedding_cache_status": cache_status,
        "selected_gate": selected_gate,
        "calibration_products": {
            "synthetic_dev": SYNTHETIC_CALIBRATION_PRODUCTS,
            "hard_near_duplicate_dev_v2": HARD_CALIBRATION_PRODUCTS,
            "generated_stress_dev_pilot32": len(pilot_split),
        },
        "gate_inputs": "full-image ranking scores only (top1, top2, top5); no ground truth/labels/crops",
        "fallback_views": list(FALLBACK_CROP_VIEWS),
        "full_embedding_recomputed_on_fallback": False,
        "benchmarks": {
            benchmark: {
                "manifest_sha256": sha256_file(_default_benchmark_manifest(project_root, benchmark, None)),
                "queries": len(manifests[benchmark]),
            }
            for benchmark in BENCHMARKS
        },
        "gate_reproducibility_mismatches_on_calibration": mismatches_total,
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report = _render_report(summary_rows, latency_rows, fallback_rows, effectiveness_rows, signal_rows, pilot_payload, selected_gate, config)
    report_path = _path(project_root, args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))
    print(f"Artifacts: {run_dir}")
    print(f"Report: {report_path}")


# ----------------------------------------------------------------------
# Inference runners
# ----------------------------------------------------------------------

def _sync(device: str) -> None:
    import torch

    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _base_row(query: dict[str, str], ranked: list[tuple[str, float]], rank: int | None,
              embed_ms: float, retrieval_ms: float) -> dict[str, object]:
    ranked_slugs = [slug for slug, _ in ranked]
    top5_scores = [score for _, score in ranked[:5]]
    return {
        "query_id": query["query_id"],
        "target_slug": query["target_slug"],
        "predicted_slug": ranked_slugs[0],
        "top1_score": ranked[0][1],
        "top5_slugs": json.dumps(ranked_slugs[:5], ensure_ascii=False, separators=(",", ":")),
        "top5_scores": top5_scores,
        "correct_top1": ranked_slugs[0] == query["target_slug"],
        "correct_top5": rank is not None and rank <= 5,
        "target_rank": rank if rank is not None else "",
        "embedding_latency_ms": _round2(embed_ms),
        "retrieval_latency_ms": _round2(retrieval_ms),
        "latency_ms": _round2(embed_ms + retrieval_ms),
    }


def _finish(rows: list[dict[str, object]]) -> dict[str, Any]:
    return {
        "rows": rows,
        "ranks": [int(row["target_rank"]) if row["target_rank"] != "" else None for row in rows],
        "latencies_ms": [float(row["latency_ms"]) for row in rows],
    }


def _run_fullview(
    encoder: VisualEncoder,
    manifest_rows: list[dict[str, str]],
    reference_matrix: Any,
    catalog_slugs: list[str],
    project_root: Path,
    device: str,
    benchmark: str,
) -> dict[str, Any]:
    import torch

    rows: list[dict[str, object]] = []
    for position, query in enumerate(manifest_rows, start=1):
        image = load_rgb_image(_path(project_root, query["query_path"]))
        _sync(device)
        started = perf_counter()
        embedding = torch.tensor(
            encoder.encode_pil([image]), dtype=torch.float32, device=device
        )
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        _sync(device)
        embed_ms = (perf_counter() - started) * 1000.0
        retrieval_started = perf_counter()
        scores = (embedding @ reference_matrix.transpose(0, 1))[0].detach().cpu().tolist()
        ranked = full_ranking(scores, catalog_slugs)
        _sync(device)
        retrieval_ms = (perf_counter() - retrieval_started) * 1000.0
        rank = target_rank_full([slug for slug, _ in ranked], query["target_slug"])
        rows.append(_base_row(query, ranked, rank, embed_ms, retrieval_ms))
        if position % 500 == 0 or position == len(manifest_rows):
            print(f"  [{benchmark}/baseline_full] {position}/{len(manifest_rows)}")
    return _finish(rows)


def _run_multicrop(
    encoder: VisualEncoder,
    manifest_rows: list[dict[str, str]],
    reference_matrix: Any,
    catalog_slugs: list[str],
    project_root: Path,
    device: str,
    benchmark: str,
) -> dict[str, Any]:
    import torch

    rows: list[dict[str, object]] = []
    for position, query in enumerate(manifest_rows, start=1):
        image = load_rgb_image(_path(project_root, query["query_path"]))
        views = strategy_views(image, PREPROCESSING_V1)
        _sync(device)
        started = perf_counter()
        embeddings = torch.tensor(
            encoder.encode_pil(views), dtype=torch.float32, device=device
        )
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        _sync(device)
        embed_ms = (perf_counter() - started) * 1000.0
        retrieval_started = perf_counter()
        view_scores = (embeddings @ reference_matrix.transpose(0, 1)).detach().cpu().tolist()
        final_scores = aggregate_strategy_scores(view_scores, PREPROCESSING_V1)
        ranked = full_ranking(final_scores, catalog_slugs)
        _sync(device)
        retrieval_ms = (perf_counter() - retrieval_started) * 1000.0
        rank = target_rank_full([slug for slug, _ in ranked], query["target_slug"])
        rows.append(_base_row(query, ranked, rank, embed_ms, retrieval_ms))
        if position % 500 == 0 or position == len(manifest_rows):
            print(f"  [{benchmark}/preprocessing_v1] {position}/{len(manifest_rows)}")
    return _finish(rows)


def _run_gated(
    encoder: VisualEncoder,
    manifest_rows: list[dict[str, str]],
    reference_matrix: Any,
    catalog_slugs: list[str],
    project_root: Path,
    device: str,
    rule: GateRule,
    benchmark: str,
) -> dict[str, Any]:
    rows: list[dict[str, object]] = []
    for position, query in enumerate(manifest_rows, start=1):
        image = load_rgb_image(_path(project_root, query["query_path"]))
        ranked, diagnostics, latencies = retrieve_with_confidence_gate(
            encoder, reference_matrix, catalog_slugs, image, rule, device
        )
        rank = target_rank_full([slug for slug, _ in ranked], query["target_slug"])
        row = _base_row(
            query,
            ranked,
            rank,
            latencies["embed_full_ms"],
            latencies["retrieval_gate_ms"] + latencies["embed_crops_ms"] + latencies["retrieval_fallback_ms"],
        )
        row["used_fallback"] = diagnostics.used_fallback
        row["gate_reason"] = diagnostics.gate_reason
        row["views_used"] = diagnostics.views_used
        row["path"] = "fallback" if diagnostics.used_fallback else "confident"
        rows.append(row)
        if position % 500 == 0 or position == len(manifest_rows):
            print(f"  [{benchmark}/confidence_gated_v1] {position}/{len(manifest_rows)}")
    return {
        "rows": rows,
        "ranks": [int(row["target_rank"]) if row["target_rank"] != "" else None for row in rows],
        "latencies_ms": [float(row["latency_ms"]) for row in rows],
    }


# ----------------------------------------------------------------------
# Candidate evaluation (calibration subsets only)
# ----------------------------------------------------------------------

def _evaluate_candidate(
    rule: GateRule,
    cal_results: dict[str, dict[str, dict[str, Any]]],
    cal_query_ids: dict[str, set[str]],
    manifests: dict[str, list[dict[str, str]]],
) -> dict[str, Any]:
    evaluation: dict[str, Any] = {}
    for key, benchmark in (("syn", "synthetic_dev"), ("hard", "hard_near_duplicate_dev_v2"), ("gen", "generated_stress_dev_pilot32")):
        baseline_rows = cal_results[benchmark][BASELINE_FULL.name]["rows"]
        v1_rows = {row["query_id"]: row for row in cal_results[benchmark][PREPROCESSING_V1.name]["rows"]}
        ranks, fallback_flags = [], []
        for row in baseline_rows:
            features = confidence_features(row["top5_scores"])
            fallback = rule.should_fallback(
                features["top1_score"], features["top2_score"], features["top5_score"]
            )
            fallback_flags.append(fallback)
            ranks.append(
                v1_rows[row["query_id"]]["target_rank"]
                if fallback else row["target_rank"]
            )
        composite_metrics = rank_metrics([int(rank) if rank != "" else None for rank in ranks])
        baseline_metrics = rank_metrics(cal_results[benchmark][BASELINE_FULL.name]["ranks"])
        evaluation[key] = {
            "top1": composite_metrics["top1_accuracy"],
            "r5": composite_metrics["recall_at_5"],
            "r10": composite_metrics["recall_at_10"],
            "top1_delta": composite_metrics["top1_accuracy"] - baseline_metrics["top1_accuracy"],
            "r5_delta": composite_metrics["recall_at_5"] - baseline_metrics["recall_at_5"],
            "fallback_rate": sum(fallback_flags) / len(fallback_flags) if fallback_flags else 0.0,
        }
    evaluation["passes_guard"] = (
        evaluation["syn"]["top1_delta"] >= -CLEAN_TOP1_GUARD
        and evaluation["hard"]["top1_delta"] >= -CLEAN_TOP1_GUARD
    )
    return evaluation


# ----------------------------------------------------------------------
# Analysis helpers
# ----------------------------------------------------------------------

def _signal_analysis(full_runs: dict[str, dict[str, dict[str, Any]]], manifests: dict[str, list[dict[str, str]]]) -> list[dict[str, object]]:
    signal_rows = []
    signals = ("top1_score", "top1_top2_margin", "top1_top5_margin")
    for benchmark in BENCHMARKS:
        baseline_rows = full_runs[benchmark][BASELINE_FULL.name]["rows"]
        scopes: list[tuple[str, list[dict[str, object]]]] = [("overall", baseline_rows)]
        if benchmark == "generated_stress_dev_pilot32":
            subset_by_query_diag = {row["query_id"]: row.get("subset_role", "") for row in manifests[benchmark]}
            for subset in SUBSETS:
                scopes.append((subset, [row for row in baseline_rows if subset_by_query_diag[row["query_id"]] == subset]))
        for scope_name, scope_rows in scopes:
            correct_flags = [bool(row["correct_top1"]) for row in scope_rows]
            top5_flags = [bool(row["correct_top5"]) for row in scope_rows]
            for signal in signals:
                values = [confidence_features(row["top5_scores"])[signal] for row in scope_rows]
                correct_values = [value for value, flag in zip(values, correct_flags) if flag]
                incorrect_values = [value for value, flag in zip(values, correct_flags) if not flag]
                inside_values = [value for value, flag in zip(values, top5_flags) if flag]
                outside_values = [value for value, flag in zip(values, top5_flags) if not flag]
                summary = distribution_summary(correct_values, incorrect_values)
                signal_rows.append(
                    {
                        "benchmark": benchmark,
                        "scope": scope_name,
                        "signal": signal,
                        "queries": len(scope_rows),
                        "auc_top1_correct": _round4(roc_auc(correct_values, incorrect_values)),
                        "auc_target_in_top5": _round4(roc_auc(inside_values, outside_values)),
                        "median_correct": _round4(summary["correct"]["median"]),
                        "median_incorrect": _round4(summary["incorrect"]["median"]),
                        "q10_correct": _round4(summary["correct"]["q10"]),
                        "q90_correct": _round4(summary["correct"]["q90"]),
                        "q10_incorrect": _round4(summary["incorrect"]["q10"]),
                        "q90_incorrect": _round4(summary["incorrect"]["q90"]),
                    }
                )
    return signal_rows


def _fallback_confusion(baseline_rows: list[dict[str, object]], gated_rows: list[dict[str, object]]) -> dict[str, object]:
    counts = {"trusted_correct": 0, "unnecessary_fallback": 0, "useful_trigger": 0, "missed_opportunity": 0}
    for baseline_row, gated_row in zip(baseline_rows, gated_rows):
        full_correct = bool(baseline_row["correct_top1"])
        fallback = bool(gated_row["used_fallback"])
        if full_correct and not fallback:
            counts["trusted_correct"] += 1
        elif full_correct and fallback:
            counts["unnecessary_fallback"] += 1
        elif not full_correct and fallback:
            counts["useful_trigger"] += 1
        else:
            counts["missed_opportunity"] += 1
    total = len(baseline_rows) or 1
    return {
        **counts,
        "trusted_correct_share": _round4(counts["trusted_correct"] / total),
        "unnecessary_fallback_share": _round4(counts["unnecessary_fallback"] / total),
        "useful_trigger_share": _round4(counts["useful_trigger"] / total),
        "missed_opportunity_share": _round4(counts["missed_opportunity"] / total),
    }


def _fallback_effectiveness(baseline_rows: list[dict[str, object]], gated_rows: list[dict[str, object]]) -> dict[str, object]:
    counts = {
        "fallback_queries": 0,
        "top1_wrong_to_correct": 0,
        "top1_correct_to_wrong": 0,
        "top5_outside_to_inside": 0,
        "top5_inside_to_outside": 0,
    }
    for baseline_row, gated_row in zip(baseline_rows, gated_rows):
        if not bool(gated_row["used_fallback"]):
            continue
        counts["fallback_queries"] += 1
        if bool(gated_row["correct_top1"]) and not bool(baseline_row["correct_top1"]):
            counts["top1_wrong_to_correct"] += 1
        if bool(baseline_row["correct_top1"]) and not bool(gated_row["correct_top1"]):
            counts["top1_correct_to_wrong"] += 1
        if bool(gated_row["correct_top5"]) and not bool(baseline_row["correct_top5"]):
            counts["top5_outside_to_inside"] += 1
        if bool(baseline_row["correct_top5"]) and not bool(gated_row["correct_top5"]):
            counts["top5_inside_to_outside"] += 1
    return counts


def _fallback_rate(rows: list[dict[str, object]]) -> float:
    if not rows:
        return 0.0
    return sum(bool(row.get("used_fallback", False)) for row in rows) / len(rows)


def _effective_views(rows: list[dict[str, object]]) -> float:
    if not rows:
        return 0.0
    return sum(int(row.get("views_used", 1)) for row in rows) / len(rows)


def _metric_cells(ranks: list[int | None]) -> dict[str, object]:
    metrics = rank_metrics(ranks)
    return {
        "top1_accuracy": metrics["top1_accuracy"],
        "recall_at_5": metrics["recall_at_5"],
        "recall_at_10": metrics["recall_at_10"],
        "mrr": metrics["mrr"],
        "median_target_rank_found": metrics["median_target_rank_found"],
        "missing_count": metrics["missing_count"],
    }


def _metrics_payload(benchmark: str, method: str, run: dict[str, Any], query_count: int) -> dict[str, object]:
    return {
        "benchmark": benchmark,
        "method": method,
        "query_count": query_count,
        **_metric_cells(run["ranks"]),
        **_latency_cells(run["latencies_ms"]),
        "note": "full-catalog ranking metrics; MRR uses the full-catalog target rank",
    }


def _latency_cells(latencies_ms: list[float]) -> dict[str, object]:
    return {
        "mean_latency_ms": _round2(_mean(latencies_ms)),
        "p50_latency_ms": _round2(statistics.median(latencies_ms)),
        "p95_latency_ms": _round2(_percentile(latencies_ms, 95)),
    }


def _pilot32_breakdown(
    run_dir: Path,
    manifest_rows: list[dict[str, str]],
    full_runs: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    runs = full_runs["generated_stress_dev_pilot32"]
    scenario_by_query = {row["query_id"]: row.get("scenario_id", "") for row in manifest_rows}
    subset_by_query = {row["query_id"]: row.get("subset_role", "") for row in manifest_rows}
    family_type_by_query = {row["query_id"]: row.get("family_type", "") for row in manifest_rows}
    for run in runs.values():
        for row in run["rows"]:
            row["scenario_id"] = scenario_by_query[str(row["query_id"])]
            row["subset_role"] = subset_by_query[str(row["query_id"])]
            row["family_type"] = family_type_by_query[str(row["query_id"])]

    def scoped_rows(run: dict[str, Any], key: str, value: str) -> dict[str, Any]:
        positions = [index for index, row in enumerate(run["rows"]) if row[key] == value]
        return {
            "rows": [run["rows"][index] for index in positions],
            "ranks": [run["ranks"][index] for index in positions],
        }

    method_names = (BASELINE_FULL.name, PREPROCESSING_V1.name, "confidence_gated_v1")
    scenario_rows, subset_rows, family_rows = [], [], []
    for scope_key, scope_values, csv_name in (
        ("scenario_id", SCENARIOS, "scenario_metrics.csv"),
        ("subset_role", SUBSETS, "subset_metrics.csv"),
    ):
        output_rows = []
        for value in scope_values:
            for method in method_names:
                scoped = scoped_rows(runs[method], scope_key, value)
                output_rows.append(
                    {
                        "method": method,
                        "scope_value": value,
                        **_metric_cells(scoped["ranks"]),
                        "fallback_rate": _round4(_fallback_rate(scoped["rows"])),
                    }
                )
        write_csv(
            run_dir / csv_name,
            ["method", "scope_value", "top1_accuracy", "recall_at_5", "recall_at_10", "mrr",
             "median_target_rank_found", "missing_count", "fallback_rate"],
            output_rows,
        )
        if scope_key == "scenario_id":
            scenario_rows = output_rows
        else:
            subset_rows = output_rows

    for method in method_names:
        diagnostics = family_diagnostics(runs[method]["rows"], manifest_rows)
        for family_type, breakdown in sorted(diagnostics["family_type_breakdown"].items()):
            family_rows.append(
                {
                    "method": method,
                    "family_type": family_type,
                    "query_count": breakdown["query_count"],
                    "exact_top1": breakdown["top1_accuracy"],
                    "family_top1": breakdown["family_top1"],
                    "family_recall_at_5": breakdown["family_recall_at_5"],
                    "within_family_disambiguation_top1": breakdown["within_family_disambiguation_top1"],
                    "fallback_rate": _round4(
                        _fallback_rate(
                            [row for row in runs[method]["rows"] if row.get("family_type") == family_type]
                        )
                    ),
                }
            )
        family_rows.append(
            {
                "method": method,
                "family_type": "ALL",
                "query_count": diagnostics["hard_query_count"],
                "exact_top1": _metric_cells(scoped_rows(runs[method], "subset_role", "hard")["ranks"])["top1_accuracy"],
                "family_top1": diagnostics["family_top1"],
                "family_recall_at_5": diagnostics["family_recall_at_5"],
                "within_family_disambiguation_top1": diagnostics["within_family_disambiguation_top1"],
                "fallback_rate": _round4(_fallback_rate(scoped_rows(runs[method], "subset_role", "hard")["rows"])),
            }
        )
    write_csv(
        run_dir / "family_metrics.csv",
        ["method", "family_type", "query_count", "exact_top1", "family_top1", "family_recall_at_5",
         "within_family_disambiguation_top1", "fallback_rate"],
        family_rows,
    )

    transition_rows = []
    baseline_run = runs[BASELINE_FULL.name]
    gated_run = runs["confidence_gated_v1"]
    for scope_name, scope_key, scope_values in (
        ("overall", "", [""]),
        ("scenario", "scenario_id", SCENARIOS),
        ("subset", "subset_role", SUBSETS),
    ):
        for value in scope_values:
            if scope_name == "overall":
                scoped_baseline, scoped_gated = baseline_run, gated_run
            else:
                positions = [index for index, row in enumerate(baseline_run["rows"]) if row[scope_key] == value]
                scoped_baseline = {
                    "rows": [baseline_run["rows"][i] for i in positions],
                    "ranks": [baseline_run["ranks"][i] for i in positions],
                }
                scoped_gated = {
                    "rows": [gated_run["rows"][i] for i in positions],
                    "ranks": [gated_run["ranks"][i] for i in positions],
                }
            top1 = top1_transition_matrix(
                [bool(row["correct_top1"]) for row in scoped_baseline["rows"]],
                [bool(row["correct_top1"]) for row in scoped_gated["rows"]],
            )
            top5 = top5_transition_matrix(
                [bool(row["correct_top5"]) for row in scoped_baseline["rows"]],
                [bool(row["correct_top5"]) for row in scoped_gated["rows"]],
            )
            rank_counts = rank_change_counts(scoped_baseline["ranks"], scoped_gated["ranks"])
            transition_rows.append(
                {
                    "scope": scope_name,
                    "scope_value": value,
                    "fallback_rate": _round4(_fallback_rate(scoped_gated["rows"])),
                    "top1_wrong_to_correct": top1["wrong_to_correct"],
                    "top1_correct_to_wrong": top1["correct_to_wrong"],
                    "top5_outside_to_inside": top5["outside_to_inside"],
                    "top5_inside_to_outside": top5["inside_to_outside"],
                    "rank_improved": rank_counts["improved"],
                    "rank_degraded": rank_counts["degraded"],
                    "rank_unchanged": rank_counts["unchanged"],
                }
            )
    write_csv(
        run_dir / "fallback_transitions_pilot32.csv",
        ["scope", "scope_value", "fallback_rate", "top1_wrong_to_correct", "top1_correct_to_wrong",
         "top5_outside_to_inside", "top5_inside_to_outside", "rank_improved", "rank_degraded", "rank_unchanged"],
        transition_rows,
    )
    return {
        "scenario_rows": scenario_rows,
        "subset_rows": subset_rows,
        "family_rows": family_rows,
        "transition_rows": transition_rows,
    }


def _render_report(
    summary_rows: list[dict[str, object]],
    latency_rows: list[dict[str, object]],
    fallback_rows: list[dict[str, object]],
    effectiveness_rows: list[dict[str, object]],
    signal_rows: list[dict[str, object]],
    pilot_payload: dict[str, Any],
    selected_gate: dict[str, Any],
    config: dict[str, object],
) -> str:
    lines: list[str] = []
    lines.append("# confidence_gated_v1 evaluation (frozen SigLIP2)")
    lines.append("")
    lines.append("Controlled milestone: baseline_full vs unconditional preprocessing_v1 vs confidence_gated_v1.")
    lines.append("The gate uses only the full-image ranking score distribution (top1 score / top1-top2 margin);")
    lines.append("fallback encodes exactly center_85 + center_70 in one batched forward and never recomputes")
    lines.append("the full embedding; aggregation is identical to preprocessing_v1 (mean of three views).")
    lines.append("")
    lines.append("## Selected gate (frozen)")
    lines.append("")
    lines.append(f"- Rule: `{selected_gate['rule_description']}`")
    lines.append(f"- Calibration seed: {selected_gate['calibration_seed']}, fingerprint `{str(selected_gate['calibration_fingerprint'])[:12]}...`")
    lines.append(f"- Calibration query sizes: {selected_gate['calibration_sizes']}")
    lines.append(f"- Candidates passing the clean guard: {selected_gate['survivors_of_clean_guard']}")
    lines.append("")
    lines.append("## Main comparison")
    lines.append("")
    lines.append("| Benchmark | Method | Top1 | R@5 | R@10 | MRR | Fallback% | Mean latency ms |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in summary_rows:
        lines.append(
            f"| {row['benchmark']} | {row['method']} | {_pct(row['top1_accuracy'])} | {_pct(row['recall_at_5'])} | "
            f"{_pct(row['recall_at_10'])} | {_num(row['mrr'], 4)} | {_pct(row['fallback_rate'])} | {_num(row['mean_latency_ms'], 1)} |"
        )
    lines.append("")
    lines.append("## Confidence signal analysis (Part B)")
    lines.append("")
    lines.append("| Benchmark | Scope | Signal | AUC top1-correct | AUC target-in-top5 | median correct | median incorrect |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in signal_rows:
        lines.append(
            f"| {row['benchmark']} | {row['scope']} | {row['signal']} | {_num(row['auc_top1_correct'], 3)} | "
            f"{_num(row['auc_target_in_top5'], 3)} | {_num(row['median_correct'], 4)} | {_num(row['median_incorrect'], 4)} |"
        )
    lines.append("")
    lines.append("## pilot32 per-scenario (baseline / v1 / gated, with gated fallback rate)")
    lines.append("")
    lines.append("| Scenario | baseline Top1 | v1 Top1 | gated Top1 | baseline R@5 | v1 R@5 | gated R@5 | gated fallback% |")
    lines.append("|---|---|---|---|---|---|---|---|")
    scenario_rows = {(row["scope_value"], row["method"]): row for row in pilot_payload["scenario_rows"]}
    for scenario in SCENARIOS:
        base = scenario_rows.get((scenario, BASELINE_FULL.name), {})
        v1 = scenario_rows.get((scenario, PREPROCESSING_V1.name), {})
        gated = scenario_rows.get((scenario, "confidence_gated_v1"), {})
        lines.append(
            f"| {scenario} | {_pct(base.get('top1_accuracy'))} | {_pct(v1.get('top1_accuracy'))} | {_pct(gated.get('top1_accuracy'))} | "
            f"{_pct(base.get('recall_at_5'))} | {_pct(v1.get('recall_at_5'))} | {_pct(gated.get('recall_at_5'))} | {_pct(gated.get('fallback_rate'))} |"
        )
    lines.append("")
    lines.append("## pilot32 per-subset")
    lines.append("")
    lines.append("| Subset | baseline Top1 | v1 Top1 | gated Top1 | baseline R@5 | v1 R@5 | gated R@5 | gated fallback% |")
    lines.append("|---|---|---|---|---|---|---|---|")
    subset_rows = {(row["scope_value"], row["method"]): row for row in pilot_payload["subset_rows"]}
    for subset in SUBSETS:
        base = subset_rows.get((subset, BASELINE_FULL.name), {})
        v1 = subset_rows.get((subset, PREPROCESSING_V1.name), {})
        gated = subset_rows.get((subset, "confidence_gated_v1"), {})
        lines.append(
            f"| {subset} | {_pct(base.get('top1_accuracy'))} | {_pct(v1.get('top1_accuracy'))} | {_pct(gated.get('top1_accuracy'))} | "
            f"{_pct(base.get('recall_at_5'))} | {_pct(v1.get('recall_at_5'))} | {_pct(gated.get('recall_at_5'))} | {_pct(gated.get('fallback_rate'))} |"
        )
    lines.append("")
    lines.append("## pilot32 family metrics (hard subset)")
    lines.append("")
    lines.append("| Method | family_type | exact Top1 | family Top1 | family R@5 | disambiguation | fallback% |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in pilot_payload["family_rows"]:
        lines.append(
            f"| {row['method']} | {row['family_type']} | {_pct(row['exact_top1'])} | {_pct(row['family_top1'])} | "
            f"{_pct(row['family_recall_at_5'])} | {_pct(row['within_family_disambiguation_top1'])} | {_pct(row['fallback_rate'])} |"
        )
    lines.append("")
    lines.append("## Gate confusion matrix (Part M)")
    lines.append("")
    lines.append("| Benchmark | trusted correct | unnecessary fallback | useful trigger | missed opportunity |")
    lines.append("|---|---|---|---|---|")
    for row in fallback_rows:
        lines.append(
            f"| {row['benchmark']} | {row['trusted_correct']} ({_pct(row['trusted_correct_share'])}) | "
            f"{row['unnecessary_fallback']} ({_pct(row['unnecessary_fallback_share'])}) | "
            f"{row['useful_trigger']} ({_pct(row['useful_trigger_share'])}) | "
            f"{row['missed_opportunity']} ({_pct(row['missed_opportunity_share'])}) |"
        )
    lines.append("")
    lines.append("## Fallback effectiveness among triggered queries (Part N)")
    lines.append("")
    lines.append("| Benchmark | fallback queries | Top1 wrong->correct | Top1 correct->wrong | Top5 out->in | Top5 in->out |")
    lines.append("|---|---|---|---|---|---|")
    for row in effectiveness_rows:
        lines.append(
            f"| {row['benchmark']} | {row['fallback_queries']} | {row['top1_wrong_to_correct']} | "
            f"{row['top1_correct_to_wrong']} | {row['top5_outside_to_inside']} | {row['top5_inside_to_outside']} |"
        )
    lines.append("")
    lines.append("## Latency (production-like)")
    lines.append("")
    lines.append("| Benchmark | Method | mean ms | p50 ms | p95 ms | fallback% | views/query |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in latency_rows:
        lines.append(
            f"| {row['benchmark']} | {row['method']} | {_num(row['mean_latency_ms'], 1)} | {_num(row['p50_latency_ms'], 1)} | "
            f"{_num(row['p95_latency_ms'], 1)} | {_pct(row['fallback_rate'])} | {_num(row['effective_views_per_query'], 2)} |"
        )
    lines.append("")
    lines.append(f"Gate reproducibility mismatches on calibration queries: {config['gate_reproducibility_mismatches_on_calibration']}.")
    lines.append(f"Run directory: `artifacts/experiments/{config['run_id']}/`.")
    lines.append("")
    return "\n".join(lines) + "\n"


def _pct(value: object) -> str:
    if value in (None, ""):
        return "-"
    return f"{float(value) * 100:.2f}%"


def _num(value: object, digits: int) -> str:
    if value in (None, ""):
        return "-"
    return f"{float(value):.{digits}f}"


def _round2(value: float) -> float:
    return round(float(value), 2)


def _round4(value: object) -> float | str:
    if value in (None, ""):
        return ""
    return round(float(value), 4)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], percentage: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100.0
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


if __name__ == "__main__":
    main()
