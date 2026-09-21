#!/usr/bin/env python3
"""Compare baseline_full vs preprocessing_v1 on the frozen benchmarks.

preprocessing_v1 encodes three deterministic views of every query image
(full, center 85%, center 70%) in one batched forward pass with the same
frozen SigLIP2, then ranks the unchanged cached reference embeddings by the
mean similarity across the three views.

Guarantees:
- one fixed strategy per method for all queries of a benchmark;
- no target label, scenario, or subset is consulted during inference;
- catalog references and cached reference embeddings are unchanged;
- benchmark manifests are validated for alignment before any inference.
"""

from __future__ import annotations

import argparse
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
from recognition.preprocessing import load_rgb_image  # noqa: E402
from recognition.reporting import write_csv  # noqa: E402
from recognition.view_strategy import (  # noqa: E402
    BASELINE_FULL,
    PREPROCESSING_V1,
    STRATEGIES,
    QueryViewStrategy,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preprocessing_v1 comparison on frozen benchmarks")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--catalog-version", default="data/processed/catalog_v1.json")
    parser.add_argument("--catalog-manifest", default="data/processed/catalog_manifest.csv")
    parser.add_argument("--cache", default="artifacts/catalog_embeddings/siglip2_catalog.pt")
    parser.add_argument("--artifacts-root", default="artifacts/experiments")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(BENCHMARKS),
        choices=BENCHMARKS,
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=[BASELINE_FULL.name, PREPROCESSING_V1.name],
        choices=list(STRATEGIES),
    )
    parser.add_argument(
        "--report",
        default="reports/preprocessing_v1_evaluation.md",
    )
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

    run_id = datetime.now(timezone.utc).strftime("siglip2_preprocessing_v1_%Y%m%dT%H%M%SZ")
    run_dir = _path(project_root, args.artifacts_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    strategies = [STRATEGIES[name] for name in args.strategies]
    summary_rows: list[dict[str, object]] = []
    latency_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    per_benchmark_payload: dict[str, dict[str, Any]] = {}

    for benchmark in args.benchmarks:
        manifest_path = _default_benchmark_manifest(project_root, benchmark, None)
        manifest_rows = _read_csv(manifest_path)
        _validate_benchmark(project_root, manifest_rows, catalog_items, benchmark)
        # Alignment is enforced again against query_id/target_slug pairs.
        manifest_by_id: dict[str, dict[str, str]] = {}
        for row in manifest_rows:
            if row["query_id"] in manifest_by_id:
                raise ValueError(f"Duplicate query_id in manifest: {row['query_id']}")
            manifest_by_id[row["query_id"]] = row

        results: dict[str, dict[str, Any]] = {}
        for strategy in strategies:
            print(f"[{benchmark}] running strategy {strategy.name} ...")
            results[strategy.name] = _run_strategy(
                encoder, benchmark, manifest_rows, reference_matrix, catalog_slugs, strategy, device, project_root
            )
        baseline_result = results[BASELINE_FULL.name]
        for strategy in strategies:
            result = results[strategy.name]
            metrics = _metrics_payload(
                benchmark, strategy, result, len(manifest_rows), len({row["target_slug"] for row in manifest_rows})
            )
            out_prefix = f"{benchmark}_{strategy.name}"
            write_csv(
                run_dir / f"predictions_{out_prefix}.csv",
                [
                    "query_id", "target_slug", "predicted_slug", "top1_score",
                    "top5_slugs", "top5_scores", "correct_top1", "correct_top5",
                    "target_rank", "target_score", "embedding_latency_ms",
                    "retrieval_latency_ms", "latency_ms",
                ],
                result["prediction_rows"],
            )
            (run_dir / f"metrics_{out_prefix}.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            summary_rows.append(
                {
                    "benchmark": benchmark,
                    "method": strategy.name,
                    "queries": len(manifest_rows),
                    "unique_target_slugs": len({row["target_slug"] for row in manifest_rows}),
                    **_metric_cells(result["ranks"]),
                    **_latency_cells(result["latencies_ms"]),
                }
            )
            latency_rows.append(
                {
                    "benchmark": benchmark,
                    "method": strategy.name,
                    **_latency_cells(result["latencies_ms"]),
                    "embedding_mean_ms": _round2(_mean([row["embedding_latency_ms"] for row in result["prediction_rows"]])),
                    "retrieval_mean_ms": _round2(_mean([row["retrieval_latency_ms"] for row in result["prediction_rows"]])),
                }
            )

        baseline_ranks = baseline_result["ranks"]
        for strategy in strategies:
            if strategy.name == BASELINE_FULL.name:
                continue
            result = results[strategy.name]
            transitions = _transition_payload(baseline_result, result)
            transition_rows.append({"benchmark": benchmark, "method": strategy.name, **transitions})

        benchmark_payload: dict[str, Any] = {
            "metrics": {
                strategy.name: _metrics_payload(
                    benchmark, strategy, results[strategy.name], len(manifest_rows),
                    len({row["target_slug"] for row in manifest_rows}),
                )
                for strategy in strategies
            }
        }
        if benchmark == "generated_stress_dev_pilot32":
            benchmark_payload.update(
                _write_pilot32_breakdown(
                    run_dir,
                    manifest_rows,
                    results,
                    strategies,
                    baseline_result,
                )
            )
            benchmark_payload["family_diagnostics"] = {
                strategy.name: family_diagnostics(
                    results[strategy.name]["prediction_rows"], manifest_rows
                )
                for strategy in strategies
            }
        per_benchmark_payload[benchmark] = benchmark_payload

    write_csv(
        run_dir / "benchmark_summary.csv",
        ["benchmark", "method", "queries", "unique_target_slugs", "top1_accuracy", "recall_at_5",
         "recall_at_10", "mrr", "median_target_rank_found", "missing_count",
         "mean_latency_ms", "p50_latency_ms", "p95_latency_ms"],
        summary_rows,
    )
    write_csv(
        run_dir / "transition_summary.csv",
        ["benchmark", "method", "top1_wrong_to_correct", "top1_correct_to_wrong",
         "top1_unchanged_correct", "top1_unchanged_wrong", "top5_outside_to_inside",
         "top5_inside_to_outside", "top5_stayed_inside", "top5_stayed_outside",
         "rank_improved", "rank_degraded", "rank_unchanged"],
        transition_rows,
    )
    write_csv(
        run_dir / "latency_summary.csv",
        ["benchmark", "method", "mean_latency_ms", "p50_latency_ms", "p95_latency_ms",
         "embedding_mean_ms", "retrieval_mean_ms"],
        latency_rows,
    )

    config = {
        "run_id": run_id,
        "purpose": "controlled comparison of baseline_full vs preprocessing_v1 on frozen benchmarks",
        "model_id": encoder.model_id,
        "model_version": encoder.model_version,
        "device": device,
        "catalog_version": catalog_meta["catalog_version"],
        "catalog_manifest_sha256": catalog_meta["manifest_sha256"],
        "catalog_embedding_cache_status": cache_status,
        "reference_side": "unchanged: one embedding per canonical reference image from the validated cache",
        "strategies": {
            BASELINE_FULL.name: {
                "views": list(BASELINE_FULL.view_names),
                "aggregation": BASELINE_FULL.aggregation,
            },
            PREPROCESSING_V1.name: {
                "views": list(PREPROCESSING_V1.view_names),
                "aggregation": PREPROCESSING_V1.aggregation,
                "view_batching": "three views of one query are encoded in one forward pass (batch of 3)",
                "crop_policy": "deterministic centered crops (round-half-even int scaling), identical for all queries",
            },
        },
        "score_aggregation": "final_score = mean(similarity_full, similarity_crop85, similarity_crop70)",
        "leakage_guards": [
            "strategy views depend only on image pixels",
            "same strategy for every query; no scenario/subset-specific parameters",
            "ground truth used only after inference for scoring",
        ],
        "benchmarks": {
            benchmark: {
                "manifest": str(_default_benchmark_manifest(project_root, benchmark, None).relative_to(project_root)),
                "manifest_sha256": sha256_file(_default_benchmark_manifest(project_root, benchmark, None)),
                "queries": len(_read_csv(_default_benchmark_manifest(project_root, benchmark, None))),
            }
            for benchmark in args.benchmarks
        },
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report = _render_report(summary_rows, latency_rows, transition_rows, per_benchmark_payload, config)
    report_path = _path(project_root, args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))
    print(f"Artifacts: {run_dir}")
    print(f"Report: {report_path}")


def _run_strategy(
    encoder: VisualEncoder,
    benchmark: str,
    manifest_rows: list[dict[str, str]],
    reference_matrix: Any,
    catalog_slugs: list[str],
    strategy: QueryViewStrategy,
    device: str,
    project_root: Path,
) -> dict[str, Any]:
    """Run one fixed strategy over one benchmark manifest with per-query latency."""

    import torch

    prediction_rows: list[dict[str, object]] = []
    ranks: list[int | None] = []
    latencies_ms: list[float] = []
    for position, manifest_row in enumerate(manifest_rows, start=1):
        image = load_rgb_image(_path(project_root, manifest_row["query_path"]))
        views = strategy_views(image, strategy)

        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        started = perf_counter()
        embeddings = torch.tensor(encoder.encode_pil(views), dtype=torch.float32, device=device)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        embedding_ms = (perf_counter() - started) * 1000.0

        retrieval_started = perf_counter()
        view_score_vectors = (embeddings @ reference_matrix.transpose(0, 1)).detach().cpu().tolist()
        final_scores = aggregate_strategy_scores(view_score_vectors, strategy)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        retrieval_ms = (perf_counter() - retrieval_started) * 1000.0

        ranked = full_ranking(final_scores, catalog_slugs)
        ranked_slugs = [slug for slug, _ in ranked]
        target_slug = manifest_row["target_slug"]
        rank = target_rank_full(ranked_slugs, target_slug)
        ranks.append(rank)
        total_ms = embedding_ms + retrieval_ms
        latencies_ms.append(total_ms)
        top5_scores = [score for _, score in ranked[:5]]
        prediction_rows.append(
            {
                "query_id": manifest_row["query_id"],
                "target_slug": target_slug,
                "predicted_slug": ranked_slugs[0],
                "top1_score": ranked[0][1],
                "top5_slugs": json.dumps(ranked_slugs[:5], ensure_ascii=False, separators=(",", ":")),
                "top5_scores": json.dumps(top5_scores, separators=(",", ":")),
                "correct_top1": ranked_slugs[0] == target_slug,
                "correct_top5": rank is not None and rank <= 5,
                "target_rank": rank if rank is not None else "",
                "target_score": top5_scores[rank - 1] if rank is not None and rank <= 5 else "",
                "embedding_latency_ms": _round2(embedding_ms),
                "retrieval_latency_ms": _round2(retrieval_ms),
                "latency_ms": _round2(total_ms),
            }
        )
        if position % 500 == 0 or position == len(manifest_rows):
            print(f"  {position}/{len(manifest_rows)}")
    return {
        "prediction_rows": prediction_rows,
        "ranks": ranks,
        "latencies_ms": latencies_ms,
    }


def _metrics_payload(
    benchmark: str,
    strategy: QueryViewStrategy,
    result: dict[str, Any],
    query_count: int,
    unique_targets: int,
) -> dict[str, object]:
    metrics = rank_metrics(result["ranks"])
    latencies = result["latencies_ms"]
    return {
        "benchmark": benchmark,
        "method": strategy.name,
        "views": list(strategy.view_names),
        "aggregation": strategy.aggregation,
        "query_count": query_count,
        "unique_target_slugs": unique_targets,
        **metrics,
        "latency_ms": {
            "mean": _round2(_mean(latencies)),
            "p50": _round2(statistics.median(latencies)),
            "p95": _round2(_percentile(latencies, 95)),
        },
        "note": "full-catalog ranking metrics; MRR uses the full-catalog target rank",
    }


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


def _latency_cells(latencies_ms: list[float]) -> dict[str, object]:
    return {
        "mean_latency_ms": _round2(_mean(latencies_ms)),
        "p50_latency_ms": _round2(statistics.median(latencies_ms)),
        "p95_latency_ms": _round2(_percentile(latencies_ms, 95)),
    }


def _transition_payload(
    baseline_result: dict[str, Any], result: dict[str, Any]
) -> dict[str, object]:
    baseline_rows = baseline_result["prediction_rows"]
    new_rows = result["prediction_rows"]
    top1 = top1_transition_matrix(
        [bool(row["correct_top1"]) for row in baseline_rows],
        [bool(row["correct_top1"]) for row in new_rows],
    )
    top5 = top5_transition_matrix(
        [bool(row["correct_top5"]) for row in baseline_rows],
        [bool(row["correct_top5"]) for row in new_rows],
    )
    rank_counts = rank_change_counts(baseline_result["ranks"], result["ranks"])
    return {
        "top1_wrong_to_correct": top1["wrong_to_correct"],
        "top1_correct_to_wrong": top1["correct_to_wrong"],
        "top1_unchanged_correct": top1["unchanged_correct"],
        "top1_unchanged_wrong": top1["unchanged_wrong"],
        "top5_outside_to_inside": top5["outside_to_inside"],
        "top5_inside_to_outside": top5["inside_to_outside"],
        "top5_stayed_inside": top5["stayed_inside"],
        "top5_stayed_outside": top5["stayed_outside"],
        "rank_improved": rank_counts["improved"],
        "rank_degraded": rank_counts["degraded"],
        "rank_unchanged": rank_counts["unchanged"],
    }


def _subset_rows(rows: list[dict[str, object]], subset: str) -> list[dict[str, object]]:
    return [row for row in rows if row.get("subset_role") == subset]


def _write_pilot32_breakdown(
    run_dir: Path,
    manifest_rows: list[dict[str, str]],
    results: dict[str, dict[str, Any]],
    strategies: list[QueryViewStrategy],
    baseline_result: dict[str, Any],
) -> dict[str, Any]:
    """Write scenario/subset/family breakdowns and scoped transitions for pilot32."""

    scenario_by_query = {row["query_id"]: row.get("scenario_id", "") for row in manifest_rows}
    subset_by_query = {row["query_id"]: row.get("subset_role", "") for row in manifest_rows}
    for result in results.values():
        for row in result["prediction_rows"]:
            row["scenario_id"] = scenario_by_query[str(row["query_id"])]
            row["subset_role"] = subset_by_query[str(row["query_id"])]

    def scoped_metrics(strategy: QueryViewStrategy, key: str, value: str) -> dict[str, object]:
        rows = results[strategy.name]["prediction_rows"]
        ranks = results[strategy.name]["ranks"]
        pairs = [(row, rank) for row, rank in zip(rows, ranks) if row[key] == value]
        return _metric_cells([rank for _, rank in pairs])

    scenario_rows = []
    scenario_report_rows = []
    for scenario in SCENARIOS:
        for strategy in strategies:
            scenario_rows.append(
                {"method": strategy.name, "scenario_id": scenario, **scoped_metrics(strategy, "scenario_id", scenario)}
            )
        baseline_cell = scoped_metrics(BASELINE_FULL, "scenario_id", scenario)
        v1_cell = scoped_metrics(PREPROCESSING_V1, "scenario_id", scenario) if PREPROCESSING_V1 in strategies else {}
        scenario_report_rows.append(
            {
                "scenario": scenario,
                "baseline_top1": baseline_cell.get("top1_accuracy"),
                "baseline_r5": baseline_cell.get("recall_at_5"),
                "v1_top1": v1_cell.get("top1_accuracy"),
                "v1_r5": v1_cell.get("recall_at_5"),
            }
        )
    write_csv(
        run_dir / "scenario_metrics.csv",
        ["method", "scenario_id", "top1_accuracy", "recall_at_5", "recall_at_10", "mrr", "median_target_rank_found", "missing_count"],
        scenario_rows,
    )

    subset_rows = []
    subset_report_rows = []
    for subset in SUBSETS:
        for strategy in strategies:
            subset_rows.append(
                {"method": strategy.name, "subset_role": subset, **scoped_metrics(strategy, "subset_role", subset)}
            )
        baseline_cell = scoped_metrics(BASELINE_FULL, "subset_role", subset)
        v1_cell = scoped_metrics(PREPROCESSING_V1, "subset_role", subset) if PREPROCESSING_V1 in strategies else {}
        subset_report_rows.append(
            {
                "subset": subset,
                "baseline_top1": baseline_cell.get("top1_accuracy"),
                "baseline_r5": baseline_cell.get("recall_at_5"),
                "v1_top1": v1_cell.get("top1_accuracy"),
                "v1_r5": v1_cell.get("recall_at_5"),
            }
        )
    write_csv(
        run_dir / "subset_metrics.csv",
        ["method", "subset_role", "top1_accuracy", "recall_at_5", "recall_at_10", "mrr", "median_target_rank_found", "missing_count"],
        subset_rows,
    )

    family_rows = []
    for strategy in strategies:
        diagnostics = family_diagnostics(results[strategy.name]["prediction_rows"], manifest_rows)
        for family_type, breakdown in sorted(diagnostics["family_type_breakdown"].items()):
            family_rows.append(
                {
                    "method": strategy.name,
                    "family_type": family_type,
                    "query_count": breakdown["query_count"],
                    "exact_top1": breakdown["top1_accuracy"],
                    "family_top1": breakdown["family_top1"],
                    "family_recall_at_5": breakdown["family_recall_at_5"],
                    "within_family_disambiguation_top1": breakdown["within_family_disambiguation_top1"],
                }
            )
        family_rows.append(
            {
                "method": strategy.name,
                "family_type": "ALL",
                "query_count": diagnostics["hard_query_count"],
                "exact_top1": scoped_metrics(strategy, "subset_role", "hard")["top1_accuracy"],
                "family_top1": diagnostics["family_top1"],
                "family_recall_at_5": diagnostics["family_recall_at_5"],
                "within_family_disambiguation_top1": diagnostics["within_family_disambiguation_top1"],
            }
        )
    write_csv(
        run_dir / "family_metrics.csv",
        ["method", "family_type", "query_count", "exact_top1", "family_top1", "family_recall_at_5", "within_family_disambiguation_top1"],
        family_rows,
    )

    transition_rows = []
    baseline_rows = baseline_result["prediction_rows"]
    if PREPROCESSING_V1 in strategies:
        strategy_rows = results[PREPROCESSING_V1.name]["prediction_rows"]
        transition_rows.append(
            {
                "scope": "overall",
                "scope_value": "",
                "method": PREPROCESSING_V1.name,
                **_transition_payload(baseline_result, results[PREPROCESSING_V1.name]),
            }
        )
        for scenario in SCENARIOS:
            positions = [index for index, row in enumerate(baseline_rows) if row["scenario_id"] == scenario]
            transition_rows.append(
                {
                    "scope": "scenario",
                    "scope_value": scenario,
                    "method": PREPROCESSING_V1.name,
                    **_transition_payload(
                        _subset_result(baseline_result, positions),
                        _subset_result(results[PREPROCESSING_V1.name], positions),
                    ),
                }
            )
        for subset in SUBSETS:
            positions = [index for index, row in enumerate(baseline_rows) if row["subset_role"] == subset]
            transition_rows.append(
                {
                    "scope": "subset",
                    "scope_value": subset,
                    "method": PREPROCESSING_V1.name,
                    **_transition_payload(
                        _subset_result(baseline_result, positions),
                        _subset_result(results[PREPROCESSING_V1.name], positions),
                    ),
                }
            )
    write_csv(
        run_dir / "transitions.csv",
        ["scope", "scope_value", "method", "top1_wrong_to_correct", "top1_correct_to_wrong",
         "top1_unchanged_correct", "top1_unchanged_wrong", "top5_outside_to_inside",
         "top5_inside_to_outside", "top5_stayed_inside", "top5_stayed_outside",
         "rank_improved", "rank_degraded", "rank_unchanged"],
        transition_rows,
    )
    return {
        "scenario_metrics": scenario_report_rows,
        "subset_metrics": subset_report_rows,
        "family_metrics": family_rows,
    }


def _subset_result(result: dict[str, Any], positions: list[int]) -> dict[str, Any]:
    return {
        "prediction_rows": [result["prediction_rows"][index] for index in positions],
        "ranks": [result["ranks"][index] for index in positions],
    }


def _render_report(
    summary_rows: list[dict[str, object]],
    latency_rows: list[dict[str, object]],
    transition_rows: list[dict[str, object]],
    per_benchmark_payload: dict[str, dict[str, Any]],
    config: dict[str, object],
) -> str:
    lines: list[str] = []
    lines.append("# preprocessing_v1 evaluation (frozen SigLIP2)")
    lines.append("")
    lines.append("Controlled comparison of `baseline_full` vs `preprocessing_v1` on the frozen benchmarks.")
    lines.append("preprocessing_v1 = full + center 85% + center 70% views of the query, encoded with the")
    lines.append("same frozen SigLIP2 in one batched forward pass, ranked by mean similarity against the")
    lines.append("unchanged cached reference embeddings. One fixed strategy for every query; no target,")
    lines.append("scenario, or subset information is used during inference.")
    lines.append("")
    lines.append("Fixed reasoning (from the pilot32 crop diagnostics, not reinterpreted):")
    lines.append("")
    lines.append("- full image keeps overall context;")
    lines.append("- crop85 gives a mild zoom-in;")
    lines.append("- crop70 further reduces background influence;")
    lines.append("- crop55 is excluded: the diagnostics showed it helps distance_crop but clearly degrades glare/normal cases;")
    lines.append("- mean aggregation is preferred over max as the more stable strategy.")
    lines.append("")
    lines.append("## Main comparison")
    lines.append("")
    lines.append("| Benchmark | Method | Top1 | R@5 | R@10 | MRR | MedianRank | MeanLatency ms |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in summary_rows:
        lines.append(
            f"| {row['benchmark']} | {row['method']} | {_pct(row['top1_accuracy'])} | {_pct(row['recall_at_5'])} | "
            f"{_pct(row['recall_at_10'])} | {_num(row['mrr'], 4)} | {_num(row['median_target_rank_found'], 1)} | "
            f"{_num(row['mean_latency_ms'], 1)} |"
        )
    lines.append("")
    lines.append("## Generated stress pilot32: per-scenario")
    lines.append("")
    lines.append("| Scenario | baseline Top1 | v1 Top1 | baseline R@5 | v1 R@5 |")
    lines.append("|---|---|---|---|---|")
    pilot_payload = per_benchmark_payload.get("generated_stress_dev_pilot32", {})
    for row in pilot_payload.get("scenario_metrics", []):
        lines.append(
            f"| {row['scenario']} | {_pct(row['baseline_top1'])} | "
            f"{_pct(row['v1_top1'])} | {_pct(row['baseline_r5'])} | {_pct(row['v1_r5'])} |"
        )
    lines.append("")
    lines.append("## Generated stress pilot32: per-subset")
    lines.append("")
    lines.append("| Subset | baseline Top1 | v1 Top1 | baseline R@5 | v1 R@5 |")
    lines.append("|---|---|---|---|---|")
    for row in pilot_payload.get("subset_metrics", []):
        lines.append(
            f"| {row['subset']} | {_pct(row['baseline_top1'])} | "
            f"{_pct(row['v1_top1'])} | {_pct(row['baseline_r5'])} | {_pct(row['v1_r5'])} |"
        )
    lines.append("")
    lines.append("## Generated stress pilot32: family metrics (hard subset)")
    lines.append("")
    lines.append("| Method | family_type | exact Top1 | family Top1 | family R@5 | within-family disambiguation |")
    lines.append("|---|---|---|---|---|---|")
    for row in pilot_payload.get("family_metrics", []):
        lines.append(
            f"| {row['method']} | {row['family_type']} | {_pct(row['exact_top1'])} | {_pct(row['family_top1'])} | "
            f"{_pct(row['family_recall_at_5'])} | {_pct(row['within_family_disambiguation_top1'])} |"
        )
    lines.append("")
    lines.append("## Transitions (baseline_full -> preprocessing_v1)")
    lines.append("")
    lines.append("| Benchmark | Top1 wrong->correct | correct->wrong | unchanged correct | unchanged wrong | Top5 out->in | in->out | stayed in | stayed out | rank improved | degraded | unchanged |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in transition_rows:
        lines.append(
            f"| {row['benchmark']} | {row['top1_wrong_to_correct']} | {row['top1_correct_to_wrong']} | "
            f"{row['top1_unchanged_correct']} | {row['top1_unchanged_wrong']} | {row['top5_outside_to_inside']} | "
            f"{row['top5_inside_to_outside']} | {row['top5_stayed_inside']} | {row['top5_stayed_outside']} | "
            f"{row['rank_improved']} | {row['rank_degraded']} | {row['rank_unchanged']} |"
        )
    lines.append("")
    lines.append("## Latency")
    lines.append("")
    lines.append("| Benchmark | Method | mean ms | p50 ms | p95 ms |")
    lines.append("|---|---|---|---|---|")
    for row in latency_rows:
        lines.append(
            f"| {row['benchmark']} | {row['method']} | {_num(row['mean_latency_ms'], 1)} | "
            f"{_num(row['p50_latency_ms'], 1)} | {_num(row['p95_latency_ms'], 1)} |"
        )
    lines.append("")
    lines.append("## Acceptance criteria assessment")
    lines.append("")
    lines.append(_acceptance_assessment(summary_rows, latency_rows))
    lines.append("")
    lines.append("Note: metrics use full-catalog target ranks. Frozen-run MRR values published earlier for")
    lines.append("pilot32 (0.3251) were computed from Top-5 truncated rankings; full-rank baseline MRR here is")
    lines.append("directly comparable only to other full-rank rows of this report.")
    lines.append("")
    lines.append(f"Run directory: `artifacts/experiments/{config['run_id']}/`.")
    lines.append("")
    return "\n".join(lines) + "\n"


def _acceptance_assessment(
    summary_rows: list[dict[str, object]], latency_rows: list[dict[str, object]]
) -> str:
    def cell(benchmark: str, method: str, key: str) -> object:
        for row in summary_rows:
            if row["benchmark"] == benchmark and row["method"] == method:
                return row[key]
        return None

    def latency_cell(benchmark: str, method: str, key: str) -> object:
        for row in latency_rows:
            if row["benchmark"] == benchmark and row["method"] == method:
                return row[key]
        return None

    pilot_v1_r5 = cell("generated_stress_dev_pilot32", PREPROCESSING_V1.name, "recall_at_5")
    pilot_base_r5 = cell("generated_stress_dev_pilot32", BASELINE_FULL.name, "recall_at_5")
    synthetic_base_top1 = cell("synthetic_dev", BASELINE_FULL.name, "top1_accuracy")
    synthetic_v1_top1 = cell("synthetic_dev", PREPROCESSING_V1.name, "top1_accuracy")
    synthetic_base_r5 = cell("synthetic_dev", BASELINE_FULL.name, "recall_at_5")
    synthetic_v1_r5 = cell("synthetic_dev", PREPROCESSING_V1.name, "recall_at_5")
    hard_base_top1 = cell("hard_near_duplicate_dev_v2", BASELINE_FULL.name, "top1_accuracy")
    hard_v1_top1 = cell("hard_near_duplicate_dev_v2", PREPROCESSING_V1.name, "top1_accuracy")
    hard_base_r5 = cell("hard_near_duplicate_dev_v2", BASELINE_FULL.name, "recall_at_5")
    hard_v1_r5 = cell("hard_near_duplicate_dev_v2", PREPROCESSING_V1.name, "recall_at_5")
    v1_mean = latency_cell("synthetic_dev", PREPROCESSING_V1.name, "mean_latency_ms")
    base_mean = latency_cell("synthetic_dev", BASELINE_FULL.name, "mean_latency_ms")

    checks = [
        (
            "generated stress R@5 growth vs baseline",
            pilot_v1_r5 is not None and pilot_base_r5 is not None and float(pilot_v1_r5) > float(pilot_base_r5),
            f"{_pct(pilot_base_r5)} -> {_pct(pilot_v1_r5)}",
        ),
        (
            "synthetic_dev minimal degradation (Top-1 drop <= 1pp and R@5 drop <= 0.2pp)",
            (
                synthetic_base_top1 is not None and synthetic_v1_top1 is not None
                and float(synthetic_base_top1) - float(synthetic_v1_top1) <= 0.01
                and float(synthetic_base_r5) - float(synthetic_v1_r5) <= 0.002
            ),
            f"Top1 {_pct(synthetic_base_top1)} -> {_pct(synthetic_v1_top1)}, R@5 {_pct(synthetic_base_r5)} -> {_pct(synthetic_v1_r5)}",
        ),
        (
            "hard_v2 no substantial regression (Top-1 drop <= 1pp and R@5 drop <= 0.2pp)",
            (
                hard_base_top1 is not None and hard_v1_top1 is not None
                and float(hard_base_top1) - float(hard_v1_top1) <= 0.01
                and float(hard_base_r5) - float(hard_v1_r5) <= 0.002
            ),
            f"Top1 {_pct(hard_base_top1)} -> {_pct(hard_v1_top1)}, R@5 {_pct(hard_base_r5)} -> {_pct(hard_v1_r5)}",
        ),
        (
            "latency stays far below the 3 s hackathon SLA",
            v1_mean is not None and float(v1_mean) < 3000,
            f"v1 mean latency {_num(v1_mean, 1)} ms (baseline {_num(base_mean, 1)} ms)",
        ),
    ]
    lines = []
    for name, passed, evidence in checks:
        lines.append(f"- {'PASS' if passed else 'FAIL'}: {name} ({evidence})")
    return "\n".join(lines)


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
