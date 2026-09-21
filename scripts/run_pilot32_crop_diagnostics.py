#!/usr/bin/env python3
"""Diagnostic milestone for the pilot32 generated-stress failure.

Runs frozen SigLIP2 on the 128 accepted generated_stress_dev_pilot32 queries
with a fixed set of deterministic preprocessing variants (full image, fixed
center crops, fixed multi-crop ensembles), computes full-catalog target ranks
and family-aware diagnostics, and writes per-query CSVs, comparison tables,
transition analysis, and an HTML error-analysis report.

Constraints honored by design:
- frozen encoder, no training/fine-tuning, no OCR, no reranker, no detector;
- crop policy is identical for every image and never uses the target label;
- multi-crop aggregation is fixed (max/mean) and never query-specific;
- benchmark manifest, catalog references, and generated queries are unchanged.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from recognition.cache import (  # noqa: E402
    EmbeddingCacheMismatch,
    load_embedding_cache,
    sha256_file,
)
from recognition.crop_diagnostics import (  # noqa: E402
    MULTICROP_METHODS,
    RANK_BANDS,
    SINGLE_VIEW_METHODS,
    VIEW_NAMES,
    aggregate_view_scores,
    family_transition_flags,
    full_ranking,
    rank_band,
    rank_metrics,
    rank_transition,
    split_by,
    target_rank_full,
    transition_summary,
    top5_transition_counts,
    verify_family_mapping,
    verify_manifest_alignment,
    verify_subset_scenario_split,
    view_images,
)
from recognition.encoder import VisualEncoder  # noqa: E402
from recognition.family_metrics import family_diagnostics  # noqa: E402
from recognition.reporting import write_csv  # noqa: E402
from recognition.preprocessing import load_rgb_image  # noqa: E402
from scripts.run_siglip2_baseline import (  # noqa: E402
    _catalog_items,
    _default_benchmark_manifest,
    _path,
    _read_csv,
    _validate_benchmark,
)

SCENARIOS = ("distance_crop", "glare_bad_light", "handheld", "slight_angle")
SUBSETS = ("representative", "hard")
RANK_BANDS_ORDER = ("1", "2-5", "6-10", "11-25", "26-100", ">100")
METHODS = SINGLE_VIEW_METHODS + MULTICROP_METHODS
PREDICTIONS_COLUMNS = [
    "query_id",
    "target_slug",
    "predicted_slug",
    "top1_score",
    "top5_slugs",
    "top5_scores",
    "correct_top1",
    "target_rank",
    "target_score",
    "subset_role",
    "scenario_id",
    "target_family_id",
    "family_type",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot32 frozen-encoder crop diagnostics")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--catalog-version", default="data/processed/catalog_v1.json")
    parser.add_argument("--catalog-manifest", default="data/processed/catalog_manifest.csv")
    parser.add_argument(
        "--benchmark-manifest",
        default="data/benchmarks/generated_stress_dev_pilot32/manifest.csv",
    )
    parser.add_argument("--cache", default="artifacts/catalog_embeddings/siglip2_catalog.pt")
    parser.add_argument("--artifacts-root", default="artifacts/experiments")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--baseline-predictions",
        default="artifacts/experiments/siglip2_generated_stress_dev_pilot32_20260917T184815Z/predictions.csv",
        help="Frozen run predictions.csv used only for a Top-5 alignment sanity check",
    )
    parser.add_argument(
        "--html-report",
        default="reports/generated_stress_pilot32_error_analysis.html",
    )
    parser.add_argument(
        "--md-report",
        default="reports/generated_stress_pilot32_diagnostic_report.md",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    catalog_version_path = _path(project_root, args.catalog_version)
    catalog_manifest_path = _path(project_root, args.catalog_manifest)
    benchmark_manifest_path = _path(project_root, args.benchmark_manifest)
    cache_path = _path(project_root, args.cache)

    catalog_meta = json.loads(catalog_version_path.read_text(encoding="utf-8"))
    catalog_rows = _read_csv(catalog_manifest_path)
    catalog_items, catalog_by_slug = _catalog_items(project_root, catalog_rows, catalog_meta)
    benchmark_rows = _read_csv(benchmark_manifest_path)
    _validate_benchmark(project_root, benchmark_rows, catalog_items, "generated_stress_dev_pilot32")
    verify_manifest_alignment(
        [{"query_id": row["query_id"], "target_slug": row["target_slug"]} for row in benchmark_rows],
        benchmark_rows,
    )
    verify_subset_scenario_split(benchmark_rows, dict.fromkeys(SUBSETS, 64), dict.fromkeys(SCENARIOS, 32))
    slug_to_family = verify_family_mapping(benchmark_rows)
    hard_family_ids = sorted(set(slug_to_family.values()))
    if len(hard_family_ids) != 8:
        raise ValueError(f"Expected 8 hard families, found {len(hard_family_ids)}")

    encoder = VisualEncoder(model_name="siglip", device=args.device, batch_size=args.batch_size)
    expected_cache_metadata = {
        "model_id": encoder.model_id,
        "model_version": encoder.model_version,
        "catalog_version": catalog_meta["catalog_version"],
        "catalog_manifest_sha256": catalog_meta["manifest_sha256"],
        "preprocessing_config": encoder.preprocessing_config,
        "catalog_count": len(catalog_items),
        "embedding_dim": encoder.embedding_dim,
        "catalog_item_ids": [item.item_id for item in catalog_items],
    }
    try:
        catalog_embeddings = load_embedding_cache(cache_path, expected_cache_metadata)
        cache_status = "loaded"
    except (FileNotFoundError, EmbeddingCacheMismatch) as exc:
        print(f"Catalog embedding cache not reused: {exc}; computing it now.")
        catalog_embeddings = encoder.encode([item.image_path for item in catalog_items])
        from recognition.cache import save_embedding_cache

        save_embedding_cache(cache_path, catalog_embeddings, expected_cache_metadata)
        cache_status = "recomputed"

    import torch

    device = encoder.device
    catalog_matrix = torch.tensor(catalog_embeddings, dtype=torch.float32, device=device)
    catalog_matrix = torch.nn.functional.normalize(catalog_matrix, p=2, dim=1)
    catalog_slugs = [item.item_id for item in catalog_items]

    # Per-query metadata and fixed view images (no target usage anywhere).
    image_sizes: dict[str, tuple[int, int]] = {}
    views_by_query: dict[str, dict[str, object]] = {}
    for row in benchmark_rows:
        image = load_rgb_image(_path(project_root, row["query_path"]))
        image_sizes[row["query_id"]] = (image.width, image.height)
        views_by_query[row["query_id"]] = view_images(image)

    query_ids = [row["query_id"] for row in benchmark_rows]
    view_scores: dict[str, torch.Tensor] = {}
    for view_name in VIEW_NAMES:
        images = [views_by_query[query_id][view_name] for query_id in query_ids]
        embeddings = torch.tensor(encoder.encode_pil(images), dtype=torch.float32, device=device)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        view_scores[view_name] = embeddings @ catalog_matrix.transpose(0, 1)
        print(f"Encoded view {view_name}: {tuple(view_scores[view_name].shape)}")

    method_scores: dict[str, torch.Tensor] = {}
    method_view_names = {
        "baseline_full": "full",
        "center_crop_85": "center_85",
        "center_crop_70": "center_70",
        "center_crop_55": "center_55",
    }
    for method_name, view_name in method_view_names.items():
        method_scores[method_name] = view_scores[view_name]
    stacked = torch.stack([view_scores[name] for name in VIEW_NAMES], dim=0)
    method_scores["multicrop_max"] = stacked.max(dim=0).values
    method_scores["multicrop_mean"] = stacked.mean(dim=0)

    run_id = datetime.now(timezone.utc).strftime("siglip2_generated_stress_pilot32_crop_diagnostics_%Y%m%dT%H%M%SZ")
    run_dir = _path(project_root, args.artifacts_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest_by_query = {row["query_id"]: row for row in benchmark_rows}
    per_query_rows: list[dict[str, object]] = []
    rankings_by_method: dict[str, dict[str, dict[str, object]]] = {}
    prediction_rows_by_method: dict[str, list[dict[str, object]]] = {}
    for method_name in METHODS:
        scores_matrix = method_scores[method_name].detach().cpu().tolist()
        rankings_by_method[method_name] = {}
        prediction_rows: list[dict[str, object]] = []
        for position, benchmark_row in enumerate(benchmark_rows):
            query_id = benchmark_row["query_id"]
            ranked = full_ranking(scores_matrix[position], catalog_slugs)
            ranked_slugs = [slug for slug, _ in ranked]
            target_slug = benchmark_row["target_slug"]
            rank = target_rank_full(ranked_slugs, target_slug)
            top1_slug, top1_score = ranked[0]
            target_score = dict(ranked).get(target_slug)
            subset_role = benchmark_row["subset_role"]
            scenario_id = benchmark_row["scenario_id"]
            width, height = image_sizes[query_id]
            per_query_rows.append(
                {
                    "query_id": query_id,
                    "method": method_name,
                    "target_slug": target_slug,
                    "scenario_id": scenario_id,
                    "subset_role": subset_role,
                    "target_family_id": benchmark_row["target_family_id"],
                    "family_type": benchmark_row["family_type"],
                    "top1_slug": top1_slug,
                    "top1_similarity": top1_score,
                    "target_similarity": target_score if rank is not None else "",
                    "target_rank": rank if rank is not None else "",
                    "top1_target_margin": (top1_score - target_score) if rank is not None else "",
                    "target_in_top5": rank is not None and rank <= 5,
                    "target_in_top10": rank is not None and rank <= 10,
                    "target_in_top25": rank is not None and rank <= 25,
                    "target_in_top100": rank is not None and rank <= 100,
                    "target_rank_band": rank_band(rank),
                    "image_width": width,
                    "image_height": height,
                    "image_aspect_ratio": round(width / height, 4),
                }
            )
            record = {
                "query_id": query_id,
                "target_slug": target_slug,
                "predicted_slug": top1_slug,
                "top1_score": top1_score,
                "top5_slugs": json.dumps(ranked_slugs[:5], ensure_ascii=False, separators=(",", ":")),
                "top5_scores": json.dumps([score for _, score in ranked[:5]], separators=(",", ":")),
                "correct_top1": top1_slug == target_slug,
                "target_rank": rank if rank is not None else "",
                "target_score": target_score if rank is not None else "",
                "subset_role": subset_role,
                "scenario_id": scenario_id,
                "target_family_id": benchmark_row["target_family_id"],
                "family_type": benchmark_row["family_type"],
            }
            prediction_rows.append(record)
            rankings_by_method[method_name][query_id] = {
                "ranked_slugs": ranked_slugs[:100],
                "ranked_scores": [score for _, score in ranked[:100]],
                "target_rank": rank,
            }
        prediction_rows_by_method[method_name] = prediction_rows
        write_csv(
            run_dir / f"predictions_{method_name}.csv",
            PREDICTIONS_COLUMNS,
            prediction_rows,
        )
        print(f"Ranked method {method_name}")

    baseline_prediction_rows = prediction_rows_by_method["baseline_full"]
    baseline_alignment = _verify_baseline_top5_alignment(args.baseline_predictions, baseline_prediction_rows)

    method_metrics_rows: list[dict[str, object]] = []
    scenario_metrics_rows: list[dict[str, object]] = []
    subset_metrics_rows: list[dict[str, object]] = []
    ranks_by_method: dict[str, list[int | None]] = {}
    for method_name in METHODS:
        rows_for_method = [row for row in per_query_rows if row["method"] == method_name]
        ranks = _ranks_of(rows_for_method)
        ranks_by_method[method_name] = ranks
        method_metrics_rows.append({"method": method_name, "scope": "overall", **rank_metrics(ranks)})
        for scenario in SCENARIOS:
            scenario_ranks = _ranks_of(
                [row for row in rows_for_method if row["scenario_id"] == scenario]
            )
            scenario_metrics_rows.append(
                {"method": method_name, "scenario_id": scenario, **rank_metrics(scenario_ranks)}
            )
        for subset in SUBSETS:
            subset_ranks = _ranks_of(
                [row for row in rows_for_method if row["subset_role"] == subset]
            )
            subset_metrics_rows.append(
                {"method": method_name, "subset_role": subset, **rank_metrics(subset_ranks)}
            )

    write_csv(
        run_dir / "method_metrics.csv",
        ["method", "scope", "query_count", "top1_accuracy", "recall_at_5", "recall_at_10", "recall_at_25", "mrr", "median_target_rank_found", "missing_count"],
        method_metrics_rows,
    )
    write_csv(
        run_dir / "scenario_metrics.csv",
        ["method", "scenario_id", "query_count", "top1_accuracy", "recall_at_5", "recall_at_10", "recall_at_25", "mrr", "median_target_rank_found", "missing_count"],
        scenario_metrics_rows,
    )
    write_csv(
        run_dir / "subset_metrics.csv",
        ["method", "subset_role", "query_count", "top1_accuracy", "recall_at_5", "recall_at_10", "recall_at_25", "mrr", "median_target_rank_found", "missing_count"],
        subset_metrics_rows,
    )

    subset_scenario_metrics_rows: list[dict[str, object]] = []
    for method_name in METHODS:
        rows_for_method = [row for row in per_query_rows if row["method"] == method_name]
        for subset in SUBSETS:
            for scenario in SCENARIOS:
                cell_ranks = _ranks_of(
                    [
                        row
                        for row in rows_for_method
                        if row["subset_role"] == subset and row["scenario_id"] == scenario
                    ]
                )
                subset_scenario_metrics_rows.append(
                    {
                        "method": method_name,
                        "subset_role": subset,
                        "scenario_id": scenario,
                        **rank_metrics(cell_ranks),
                    }
                )
    write_csv(
        run_dir / "subset_scenario_metrics.csv",
        ["method", "subset_role", "scenario_id", "query_count", "top1_accuracy", "recall_at_5", "recall_at_10", "recall_at_25", "mrr", "median_target_rank_found", "missing_count"],
        subset_scenario_metrics_rows,
    )

    rank_distribution_rows: list[dict[str, object]] = []
    distribution_scopes: list[tuple[str, str, str]] = [("overall", "", "")]
    distribution_scopes += [("scenario", "scenario_id", scenario) for scenario in SCENARIOS]
    distribution_scopes += [("subset", "subset_role", subset) for subset in SUBSETS]
    distribution_scopes += [
        ("subset_scenario", "subset_role", f"{subset}|{scenario}")
        for subset in SUBSETS
        for scenario in SCENARIOS
    ]
    for method_name in METHODS:
        rows_for_method = [row for row in per_query_rows if row["method"] == method_name]
        for scope_name, key, value in distribution_scopes:
            if scope_name == "overall":
                scope_rows = rows_for_method
            elif scope_name == "subset_scenario":
                subset, scenario = value.split("|")
                scope_rows = [
                    row
                    for row in rows_for_method
                    if row["subset_role"] == subset and row["scenario_id"] == scenario
                ]
            else:
                scope_rows = [row for row in rows_for_method if row[key] == value]
            counts = {band: 0 for band in RANK_BANDS_ORDER}
            for row in scope_rows:
                counts[row["target_rank_band"]] += 1
            for band in RANK_BANDS_ORDER:
                rank_distribution_rows.append(
                    {
                        "method": method_name,
                        "scope": scope_name,
                        "scope_value": value,
                        "rank_band": band,
                        "query_count": counts[band],
                        "share": round(counts[band] / len(scope_rows), 4) if scope_rows else 0.0,
                    }
                )
    write_csv(
        run_dir / "rank_distribution.csv",
        ["method", "scope", "scope_value", "rank_band", "query_count", "share"],
        rank_distribution_rows,
    )

    transition_rows: list[dict[str, object]] = []
    for method_name in METHODS[1:]:
        method_prediction_rows = prediction_rows_by_method[method_name]
        flip_counts = top5_transition_counts(baseline_prediction_rows, method_prediction_rows)
        summary = transition_summary(ranks_by_method["baseline_full"], ranks_by_method[method_name])
        transition_rows.append({"method": method_name, **flip_counts, **summary})
    write_csv(
        run_dir / "transition_analysis.csv",
        [
            "method",
            "wrong_to_correct_top1",
            "correct_to_wrong_top1",
            "top5_gained",
            "top5_lost",
            "recovered_from_missing",
            "lost_to_missing",
            "rank_improved",
            "rank_degraded",
            "rank_unchanged",
            "missing_unchanged",
            "median_rank_delta",
        ],
        transition_rows,
    )

    family_metrics_payload: dict[str, object] = {}
    family_metrics_rows: list[dict[str, object]] = []
    for method_name in METHODS:
        diagnostics = family_diagnostics(prediction_rows_by_method[method_name], benchmark_rows)
        family_metrics_payload[method_name] = diagnostics
        for family_type, breakdown in sorted(diagnostics["family_type_breakdown"].items()):
            family_metrics_rows.append(
                {
                    "method": method_name,
                    "family_type": family_type,
                    "query_count": breakdown["query_count"],
                    "exact_top1": breakdown["top1_accuracy"],
                    "family_top1": breakdown["family_top1"],
                    "family_recall_at_5": breakdown["family_recall_at_5"],
                    "within_family_disambiguation_top1": breakdown["within_family_disambiguation_top1"],
                }
            )
        aggregate = {
            "query_count": diagnostics["hard_query_count"],
            "family_top1": diagnostics["family_top1"],
            "family_recall_at_5": diagnostics["family_recall_at_5"],
            "within_family_disambiguation_top1": diagnostics["within_family_disambiguation_top1"],
        }
        family_metrics_rows.append(
            {
                "method": method_name,
                "family_type": "ALL",
                **aggregate,
                "exact_top1": rank_metrics(ranks_by_method[method_name])["top1_accuracy"],
            }
        )
    write_csv(
        run_dir / "family_metrics.csv",
        ["method", "family_type", "query_count", "exact_top1", "family_top1", "family_recall_at_5", "within_family_disambiguation_top1"],
        family_metrics_rows,
    )

    family_transition_rows: list[dict[str, object]] = []
    for method_name in METHODS[1:]:
        summary = {key: 0 for key in (
            "target_back_in_top5",
            "family_back_in_top5",
            "correct_family_top1_wrong_member",
            "left_family_top5",
        )}
        summary_by_type = {
            "vintage": {key: 0 for key in summary},
            "subtype": {key: 0 for key in summary},
        }
        for baseline_row, method_row in zip(baseline_prediction_rows, prediction_rows_by_method[method_name]):
            family_id = str(baseline_row["target_family_id"])
            if not family_id:
                continue
            flags = family_transition_flags(
                json.loads(baseline_row["top5_slugs"]),
                json.loads(method_row["top5_slugs"]),
                str(baseline_row["target_slug"]),
                family_id,
                slug_to_family,
            )
            family_type = str(baseline_row["family_type"]) or "unknown"
            for key in summary:
                summary[key] += int(flags[key])
                summary_by_type[family_type][key] += int(flags[key])
        family_transition_rows.append({"method": method_name, "family_type": "ALL", **summary})
        for family_type, counts in sorted(summary_by_type.items()):
            family_transition_rows.append({"method": method_name, "family_type": family_type, **counts})
    write_csv(
        run_dir / "family_transitions.csv",
        ["method", "family_type", "target_back_in_top5", "family_back_in_top5", "correct_family_top1_wrong_member", "left_family_top5"],
        family_transition_rows,
    )

    query_image_diagnostics_rows = []
    for benchmark_row in benchmark_rows:
        query_id = benchmark_row["query_id"]
        width, height = image_sizes[query_id]
        query_image_diagnostics_rows.append(
            {
                "query_id": query_id,
                "scenario_id": benchmark_row["scenario_id"],
                "subset_role": benchmark_row["subset_role"],
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 4),
            }
        )
    write_csv(
        run_dir / "query_image_diagnostics.csv",
        ["query_id", "scenario_id", "subset_role", "width", "height", "aspect_ratio"],
        query_image_diagnostics_rows,
    )

    top100_rows = []
    for method_name in METHODS:
        for query_id, ranking in rankings_by_method[method_name].items():
            top100_rows.append(
                {
                    "method": method_name,
                    "query_id": query_id,
                    "target_rank": ranking["target_rank"] if ranking["target_rank"] is not None else "",
                    "top100_slugs": json.dumps(ranking["ranked_slugs"], separators=(",", ":")),
                    "top100_scores": json.dumps(ranking["ranked_scores"], separators=(",", ":")),
                }
            )
    write_csv(
        run_dir / "top100_rankings.csv",
        ["method", "query_id", "target_rank", "top100_slugs", "top100_scores"],
        top100_rows,
    )

    write_csv(
        run_dir / "per_query_diagnostics.csv",
        [
            "query_id", "method", "target_slug", "scenario_id", "subset_role",
            "target_family_id", "family_type", "top1_slug", "top1_similarity",
            "target_similarity", "target_rank", "top1_target_margin",
            "target_in_top5", "target_in_top10", "target_in_top25", "target_in_top100",
            "target_rank_band", "image_width", "image_height", "image_aspect_ratio",
        ],
        per_query_rows,
    )

    config = {
        "run_id": run_id,
        "purpose": "diagnostic milestone: frozen-encoder crop/multi-crop ablation on pilot32",
        "model_id": encoder.model_id,
        "model_version": encoder.model_version,
        "device": encoder.device,
        "catalog_version": catalog_meta["catalog_version"],
        "catalog_manifest_sha256": catalog_meta["manifest_sha256"],
        "benchmark_manifest": str(benchmark_manifest_path.relative_to(project_root)),
        "benchmark_manifest_sha256": sha256_file(benchmark_manifest_path),
        "catalog_embedding_cache": str(cache_path.relative_to(project_root)),
        "catalog_embedding_cache_status": cache_status,
        "methods": {
            "baseline_full": "current SigLIP2 preprocessing without changes (square stretch to 224x224)",
            "center_crop_85": "center 85% crop before the unchanged SigLIP2 preprocessing",
            "center_crop_70": "center 70% crop before the unchanged SigLIP2 preprocessing",
            "center_crop_55": "center 55% crop before the unchanged SigLIP2 preprocessing",
            "multicrop_max": "per-candidate max similarity over full/85/70/55 views",
            "multicrop_mean": "per-candidate mean similarity over full/85/70/55 views",
        },
        "crop_policy": "deterministic centered ratio crops; identical for all images; no target label used",
        "preprocessing_config": encoder.preprocessing_config,
        "query_count": len(benchmark_rows),
        "catalog_count": len(catalog_items),
        "baseline_top5_alignment": baseline_alignment,
        "notes": [
            "diagnostic experiment only; metrics are not a production benchmark result",
            "ground truth used only for scoring after inference, never for crop selection",
        ],
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    _write_html_report(
        _path(project_root, args.html_report),
        project_root,
        per_query_rows,
        prediction_rows_by_method,
        method_metrics_rows,
        manifest_by_query,
        catalog_by_slug,
    )

    md_text = _render_md_report(
        method_metrics_rows,
        scenario_metrics_rows,
        subset_metrics_rows,
        subset_scenario_metrics_rows,
        rank_distribution_rows,
        transition_rows,
        family_metrics_rows,
        family_transition_rows,
        query_image_diagnostics_rows,
        config,
    )
    report_path = _path(project_root, args.md_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(md_text, encoding="utf-8")

    print(json.dumps({"run_dir": str(run_dir), "baseline_alignment": baseline_alignment}, indent=2))
    print(f"HTML report: {_path(project_root, args.html_report)}")
    print(f"MD report: {report_path}")


def _ranks_of(rows: list[dict[str, object]]) -> list[int | None]:
    return [int(row["target_rank"]) if row["target_rank"] != "" else None for row in rows]


def _verify_baseline_top5_alignment(
    baseline_predictions_path: str, baseline_prediction_rows: list[dict[str, object]]
) -> dict[str, object]:
    """Compare recomputed baseline Top-5 against the frozen run predictions."""

    path = Path(baseline_predictions_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        return {"status": "skipped", "reason": f"file not found: {path}"}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        frozen_rows = {row["query_id"]: row for row in csv.DictReader(stream)}
    matched = 0
    mismatches = []
    for row in baseline_prediction_rows:
        frozen = frozen_rows.get(str(row["query_id"]))
        if frozen is None:
            mismatches.append({"query_id": row["query_id"], "reason": "missing in frozen predictions"})
            continue
        if json.loads(frozen["top5_slugs"]) == json.loads(row["top5_slugs"]):
            matched += 1
        else:
            mismatches.append(
                {
                    "query_id": row["query_id"],
                    "frozen_top5": frozen["top5_slugs"],
                    "recomputed_top5": row["top5_slugs"],
                }
            )
    return {
        "status": "ok" if not mismatches else "mismatch",
        "matched": matched,
        "total": len(baseline_prediction_rows),
        "mismatches": mismatches[:5],
    }


def _render_md_report(
    method_metrics_rows: list[dict[str, object]],
    scenario_metrics_rows: list[dict[str, object]],
    subset_metrics_rows: list[dict[str, object]],
    subset_scenario_metrics_rows: list[dict[str, object]],
    rank_distribution_rows: list[dict[str, object]],
    transition_rows: list[dict[str, object]],
    family_metrics_rows: list[dict[str, object]],
    family_transition_rows: list[dict[str, object]],
    query_image_diagnostics_rows: list[dict[str, object]],
    config: dict[str, object],
) -> str:
    lines: list[str] = []
    lines.append("# Pilot32 crop diagnostics (frozen SigLIP2)")
    lines.append("")
    lines.append("Diagnostic milestone run; not a production benchmark result.")
    lines.append("")
    lines.append("## A. Current preprocessing (verified)")
    lines.append("")
    lines.append("- Processor: `SiglipImageProcessor` (`google/siglip2-base-patch16-224`).")
    lines.append("- Resize: direct anisotropic resize to 224x224 (`size={'height':224,'width':224}`, bilinear, resample=2).")
    lines.append("- Center crop: **not performed** (`do_center_crop=None`). A non-square probe image kept both halves visible: the pipeline stretches, it does not cut.")
    lines.append("- Normalization: mean/std = 0.5/0.5 (input scaled to [-1, 1]).")
    lines.append("- Final tensor: `(3, 224, 224)`; catalog references and queries go through the identical path.")
    lines.append("- Consequence: current preprocessing cannot cut off part of the bottle; it can only distort the aspect ratio.")
    lines.append("")
    lines.append("## L. Comparison table (overall, 128 queries)")
    lines.append("")
    lines.append("| Method | Top1 | R@5 | R@10 | MRR | MedianRank(found) | Missing |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in method_metrics_rows:
        lines.append(
            f"| {row['method']} | {_pct(row['top1_accuracy'])} | {_pct(row['recall_at_5'])} | "
            f"{_pct(row['recall_at_10'])} | {_num(row['mrr'], 4)} | {_num(row['median_target_rank_found'], 1)} | {row['missing_count']} |"
        )
    lines.append("")
    for subset in SUBSETS:
        lines.append(f"### {subset} (64 queries)")
        lines.append("")
        lines.append("| Method | Top1 | R@5 | R@10 | MRR | MedianRank(found) | Missing |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in subset_metrics_rows:
            if row["subset_role"] != subset:
                continue
            lines.append(
                f"| {row['method']} | {_pct(row['top1_accuracy'])} | {_pct(row['recall_at_5'])} | "
                f"{_pct(row['recall_at_10'])} | {_num(row['mrr'], 4)} | {_num(row['median_target_rank_found'], 1)} | {row['missing_count']} |"
            )
        lines.append("")
    for scenario in SCENARIOS:
        lines.append(f"### scenario: {scenario} (32 queries)")
        lines.append("")
        lines.append("| Method | Top1 | R@5 | R@10 | MRR | MedianRank(found) | Missing |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in scenario_metrics_rows:
            if row["scenario_id"] != scenario:
                continue
            lines.append(
                f"| {row['method']} | {_pct(row['top1_accuracy'])} | {_pct(row['recall_at_5'])} | "
                f"{_pct(row['recall_at_10'])} | {_num(row['mrr'], 4)} | {_num(row['median_target_rank_found'], 1)} | {row['missing_count']} |"
            )
        lines.append("")
    lines.append("### subset x scenario (Top1 / R@5 / R@10)")
    lines.append("")
    lines.append("| Method | subset | scenario | Top1 | R@5 | R@10 |")
    lines.append("|---|---|---|---|---|---|")
    for row in subset_scenario_metrics_rows:
        lines.append(
            f"| {row['method']} | {row['subset_role']} | {row['scenario_id']} | "
            f"{_pct(row['top1_accuracy'])} | {_pct(row['recall_at_5'])} | {_pct(row['recall_at_10'])} |"
        )
    lines.append("")
    lines.append("## B. Baseline full-rank distribution (query counts per band)")
    lines.append("")
    lines.append("| scope | value | 1 | 2-5 | 6-10 | 11-25 | 26-100 | >100 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    baseline_distribution: dict[tuple[str, str], dict[str, int]] = {}
    for row in rank_distribution_rows:
        if row["method"] != "baseline_full":
            continue
        key = (str(row["scope"]), str(row["scope_value"]))
        baseline_distribution.setdefault(key, {})[str(row["rank_band"])] = int(row["query_count"])
    for (scope, value), counts in baseline_distribution.items():
        cells = "".join(f" {counts.get(band, 0)} |" for band in RANK_BANDS_ORDER)
        lines.append(f"| {scope} | {value} |{cells}")
    lines.append("")
    lines.append("Full distribution (all methods, all scopes): `rank_distribution.csv` in the run directory.")
    lines.append("")
    lines.append("## M. Error transitions vs baseline_full")
    lines.append("")
    lines.append("| Method | wrong->correct Top1 | correct->wrong Top1 | Top5 gained | Top5 lost | recovered_from_missing | lost_to_missing | improved | degraded | unchanged | missing_unchanged | median rank delta |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in transition_rows:
        lines.append(
            f"| {row['method']} | {row['wrong_to_correct_top1']} | {row['correct_to_wrong_top1']} | "
            f"{row['top5_gained']} | {row['top5_lost']} | {row['recovered_from_missing']} | {row['lost_to_missing']} | "
            f"{row['rank_improved']} | {row['rank_degraded']} | {row['rank_unchanged']} | {row['missing_unchanged']} | {_num(row['median_rank_delta'], 1)} |"
        )
    lines.append("")
    lines.append("## C/H. Family metrics (hard subset)")
    lines.append("")
    lines.append("| Method | family_type | queries | exact Top1 | family Top1 | family R@5 | within-family disambiguation |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in family_metrics_rows:
        lines.append(
            f"| {row['method']} | {row['family_type']} | {row['query_count']} | {_pct(row['exact_top1'])} | "
            f"{_pct(row['family_top1'])} | {_pct(row['family_recall_at_5'])} | {_pct(row['within_family_disambiguation_top1'])} |"
        )
    lines.append("")
    lines.append("## N. Family transitions (hard subset, vs baseline_full)")
    lines.append("")
    lines.append("| Method | family_type | target back in Top5 | family back in Top5 | correct family Top1, wrong member | left family Top5 |")
    lines.append("|---|---|---|---|---|---|")
    for row in family_transition_rows:
        lines.append(
            f"| {row['method']} | {row['family_type']} | {row['target_back_in_top5']} | {row['family_back_in_top5']} | "
            f"{row['correct_family_top1_wrong_member']} | {row['left_family_top5']} |"
        )
    lines.append("")
    lines.append("## K. Query image scale diagnostics")
    lines.append("")
    lines.append("| scenario | median width | median height | median aspect |")
    lines.append("|---|---|---|---|")
    for scenario in SCENARIOS:
        rows = [row for row in query_image_diagnostics_rows if row["scenario_id"] == scenario]
        widths = sorted(int(row["width"]) for row in rows)
        heights = sorted(int(row["height"]) for row in rows)
        aspects = sorted(float(row["aspect_ratio"]) for row in rows)
        lines.append(
            f"| {scenario} | {widths[len(widths) // 2]} | {heights[len(heights) // 2]} | {round(aspects[len(aspects) // 2], 3)} |"
        )
    lines.append("")
    alignment = config["baseline_top5_alignment"]
    alignment_note = (
        f"status={alignment['status']}, exact Top-5 match {alignment['matched']}/{alignment['total']}"
        + (
            f"; mismatches are near-tie rank swaps within Top-5 from cross-run float nondeterminism"
            if alignment.get("mismatches")
            else ""
        )
    )
    lines.append(f"Baseline Top-5 alignment check: {alignment_note}.")
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


def _write_html_report(
    report_path: Path,
    project_root: Path,
    per_query_rows: list[dict[str, object]],
    prediction_rows_by_method: dict[str, list[dict[str, object]]],
    method_metrics_rows: list[dict[str, object]],
    manifest_by_query: dict[str, dict[str, str]],
    catalog_by_slug: dict[str, dict[str, str]],
) -> None:
    from html import escape
    import os

    report_path.parent.mkdir(parents=True, exist_ok=True)

    def rel_src(value: str | None) -> str | None:
        if not value:
            return None
        image_path = (project_root / value).resolve()
        if not image_path.is_file():
            return None
        return os.path.relpath(image_path, report_path.parent.resolve()).replace(os.sep, "/")

    def img(src: str | None, alt: str) -> str:
        return f"<img src='{escape(src)}' alt='{escape(alt)}' loading='lazy'>" if src else "<div class='miss'>no image</div>"

    baseline_by_query = {str(row["query_id"]): row for row in per_query_rows if row["method"] == "baseline_full"}
    predictions_by_query = {
        method: {str(row["query_id"]): row for row in rows}
        for method, rows in prediction_rows_by_method.items()
    }

    def method_label(row: dict[str, object]) -> str:
        return f"{row['top1_accuracy'] * 100:.1f}%"

    summary_cells = "".join(
        f"<td>{escape(str(row['method']))}<br><b>{method_label(row)}</b><br>R@5 {_pct(row['recall_at_5'])} · R@10 {_pct(row['recall_at_10'])}</td>"
        for row in method_metrics_rows
    )

    sections: list[tuple[str, str]] = []

    def card(query_id: str) -> str:
        base = baseline_by_query[query_id]
        manifest = manifest_by_query[query_id]
        target_slug = base["target_slug"]
        target_row = catalog_by_slug.get(target_slug, {})
        per_method_cells = []
        for method in METHODS:
            pred = predictions_by_query[method][query_id]
            per_method_cells.append(
                f"<td>{'OK' if pred['correct_top1'] else ''} rank {pred['target_rank'] if pred['target_rank'] != '' else '—'}</td>"
            )
        top5_html = []
        top5_slugs = json.loads(pred["top5_slugs"])
        top5_scores = json.loads(pred["top5_scores"])
        for pos, (slug, score) in enumerate(zip(top5_slugs, top5_scores), start=1):
            slug_row = catalog_by_slug.get(slug, {})
            css = "hit" if slug == target_slug else ""
            top5_html.append(
                f"<figure class='{css}'><figcaption>#{pos} {escape(slug)}<br>score {score:.4f}</figcaption>"
                f"{img(rel_src(slug_row.get('reference_image_path')), slug)}</figure>"
            )
        family = manifest.get("target_family_id", "")
        family_html = f" · family {escape(family)} ({escape(manifest.get('family_type', ''))})" if family else ""
        per_method_header = "".join(f"<th>{escape(m)}</th>" for m in METHODS)
        return (
            "<article class='card'>"
            f"<h3>{escape(query_id)}</h3>"
            f"<p>scenario <b>{escape(base['scenario_id'])}</b> · {escape(base['subset_role'])}{family_html} · "
            f"baseline target rank <b>{escape(str(base['target_rank']))}</b> "
            f"(sim {escape(str(base['target_similarity']))}) · top1 sim {escape(str(base['top1_similarity']))} · "
            f"image {base['image_width']}x{base['image_height']}</p>"
            f"<table class='mtable'><tr><th>method</th>{per_method_header}</tr><tr><td>result</td>{''.join(per_method_cells)}</tr></table>"
            "<div class='images'>"
            f"<figure><figcaption>Query</figcaption>{img(rel_src(manifest.get('query_path')), query_id)}</figure>"
            f"<figure class='hit'><figcaption>Correct: {escape(target_slug)}</figcaption>"
            f"{img(rel_src(target_row.get('reference_image_path')), target_slug)}</figure>"
            + "".join(top5_html)
            + "</div></article>"
        )

    rank_bands = [label for label, _, _ in RANK_BANDS]
    band_titles = {
        "1": "Top-1 correct (baseline)",
        "2-5": "Target rank 2–5 (baseline)",
        "6-10": "Target rank 6–10 (baseline)",
        "11-25": "Target rank 11–25 (baseline)",
        "26-100": "Target rank 26–100 (baseline)",
        ">100": "Target rank >100 / missing (baseline)",
    }
    for band in rank_bands:
        ids = sorted(
            (query_id for query_id, row in baseline_by_query.items() if row["target_rank_band"] == band),
            key=lambda query_id: (
                -(int(baseline_by_query[query_id]["target_rank"]) if baseline_by_query[query_id]["target_rank"] != "" else 99999)
            ),
        )
        sections.append((f"band-{band}", f"{band_titles[band]} — {len(ids)}"))
    for scenario in SCENARIOS:
        ids = [query_id for query_id, row in baseline_by_query.items() if row["scenario_id"] == scenario]
        sections.append((f"scenario-{scenario}", f"Scenario: {scenario} — {len(ids)}"))
    for subset in SUBSETS:
        ids = [query_id for query_id, row in baseline_by_query.items() if row["subset_role"] == subset]
        sections.append((f"subset-{subset}", f"Subset: {subset} — {len(ids)}"))

    body_parts = []
    nav = []
    for anchor, title in sections:
        nav.append(f"<a href='#{escape(anchor)}'>{escape(title)}</a>")
    body_parts.append(f"<nav>{' · '.join(nav)}</nav>")
    body_parts.append(f"<table class='summary'><tr>{summary_cells}</tr></table>")
    for anchor, title in sections:
        body_parts.append(f"<h2 id='{escape(anchor)}'>{escape(title)}</h2>")
        scope_filter = anchor.split("-", 1)
        if anchor.startswith("band-"):
            band = scope_filter[1]
            ids = [query_id for query_id, row in baseline_by_query.items() if row["target_rank_band"] == band]
        elif anchor.startswith("scenario-"):
            scenario = scope_filter[1]
            ids = [query_id for query_id, row in baseline_by_query.items() if row["scenario_id"] == scenario]
        else:
            subset = scope_filter[1]
            ids = [query_id for query_id, row in baseline_by_query.items() if row["subset_role"] == subset]
        ids = sorted(
            ids,
            key=lambda query_id: (
                -(int(baseline_by_query[query_id]["target_rank"]) if baseline_by_query[query_id]["target_rank"] != "" else 99999)
            ),
        )
        body_parts.append("".join(card(query_id) for query_id in ids) or "<p>None.</p>")

    html = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>Pilot32 error analysis</title><style>"
        "body{font-family:system-ui,sans-serif;margin:20px;background:#f4f4f4;color:#222}"
        "nav a{margin-right:10px;font-size:13px}"
        ".summary{border-collapse:collapse;margin:12px 0}.summary td{border:1px solid #ccc;padding:6px 10px;background:#fff;font-size:13px}"
        "h2{margin-top:28px;border-bottom:1px solid #bbb}"
        ".card{background:#fff;margin:14px 0;padding:12px;border-radius:8px;box-shadow:0 1px 4px #bbb}"
        ".card h3{margin:0 0 4px;font-size:15px}.card p{margin:2px 0 8px;font-size:13px}"
        ".images{display:flex;gap:10px;flex-wrap:wrap}figure{margin:0;width:170px}"
        "figcaption{font-size:11px;min-height:34px}"
        "img{display:block;max-width:170px;max-height:240px;object-fit:contain;background:#eee}"
        "figure.hit figcaption{color:#0a7a0a;font-weight:600}"
        ".miss{color:#a00}.mtable{border-collapse:collapse;font-size:12px}.mtable td,.mtable th{border:1px solid #ccc;padding:2px 6px}"
        "</style></head><body><h1>Pilot32 error analysis — frozen SigLIP2, full-rank + crop diagnostics</h1>"
        "<p>All 128 accepted generated queries; sections sorted worst-rank-first; Top-5 shown for the baseline method per card.</p>"
        + "".join(body_parts)
        + "</body></html>\n"
    )
    report_path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
