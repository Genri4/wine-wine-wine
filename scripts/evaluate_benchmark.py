#!/usr/bin/env python3
"""Evaluate any ranked-predictions CSV using the universal retrieval evaluator."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recognition.benchmark import RankingRecord, evaluate_rankings, target_rank  # noqa: E402
from recognition.family_metrics import family_diagnostics  # noqa: E402
from recognition.reporting import write_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ranked retrieval predictions")
    parser.add_argument("--predictions", required=True, help="predictions.csv from an experiment")
    parser.add_argument(
        "--benchmark",
        default="synthetic_dev",
        choices=("synthetic_dev", "hard_near_duplicate_dev", "hard_near_duplicate_dev_v2", "generated_stress_dev", "generated_stress_dev_pilot32", "web_extra_dev"),
        help="Benchmark taxonomy label; metrics are never pooled across labels",
    )
    parser.add_argument("--benchmark-manifest", default=None)
    parser.add_argument(
        "--synthetic-metrics",
        default="artifacts/experiments/siglip2_synthetic_dev_20260916T062142Z/metrics.json",
        help="Synthetic DEV metrics used for generated-stress degradation comparison",
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--output-dir", default=None, help="Write metrics/errors under this directory")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    manifest_path = _default_manifest(project_root, args.benchmark, args.benchmark_manifest)
    manifest_rows = _read_manifest(manifest_path)
    predictions_path = Path(args.predictions)
    rows = _read_rows(predictions_path)
    _validate_prediction_alignment(rows, manifest_rows, args.benchmark)
    records = [_record_from_row(row) for row in rows]
    metrics = evaluate_rankings(records)
    payload = {
        "benchmark": args.benchmark,
        "benchmark_manifest": str(manifest_path.relative_to(project_root)),
        "benchmark_queries": len(manifest_rows),
        "benchmark_unique_target_slugs": len({row["target_slug"] for row in manifest_rows}),
        "unique_target_slugs_in_predictions": len({row["target_slug"] for row in rows}),
        **metrics,
    }
    if args.benchmark.startswith("hard_near_duplicate_dev"):
        payload["hard_diagnostics"] = _hard_diagnostics(rows, manifest_rows)
    if args.benchmark in {"generated_stress_dev", "generated_stress_dev_pilot32"}:
        synthetic_metrics_path = Path(args.synthetic_metrics)
        if not synthetic_metrics_path.is_absolute():
            synthetic_metrics_path = project_root / synthetic_metrics_path
        payload["generated_stress_diagnostics"] = _generated_stress_diagnostics(
            rows, manifest_rows, _optional_json(synthetic_metrics_path)
        )
    if args.benchmark == "generated_stress_dev_pilot32":
        payload["family_diagnostics"] = family_diagnostics(rows, manifest_rows)
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        errors = _error_rows(rows)
        write_csv(
            output_dir / "errors.csv",
            [
                "query_id",
                "target_slug",
                "predicted_slug",
                "target_rank",
                "top1_score",
                "target_score",
                "top1_top2_margin",
                "transform_type",
            ],
            errors,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        rows = list(reader)
    required = {"query_id", "target_slug", "predicted_slug", "top5_slugs", "latency_ms"}
    if required - set(fields):
        raise ValueError(f"Predictions CSV is missing required columns: {sorted(required)}")
    return rows


def _default_manifest(project_root: Path, benchmark: str, requested: str | None) -> Path:
    if requested:
        path = Path(requested)
        return path if path.is_absolute() else project_root / path
    names = {
        "synthetic_dev": "data/benchmarks/synthetic_dev/manifest.csv",
        "hard_near_duplicate_dev": "data/benchmarks/hard_near_duplicate_dev/manifest.csv",
        "hard_near_duplicate_dev_v2": "data/benchmarks/hard_near_duplicate_dev_v2/manifest.csv",
        "generated_stress_dev": "data/benchmarks/generated_stress_dev/manifest.csv",
        "generated_stress_dev_pilot32": "data/benchmarks/generated_stress_dev_pilot32/manifest.csv",
        "web_extra_dev": "data/benchmarks/web_extra_dev/manifest.csv",
    }
    return project_root / names[benchmark]


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        rows = list(reader)
    required = {"query_id", "target_slug"}
    if required - set(fields):
        raise ValueError(f"Benchmark manifest is missing columns: {sorted(required)}")
    return rows


def _validate_prediction_alignment(
    prediction_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
    benchmark: str,
) -> None:
    """Prevent silently scoring a different split or a pooled benchmark."""

    manifest_by_id = _unique_by_query_id(manifest_rows, "benchmark manifest")
    prediction_by_id = _unique_by_query_id(prediction_rows, "predictions")
    missing = sorted(set(manifest_by_id) - set(prediction_by_id))
    extra = sorted(set(prediction_by_id) - set(manifest_by_id))
    if missing or extra:
        raise ValueError(
            f"{benchmark} predictions do not match the manifest: "
            f"missing={missing[:3]} ({len(missing)} total), "
            f"extra={extra[:3]} ({len(extra)} total)"
        )
    for query_id, manifest_row in manifest_by_id.items():
        prediction_row = prediction_by_id[query_id]
        if prediction_row.get("target_slug") != manifest_row.get("target_slug"):
            raise ValueError(
                f"Target mismatch for {query_id}: "
                f"manifest={manifest_row.get('target_slug')!r}, "
                f"prediction={prediction_row.get('target_slug')!r}"
            )


def _unique_by_query_id(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        query_id = row.get("query_id", "")
        if not query_id:
            raise ValueError(f"{label} contains an empty query_id")
        if query_id in indexed:
            raise ValueError(f"{label} contains duplicate query_id: {query_id}")
        indexed[query_id] = row
    return indexed


def _hard_diagnostics(
    prediction_rows: list[dict[str, str]], manifest_rows: list[dict[str, str]]
) -> dict[str, object]:
    family_by_query = {row["query_id"]: row.get("family_id", "") for row in manifest_rows}
    rank_distribution: Counter[str] = Counter()
    family_total: Counter[str] = Counter()
    family_correct: Counter[str] = Counter()
    confusion_pairs: Counter[tuple[str, str]] = Counter()
    margins: list[float] = []
    for row in prediction_rows:
        family_id = family_by_query.get(row["query_id"], "unknown")
        family_total[family_id] += 1
        rank = row.get("target_rank") or "not-found"
        rank_distribution[rank] += 1
        if row.get("predicted_slug") == row.get("target_slug"):
            family_correct[family_id] += 1
        else:
            confusion_pairs[(row.get("target_slug", ""), row.get("predicted_slug", ""))] += 1
        try:
            margins.append(float(row["top1_top2_margin"]))
        except (KeyError, TypeError, ValueError):
            pass
    accuracy_per_family = {
        family: family_correct[family] / total
        for family, total in sorted(family_total.items())
    }
    return {
        "target_rank_distribution": dict(sorted(rank_distribution.items(), key=lambda item: item[0])),
        "top1_top2_margin": {
            "mean": sum(margins) / len(margins) if margins else None,
            "min": min(margins) if margins else None,
            "max": max(margins) if margins else None,
        },
        "accuracy_per_family": accuracy_per_family,
        "top_confusion_pairs": [
            {"target_slug": target, "predicted_slug": predicted, "count": count}
            for (target, predicted), count in confusion_pairs.most_common(20)
        ],
    }


def _generated_stress_diagnostics(
    prediction_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
    synthetic_metrics: dict[str, object] | None,
) -> dict[str, object]:
    """Return scenario metrics and deltas without pooling benchmark labels."""

    scenario_by_query: dict[str, str] = {}
    for row in manifest_rows:
        scenario = row.get("scenario_id", "").strip()
        if not scenario:
            raise ValueError(f"generated_stress manifest has empty scenario_id: {row.get('query_id')}")
        if row["query_id"] in scenario_by_query:
            raise ValueError(f"generated_stress manifest has duplicate query_id: {row['query_id']}")
        scenario_by_query[row["query_id"]] = scenario
    grouped: dict[str, list[RankingRecord]] = {}
    for row in prediction_rows:
        scenario = scenario_by_query.get(row["query_id"])
        if scenario is None:
            raise ValueError(f"generated_stress prediction missing scenario mapping: {row['query_id']}")
        grouped.setdefault(scenario, []).append(_record_from_row(row))
    per_scenario = {
        scenario: evaluate_rankings(records)
        for scenario, records in sorted(grouped.items())
    }
    metrics = ("top1_accuracy", "recall_at_5", "mrr")
    degradation = None
    if synthetic_metrics:
        degradation = {
            metric: {
                "synthetic_dev": synthetic_metrics.get(metric),
                "generated_stress_dev": evaluate_rankings(
                    [_record_from_row(row) for row in prediction_rows]
                ).get(metric),
                "delta_generated_minus_synthetic": _delta(
                    evaluate_rankings([_record_from_row(row) for row in prediction_rows]).get(metric),
                    synthetic_metrics.get(metric),
                ),
            }
            for metric in metrics
        }
    return {"per_scenario": per_scenario, "degradation_vs_synthetic_dev": degradation}


def _optional_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read synthetic metrics: {path}: {exc}") from exc


def _delta(generated: object, synthetic: object) -> float | None:
    if generated is None or synthetic is None:
        return None
    return float(generated) - float(synthetic)


def _record_from_row(row: dict[str, str]) -> RankingRecord:
    candidates = json.loads(row["top5_slugs"])
    scores = json.loads(row.get("top5_scores", "[]"))
    return RankingRecord(
        query_id=row["query_id"],
        target_slug=row["target_slug"],
        ranked_candidates=candidates,
        scores=scores,
        latency_ms=float(row["latency_ms"]),
        transform_type=row.get("transform_type", ""),
    )


def _error_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    errors: list[dict[str, object]] = []
    for row in rows:
        candidates = json.loads(row["top5_slugs"])
        scores = [float(value) for value in json.loads(row.get("top5_scores", "[]"))]
        rank = target_rank(row["target_slug"], candidates)
        if row.get("predicted_slug") == row["target_slug"]:
            continue
        target_score = scores[rank - 1] if rank and rank <= len(scores) else ""
        errors.append(
            {
                "query_id": row["query_id"],
                "target_slug": row["target_slug"],
                "predicted_slug": row.get("predicted_slug", ""),
                "target_rank": rank or "",
                "top1_score": row.get("top1_score", ""),
                "target_score": target_score,
                "top1_top2_margin": row.get("top1_top2_margin", ""),
                "transform_type": row.get("transform_type", ""),
            }
        )
    return errors


if __name__ == "__main__":
    main()
