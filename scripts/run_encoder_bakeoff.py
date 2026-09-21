#!/usr/bin/env python3
"""Sequential encoder bake-off over all available models and frozen benchmarks.

Models run one at a time; a failing model is recorded as failed with its
error and the bake-off continues. Results are collected into one namespace
with pairwise-vs-SigLIP and oracle-complementarity analyses.
"""

from __future__ import annotations

import argparse
import traceback
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from recognition.encoder_adapters import ADAPTER_REGISTRY  # noqa: E402

BENCHMARKS = ("synthetic_dev", "hard_near_duplicate_dev_v2", "generated_stress_dev_pilot32")
DEFAULT_ORDER = (
    "current_siglip2",
    "dinov2_vitl14_reg",
    "pe_core_l14_336",
    "dfn5b_h14_378",
    "siglip2_so400m_384",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential encoder bake-off")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_ORDER))
    parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS))
    parser.add_argument("--artifacts-root", default="artifacts/experiments")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root if hasattr(args, "project_root") else PROJECT_ROOT).resolve()
    run_id = datetime.now(timezone.utc).strftime("encoder_bakeoff_%Y%m%dT%H%M%SZ")
    run_dir = project_root / args.artifacts_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    import shutil

    runner_script = PROJECT_ROOT / "scripts" / "run_encoder_benchmark.py"
    config = {
        "run_id": run_id,
        "purpose": "pure encoder bake-off: one model swap, identical frozen benchmarks, no crops/gate/OCR/reranking",
        "models_requested": args.models,
        "benchmarks": args.benchmarks,
        "protocol": "per-model reference index (2042 refs) -> cosine over L2-normalized global embeddings -> single full-image query",
        "adapter_registry": sorted(ADAPTER_REGISTRY),
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    statuses = {}
    for model_key in args.models:
        if model_key not in ADAPTER_REGISTRY:
            statuses[model_key] = {"status": "failed", "error": f"unknown model key; available: {sorted(ADAPTER_REGISTRY)}"}
            continue
        model_dir = run_dir / "models" / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        model_statuses = {}
        for benchmark in args.benchmarks:
            benchmark_dir = model_dir / benchmark
            command = [
                sys.executable,
                str(runner_script),
                "--model", model_key,
                "--benchmark", benchmark,
                "--output-dir", str(benchmark_dir),
            ]
            if args.device:
                command += ["--device", args.device]
            if args.batch_size:
                command += ["--batch-size", str(args.batch_size)]
            completed = None
            try:
                import subprocess

                completed = subprocess.run(command, capture_output=True, text=True, cwd=str(project_root))
                if completed.returncode != 0:
                    model_statuses[benchmark] = {
                        "status": "failed",
                        "error": (completed.stderr or completed.stdout)[-2000:],
                    }
                else:
                    model_statuses[benchmark] = {"status": "ok"}
            except Exception:
                model_statuses[benchmark] = {"status": "failed", "error": traceback.format_exc()[-2000:]}
        statuses[model_key] = {"status": "ok" if all(s["status"] == "ok" for s in model_statuses.values()) else "failed", "benchmarks": model_statuses}
        (run_dir / "model_statuses.json").write_text(
            json.dumps(statuses, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"[{model_key}] {statuses[model_key]['status']}")

    # Aggregate comparison artifacts from per-model metrics files.
    _build_summaries(run_dir, args.benchmarks)
    print(f"Bake-off artifacts: {run_dir}")


def _build_summaries(run_dir: Path, benchmarks: tuple[str, ...]) -> None:
    import csv

    def read_metrics(model_key: str, benchmark: str) -> dict | None:
        path = run_dir / "models" / model_key / benchmark / "metrics.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def read_resource(model_key: str) -> dict | None:
        path = run_dir / "models" / model_key / "synthetic_dev" / "resource.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    models = sorted({path.parent.parent.name for path in run_dir.glob("models/*/synthetic_dev/metrics.json")})

    # benchmark_summary.csv
    summary_rows = []
    for model_key in models:
        for benchmark in benchmarks:
            metrics = read_metrics(model_key, benchmark)
            if metrics is None:
                continue
            summary_rows.append(
                {
                    "model": model_key,
                    "benchmark": benchmark,
                    "queries": metrics["query_count"],
                    "top1_accuracy": metrics["top1_accuracy"],
                    "recall_at_5": metrics["recall_at_5"],
                    "recall_at_10": metrics["recall_at_10"],
                    "mrr": metrics["mrr"],
                    "median_target_rank_found": metrics["median_target_rank_found"],
                    "mean_target_rank_found": metrics.get("mean_target_rank_found"),
                    "mean_query_latency_ms": metrics.get("mean_query_latency_ms"),
                    "p95_query_latency_ms": metrics.get("p95_query_latency_ms"),
                }
            )
    if summary_rows:
        with (run_dir / "benchmark_summary.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)

    # resource_summary.csv
    resource_rows = []
    for model_key in models:
        resource = read_resource(model_key)
        if resource:
            resource_rows.append(resource)
    if resource_rows:
        with (run_dir / "resource_summary.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(resource_rows[0]))
            writer.writeheader()
            writer.writerows(resource_rows)

    # pairwise_vs_siglip.csv and oracle_complementarity.csv
    pairwise_rows = []
    oracle_rows = []
    base_model = "current_siglip2"
    for benchmark in benchmarks:
        base_rows = _load_predictions(run_dir, base_model, benchmark)
        if base_rows is None:
            continue
        for model_key in models:
            if model_key == base_model:
                continue
            other_rows = _load_predictions(run_dir, model_key, benchmark)
            if other_rows is None:
                continue
            pairwise_rows.append(
                {"benchmark": benchmark, "model": model_key, **_pairwise_counts(base_rows, other_rows)}
            )
            oracle_rows.append(
                {"benchmark": benchmark, "pair": f"{base_model}+{model_key}", **_oracle_counts(base_rows, other_rows)}
            )
    if pairwise_rows:
        with (run_dir / "pairwise_vs_siglip.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(pairwise_rows[0]))
            writer.writeheader()
            writer.writerows(pairwise_rows)
    if oracle_rows:
        with (run_dir / "oracle_complementarity.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(oracle_rows[0]))
            writer.writeheader()
            writer.writerows(oracle_rows)


def _load_predictions(run_dir: Path, model_key: str, benchmark: str) -> list[dict] | None:
    path = run_dir / "models" / model_key / benchmark / "predictions.csv"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream)) if (csv := __import__("csv")) else []
    return rows


def _pairwise_counts(baseline_rows: list[dict], other_rows: list[dict]) -> dict:
    counts = {
        "top1_siglip_wrong_model_correct": 0,
        "top1_siglip_correct_model_wrong": 0,
        "top1_both_correct": 0,
        "top1_both_wrong": 0,
        "top5_siglip_outside_model_inside": 0,
        "top5_siglip_inside_model_outside": 0,
        "rank_improved": 0,
        "rank_degraded": 0,
        "rank_same": 0,
    }
    for base, other in zip(baseline_rows, other_rows):
        base_correct = base["correct_top1"] == "True"
        other_correct = other["correct_top1"] == "True"
        if other_correct and not base_correct:
            counts["top1_siglip_wrong_model_correct"] += 1
        elif base_correct and not other_correct:
            counts["top1_siglip_correct_model_wrong"] += 1
        elif base_correct and other_correct:
            counts["top1_both_correct"] += 1
        else:
            counts["top1_both_wrong"] += 1
        base_inside = base["correct_top5"] == "True"
        other_inside = other["correct_top5"] == "True"
        if other_inside and not base_inside:
            counts["top5_siglip_outside_model_inside"] += 1
        if base_inside and not other_inside:
            counts["top5_siglip_inside_model_outside"] += 1
        base_rank = int(base["target_rank"]) if base["target_rank"] else None
        other_rank = int(other["target_rank"]) if other["target_rank"] else None
        if base_rank is None and other_rank is None:
            counts["rank_same"] += 1
        elif base_rank is None:
            counts["rank_improved"] += 1
        elif other_rank is None:
            counts["rank_degraded"] += 1
        elif other_rank < base_rank:
            counts["rank_improved"] += 1
        elif other_rank > base_rank:
            counts["rank_degraded"] += 1
        else:
            counts["rank_same"] += 1
    return counts


def _oracle_counts(baseline_rows: list[dict], other_rows: list[dict]) -> dict:
    queries = len(baseline_rows)
    top1 = sum(
        1 for base, other in zip(baseline_rows, other_rows)
        if base["correct_top1"] == "True" or other["correct_top1"] == "True"
    )
    top5 = sum(
        1 for base, other in zip(baseline_rows, other_rows)
        if base["correct_top5"] == "True" or other["correct_top5"] == "True"
    )
    return {
        "queries": queries,
        "oracle_top1_coverage": round(top1 / queries, 4),
        "oracle_recall_at_5_coverage": round(top5 / queries, 4),
        "baseline_top1": round(sum(1 for row in baseline_rows if row["correct_top1"] == "True") / queries, 4),
        "other_top1": round(sum(1 for row in other_rows if row["correct_top1"] == "True") / queries, 4),
    }


if __name__ == "__main__":
    main()
