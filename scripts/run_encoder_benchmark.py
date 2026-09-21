#!/usr/bin/env python3
"""Run one frozen encoder adapter on one frozen benchmark (pure model swap)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from recognition.crop_diagnostics import (  # noqa: E402
    full_ranking,
    rank_metrics,
    target_rank_full,
)
from recognition.data import CatalogItem  # noqa: E402
from recognition.encoder_adapters import (  # noqa: E402
    create_adapter,
    load_reference_cache,
    reference_cache_dir,
    save_reference_cache,
)
from recognition.family_metrics import family_diagnostics  # noqa: E402
from recognition.preprocessing import load_rgb_image  # noqa: E402
from recognition.reporting import write_csv  # noqa: E402
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


def rank_band(rank: int | None) -> str:
    if rank is None:
        return ">100"
    for label, low, high in (("1", 1, 1), ("2-5", 2, 5), ("6-10", 6, 10), ("11-25", 11, 25), ("26-100", 26, 100), (">100", 101, 10**9)):
        if low <= rank <= high:
            return label
    return ">100"


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-model encoder benchmark")
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", required=True, choices=("synthetic_dev", "hard_near_duplicate_dev_v2", "generated_stress_dev_pilot32"))
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--cache-root", default="artifacts/reference_embeddings")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", required=True, help="artifact dir for this model/benchmark")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    import torch

    catalog_meta = json.loads(_path(project_root, "data/processed/catalog_v1.json").read_text(encoding="utf-8"))
    catalog_items, catalog_by_slug = _catalog_items(
        project_root, _read_csv(_path(project_root, "data/processed/catalog_manifest.csv")), catalog_meta
    )
    catalog_slugs = [item.item_id for item in catalog_items]
    manifest_rows = _read_csv(_default_benchmark_manifest(project_root, args.benchmark, None))
    _validate_benchmark(project_root, manifest_rows, catalog_items, args.benchmark)

    print(f"Loading adapter {args.model} ...")
    adapter = create_adapter(args.model, device=args.device, batch_size=args.batch_size)
    fingerprint = adapter.fingerprint(catalog_meta["manifest_sha256"], len(catalog_items))

    cache_dir = reference_cache_dir(project_root, adapter.model_key)
    from recognition.encoder_adapters import ReferenceCacheMismatch

    try:
        reference_embeddings = load_reference_cache(cache_dir, fingerprint, catalog_slugs)
        cache_status = "loaded"
        reference_encode_seconds = None
    except (FileNotFoundError, ReferenceCacheMismatch) as exc:
        print(f"Reference cache not reused ({type(exc).__name__}); encoding 2042 references ...")
        started = perf_counter()
        reference_embeddings = adapter.encode_paths([item.image_path for item in catalog_items])
        reference_encode_seconds = perf_counter() - started
        save_reference_cache(
            cache_dir, reference_embeddings, catalog_slugs, adapter.metadata(), fingerprint
        )
        cache_status = "computed"

    reference_matrix = torch.tensor(reference_embeddings, dtype=torch.float32, device=adapter.device)
    reference_matrix = torch.nn.functional.normalize(reference_matrix, p=2, dim=1)

    rows = []
    checkpoint_path = run_dir / "predictions_partial.csv"
    if checkpoint_path.is_file():
        with checkpoint_path.open("r", encoding="utf-8-sig", newline="") as stream:
            import csv as csv_module

            rows = list(csv_module.DictReader(stream))
        print(f"Resuming from checkpoint: {len(rows)} rows already computed")
    completed_ids = {row["query_id"] for row in rows}

    for position, query in enumerate(manifest_rows, start=1):
        if query["query_id"] in completed_ids:
            continue
        image = load_rgb_image(_path(project_root, query["query_path"]))
        if str(adapter.device).startswith("cuda"):
            torch.cuda.synchronize(adapter.device)
        started = perf_counter()
        embedding = torch.tensor(adapter.encode_images([image]), dtype=torch.float32, device=adapter.device)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        if str(adapter.device).startswith("cuda"):
            torch.cuda.synchronize(adapter.device)
        embed_ms = (perf_counter() - started) * 1000.0
        retrieval_started = perf_counter()
        scores = (embedding @ reference_matrix.transpose(0, 1))[0].detach().cpu().tolist()
        ranked = full_ranking(scores, catalog_slugs)
        if str(adapter.device).startswith("cuda"):
            torch.cuda.synchronize(adapter.device)
        retrieval_ms = (perf_counter() - retrieval_started) * 1000.0
        ranked_slugs = [slug for slug, _ in ranked]
        rank = target_rank_full(ranked_slugs, query["target_slug"])
        top5_scores = [score for _, score in ranked[:5]]
        rows.append(
            {
                "query_id": query["query_id"],
                "target_slug": query["target_slug"],
                "predicted_slug": ranked_slugs[0],
                "top1_score": ranked[0][1],
                "top5_slugs": json.dumps(ranked_slugs[:5], ensure_ascii=False, separators=(",", ":")),
                "top5_scores": json.dumps(top5_scores, separators=(",", ":")),
                "correct_top1": ranked_slugs[0] == query["target_slug"],
                "correct_top5": rank is not None and rank <= 5,
                "target_rank": rank if rank is not None else "",
                "target_rank_band": rank_band(rank),
                "target_score": top5_scores[rank - 1] if rank is not None and rank <= 5 else "",
                "embedding_latency_ms": round(embed_ms, 2),
                "retrieval_latency_ms": round(retrieval_ms, 2),
                "latency_ms": round(embed_ms + retrieval_ms, 2),
            }
        )
        if position % 500 == 0 or position == len(manifest_rows):
            write_csv(
                checkpoint_path,
                [key for key in rows[0] if key not in ()] if rows else ["query_id"],
                rows,
            )
            print(f"  {position}/{len(manifest_rows)} (checkpointed)")

    ranks = [int(row["target_rank"]) if row["target_rank"] != "" else None for row in rows]
    if checkpoint_path.is_file():
        checkpoint_path.unlink()
    metrics = rank_metrics(ranks)
    latencies = [float(row["latency_ms"]) for row in rows]
    embed_latencies = [float(row["embedding_latency_ms"]) for row in rows]
    payload = {
        "benchmark": args.benchmark,
        "model": adapter.model_key,
        "model_id": adapter.hf_model_id,
        "checkpoint_revision": adapter.checkpoint_revision,
        "reference_cache": cache_status,
        "reference_encode_seconds": round(reference_encode_seconds, 1) if reference_encode_seconds else None,
        "query_count": len(manifest_rows),
        "unique_target_slugs": len({row["target_slug"] for row in manifest_rows}),
        **metrics,
        "mean_target_rank_found": _mean_found(ranks),
        "mean_query_latency_ms": round(sum(latencies) / len(latencies), 2),
        "p50_query_latency_ms": _percentile(latencies, 50),
        "p95_query_latency_ms": _percentile(latencies, 95),
        "mean_query_embed_ms": round(sum(embed_latencies) / len(embed_latencies), 2),
        "note": "single full image, official model preprocessing, cosine retrieval; no crops/gate/OCR",
    }
    if args.benchmark == "generated_stress_dev_pilot32":
        scenario_by_query = {row["query_id"]: row.get("scenario_id", "") for row in manifest_rows}
        subset_by_query = {row["query_id"]: row.get("subset_role", "") for row in manifest_rows}
        family_type_by_query = {row["query_id"]: row.get("family_type", "") for row in manifest_rows}
        for row in rows:
            row["scenario_id"] = scenario_by_query[row["query_id"]]
            row["subset_role"] = subset_by_query[row["query_id"]]
            row["family_type"] = family_type_by_query[row["query_id"]]
        payload["per_scenario"] = {
            scenario: _scoped_metrics(rows, "scenario_id", scenario)
            for scenario in SCENARIOS
        }
        payload["per_subset"] = {
            subset: _scoped_metrics(rows, "subset_role", subset)
            for subset in SUBSETS
        }
        payload["family_diagnostics"] = family_diagnostics(rows, manifest_rows)
        write_csv(
            run_dir / "scenario_metrics.csv",
            ["scope_value", "top1_accuracy", "recall_at_5", "recall_at_10", "mrr", "median_target_rank_found"],
            [
                {"scope_value": scenario, **_scoped_metrics(rows, "scenario_id", scenario)}
                for scenario in SCENARIOS
            ],
        )
        write_csv(
            run_dir / "subset_metrics.csv",
            ["scope_value", "top1_accuracy", "recall_at_5", "recall_at_10", "mrr", "median_target_rank_found"],
            [
                {"scope_value": subset, **_scoped_metrics(rows, "subset_role", subset)}
                for subset in SUBSETS
            ],
        )
        family_rows = []
        for family_type, breakdown in sorted(payload["family_diagnostics"]["family_type_breakdown"].items()):
            family_rows.append(
                {
                    "family_type": family_type,
                    "query_count": breakdown["query_count"],
                    "exact_top1": breakdown["top1_accuracy"],
                    "family_top1": breakdown["family_top1"],
                    "family_recall_at_5": breakdown["family_recall_at_5"],
                    "within_family_disambiguation_top1": breakdown["within_family_disambiguation_top1"],
                }
            )
        write_csv(
            run_dir / "family_metrics.csv",
            ["family_type", "query_count", "exact_top1", "family_top1", "family_recall_at_5", "within_family_disambiguation_top1"],
            family_rows,
        )
    if args.benchmark == "hard_near_duplicate_dev_v2":
        payload["hard_diagnostics"] = _hard_diagnostics(rows)

    write_csv(
        run_dir / "predictions.csv",
        [key for key in rows[0] if key not in ()],
        rows,
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(
        run_dir / "target_ranks.csv",
        ["query_id", "target_slug", "target_rank", "target_rank_band", "predicted_slug", "top1_score"],
        [
            {
                "query_id": row["query_id"],
                "target_slug": row["target_slug"],
                "target_rank": row["target_rank"],
                "target_rank_band": row["target_rank_band"],
                "predicted_slug": row["predicted_slug"],
                "top1_score": row["top1_score"],
            }
            for row in rows
        ],
    )
    resource = {
        "model": adapter.model_key,
        "model_id": adapter.hf_model_id,
        "embedding_dim": adapter.embedding_dim,
        "dtype": adapter.dtype,
        "batch_size_used": adapter.batch_size or 16,
        "parameters": adapter.metadata().get("parameter_count"),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if str(adapter.device).startswith("cuda") else None,
    }
    (run_dir / "resource.json").write_text(
        json.dumps(resource, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    adapter.release()
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _scoped_metrics(rows: list[dict], key: str, value: str) -> dict:
    scoped = [row for row in rows if row[key] == value]
    ranks = [int(row["target_rank"]) if row["target_rank"] != "" else None for row in scoped]
    metrics = rank_metrics(ranks)
    return {k: v for k, v in metrics.items() if k in ("top1_accuracy", "recall_at_5", "recall_at_10", "mrr", "median_target_rank_found")}


def _hard_diagnostics(rows: list[dict]) -> dict:
    from collections import Counter

    rank_distribution = Counter(row["target_rank_band"] for row in rows)
    family_counter: Counter = Counter()
    family_correct: Counter = Counter()
    confusion_pairs: Counter = Counter()
    for row in rows:
        family_id = row["query_id"].rsplit("__", 1)[0]
        family_counter[family_id] += 1
        if row["correct_top1"]:
            family_correct[family_id] += 1
        else:
            confusion_pairs[(row["target_slug"], row["predicted_slug"])] += 1
    return {
        "target_rank_distribution": {band: rank_distribution.get(band, 0) for band in RANK_BANDS_ORDER},
        "accuracy_per_family": {
            family: family_correct[family] / total
            for family, total in sorted(family_counter.items())
        },
        "top_confusion_pairs": [
            {"target_slug": target, "predicted_slug": predicted, "count": count}
            for (target, predicted), count in confusion_pairs.most_common(20)
        ],
    }


def _mean_found(ranks: list[int | None]) -> float | None:
    found = [rank for rank in ranks if rank is not None]
    return round(sum(found) / len(found), 2) if found else None


def _percentile(values: list[float], percentage: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100.0
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 2)


if __name__ == "__main__":
    main()
