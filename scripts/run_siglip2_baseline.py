#!/usr/bin/env python3
"""Run the reproducible zero-shot SigLIP 2 benchmark baseline."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recognition.benchmark import (  # noqa: E402
    RankingRecord,
    assert_no_reference_query_leakage,
    evaluate_rankings,
    target_rank,
)
from recognition.cache import (  # noqa: E402
    EmbeddingCacheMismatch,
    load_embedding_cache,
    save_embedding_cache,
    sha256_file,
)
from recognition.data import CatalogItem  # noqa: E402
from recognition.encoder import VisualEncoder  # noqa: E402
from recognition.index import CatalogIndex  # noqa: E402
from recognition.reporting import write_csv, write_error_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SigLIP2 zero-shot retrieval baseline")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--catalog-version", default="data/processed/catalog_v1.json")
    parser.add_argument("--catalog-manifest", default="data/processed/catalog_manifest.csv")
    parser.add_argument(
        "--benchmark",
        default="synthetic_dev",
        choices=("synthetic_dev", "hard_near_duplicate_dev", "hard_near_duplicate_dev_v2", "generated_stress_dev", "generated_stress_dev_pilot32", "web_extra_dev"),
        help="Run one benchmark taxonomy member; results are never pooled",
    )
    parser.add_argument(
        "--benchmark-manifest",
        default=None,
        help="Override the manifest path; otherwise use the selected benchmark default",
    )
    parser.add_argument(
        "--cache",
        default="artifacts/catalog_embeddings/siglip2_catalog.pt",
        help="Validated shared catalog embedding cache",
    )
    parser.add_argument("--artifacts-root", default="artifacts/experiments")
    parser.add_argument("--device", default=None, help="torch device, e.g. cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    catalog_version_path = _path(project_root, args.catalog_version)
    catalog_manifest_path = _path(project_root, args.catalog_manifest)
    benchmark_manifest_path = _default_benchmark_manifest(
        project_root, args.benchmark, args.benchmark_manifest
    )
    cache_path = _path(project_root, args.cache)
    catalog_meta = json.loads(catalog_version_path.read_text(encoding="utf-8"))
    catalog_rows = _read_csv(catalog_manifest_path)
    catalog_items, catalog_by_slug = _catalog_items(project_root, catalog_rows, catalog_meta)
    benchmark_rows = _read_csv(benchmark_manifest_path)
    _validate_benchmark(project_root, benchmark_rows, catalog_items, args.benchmark)

    benchmark_checksum = sha256_file(benchmark_manifest_path)
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
    cache_status = "loaded"
    try:
        catalog_embeddings = load_embedding_cache(cache_path, expected_cache_metadata)
    except (FileNotFoundError, EmbeddingCacheMismatch) as exc:
        cache_status = f"recomputed ({type(exc).__name__})"
        print(f"Catalog embedding cache not reused: {exc}; computing it now.")
        catalog_embeddings = encoder.encode([item.image_path for item in catalog_items])
        save_embedding_cache(cache_path, catalog_embeddings, expected_cache_metadata)

    index = CatalogIndex.from_items(
        catalog_items,
        catalog_embeddings,
        encoder_name=encoder.encoder_name,
        model_name="siglip2",
    )
    index.prepare_device(encoder.device)
    _synchronize(encoder.device)

    run_id = _new_run_id(project_root, args.artifacts_root, args.benchmark)
    run_dir = _path(project_root, args.artifacts_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "run_id": run_id,
        "baseline": "zero-shot visual retrieval",
        "model": "SigLIP2",
        "model_name": encoder.model_name,
        "model_id": encoder.model_id,
        "model_version": encoder.model_version,
        "device": encoder.device,
        "batch_size_for_catalog_indexing": args.batch_size,
        "query_batch_size": 1,
        "catalog_version": catalog_meta["catalog_version"],
        "catalog_manifest": _relative(catalog_manifest_path, project_root),
        "catalog_manifest_sha256": catalog_meta["manifest_sha256"],
        "benchmark_manifest": _relative(benchmark_manifest_path, project_root),
        "benchmark": args.benchmark,
        "benchmark_manifest_sha256": benchmark_checksum,
        "catalog_embedding_cache": _relative(cache_path, project_root),
        "catalog_embedding_cache_status": cache_status,
        "preprocessing_config": encoder.preprocessing_config,
        "retrieval": "normalized cosine similarity via matrix multiplication",
        "ocr": False,
        "fine_tuning": False,
        "inference_augmentation": False,
        "reranking": False,
        "synthetic_dev_is_internal_only": args.benchmark == "synthetic_dev",
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    prediction_rows: list[dict[str, object]] = []
    ranking_records: list[RankingRecord] = []
    for position, benchmark_row in enumerate(benchmark_rows, start=1):
        query_path = _path(project_root, benchmark_row["query_path"])
        total_started = perf_counter()
        _synchronize(encoder.device)
        embedding_started = perf_counter()
        embeddings = encoder.encode([query_path])
        _synchronize(encoder.device)
        embedding_latency_ms = (perf_counter() - embedding_started) * 1000.0
        if len(embeddings) != 1:
            raise ValueError(f"Encoder returned {len(embeddings)} embeddings for {query_path}")

        retrieval_started = perf_counter()
        results = index.search(embeddings[0], top_k=5, device=encoder.device)
        _synchronize(encoder.device)
        retrieval_latency_ms = (perf_counter() - retrieval_started) * 1000.0
        total_latency_ms = (perf_counter() - total_started) * 1000.0

        slugs = [result.item_id for result in results]
        scores = [float(result.score) for result in results]
        top1_score = scores[0] if scores else ""
        top2_score = scores[1] if len(scores) > 1 else ""
        margin = (scores[0] - scores[1]) if len(scores) > 1 else ""
        target = benchmark_row["target_slug"]
        rank = target_rank(target, slugs)
        target_score = scores[rank - 1] if rank and rank <= len(scores) else ""
        transform_type = _benchmark_label(benchmark_row, args.benchmark)
        prediction_rows.append(
            {
                "query_id": benchmark_row["query_id"],
                "target_slug": target,
                "predicted_slug": slugs[0] if slugs else "",
                "top1_score": top1_score,
                "top2_score": top2_score,
                "top1_top2_margin": margin,
                "top5_slugs": json.dumps(slugs, ensure_ascii=False, separators=(",", ":")),
                "top5_scores": json.dumps(scores, separators=(",", ":")),
                "correct_top1": bool(slugs and slugs[0] == target),
                "correct_top5": bool(rank is not None and rank <= 5),
                "target_rank": rank or "",
                "target_score": target_score,
                "embedding_latency_ms": round(embedding_latency_ms, 3),
                "retrieval_latency_ms": round(retrieval_latency_ms, 3),
                "latency_ms": round(total_latency_ms, 3),
                "transform_type": transform_type,
            }
        )
        ranking_records.append(
            RankingRecord(
                query_id=benchmark_row["query_id"],
                target_slug=target,
                ranked_candidates=slugs,
                scores=scores,
                latency_ms=total_latency_ms,
                transform_type=transform_type,
            )
        )
        if position % 100 == 0 or position == len(benchmark_rows):
            print(f"Processed {position}/{len(benchmark_rows)} queries")

    metrics = evaluate_rankings(ranking_records)
    errors = _errors(prediction_rows)
    metrics_payload = {
        "model": "SigLIP2",
        "model_id": encoder.model_id,
        "model_version": encoder.model_version,
        "catalog_version": catalog_meta["catalog_version"],
        "benchmark": args.benchmark,
        "benchmark_version": _benchmark_version(benchmark_manifest_path, args.benchmark),
        "benchmark_manifest_sha256": benchmark_checksum,
        "query_count": len(benchmark_rows),
        **metrics,
        "latency_breakdown_ms": {
            "query_embedding": _latency_summary(prediction_rows, "embedding_latency_ms"),
            "retrieval": _latency_summary(prediction_rows, "retrieval_latency_ms"),
            "total": _latency_summary(prediction_rows, "latency_ms"),
        },
        "top1_error_count": len(errors),
        "most_common_query_labels": Counter(
            _benchmark_label(row, args.benchmark) for row in benchmark_rows
        ).most_common(),
        "note": _benchmark_note(args.benchmark),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(
        run_dir / "predictions.csv",
        [
            "query_id",
            "target_slug",
            "predicted_slug",
            "top1_score",
            "top2_score",
            "top1_top2_margin",
            "top5_slugs",
            "top5_scores",
            "correct_top1",
            "correct_top5",
            "target_rank",
            "target_score",
            "embedding_latency_ms",
            "retrieval_latency_ms",
            "latency_ms",
            "transform_type",
        ],
        prediction_rows,
    )
    write_csv(
        run_dir / "errors.csv",
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
    write_error_report(
        run_dir / "error_report.html",
        errors,
        project_root=project_root,
        catalog_by_slug=catalog_by_slug,
        benchmark_rows_by_query={row["query_id"]: row for row in benchmark_rows},
    )
    print(json.dumps(metrics_payload, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Artifacts: {run_dir}")


def _catalog_items(
    project_root: Path, rows: list[dict[str, str]], catalog_meta: dict[str, object]
) -> tuple[list[CatalogItem], dict[str, dict[str, str]]]:
    current_checksum = sha256_file(_path(project_root, str(catalog_meta["source_manifest"])))
    if current_checksum != catalog_meta["manifest_sha256"]:
        raise ValueError("Catalog manifest checksum does not match catalog_v1.json")
    items: list[CatalogItem] = []
    by_slug: dict[str, dict[str, str]] = {}
    for row in sorted(rows, key=lambda value: value["slug"]):
        reference = row.get("reference_image_path", "").strip()
        if row.get("mapping_status") != "matched" or not reference:
            continue
        image_path = _path(project_root, reference)
        if not image_path.is_file():
            continue
        slug = row["slug"]
        if slug in by_slug:
            raise ValueError(f"Duplicate usable slug in catalog manifest: {slug}")
        by_slug[slug] = row
        items.append(CatalogItem(slug, image_path, metadata=row))
    expected_count = int(catalog_meta["usable_products"])
    if len(items) != expected_count:
        raise ValueError(f"catalog_v1 expects {expected_count} usable products, found {len(items)}")
    return items, by_slug


def _validate_benchmark(
    project_root: Path,
    rows: list[dict[str, str]],
    items: list[CatalogItem],
    benchmark: str,
) -> None:
    required = {"query_id", "query_path", "target_slug"}
    if not rows or required - set(rows[0]):
        raise ValueError(f"Benchmark manifest missing required columns: {sorted(required)}")
    if benchmark == "synthetic_dev":
        synthetic_required = {"source_reference_path", "transform_type", "transform_params", "seed"}
        if synthetic_required - set(rows[0]):
            raise ValueError(
                "synthetic_dev manifest missing required columns: "
                f"{sorted(synthetic_required - set(rows[0]))}"
            )
    if benchmark in {"generated_stress_dev", "generated_stress_dev_pilot32"}:
        generated_required = {"scenario_id", "generation_id", "source_reference", "manual_review_status"}
        if generated_required - set(rows[0]):
            raise ValueError(
                f"{benchmark} manifest missing required columns: "
                f"{sorted(generated_required - set(rows[0]))}"
            )
        non_accepted = [row["query_id"] for row in rows if row.get("manual_review_status") != "accepted"]
        if non_accepted:
            raise ValueError(
                f"{benchmark} manifest contains non-accepted rows: "
                f"{non_accepted[:3]} ({len(non_accepted)} total)"
            )
        if benchmark == "generated_stress_dev_pilot32":
            pilot_required = {"subset_role", "target_family_id", "family_type", "family_size", "family_member_order"}
            if pilot_required - set(rows[0]):
                raise ValueError(
                    "generated_stress_dev_pilot32 manifest missing required columns: "
                    f"{sorted(pilot_required - set(rows[0]))}"
                )
            hard_rows = [row for row in rows if row.get("target_family_id")]
            if any(row.get("subset_role") != "hard" for row in hard_rows):
                raise ValueError("generated_stress_dev_pilot32 family rows must have subset_role=hard")
            if any(not row.get("family_type") or not row.get("family_size") for row in hard_rows):
                raise ValueError("generated_stress_dev_pilot32 family rows need family_type and family_size")
    reference_paths = {item.image_path.resolve() for item in items}
    query_paths = set()
    catalog_slugs = {item.item_id for item in items}
    query_ids: set[str] = set()
    for row in rows:
        if row["query_id"] in query_ids:
            raise ValueError(f"Duplicate query_id: {row['query_id']}")
        query_ids.add(row["query_id"])
        query = _path(project_root, row["query_path"]).resolve()
        if not query.is_file():
            raise FileNotFoundError(f"Benchmark path missing for {row['query_id']}")
        query_paths.add(query)
        if row["target_slug"] not in catalog_slugs:
            raise ValueError(f"Benchmark target is not in catalog: {row['target_slug']}")
        source_reference = row.get("source_reference_path") or row.get("source_reference")
        if source_reference:
            source = _path(project_root, source_reference).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"Benchmark source path missing for {row['query_id']}")
            if source not in reference_paths:
                raise ValueError(f"Benchmark source is not in reference index: {source}")
    assert_no_reference_query_leakage(query_paths, reference_paths)


def _errors(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    errors = []
    for row in rows:
        if bool(row["correct_top1"]):
            continue
        errors.append(
            {
                "query_id": row["query_id"],
                "target_slug": row["target_slug"],
                "predicted_slug": row["predicted_slug"],
                "target_rank": row["target_rank"],
                "top1_score": row["top1_score"],
                "target_score": row["target_score"],
                "top1_top2_margin": row["top1_top2_margin"],
                "transform_type": row["transform_type"],
            }
        )
    return errors


def _latency_summary(rows: list[dict[str, object]], field: str) -> dict[str, float | None]:
    values = sorted(float(row[field]) for row in rows)
    if not values:
        return {"mean_ms": None, "p50_ms": None, "p95_ms": None}
    return {
        "mean_ms": sum(values) / len(values),
        "p50_ms": _percentile(values, 50),
        "p95_ms": _percentile(values, 95),
    }


def _percentile(values: list[float], percentage: float) -> float:
    position = (len(values) - 1) * percentage / 100.0
    lower = int(position)
    upper = min(len(values) - 1, lower + 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _default_benchmark_manifest(root: Path, benchmark: str, requested: str | None) -> Path:
    if requested:
        return _path(root, requested)
    names = {
        "synthetic_dev": "data/benchmarks/synthetic_dev/manifest.csv",
        "hard_near_duplicate_dev": "data/benchmarks/hard_near_duplicate_dev/manifest.csv",
        "hard_near_duplicate_dev_v2": "data/benchmarks/hard_near_duplicate_dev_v2/manifest.csv",
        "generated_stress_dev": "data/benchmarks/generated_stress_dev/manifest.csv",
        "generated_stress_dev_pilot32": "data/benchmarks/generated_stress_dev_pilot32/manifest.csv",
        "web_extra_dev": "data/benchmarks/web_extra_dev/manifest.csv",
    }
    return _path(root, names[benchmark])


def _benchmark_label(row: dict[str, str], benchmark: str) -> str:
    return row.get("transform_type", "") or row.get("scenario_id", "") or row.get("family_id", "") or benchmark


def _benchmark_version(manifest_path: Path, benchmark: str) -> str:
    metadata_path = manifest_path.parent / "metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            version = metadata.get("benchmark_version")
            if version:
                return str(version)
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "synthetic_dev": "synthetic-dev-v1",
        "hard_near_duplicate_dev": "hard-near-duplicate-v1",
        "hard_near_duplicate_dev_v2": "hard-near-duplicate-v2",
        "generated_stress_dev": "generated-stress-v1",
        "generated_stress_dev_pilot32": "generated-stress-pilot32-v1",
        "web_extra_dev": "web-extra-v1",
    }[benchmark]


def _benchmark_note(benchmark: str) -> str:
    return {
        "synthetic_dev": "Synthetic DEV is an internal comparative benchmark, not official evaluation and not real user photos.",
        "hard_near_duplicate_dev": "Hard benchmark diagnoses fine-grained confusion; v1 reuses synthetic DEV query images and is not independent.",
        "hard_near_duplicate_dev_v2": "Strict hard benchmark diagnoses fine-grained near-duplicate confusion; v2 reuses synthetic DEV query images and is not independent.",
        "generated_stress_dev": "Generated stress benchmark tests capture/domain robustness; generated images require manual review and are not real-world independent data.",
        "generated_stress_dev_pilot32": "Isolated 32-product generated-stress pilot; family-aware metrics apply to the hard half, and generated images require manual review.",
        "web_extra_dev": "Web-extra metrics cover only accepted exact-slug official-site images after conservative reference duplicate checks.",
    }[benchmark]


def _new_run_id(root: Path, artifacts_root: str, benchmark: str) -> str:
    base = datetime.now(timezone.utc).strftime(f"siglip2_{benchmark}_%Y%m%dT%H%M%SZ")
    target = _path(root, artifacts_root) / base
    if not target.exists():
        return base
    suffix = 2
    while (_path(root, artifacts_root) / f"{base}-{suffix}").exists():
        suffix += 1
    return f"{base}-{suffix}"


def _synchronize(device: str) -> None:
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)


if __name__ == "__main__":
    main()
