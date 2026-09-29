#!/usr/bin/env python3
"""Build the local OCR caches for the OCR reranking milestone.

OCR runs exactly once per unique image and is cached; reranking experiments
never re-run OCR. Caches live under:

    artifacts/ocr_cache/<ocr_model>/<benchmark>/query_ocr.jsonl
    artifacts/ocr_cache/<ocr_model>/catalog_references/reference_ocr.jsonl

Images are de-duplicated by sha256 across the whole invocation (hard_v2 query
bytes are reused synthetic_dev queries), so duplicated files are OCR'd once
and the duplicate records reference the same lines.

Also writes (Parts Q and X of the brief):
- per-benchmark OCR coverage reports (token counts, vintage detection,
  winery/name detection against the target's own catalog fields);
- a reference OCR quality audit + a 50-reference random visual sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from recognition.cache import sha256_file  # noqa: E402
from recognition.ocr_engine import (  # noqa: E402
    OCR_CONFIGS,
    OCR_MODEL_KEY,
    create_ocr_engine,
    engine_metadata,
)
from recognition.text_signals import (  # noqa: E402
    build_candidate_text_index,
    build_query_text_evidence,
    compute_text_signals,
)
from scripts.run_siglip2_baseline import (  # noqa: E402
    _catalog_items,
    _default_benchmark_manifest,
    _path,
    _read_csv,
)

BENCHMARKS = ("synthetic_dev", "hard_near_duplicate_dev_v2", "generated_stress_dev_pilot32")
WELL_DETECTED_THRESHOLD = 0.85


def lines_to_records(lines):
    return [
        {"text": line.text, "confidence": round(line.confidence, 4), "box": list(line.box) if line.box else None}
        for line in lines
    ]


def sha256_of_image(path: Path) -> str:
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OCR caches (queries + catalog references)")
    parser.add_argument("--ocr-config", default="current_eslav", choices=sorted(OCR_CONFIGS))
    parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=BENCHMARKS)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--skip-references", action="store_true")
    parser.add_argument("--skip-queries", action="store_true")
    parser.add_argument("--audit-sample-size", type=int, default=50)
    args = parser.parse_args()

    ocr_config = OCR_CONFIGS[args.ocr_config]
    project_root = PROJECT_ROOT
    cache_root = project_root / "artifacts" / "ocr_cache" / ocr_config.cache_key
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "ocr_engine.json").write_text(
        json.dumps(engine_metadata(args.device, ocr_config), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    catalog_meta = json.loads(_path(project_root, "data/processed/catalog_v1.json").read_text(encoding="utf-8"))
    catalog_items, catalog_by_slug = _catalog_items(
        project_root, _read_csv(_path(project_root, "data/processed/catalog_manifest.csv")), catalog_meta
    )

    import os

    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    print("Initializing OCR engine ...", flush=True)
    engine = create_ocr_engine(ocr_config, device=args.device)

    # ------------------------------------------------------------------
    # Reference OCR cache (Part C) + audit (Part X)
    # ------------------------------------------------------------------
    reference_cache_path = cache_root / "catalog_references" / "reference_ocr.jsonl"
    if not args.skip_references:
        if reference_cache_path.is_file() and _count_lines(reference_cache_path) == len(catalog_items):
            print(f"Reference cache complete, skipping ({reference_cache_path}); delete to rebuild.")
        else:
            incremental_build(
                reference_cache_path,
                identity_key="slug",
                identities=[(item.item_id, item.image_path) for item in catalog_items],
                engine=engine,
                project_root=project_root,
                progress_label="references",
            )

        # Audit (runs whether the cache was just built or loaded).
        records = [json.loads(line) for line in reference_cache_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        audit = reference_audit(records)
        (cache_root / "catalog_references" / "reference_ocr_audit.json").write_text(
            json.dumps(audit["summary"], ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_reference_sample_html(
            cache_root / "catalog_references" / "reference_ocr_sample.html",
            random.Random(20260920).sample(records, min(args.audit_sample_size, len(records))),
        )
        print(f"Reference audit: {audit['summary']}")

    # ------------------------------------------------------------------
    # Query OCR caches (Part B) + coverage diagnostics (Part Q)
    # ------------------------------------------------------------------
    candidate_indexes = {item.item_id: build_candidate_text_index(item.metadata) for item in catalog_items}

    for benchmark in args.benchmarks:
        benchmark_cache_dir = cache_root / benchmark
        query_cache_path = benchmark_cache_dir / "query_ocr.jsonl"
        manifest_rows = _read_csv(_default_benchmark_manifest(project_root, benchmark, None))
        if not query_cache_path.is_file():
            incremental_build(
                query_cache_path,
                identity_key="query_id",
                identities=[(row["query_id"], _path(project_root, row["query_path"])) for row in manifest_rows],
                engine=engine,
                project_root=project_root,
                progress_label=benchmark,
                extra_fields=lambda row, image_path: {"target_slug": row["target_slug"], "image_path": row["query_path"]},
                manifest_rows=manifest_rows,
            )
        records = [json.loads(line) for line in query_cache_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        coverage = coverage_report(records, manifest_rows, candidate_indexes, benchmark)
        (benchmark_cache_dir / "ocr_coverage.json").write_text(
            json.dumps(coverage, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Coverage[{benchmark}]: {json.dumps(coverage['overall'])}")


def _count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def incremental_build(
    cache_path: Path,
    identity_key: str,
    identities: list[tuple[str, Path]],
    engine,
    project_root: Path,
    progress_label: str,
    extra_fields=None,
    manifest_rows: list[dict] | None = None,
) -> None:
    """Crash-safe incremental OCR build: append each record immediately and
    resume from the existing partial file (identity-keyed)."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if cache_path.is_file():
        with cache_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    done.add(json.loads(line)[identity_key])
        print(f"  resuming {progress_label}: {len(done)} records already cached", flush=True)

    manifest_by_id = {row["query_id"]: row for row in (manifest_rows or [])}
    dedup: dict[str, tuple] = {}
    started_all = time.time()
    pending = [(identity, path) for identity, path in identities if identity not in done]
    with cache_path.open("a", encoding="utf-8") as stream:
        for position, (identity, image_path) in enumerate(pending, start=1):
            digest = sha256_of_image(image_path)
            cached = dedup.get(digest)
            if cached is not None:
                lines, seconds = cached
                source = "dedup"
            else:
                started = time.time()
                lines = engine.predict_lines(image_path)
                seconds = time.time() - started
                dedup[digest] = (lines, seconds)
                source = "ocr"
            record = {
                identity_key: identity,
                "image_sha256": digest,
                "lines": lines_to_records(lines),
                "ocr_seconds": round(seconds, 3) if seconds is not None else None,
                "source": source,
            }
            if identity_key == "slug":
                record["image_path"] = str(image_path.relative_to(project_root))
            else:
                manifest_row = manifest_by_id.get(identity, {})
                record["target_slug"] = manifest_row.get("target_slug", "")
                record["image_path"] = manifest_row.get("query_path", "")
            if extra_fields is not None:
                record.update(extra_fields(manifest_row, image_path))
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            if position % 200 == 0 or position == len(pending):
                print(
                    f"  {progress_label} {position}/{len(pending)} ({time.time() - started_all:.0f}s)",
                    flush=True,
                )


def reference_audit(records: list[dict]) -> dict:
    token_counts = []
    empty, with_years = 0, 0
    year_slugs = []
    for record in records:
        evidence = build_query_text_evidence([(line["text"], line["confidence"]) for line in record["lines"]])
        token_counts.append(len(evidence.tokens))
        if not evidence.tokens:
            empty += 1
        if evidence.vintage_years:
            with_years += 1
            year_slugs.append(record["slug"])
    ordered = sorted(token_counts)
    median = ordered[len(ordered) // 2] if ordered else 0
    return {
        "summary": {
            "references": len(records),
            "empty_ocr": empty,
            "empty_ocr_share": round(empty / len(records), 4) if records else None,
            "median_tokens": median,
            "references_with_detected_year": with_years,
            "year_reference_share": round(with_years / len(records), 4) if records else None,
            "example_year_references": year_slugs[:20],
            "suspicious_failures": [
                {"slug": record["slug"], "image_path": record["image_path"]}
                for record in records
                if not [line for line in record["lines"] if line["text"].strip()]
            ][:50],
        }
    }


def write_reference_sample_html(path: Path, records: list[dict]) -> None:
    rows = []
    for record in records:
        lines = "".join(
            f"<div>[{line['confidence']:.2f}] {line['text']}</div>" for line in record["lines"]
        ) or "<div><b>EMPTY OCR</b></div>"
        rows.append(
            f"<div class='card'><img src='file://{record['image_path']}' loading='lazy'>"
            f"<div class='meta'><b>{record['slug']}</b>{lines}</div></div>"
        )
    html = (
        "<!doctype html><meta charset='utf-8'><title>Reference OCR sample</title>"
        "<style>body{font-family:sans-serif;background:#fafafa}.card{display:inline-block;"
        "margin:8px;padding:8px;border:1px solid #ddd;vertical-align:top;width:420px}"
        "img{width:200px;float:left;margin-right:8px}.meta{font-size:12px}</style>"
        "<h1>Reference OCR random sample (50)</h1>" + "".join(rows)
    )
    path.write_text(html, encoding="utf-8")


def coverage_report(records: list[dict], manifest_rows: list[dict], candidate_indexes: dict, benchmark: str) -> dict:
    manifest_by_query = {row["query_id"]: row for row in manifest_rows}
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups["overall"].append(record)
        manifest_row = manifest_by_query.get(record["query_id"])
        if manifest_row is None:
            continue
        if "scenario_id" in manifest_row and manifest_row["scenario_id"]:
            groups[f"scenario:{manifest_row['scenario_id']}"].append(record)
        if "subset_role" in manifest_row and manifest_row["subset_role"]:
            groups[f"subset:{manifest_row['subset_role']}"].append(record)

    report: dict[str, dict] = {}
    for group, group_records in sorted(groups.items()):
        token_counts, with_tokens, with_years, winery_hits, name_hits = [], 0, 0, 0, 0
        for record in group_records:
            evidence = build_query_text_evidence([(line["text"], line["confidence"]) for line in record["lines"]])
            token_counts.append(len(evidence.tokens))
            if evidence.tokens:
                with_tokens += 1
            if evidence.vintage_years:
                with_years += 1
            target_index = candidate_indexes.get(record["target_slug"])
            if target_index is not None:
                signals = compute_text_signals(evidence, target_index)
                if signals["winery_score"] >= WELL_DETECTED_THRESHOLD:
                    winery_hits += 1
                if signals["metadata_name_score"] >= WELL_DETECTED_THRESHOLD:
                    name_hits += 1
        ordered = sorted(token_counts)
        report[group] = {
            "queries": len(group_records),
            "share_with_any_token": round(with_tokens / len(group_records), 4) if group_records else None,
            "median_token_count": ordered[len(ordered) // 2] if ordered else 0,
            "share_with_detected_vintage": round(with_years / len(group_records), 4) if group_records else None,
            "share_winery_detected": round(winery_hits / len(group_records), 4) if group_records else None,
            "share_name_detected": round(name_hits / len(group_records), 4) if group_records else None,
        }
    return {"benchmark": benchmark, "threshold": WELL_DETECTED_THRESHOLD, "overall": report["overall"], "groups": {k: v for k, v in report.items() if k != "overall"}}


def _write_query_csv(path: Path, records: list[dict]) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["query_id", "target_slug", "line_count", "raw_text", "detected_years", "ocr_seconds", "source"])
        for record in records:
            evidence = build_query_text_evidence([(line["text"], line["confidence"]) for line in record["lines"]])
            raw_text = " | ".join(line["text"] for line in record["lines"])
            writer.writerow(
                [
                    record["query_id"],
                    record["target_slug"],
                    len(record["lines"]),
                    raw_text,
                    " ".join(evidence.vintage_years),
                    record["ocr_seconds"] if record["ocr_seconds"] is not None else "",
                    record["source"],
                ]
            )


if __name__ == "__main__":
    main()
