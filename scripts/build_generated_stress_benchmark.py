#!/usr/bin/env python3
"""Build the scored generated_stress_dev manifest from manual review only."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recognition.reporting import write_csv  # noqa: E402
from scripts.import_generated_stress import _path, _validate_file  # noqa: E402


MANIFEST_FIELDS = [
    "query_id", "query_path", "target_slug", "scenario_id", "source_reference",
    "provenance", "generation_id", "manual_review_status",
]
OPTIONAL_PILOT_FIELDS = [
    "subset_role", "target_family_id", "family_type", "family_size", "family_member_order",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scored generated_stress_dev manifest")
    parser.add_argument("--generation-manifest", default="data/benchmarks/generated_stress_dev/generation_manifest.csv")
    parser.add_argument("--review-csv", default="data/benchmarks/generated_stress_dev/review.csv")
    parser.add_argument("--catalog-manifest", default="data/processed/catalog_manifest.csv")
    parser.add_argument("--output-dir", default="data/benchmarks/generated_stress_dev")
    args = parser.parse_args()
    result = build_benchmark(
        _path(args.generation_manifest), _path(args.review_csv), _path(args.catalog_manifest), _path(args.output_dir)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def build_benchmark(
    generation_manifest_path: Path,
    review_csv_path: Path,
    catalog_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    plan = _read_csv(generation_manifest_path)
    review = _read_csv(review_csv_path)
    catalog = _catalog_by_slug(catalog_manifest_path)
    plan_by_id = _unique(plan, "generation_id", "generation manifest")
    review_by_id = _unique(review, "generation_id", "review CSV")
    missing_review = sorted(set(plan_by_id) - set(review_by_id))
    unknown_review = sorted(set(review_by_id) - set(plan_by_id))
    if missing_review or unknown_review:
        raise ValueError(f"review does not align with generation plan: missing={missing_review[:3]}, unknown={unknown_review[:3]}")

    accepted_rows: list[dict[str, object]] = []
    invalid_accepted: list[str] = []
    reference_hashes: dict[str, str] = {}
    status_counts = Counter()
    for generation_id in sorted(plan_by_id):
        plan_row = plan_by_id[generation_id]
        review_row = review_by_id[generation_id]
        status = review_row.get("review_status", "pending")
        status_counts[status] += 1
        slug = plan_row["target_slug"]
        if slug not in catalog:
            raise ValueError(f"target is not a canonical usable slug: {slug}")
        if review_row.get("reference_image") != catalog[slug]["reference_image_path"]:
            raise ValueError(f"reference mismatch for {generation_id}")
        if status != "accepted":
            continue
        generated_path = _path(review_row.get("generated_path", ""))
        validation = _validate_file(generated_path, _path(catalog[slug]["reference_image_path"]), reference_hashes)
        if validation["validation_status"] != "valid":
            invalid_accepted.append(f"{generation_id}:{validation['validation_status']}")
            continue
        accepted_rows.append({
            "query_id": generation_id,
            "query_path": review_row["generated_path"],
            "target_slug": slug,
            "scenario_id": plan_row["scenario_id"],
            "source_reference": catalog[slug]["reference_image_path"],
            "provenance": "generated_image_edit",
            "generation_id": generation_id,
            "manual_review_status": "accepted",
        })
    if invalid_accepted:
        raise ValueError("accepted rows failed revalidation: " + ", ".join(invalid_accepted[:5]))

    optional_fields = [field for field in OPTIONAL_PILOT_FIELDS if field in plan[0]]
    output_fields = MANIFEST_FIELDS + optional_fields
    for row in accepted_rows:
        plan_row = plan_by_id[row["generation_id"]]
        for field in optional_fields:
            row[field] = plan_row.get(field, "")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = output_dir / "manifest.csv"
    write_csv(output_manifest, output_fields, accepted_rows)
    metadata_path = output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    metadata.update({
        "accepted_queries": len(accepted_rows),
        "rejected_queries": status_counts.get("rejected", 0),
        "pending_queries": status_counts.get("pending", 0),
        "manifest_queries": len(accepted_rows),
        "manifest_path": _relative(output_manifest),
        "manifest_built_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scored_manifest_manual_review_gate": "review_status=accepted AND validation_status=valid AND exact_reference_duplicate=false",
    })
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {
        "planned": len(plan),
        "accepted": len(accepted_rows),
        "rejected": status_counts.get("rejected", 0),
        "pending": status_counts.get("pending", 0),
        "manifest": _relative(output_manifest),
    }
    (output_dir / "build_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _catalog_by_slug(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(path)
    result = {}
    for row in rows:
        if row.get("mapping_status") == "matched" and row.get("reference_image_path") and _path(row["reference_image_path"]).is_file():
            result[row["slug"]] = row
    return result


def _unique(rows: list[dict[str, str]], field: str, label: str) -> dict[str, dict[str, str]]:
    result = {}
    for row in rows:
        value = row.get(field, "")
        if not value:
            raise ValueError(f"{label} contains empty {field}")
        if value in result:
            raise ValueError(f"{label} contains duplicate {field}: {value}")
        result[value] = row
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    main()
