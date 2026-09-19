#!/usr/bin/env python3
"""Build the deterministic hard near-duplicate diagnostic benchmark."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recognition.hard_families import build_hard_families, year_if_known  # noqa: E402
from recognition.reporting import write_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hard near-duplicate DEV benchmark")
    parser.add_argument("--catalog-manifest", default="data/processed/catalog_manifest.csv")
    parser.add_argument("--synthetic-manifest", default="data/benchmarks/synthetic_dev/manifest.csv")
    parser.add_argument(
        "--predictions",
        default="artifacts/experiments/siglip2_synthetic_dev_20260916T062142Z/predictions.csv",
    )
    parser.add_argument("--output-dir", default="data/benchmarks/hard_near_duplicate_dev")
    parser.add_argument("--report", default="reports/hard_near_duplicate_dev_report.md")
    args = parser.parse_args()

    catalog_path = _path(args.catalog_manifest)
    synthetic_path = _path(args.synthetic_manifest)
    predictions_path = _path(args.predictions)
    output_dir = _path(args.output_dir)
    catalog_rows = _read_csv(catalog_path)
    synthetic_rows = _read_csv(synthetic_path)
    prediction_rows = _read_csv(predictions_path)
    families = build_hard_families(catalog_rows, prediction_rows)
    if not families:
        raise ValueError("No hard families were supported by the current metadata/confusion signals")

    catalog_by_slug = {row["slug"]: row for row in catalog_rows}
    synthetic_by_target: dict[str, list[dict[str, str]]] = {}
    for row in synthetic_rows:
        synthetic_by_target.setdefault(row["target_slug"], []).append(row)

    family_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for family in families:
        for slug in family.slugs:
            row = catalog_by_slug[slug]
            family_rows.append(
                {
                    "family_id": family.family_id,
                    "slug": slug,
                    "product_name": row.get("title", ""),
                    "winery": row.get("winery", ""),
                    "category": row.get("category", ""),
                    "color": row.get("color", ""),
                    "region": row.get("region", ""),
                    "grape": row.get("grape", ""),
                    "year_if_known": year_if_known(row),
                    "family_reason": family.reason,
                }
            )
            for synthetic in sorted(synthetic_by_target.get(slug, []), key=lambda value: value["query_id"]):
                manifest_rows.append(
                    {
                        "query_id": f"{family.family_id}__{synthetic['query_id']}",
                        "query_path": synthetic["query_path"],
                        "target_slug": slug,
                        "family_id": family.family_id,
                        "source_benchmark": "synthetic_dev",
                        "source_reference": row["reference_image_path"],
                        "provenance": "reused synthetic_dev query; no new image generation",
                        "selection_reason": family.reason,
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "families.csv",
        [
            "family_id", "slug", "product_name", "winery", "category", "color",
            "region", "grape", "year_if_known", "family_reason",
        ],
        family_rows,
    )
    write_csv(
        output_dir / "manifest.csv",
        [
            "query_id", "query_path", "target_slug", "family_id", "source_benchmark",
            "source_reference", "provenance", "selection_reason",
        ],
        manifest_rows,
    )

    family_by_slug = {slug: family.family_id for family in families for slug in family.slugs}
    intersection = [
        row for row in prediction_rows
        if row.get("target_slug") in family_by_slug
        and row.get("predicted_slug") in family_by_slug
        and family_by_slug[row["target_slug"]] == family_by_slug[row["predicted_slug"]]
        and row.get("target_slug") != row.get("predicted_slug")
    ]
    family_sizes = Counter(len(family.slugs) for family in families)
    strongest_pairs = Counter(
        (row["target_slug"], row["predicted_slug"]) for row in intersection
    ).most_common(10)
    report = _report(families, manifest_rows, family_sizes, intersection, strongest_pairs)
    report_path = _path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(json.dumps({
        "families": len(families),
        "unique_products": len({row["slug"] for row in family_rows}),
        "queries": len(manifest_rows),
        "siglip2_confusion_intersection": len(intersection),
        "family_size_distribution": dict(sorted(family_sizes.items())),
    }, ensure_ascii=False, indent=2))


def _report(families, manifest_rows, family_sizes, intersection, strongest_pairs) -> str:
    lines = [
        "# hard_near_duplicate_dev report",
        "",
        "Это диагностический benchmark для fine-grained confusion. Query images переиспользованы из `synthetic_dev`; независимость от catalog reference отсутствует по дизайну.",
        "",
        "## Summary",
        "",
        f"- Hard families: **{len(families)}**.",
        f"- Unique products: **{len({slug for family in families for slug in family.slugs})}**.",
        f"- Queries: **{len(manifest_rows)}**.",
        f"- Family size distribution: `{dict(sorted(family_sizes.items()))}`.",
        f"- Current SigLIP2 errors where target and prediction are in the same family: **{len(intersection)}**.",
        "",
        "## Selection logic",
        "",
        "Families use only canonical metadata plus observed SigLIP2 confusion pairs from the synthetic DEV run. Metadata-only edges require the same winery, meaningful name overlap, and at least one shared category/color/region/grape signal. A year is recorded only when a unique explicit 19xx/20xx token is present in title/slug.",
        "",
        "## Strongest observed confusion pairs inside families",
        "",
    ]
    lines.extend(f"- `{count}`: `{target}` → `{predicted}`" for (target, predicted), count in strongest_pairs)
    if not strongest_pairs:
        lines.append("- No observed pair inside a generated family.")
    lines.extend([
        "",
        "## Limitations",
        "",
        "- This is not a real-world smartphone benchmark and must not be combined with `synthetic_dev` or `web_extra_dev` into a headline metric.",
        "- Query images originate from catalog references through synthetic transforms, so this benchmark diagnoses fine-grained retrieval confusion rather than independent-image generalization.",
        "- Family membership is a conservative reproducible diagnostic selection, not a claim that every member is visually identical.",
        "",
    ])
    return "\n".join(lines)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
