#!/usr/bin/env python3
"""Build and audit a strict v2 hard near-duplicate DEV benchmark.

The script reads the existing v1 family candidate universe and the frozen
synthetic-dev SigLIP2 predictions. It creates a new benchmark directory and
never changes v1, synthetic_dev, catalog references, or model artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
from html import escape
import itertools
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from recognition.benchmark import RankingRecord, evaluate_rankings, target_rank  # noqa: E402
from recognition.hard_families import HardFamily, year_if_known  # noqa: E402
from recognition.hard_families_v2 import (  # noqa: E402
    NAME_JACCARD_THRESHOLD,
    NAME_SEQUENCE_THRESHOLD,
    SMALL_MARGIN_THRESHOLD,
    HardFamilyV2,
    PairEvidence,
    _int_or_none,
    _pair_evidence,
    build_hard_families_v2,
)
from recognition.reporting import write_csv  # noqa: E402
from scripts.collect_web_extra import (  # noqa: E402
    _image_signature,
    _is_near_duplicate,
    _match_details,
)


STRONG_SIGNALS = (
    "real_confusion_rank_2_5",
    "small_top1_top2_margin",
    "image_similarity",
    "normalized_name_similarity",
)
FAMILY_FIELDS = [
    "family_id", "slug", "product_name", "winery", "category", "color", "region", "grape",
    "year_if_known", "selection_reason", "selection_signal_count", "family_selection_reason",
    "family_product_count", "family_edge_count", "family_confusion_count",
    "family_confusion_rank_2_5_count", "family_small_margin_count", "family_target_ranks",
    "family_min_margin", "family_max_margin", "family_visual_pairs", "family_name_pairs",
    "selection_evidence_json",
]
FAMILY_SUMMARY_FIELDS = [
    "family_id", "product_count", "slugs", "selection_reason", "edge_count",
    "confusion_count", "confusion_rank_2_5_count", "small_margin_count",
    "target_ranks", "min_margin", "max_margin", "visual_pair_count", "name_pair_count",
    "selection_evidence_json",
]
MANIFEST_FIELDS = [
    "query_id", "query_path", "target_slug", "family_id", "source_benchmark", "source_reference",
    "provenance", "selection_reason", "selection_evidence_json",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strict hard_near_duplicate_dev v2")
    parser.add_argument("--catalog-manifest", default="data/processed/catalog_manifest.csv")
    parser.add_argument("--synthetic-manifest", default="data/benchmarks/synthetic_dev/manifest.csv")
    parser.add_argument(
        "--predictions",
        default="artifacts/experiments/siglip2_synthetic_dev_20260916T062142Z/predictions.csv",
    )
    parser.add_argument("--v1-families", default="data/benchmarks/hard_near_duplicate_dev/families.csv")
    parser.add_argument("--v1-predictions", default="artifacts/experiments/siglip2_hard_near_duplicate_dev_20260916T134246Z/predictions.csv")
    parser.add_argument("--v1-metrics", default="artifacts/experiments/siglip2_hard_near_duplicate_dev_20260916T134246Z/metrics.json")
    parser.add_argument("--output-dir", default="data/benchmarks/hard_near_duplicate_dev_v2")
    parser.add_argument("--report", default="reports/hard_near_duplicate_dev_v2_report.md")
    parser.add_argument("--cases-html", default="reports/hard_near_duplicate_dev_v2_cases.html")
    args = parser.parse_args()

    catalog_rows = _read_csv(_path(args.catalog_manifest))
    synthetic_rows = _read_csv(_path(args.synthetic_manifest))
    prediction_rows = _read_csv(_path(args.predictions))
    v1_family_rows = _read_csv(_path(args.v1_families))
    v1_families = _families_from_csv(v1_family_rows)
    if not v1_families:
        raise ValueError("v1 family file is empty")

    catalog_by_slug = {row["slug"]: row for row in catalog_rows}
    products = {
        slug: row for slug, row in catalog_by_slug.items()
        if row.get("mapping_status") == "matched" and row.get("reference_image_path")
        and _path(row["reference_image_path"]).is_file()
    }
    required_slugs = {slug for family in v1_families for slug in family.slugs}
    signatures = _reference_signatures(
        {slug: products[slug] for slug in sorted(required_slugs) if slug in products}
    )
    visual_evidence = _visual_pair_evidence(v1_families, signatures)
    v2_families = build_hard_families_v2(
        catalog_rows, prediction_rows, visual_evidence, v1_families
    )
    if not v2_families:
        raise ValueError("Strict v2 selection produced no families")

    synthetic_by_target = _group_by(synthetic_rows, "target_slug")
    prediction_by_query = {row["query_id"]: row for row in prediction_rows}
    family_by_slug = {slug: family for family in v2_families for slug in family.slugs}
    family_rows, manifest_rows = _build_artifact_rows(
        v2_families, catalog_by_slug, synthetic_by_target
    )
    selected_ids = {row["query_id"] for row in manifest_rows}
    projected_predictions = [
        prediction_by_query[row["query_id"]]
        for row in manifest_rows
        if row["query_id"] in prediction_by_query
    ]
    if len(projected_predictions) != len(manifest_rows):
        missing = sorted(selected_ids - set(prediction_by_query))
        raise ValueError(f"Frozen predictions are missing v2 query ids: {missing[:5]}")

    output_dir = _path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "families.csv", FAMILY_FIELDS, family_rows)
    write_csv(output_dir / "family_summary.csv", FAMILY_SUMMARY_FIELDS, _family_summary_rows(v2_families))
    write_csv(output_dir / "manifest.csv", MANIFEST_FIELDS, manifest_rows)
    prediction_fields = list(prediction_rows[0])
    write_csv(output_dir / "predictions.csv", prediction_fields, projected_predictions)

    v2_metrics = _metrics(projected_predictions)
    v1_metrics = _read_json(_path(args.v1_metrics))
    v1_prediction_rows = _read_csv(_path(args.v1_predictions)) if _path(args.v1_predictions).is_file() else []
    audit = _audit_v1(v1_families, catalog_by_slug, prediction_rows, visual_evidence)
    metadata = _metadata(v1_families, v2_families, manifest_rows, v2_metrics, args)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(_benchmark_readme(metadata), encoding="utf-8")

    report = _render_report(
        catalog_by_slug=catalog_by_slug,
        v1_families=v1_families,
        v2_families=v2_families,
        manifest_rows=manifest_rows,
        prediction_rows=projected_predictions,
        v1_prediction_rows=v1_prediction_rows,
        v1_metrics=v1_metrics,
        v2_metrics=v2_metrics,
        audit=audit,
        report_path=_path(args.report),
        cases_html_path=_path(args.cases_html),
    )
    report_path = _path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    cases_path = _path(args.cases_html)
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    cases_path.write_text(
        _render_cases_html(catalog_by_slug, manifest_rows, projected_predictions, cases_path),
        encoding="utf-8",
    )

    print(json.dumps({
        "v1": {"families": len(v1_families), "products": len(required_slugs), "queries": 2 * len(required_slugs)},
        "v2": {
            "families": len(v2_families),
            "products": len(family_by_slug),
            "queries": len(manifest_rows),
            "catalog_share": len(family_by_slug) / len(products) if products else None,
            **v2_metrics,
        },
        "visual_pairs": sum(1 for value in visual_evidence.values() if value.get("image_similarity")),
        "report": str(report_path.relative_to(PROJECT_ROOT)),
        "cases_html": str(cases_path.relative_to(PROJECT_ROOT)),
    }, ensure_ascii=False, indent=2))


def _build_artifact_rows(
    families: list[HardFamilyV2],
    catalog_by_slug: Mapping[str, Mapping[str, str]],
    synthetic_by_target: Mapping[str, list[dict[str, str]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    family_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for family in families:
        incident: dict[str, list[PairEvidence]] = defaultdict(list)
        for edge in family.edges:
            incident[edge.left_slug].append(edge)
            incident[edge.right_slug].append(edge)
        family_evidence = [_evidence_dict(edge) for edge in family.edges]
        family_stats = _family_stats(family)
        for slug in family.slugs:
            product_edges = sorted(incident[slug], key=lambda edge: (edge.left_slug, edge.right_slug))
            product_signals = tuple(sorted({signal for edge in product_edges for signal in edge.signals}))
            product_evidence = [_evidence_dict(edge) for edge in product_edges]
            row = catalog_by_slug[slug]
            family_rows.append({
                "family_id": family.family_id,
                "slug": slug,
                "product_name": row.get("title", ""),
                "winery": row.get("winery", ""),
                "category": row.get("category", ""),
                "color": row.get("color", ""),
                "region": row.get("region", ""),
                "grape": row.get("grape", ""),
                "year_if_known": year_if_known(row),
                "selection_reason": "+".join(product_signals),
                "selection_signal_count": len(product_signals),
                "family_selection_reason": family.selection_reason,
                "family_product_count": len(family.slugs),
                **family_stats,
                "selection_evidence_json": json.dumps(product_evidence, ensure_ascii=False, sort_keys=True),
            })
            for synthetic in sorted(synthetic_by_target.get(slug, []), key=lambda value: value["query_id"]):
                manifest_rows.append({
                    "query_id": synthetic["query_id"],
                    "query_path": synthetic["query_path"],
                    "target_slug": slug,
                    "family_id": family.family_id,
                    "source_benchmark": "synthetic_dev",
                    "source_reference": row["reference_image_path"],
                    "provenance": "reused synthetic_dev query; frozen baseline prediction projection; no new image generation",
                    "selection_reason": "+".join(product_signals),
                    "selection_evidence_json": json.dumps(product_evidence, ensure_ascii=False, sort_keys=True),
                })
    manifest_rows.sort(key=lambda row: row["query_id"])
    return family_rows, manifest_rows


def _family_summary_rows(families: list[HardFamilyV2]) -> list[dict[str, object]]:
    rows = []
    for family in families:
        stats = _family_stats(family)
        rows.append({
            "family_id": family.family_id,
            "product_count": len(family.slugs),
            "slugs": json.dumps(list(family.slugs), ensure_ascii=False),
            "selection_reason": family.selection_reason,
            "edge_count": stats["family_edge_count"],
            "confusion_count": stats["family_confusion_count"],
            "confusion_rank_2_5_count": stats["family_confusion_rank_2_5_count"],
            "small_margin_count": stats["family_small_margin_count"],
            "target_ranks": stats["family_target_ranks"],
            "min_margin": stats["family_min_margin"],
            "max_margin": stats["family_max_margin"],
            "visual_pair_count": stats["family_visual_pairs"],
            "name_pair_count": stats["family_name_pairs"],
            "selection_evidence_json": json.dumps([_evidence_dict(edge) for edge in family.edges], ensure_ascii=False, sort_keys=True),
        })
    return rows


def _family_stats(family: HardFamilyV2) -> dict[str, object]:
    conf = sum(edge.confusion_count for edge in family.edges)
    rank25 = sum(edge.confusion_rank_2_5_count for edge in family.edges)
    margins = [edge.min_margin for edge in family.edges if edge.min_margin is not None]
    ranks = sorted({rank for edge in family.edges for rank in edge.target_ranks})
    return {
        "family_edge_count": len(family.edges),
        "family_confusion_count": conf,
        "family_confusion_rank_2_5_count": rank25,
        "family_small_margin_count": sum(edge.small_margin_count for edge in family.edges),
        "family_target_ranks": json.dumps(ranks, ensure_ascii=False),
        "family_min_margin": min(margins) if margins else "",
        "family_max_margin": max((edge.max_margin for edge in family.edges if edge.max_margin is not None), default=""),
        "family_visual_pairs": sum(edge.image_similarity for edge in family.edges),
        "family_name_pairs": sum(edge.name_similarity for edge in family.edges),
    }


def _evidence_dict(edge: PairEvidence) -> dict[str, object]:
    value = asdict(edge)
    value["selection_reason"] = edge.selection_reason
    return value


def _reference_signatures(products: Mapping[str, Mapping[str, str]]) -> dict[str, dict[str, object]]:
    signatures: dict[str, dict[str, object]] = {}
    for index, slug in enumerate(sorted(products), start=1):
        row = products[slug]
        path = _path(row["reference_image_path"])
        signatures[slug] = {"slug": slug, **_image_signature(path.read_bytes())}
        if index % 500 == 0 or index == len(products):
            print(f"reference signatures: {index}/{len(products)}", flush=True)
    return signatures


def _visual_pair_evidence(
    families: Iterable[HardFamily], signatures: Mapping[str, Mapping[str, object]]
) -> dict[tuple[str, str], dict[str, object]]:
    pairs = {
        tuple(sorted(pair))
        for family in families
        for pair in itertools.combinations(family.slugs, 2)
    }
    result: dict[tuple[str, str], dict[str, object]] = {}
    for left, right in sorted(pairs):
        if left not in signatures or right not in signatures:
            result[(left, right)] = {"image_similarity": False}
            continue
        details = _match_details(dict(signatures[left]), dict(signatures[right]))
        details["image_similarity"] = _is_near_duplicate(details)
        result[(left, right)] = details
    return result


def _families_from_csv(rows: list[dict[str, str]]) -> list[HardFamily]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["family_id"]].append(row)
    families = []
    for family_id in sorted(grouped):
        members = sorted(grouped[family_id], key=lambda row: row["slug"])
        reason = ";".join(sorted({row.get("family_reason", "") for row in members if row.get("family_reason")}))
        families.append(HardFamily(family_id, tuple(row["slug"] for row in members), reason, ()))
    return families


def _audit_v1(
    v1_families: list[HardFamily],
    catalog_by_slug: Mapping[str, Mapping[str, str]],
    prediction_rows: list[dict[str, str]],
    visual_evidence: Mapping[tuple[str, str], Mapping[str, object]],
) -> dict[str, object]:
    errors_by_pair: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in prediction_rows:
        target, predicted = row.get("target_slug", ""), row.get("predicted_slug", "")
        if target != predicted and target in catalog_by_slug and predicted in catalog_by_slug:
            errors_by_pair[tuple(sorted((target, predicted)))].append(row)
    pair_evidence: dict[tuple[str, str], PairEvidence] = {}
    for family in v1_families:
        for left, right in itertools.combinations(family.slugs, 2):
            pair = tuple(sorted((left, right)))
            left_row, right_row = catalog_by_slug[left], catalog_by_slug[right]
            pair_evidence[pair] = _pair_evidence(
                left, right, left_row, right_row, errors_by_pair.get(pair, []), visual_evidence.get(pair, {})
            )
    family_signal_sets: dict[str, set[str]] = {}
    product_signal_sets: dict[str, set[str]] = defaultdict(set)
    family_basis: dict[str, str] = {}
    for family in v1_families:
        edges = [pair_evidence[pair] for pair in pair_evidence if pair[0] in family.slugs and pair[1] in family.slugs]
        signals = {signal for edge in edges for signal in edge.signals}
        family_signal_sets[family.family_id] = signals
        for edge in edges:
            product_signal_sets[edge.left_slug].update(edge.signals)
            product_signal_sets[edge.right_slug].update(edge.signals)
        family_basis[family.family_id] = (
            "real_confusion" if "observed_confusion_count=" in family.reason else "metadata_only"
        )
    v1_products = {slug for family in v1_families for slug in family.slugs}
    all_confusion_rows = sum(len(rows) for rows in errors_by_pair.values())
    rank25_confusion_rows = sum(
        1 for rows in errors_by_pair.values() for row in rows
        if (rank := _int_or_none(row.get("target_rank", ""))) is not None and 2 <= rank <= 5
    )
    not_found_confusion_rows = sum(
        1 for rows in errors_by_pair.values() for row in rows
        if _int_or_none(row.get("target_rank", "")) is None
    )
    basis_counts = {
        basis: {
            "families": sum(value == basis for value in family_basis.values()),
            "products": len({slug for family in v1_families if family_basis[family.family_id] == basis for slug in family.slugs}),
        }
        for basis in ("metadata_only", "real_confusion")
    }
    signal_counts = {}
    for signal in STRONG_SIGNALS:
        signal_counts[signal] = {
            "families": sum(signal in signals for signals in family_signal_sets.values()),
            "products": sum(signal in product_signal_sets[slug] for slug in v1_products),
            "families_only": sum(signals == {signal} for signals in family_signal_sets.values()),
            "products_only": sum(product_signal_sets[slug] == {signal} for slug in v1_products),
        }
    multi = {
        "families": sum(len(signals) >= 2 for signals in family_signal_sets.values()),
        "products": sum(len(product_signal_sets[slug]) >= 2 for slug in v1_products),
    }
    combinations = Counter(
        "+".join(sorted(signals)) or "none" for signals in family_signal_sets.values()
    )
    return {
        "basis_counts": basis_counts,
        "signal_counts": signal_counts,
        "multi_signal": multi,
        "family_signal_combinations": dict(combinations.most_common()),
        "v1_pair_count": len(pair_evidence),
        "v1_strong_pairs": sum(len(edge.signals) >= 2 for edge in pair_evidence.values()),
        "all_confusion_rows": all_confusion_rows,
        "rank25_confusion_rows": rank25_confusion_rows,
        "not_found_confusion_rows": not_found_confusion_rows,
        "v1_products": len(v1_products),
        "v1_families": len(v1_families),
    }


def _metrics(rows: list[dict[str, str]]) -> dict[str, object]:
    records = []
    for row in rows:
        records.append(RankingRecord(
            query_id=row["query_id"],
            target_slug=row["target_slug"],
            ranked_candidates=json.loads(row["top5_slugs"]),
            scores=[float(value) for value in json.loads(row.get("top5_scores", "[]"))],
            latency_ms=float(row.get("latency_ms", 0.0)),
            transform_type=row.get("transform_type", ""),
        ))
    return evaluate_rankings(records)


def _metadata(v1_families, v2_families, manifest_rows, v2_metrics, args) -> dict[str, object]:
    products = len({slug for family in v2_families for slug in family.slugs})
    catalog_products = sum(
        1 for row in _read_csv(_path(args.catalog_manifest))
        if row.get("mapping_status") == "matched" and row.get("reference_image_path")
        and _path(row["reference_image_path"]).is_file()
    )
    return {
        "benchmark_version": "hard-near-duplicate-v2",
        "candidate_universe": "existing hard_near_duplicate_dev v1 families",
        "selection_rule": "at least 2 of 4 strong signals; metadata-only relation is insufficient",
        "strong_signals": {
            "real_confusion_rank_2_5": "frozen SigLIP2 Top-1 error with target rank 2..5",
            "small_top1_top2_margin": SMALL_MARGIN_THRESHOLD,
            "image_similarity": "deterministic perceptual/pixel fingerprint used by web-extra duplicate guard",
            "normalized_name_similarity": {
                "token_jaccard": NAME_JACCARD_THRESHOLD,
                "sequence_ratio": NAME_SEQUENCE_THRESHOLD,
            },
        },
        "families": len(v2_families),
        "products": products,
        "queries": len(manifest_rows),
        "catalog_products_with_reference": catalog_products,
        "catalog_share": products / catalog_products if catalog_products else None,
        "metrics": v2_metrics,
        "prediction_source": str(_path(args.predictions).relative_to(PROJECT_ROOT)),
        "query_source": "synthetic_dev manifest; query ids and bytes are reused exactly",
        "model_changed": False,
        "synthetic_dev_changed": False,
    }


def _benchmark_readme(metadata: Mapping[str, object]) -> str:
    return f"""# hard_near_duplicate_dev v2

Strict diagnostic benchmark derived from the preserved v1 family candidate
universe. A pair is retained only when it has at least two of the four strong
signals: real frozen SigLIP2 confusion with target rank 2--5, Top-1/Top-2
margin <= {SMALL_MARGIN_THRESHOLD}, deterministic reference-image similarity,
or strong normalized product-name similarity.

The `manifest.csv` query ids and query files are reused from `synthetic_dev`
exactly. `predictions.csv` is a projection of the frozen synthetic SigLIP2
predictions, so this artifact does not change the model or synthetic_dev.
Every family/product row carries `selection_reason` and JSON numerical evidence.
"""


def _render_report(
    *, catalog_by_slug, v1_families, v2_families, manifest_rows, prediction_rows,
    v1_prediction_rows, v1_metrics, v2_metrics, audit, report_path, cases_html_path,
) -> str:
    v1_products = len({slug for family in v1_families for slug in family.slugs})
    v2_products = len({slug for family in v2_families for slug in family.slugs})
    catalog_products = sum(1 for row in catalog_by_slug.values() if row.get("mapping_status") == "matched" and row.get("reference_image_path"))
    v2_errors = [row for row in prediction_rows if row.get("target_slug") != row.get("predicted_slug")]
    slug_family = {slug: family.family_id for family in v2_families for slug in family.slugs}
    same_family_errors = sum(
        row.get("target_slug") != row.get("predicted_slug")
        and slug_family.get(row.get("target_slug", "")) == slug_family.get(row.get("predicted_slug", ""))
        for row in prediction_rows
    )
    v1_same_family = _same_family_error_count(v1_prediction_rows, v1_families)
    rank_distribution = Counter(row.get("target_rank") or "not-found" for row in prediction_rows)
    v2_margin = _margin_summary(prediction_rows)
    correct_margins = _margin_values(prediction_rows, correct=True)
    error_margins = _margin_values(prediction_rows, correct=False)
    confusion_pairs = Counter((row.get("target_slug", ""), row.get("predicted_slug", "")) for row in v2_errors)
    hardest = _hardest_rows(prediction_rows, catalog_by_slug)
    family_difficulty = _family_difficulty(v2_families, manifest_rows, prediction_rows)
    cases_link = os.path.relpath(cases_html_path.resolve(), report_path.parent.resolve()).replace(os.sep, "/")
    lines = [
        "# hard_near_duplicate_dev v2 audit",
        "",
        "Строгая v2 построена как воспроизводимое уточнение сохранённого v1 candidate universe. `synthetic_dev`, модель SigLIP2 и v1 не изменялись; query ids/bytes в v2 переиспользованы из synthetic_dev, а predictions.csv — точная проекция замороженного synthetic baseline.",
        "",
        "## 1. Summary and v1 → v2",
        "",
        "| benchmark | families | products | queries | catalog share | Top-1 | Recall@5 | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| v1 | {len(v1_families)} | {v1_products} | {2*v1_products} | {_pct(v1_products, catalog_products)} | {_fmt_metric(v1_metrics.get('top1_accuracy'))} | {_fmt_metric(v1_metrics.get('recall_at_5'))} | {_fmt_metric(v1_metrics.get('mrr'))} |",
        f"| v2 | {len(v2_families)} | {v2_products} | {len(manifest_rows)} | {_pct(v2_products, catalog_products)} | {_fmt_metric(v2_metrics.get('top1_accuracy'))} | {_fmt_metric(v2_metrics.get('recall_at_5'))} | {_fmt_metric(v2_metrics.get('mrr'))} |",
        "",
        f"Визуальный разбор 20 самых сложных query: [{cases_html_path.name}]({cases_link}). В карточках показаны query, reference target и Top-5 reference images.",
        "",
        "## 2. Почему v1 получился большим",
        "",
        f"В v1 было **{audit['v1_families']} families / {audit['v1_products']} products** и **{audit['v1_pair_count']}** пар внутри уже отобранных семейств. Основное раздувание дала широкая metadata-ветка: одинаковая winery плюс слабое пересечение имени и один/несколько wine attributes. В результате **{audit['basis_counts']['metadata_only']['families']} families / {audit['basis_counts']['metadata_only']['products']} products** попали без observed SigLIP confusion; только **{audit['basis_counts']['real_confusion']['families']} / {audit['basis_counts']['real_confusion']['products']}** имели model-confusion evidence в v1 reason.",
        "",
        f"В frozen synthetic baseline было {audit['all_confusion_rows']} Top-1 confusion rows: {audit['rank25_confusion_rows']} с target rank 2–5 и {audit['not_found_confusion_rows']} с target вне Top-5. Поэтому часть v1 model evidence не соответствовала строгому strong-signal A.",
        "",
        "Дополнительные слабости v1: image similarity не была критерием отбора; confusion допускался даже при target rank вне 2–5; greedy clique добавлял collateral products, которым не обязательно соответствовала сильная прямая связь; отдельные metadata-признаки могли поддерживать широкие producer-группы.",
        "",
        "### Retrospective signal breakdown inside v1",
        "",
        "Ниже — аудит всех v1 products/families по сильным сигналам v2. `at least` означает наличие сигнала хотя бы на одном прямом pair edge; `only` — единственный сигнал на уровне family/product. Это retrospective evidence, а не утверждение, что v1 использовал image/name/margin как критерии.",
        "",
        "| signal | families at least | products at least | families only | products only |",
        "|---|---:|---:|---:|---:|",
    ]
    for signal in STRONG_SIGNALS:
        values = audit["signal_counts"][signal]
        lines.append(f"| `{signal}` | {values['families']} | {values['products']} | {values['families_only']} | {values['products_only']} |")
    lines.extend([
        f"| **several strong signals (≥2)** | **{audit['multi_signal']['families']}** | **{audit['multi_signal']['products']}** | — | — |",
        "",
        "Важная интерпретация: `image_similarity` и `small_top1_top2_margin` в v1 не были самостоятельными правилами, поэтому их retrospective-only counts не являются причиной первоначального включения. В частности, v1 image-only selection = **0** по своей реализации; v2 требует комбинацию evidence.",
        "",
        "## 3. Ужесточенные критерии v2",
        "",
        f"V2 рассматривает только пары из сохранённых v1 families и оставляет pair, если присутствуют минимум **2 из 4** сигналов:",
        "",
        f"- `real_confusion_rank_2_5`: замороженная ошибка SigLIP2, target rank 2–5;",
        f"- `small_top1_top2_margin`: margin ≤ `{SMALL_MARGIN_THRESHOLD}`;",
        "- `image_similarity`: deterministic perceptual/pixel fingerprint reference images с текущими conservative guards;",
        f"- `normalized_name_similarity`: token Jaccard ≥ `{NAME_JACCARD_THRESHOLD}` или sequence ratio ≥ `{NAME_SEQUENCE_THRESHOLD}`.",
        "",
        "Winery/category/region/grape не являются selection signals и сами по себе family не создают. После фильтра пары образуют connected components; collateral member без qualifying edge больше не добавляется.",
        "",
        f"Результат: **{len(v2_families)} families / {v2_products} products / {len(manifest_rows)} queries**, то есть {_pct(v2_products, catalog_products)} каталога с reference image.",
        "",
        "### V2 selection reasons",
        "",
        "| reason combination | families | products |",
        "|---|---:|---:|",
    ])
    v2_reason_counts = _reason_counts(v2_families)
    for reason, counts in v2_reason_counts:
        lines.append(f"| `{reason}` | {counts['families']} | {counts['products']} |")
    lines.extend([
        "",
        "Каждая строка `families.csv` содержит product-level `selection_reason`, family-level reason и `selection_evidence_json` с similarity, margins, confusion counts и target ranks.",
        "",
        "## 4. Baseline overlap и метрики",
        "",
        f"- V2 Top-1 errors: **{len(v2_errors)} / {len(prediction_rows)}**; overlap with errors where predicted slug belongs to same v2 family: **{same_family_errors}**.",
        f"- Для сравнения v1 same-family error count: **{v1_same_family}** (по сохранённому hard v1 prediction artifact).",
        f"- Mean/p50/p95 latency в v2 не переоцениваются: это projection существующих predictions; исходный SigLIP2 latency остаётся тем же, что в synthetic baseline.",
        "",
        "| metric | v1 | v2 |",
        "|---|---:|---:|",
        f"| Top-1 | {_fmt_metric(v1_metrics.get('top1_accuracy'))} | {_fmt_metric(v2_metrics.get('top1_accuracy'))} |",
        f"| Recall@5 | {_fmt_metric(v1_metrics.get('recall_at_5'))} | {_fmt_metric(v2_metrics.get('recall_at_5'))} |",
        f"| MRR | {_fmt_metric(v1_metrics.get('mrr'))} | {_fmt_metric(v2_metrics.get('mrr'))} |",
        "",
        "### Target rank distribution (v2)",
        "",
        "| target rank | queries | share |",
        "|---:|---:|---:|",
    ])
    for rank, count in sorted(rank_distribution.items(), key=lambda item: _rank_sort_key(item[0])):
        lines.append(f"| {rank} | {count} | {_pct(count, len(prediction_rows))} |")
    lines.extend([
        "",
        "### Top1–Top2 margin distributions",
        "",
        "| subset | count | min | p10 | median | p90 | max |",
        "|---|---:|---:|---:|---:|---:|---:|",
        _margin_row("correct Top-1", correct_margins),
        _margin_row("Top-1 errors", error_margins),
        "",
        f"All v2 margins: mean={_fmt_num(v2_margin.get('mean'))}, min={_fmt_num(v2_margin.get('min'))}, max={_fmt_num(v2_margin.get('max'))}.",
        "",
        "### Most frequent confusion pairs (v2)",
        "",
        "| count | target | predicted |",
        "|---:|---|---|",
    ])
    for (target, predicted), count in confusion_pairs.most_common(15):
        lines.append(f"| {count} | `{target}` | `{predicted}` |")
    if not confusion_pairs:
        lines.append("| 0 | — | — |")
    lines.extend([
        "",
        "## 5. Самые сложные families",
        "",
        "Сортировка: сначала Top-1 error count, затем error rate и размер family. Это диагностический порядок, не новая selection rule.",
        "",
        "| family | products | queries | Top-1 | errors | reason |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for item in family_difficulty[:8]:
        lines.append(f"| `{item['family_id']}` | {item['products']} | {item['queries']} | {_fmt_metric(item['top1'])} | {item['errors']} | `{item['reason']}` |")
    lines.extend([
        "",
        "### 20 hardest cases",
        "",
        "Визуальные пары query/reference и Top-5 находятся в отдельном HTML, чтобы не встраивать изображения в Markdown и не менять data artifacts.",
        "",
        "| # | query | target | rank | margin | Top-5 predictions |",
        "|---:|---|---|---:|---:|---|",
    ])
    for index, row in enumerate(hardest[:20], start=1):
        top5 = ", ".join(json.loads(row.get("top5_slugs", "[]")))
        lines.append(f"| {index} | `{row['query_id']}` | `{row['target_slug']}` | {row.get('target_rank') or 'not-found'} | {_fmt_num(row.get('top1_top2_margin'))} | `{top5}` |")
    lines.extend([
        "",
        "## 6. Открытые ограничения",
        "",
        "- V2 queries происходят из synthetic_dev и потому не являются независимым real-world test set.",
        "- Image similarity здесь — детерминированный fingerprint/dedup-style сигнал, не CV/ML recognition.",
        "- V2 не доказывает, что две бутылки genuinely near-duplicate без визуальной проверки; он формирует более узкий reproducible review set.",
        "- Baseline errors не являются labels для обучения и не должны использоваться для tuning на том же benchmark без отдельного split.",
        "",
        "## 7. Что делать дальше",
        "",
        "1. Использовать v2 как диагностический hard subset и отдельно смотреть 20 HTML cases.",
        "2. Зафиксировать эту selection policy и не смешивать v1/v2/synthetic_dev в одну headline metric.",
        "3. Для следующего эксперимента держать одинаковый query split, модель и evaluator; изменения писать в experiment history.",
        "",
        "## Files",
        "",
        f"- Benchmark: `data/benchmarks/hard_near_duplicate_dev_v2/` (`family_summary.csv` is one row per family; `families.csv` is one row per product).",
        f"- Visual cases: [{cases_html_path.name}]({cases_link}).",
        "- Preserved v1: `data/benchmarks/hard_near_duplicate_dev/`.",
        "",
    ])
    return "\n".join(lines)


def _render_cases_html(catalog_by_slug, manifest_rows, prediction_rows, output_path: Path) -> str:
    manifest_by_id = {row["query_id"]: row for row in manifest_rows}
    rows = _hardest_rows(prediction_rows, catalog_by_slug)[:20]
    cards = []
    for index, row in enumerate(rows, start=1):
        manifest = manifest_by_id[row["query_id"]]
        query_src = _image_src(manifest["query_path"], output_path)
        target = row["target_slug"]
        target_src = _image_src(catalog_by_slug.get(target, {}).get("reference_image_path", ""), output_path)
        candidate_slugs = json.loads(row.get("top5_slugs", "[]"))
        candidate_scores = json.loads(row.get("top5_scores", "[]"))
        images = [
            _figure(_image_src(catalog_by_slug.get(slug, {}).get("reference_image_path", ""), output_path), f"#{rank} {slug}", candidate_scores[rank - 1] if rank - 1 < len(candidate_scores) else "")
            for rank, slug in enumerate(candidate_slugs, start=1)
        ]
        label = _label(target, catalog_by_slug.get(target, {}))
        cards.append(
            "<article class='case'>"
            f"<h2>#{index} {escape(row['query_id'])}</h2>"
            f"<p><b>family:</b> {escape(manifest.get('family_id', ''))} · <b>target rank:</b> {escape(row.get('target_rank') or 'not-found')} · <b>margin:</b> {escape(str(row.get('top1_top2_margin', '')))}</p>"
            f"<p><b>target:</b> {escape(label)} <code>{escape(target)}</code></p>"
            "<div class='row'>"
            f"{_figure(query_src, 'query', '')}{_figure(target_src, 'correct target reference', '')}"
            "</div><h3>Top-5 reference images</h3><div class='row'>"
            + "".join(images)
            + "</div></article>"
        )
    return """<!doctype html>
<html lang='ru'><head><meta charset='utf-8'><title>hard_near_duplicate_dev v2 — 20 hardest cases</title>
<style>
body{font-family:system-ui,sans-serif;margin:24px;background:#f4f1ec;color:#25211e}.case{background:#fff;margin:0 0 24px;padding:18px;border-radius:12px;box-shadow:0 1px 5px #c9c0b7}.row{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start}figure{margin:0;width:170px}figcaption{font-size:12px;line-height:1.35;min-height:45px;overflow-wrap:anywhere}img{display:block;width:170px;height:260px;object-fit:contain;background:#eee;border:1px solid #ddd}h1{margin-bottom:8px}h2{margin:0 0 6px;font-size:18px}h3{font-size:15px;margin:18px 0 8px}code{font-size:11px;overflow-wrap:anywhere}
</style></head><body><h1>hard_near_duplicate_dev v2</h1>
<p>20 hardest frozen synthetic-dev queries. Each card shows query, correct target reference, and the five references used by the frozen SigLIP2 ranking.</p>
""" + "\n".join(cards) + "</body></html>\n"


def _hardest_rows(rows: list[dict[str, str]], catalog_by_slug) -> list[dict[str, str]]:
    def key(row):
        rank = target_rank(row["target_slug"], json.loads(row["top5_slugs"]))
        error = row.get("predicted_slug") != row.get("target_slug")
        try:
            margin = float(row.get("top1_top2_margin", "inf"))
        except ValueError:
            margin = float("inf")
        return (0 if error else 1, -(rank or 6), margin, row["query_id"])
    return sorted(rows, key=key)


def _family_difficulty(families, manifest_rows, prediction_rows):
    family_by_query = {row["query_id"]: row["family_id"] for row in manifest_rows}
    grouped = defaultdict(list)
    for row in prediction_rows:
        grouped[family_by_query[row["query_id"]]].append(row)
    result = []
    for family in families:
        rows = grouped[family.family_id]
        errors = sum(row.get("predicted_slug") != row.get("target_slug") for row in rows)
        result.append({
            "family_id": family.family_id,
            "products": len(family.slugs),
            "queries": len(rows),
            "errors": errors,
            "top1": 1 - errors / len(rows) if rows else None,
            "reason": family.selection_reason,
        })
    return sorted(result, key=lambda item: (-item["errors"], item["top1"], -item["products"], item["family_id"]))


def _reason_counts(families):
    counts = defaultdict(lambda: {"families": 0, "products": 0})
    for family in families:
        key = family.selection_reason or "(none)"
        counts[key]["families"] += 1
        counts[key]["products"] += len(family.slugs)
    return sorted(counts.items(), key=lambda item: (-item[1]["products"], item[0]))


def _same_family_error_count(rows, families):
    family_by_slug = {slug: family.family_id for family in families for slug in family.slugs}
    return sum(
        row.get("target_slug") != row.get("predicted_slug")
        and family_by_slug.get(row.get("target_slug")) == family_by_slug.get(row.get("predicted_slug"))
        for row in rows
    )


def _margin_values(rows, *, correct: bool) -> list[float]:
    values = []
    for row in rows:
        is_correct = row.get("predicted_slug") == row.get("target_slug")
        if is_correct != correct:
            continue
        try:
            values.append(float(row["top1_top2_margin"]))
        except (KeyError, TypeError, ValueError):
            pass
    return values


def _margin_summary(rows):
    values = _margin_values(rows, correct=True) + _margin_values(rows, correct=False)
    return {"mean": statistics.mean(values) if values else None, "min": min(values) if values else None, "max": max(values) if values else None}


def _margin_row(label: str, values: list[float]) -> str:
    if not values:
        return f"| {label} | 0 | — | — | — | — | — |"
    ordered = sorted(values)
    return "| " + " | ".join([
        label, str(len(values)), _fmt_num(min(values)), _fmt_num(_percentile(ordered, 10)),
        _fmt_num(statistics.median(values)), _fmt_num(_percentile(ordered, 90)), _fmt_num(max(values)),
    ]) + " |"


def _percentile(values: list[float], percentage: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentage / 100
    lower, upper = int(position), min(int(position) + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _image_src(value: str, output_path: Path) -> str | None:
    if not value:
        return None
    path = _path(value).resolve()
    if not path.is_file():
        return None
    return os.path.relpath(path, output_path.parent.resolve()).replace(os.sep, "/")


def _figure(src: str | None, caption: str, score: object) -> str:
    score_text = f"<br>score={escape(str(score))}" if score != "" else ""
    image = f"<img src='{escape(src)}' alt='{escape(caption)}'>" if src else "<div>image unavailable</div>"
    return f"<figure><figcaption>{escape(caption)}{score_text}</figcaption>{image}</figure>"


def _label(slug: str, row: Mapping[str, str]) -> str:
    title = row.get("title") or slug
    winery = row.get("winery") or ""
    category = row.get("category") or ""
    details = " · ".join(value for value in (winery, category) if value)
    return f"{title} ({details})" if details else title


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _group_by(rows: Iterable[dict[str, str]], field: str) -> dict[str, list[dict[str, str]]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[field]].append(row)
    return grouped


def _rank_sort_key(value: str):
    return (99, "") if value == "not-found" else (int(value), "")


def _evidence_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _fmt_metric(value) -> str:
    return "—" if value is None else f"{float(value) * 100:.2f}%"


def _fmt_num(value) -> str:
    if value in (None, ""):
        return "—"
    try:
        return f"{float(value):.5f}"
    except (TypeError, ValueError):
        return str(value)


def _pct(value, total) -> str:
    return "—" if not total else f"{value / total * 100:.2f}%"


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
