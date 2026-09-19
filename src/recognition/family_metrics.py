"""Family-aware diagnostics for benchmarks with explicit target families."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence


def family_diagnostics(
    prediction_rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Calculate family-level retrieval metrics without changing exact metrics.

    Rows whose target has no ``target_family_id``/``family_id`` are treated as
    representative products and excluded from the family denominator.  This
    lets a mixed representative+hard pilot keep one manifest while reporting
    the hard-family diagnostics separately.
    """

    product_meta = _family_product_metadata(manifest_rows)
    query_meta = {
        str(row["query_id"]): product_meta.get(str(row.get("target_slug", "")))
        for row in manifest_rows
    }
    hard_rows: list[tuple[Mapping[str, Any], dict[str, str]]] = []
    for row in prediction_rows:
        meta = query_meta.get(str(row.get("query_id", "")))
        if meta and meta["family_id"]:
            hard_rows.append((row, meta))

    aggregate = _aggregate(hard_rows, product_meta)
    by_family: dict[str, list[tuple[Mapping[str, Any], dict[str, str]]]] = defaultdict(list)
    by_type: dict[str, list[tuple[Mapping[str, Any], dict[str, str]]]] = defaultdict(list)
    for row, meta in hard_rows:
        by_family[meta["family_id"]].append((row, meta))
        by_type[meta["family_type"] or "unknown"].append((row, meta))

    return {
        "hard_query_count": aggregate["query_count"],
        "hard_product_count": len(product_meta),
        "hard_family_count": len({meta["family_id"] for meta in product_meta.values()}),
        "family_top1": aggregate["family_top1"],
        "family_recall_at_5": aggregate["family_recall_at_5"],
        "within_family_disambiguation_top1": aggregate["within_family_disambiguation_top1"],
        "accuracy_per_family": {
            family_id: _family_result(rows, product_meta)
            for family_id, rows in sorted(by_family.items())
        },
        "confusion_within_family": [
            {
                "target_slug": target,
                "predicted_slug": predicted,
                "count": count,
            }
            for (target, predicted), count in sorted(
                _confusions(hard_rows, product_meta).items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
        "family_type_breakdown": {
            family_type: _type_result(rows, product_meta)
            for family_type, rows in sorted(by_type.items())
        },
    }


def _family_product_metadata(
    manifest_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in manifest_rows:
        slug = str(row.get("target_slug", "")).strip()
        family_id = str(row.get("target_family_id", row.get("family_id", ""))).strip()
        if not slug or not family_id:
            continue
        metadata = {
            "target_slug": slug,
            "family_id": family_id,
            "family_type": str(row.get("family_type", "")).strip(),
        }
        previous = result.get(slug)
        if previous and (previous["family_id"], previous["family_type"]) != (
            metadata["family_id"], metadata["family_type"]
        ):
            raise ValueError(f"Product is assigned to multiple target families: {slug}")
        result[slug] = metadata
    return result


def _aggregate(
    rows: Sequence[tuple[Mapping[str, Any], dict[str, str]]],
    product_meta: Mapping[str, Mapping[str, str]],
) -> dict[str, object]:
    if not rows:
        return {
            "query_count": 0,
            "family_top1": None,
            "family_recall_at_5": None,
            "within_family_disambiguation_top1": None,
        }
    family_top1_hits = 0
    family_recall_hits = 0
    exact_top1_hits = 0
    for row, meta in rows:
        candidates = _candidates(row)
        candidate_meta = [product_meta.get(candidate, {}).get("family_id", "") for candidate in candidates]
        family_top1 = bool(candidate_meta and candidate_meta[0] == meta["family_id"])
        family_recall = any(value == meta["family_id"] for value in candidate_meta)
        exact_top1 = bool(candidates and candidates[0] == meta["target_slug"])
        family_top1_hits += family_top1
        family_recall_hits += family_recall
        exact_top1_hits += exact_top1
    return {
        "query_count": len(rows),
        "family_top1": family_top1_hits / len(rows),
        "family_recall_at_5": family_recall_hits / len(rows),
        "within_family_disambiguation_top1": (
            exact_top1_hits / family_top1_hits if family_top1_hits else None
        ),
    }


def _family_result(
    rows: Sequence[tuple[Mapping[str, Any], dict[str, str]]],
    product_meta: Mapping[str, Mapping[str, str]],
) -> dict[str, object]:
    aggregate = _aggregate(rows, product_meta)
    family_id = rows[0][1]["family_id"] if rows else ""
    family_type = rows[0][1]["family_type"] if rows else ""
    return {
        "family_id": family_id,
        "family_type": family_type,
        "query_count": aggregate["query_count"],
        "product_slugs": sorted({meta["target_slug"] for _, meta in rows}),
        "top1_accuracy": _exact_top1(rows),
        "family_top1": aggregate["family_top1"],
        "family_recall_at_5": aggregate["family_recall_at_5"],
        "within_family_disambiguation_top1": aggregate["within_family_disambiguation_top1"],
    }


def _type_result(
    rows: Sequence[tuple[Mapping[str, Any], dict[str, str]]],
    product_meta: Mapping[str, Mapping[str, str]],
) -> dict[str, object]:
    aggregate = _aggregate(rows, product_meta)
    return {
        "family_type": rows[0][1]["family_type"] if rows else "",
        "query_count": aggregate["query_count"],
        "family_count": len({meta["family_id"] for _, meta in rows}),
        "product_slugs": sorted({meta["target_slug"] for _, meta in rows}),
        "top1_accuracy": _exact_top1(rows),
        "family_top1": aggregate["family_top1"],
        "family_recall_at_5": aggregate["family_recall_at_5"],
        "within_family_disambiguation_top1": aggregate["within_family_disambiguation_top1"],
    }


def _exact_top1(rows: Sequence[tuple[Mapping[str, Any], dict[str, str]]]) -> float | None:
    if not rows:
        return None
    return sum(bool((candidates := _candidates(row)) and candidates[0] == meta["target_slug"]) for row, meta in rows) / len(rows)


def _confusions(
    rows: Sequence[tuple[Mapping[str, Any], dict[str, str]]],
    product_meta: Mapping[str, Mapping[str, str]],
) -> Counter[tuple[str, str]]:
    result: Counter[tuple[str, str]] = Counter()
    for row, meta in rows:
        candidates = _candidates(row)
        if not candidates or candidates[0] == meta["target_slug"]:
            continue
        candidate_family = product_meta.get(candidates[0], {}).get("family_id", "")
        if candidate_family == meta["family_id"]:
            result[(meta["target_slug"], candidates[0])] += 1
    return result


def _candidates(row: Mapping[str, Any]) -> list[str]:
    value = row.get("top5_slugs", row.get("ranked_candidates", []))
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid top5_slugs JSON for {row.get('query_id')}: {exc}") from exc
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"top5_slugs must be a sequence for {row.get('query_id')}")
    return [str(candidate) for candidate in value]
