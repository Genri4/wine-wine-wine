"""Deterministic construction of fine-grained hard product families."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import itertools
import re
from typing import Mapping, Sequence


YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
GENERIC_TOKENS = {
    "wine", "vino", "вино", "красное", "белое", "розовое", "оранжевое",
    "сухое", "полусухое", "полусладкое", "сладкое", "брют", "экстра",
    "reserve", "резерв", "резервное", "collection", "коллекция", "2020",
    "2021", "2022", "2023", "2024", "2025", "2026",
}


@dataclass(frozen=True)
class HardFamily:
    family_id: str
    slugs: tuple[str, ...]
    reason: str
    model_confusion_pairs: tuple[tuple[str, str, int], ...]


def build_hard_families(
    catalog_rows: Sequence[Mapping[str, str]],
    prediction_rows: Sequence[Mapping[str, str]],
) -> list[HardFamily]:
    """Build families from metadata edges supported by observed baseline errors.

    A model confusion is included only when there is also a conservative
    metadata or naming relation. Metadata-only edges require the same winery
    and meaningful product-name overlap plus a matching wine attribute. This
    avoids turning every large winery into one arbitrary family.
    """

    products = {
        row["slug"]: row
        for row in catalog_rows
        if row.get("mapping_status") == "matched" and row.get("reference_image_path")
    }
    parent = {slug: slug for slug in products}
    evidence: dict[tuple[str, str], set[str]] = defaultdict(set)
    pair_counts: Counter[tuple[str, str]] = Counter()

    def add_edge(left: str, right: str, reason: str) -> None:
        if left == right or left not in products or right not in products:
            return
        a, b = sorted((left, right))
        evidence[(a, b)].add(reason)
        _union(parent, a, b)

    for row in prediction_rows:
        target = row.get("target_slug", "")
        predicted = row.get("predicted_slug", "")
        if target == predicted or target not in products or predicted not in products:
            continue
        target_row, predicted_row = products[target], products[predicted]
        pair = tuple(sorted((target, predicted)))
        pair_counts[pair] += 1
        if _model_supported_relation(target_row, predicted_row):
            rank = row.get("target_rank") or "not-in-top5"
            margin = row.get("top1_top2_margin") or ""
            add_edge(target, predicted, f"observed_confusion_count={pair_counts[pair]};target_rank={rank};margin={margin}")

    by_winery: dict[str, list[str]] = defaultdict(list)
    for slug, row in products.items():
        winery = _norm(row.get("winery", ""))
        if winery:
            by_winery[winery].append(slug)
    for winery_slugs in by_winery.values():
        for left, right in itertools.combinations(sorted(winery_slugs), 2):
            if _metadata_near_duplicate(products[left], products[right]):
                add_edge(left, right, "same_winery;shared_name_and_wine_attributes")

    # Do not use connected components directly: one long chain of weak
    # relations would turn an entire large producer into one family. Instead,
    # greedily select disjoint, fully-supported cliques from strongest model
    # confusion evidence. This keeps each diagnostic family interpretable.
    available = set(products)
    raw_families: list[tuple[str, ...]] = []
    edge_order = sorted(
        evidence,
        key=lambda pair: (-pair_counts[pair], -len(evidence[pair]), pair[0], pair[1]),
    )
    for left, right in edge_order:
        if left not in available or right not in available:
            continue
        members = [left, right]
        for candidate in sorted(available - set(members)):
            if len(members) >= 12:
                break
            if all(tuple(sorted((candidate, member))) in evidence for member in members):
                members.append(candidate)
        raw_families.append(tuple(sorted(members)))
        available.difference_update(members)

    families: list[HardFamily] = []
    for number, slugs in enumerate(raw_families, start=1):
        family_id = f"family-{number:03d}"
        component_pairs = []
        reasons: set[str] = set()
        for pair, labels in evidence.items():
            if pair[0] in slugs and pair[1] in slugs:
                reasons.update(labels)
                component_pairs.append((pair[0], pair[1], pair_counts[pair]))
        component_pairs.sort(key=lambda value: (-value[2], value[0], value[1]))
        families.append(
            HardFamily(
                family_id=family_id,
                slugs=slugs,
                reason=";".join(sorted(reasons)),
                model_confusion_pairs=tuple(component_pairs),
            )
        )
    return families


def year_if_known(row: Mapping[str, str]) -> str:
    """Extract a year only from a unique explicit metadata/name year token."""

    values = set(YEAR_RE.findall(f"{row.get('title', '')} {row.get('slug', '')}"))
    return next(iter(values)) if len(values) == 1 else ""


def _metadata_near_duplicate(left: Mapping[str, str], right: Mapping[str, str]) -> bool:
    same_winery = _norm(left.get("winery", "")) == _norm(right.get("winery", ""))
    overlap = _jaccard(_meaningful_tokens(left), _meaningful_tokens(right))
    same_category = _norm(left.get("category", "")) == _norm(right.get("category", ""))
    same_color = _norm(left.get("color", "")) == _norm(right.get("color", ""))
    same_region = _norm(left.get("region", "")) == _norm(right.get("region", ""))
    grape_overlap = bool(_tokens(left.get("grape", "")) & _tokens(right.get("grape", "")))
    same_name_family = overlap >= 0.40
    matching_attributes = sum((same_category, same_color, same_region, grape_overlap))
    if same_winery and same_name_family and matching_attributes >= 1:
        return True
    return same_winery and overlap >= 0.25 and matching_attributes >= 2


def _model_supported_relation(left: Mapping[str, str], right: Mapping[str, str]) -> bool:
    """Allow an observed confusion to support a family with weaker name overlap."""

    same_winery = _norm(left.get("winery", "")) == _norm(right.get("winery", ""))
    overlap = _jaccard(_meaningful_tokens(left), _meaningful_tokens(right))
    same_category = _norm(left.get("category", "")) == _norm(right.get("category", ""))
    same_color = _norm(left.get("color", "")) == _norm(right.get("color", ""))
    same_region = _norm(left.get("region", "")) == _norm(right.get("region", ""))
    grape_overlap = bool(_tokens(left.get("grape", "")) & _tokens(right.get("grape", "")))
    matching_attributes = sum((same_category, same_color, same_region, grape_overlap))
    return (
        same_winery and (overlap >= 0.18 or matching_attributes >= 2)
    ) or (overlap >= 0.45 and matching_attributes >= 1)


def _meaningful_tokens(row: Mapping[str, str]) -> set[str]:
    name_tokens = _tokens(f"{row.get('title', '')} {row.get('slug', '')}")
    producer_tokens = _tokens(row.get("winery", ""))
    return name_tokens - GENERIC_TOKENS - producer_tokens


def _tokens(value: str) -> set[str]:
    return {token.casefold() for token in TOKEN_RE.findall(value) if len(token) > 2}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _norm(value: str) -> str:
    return " ".join(value.casefold().split())


def _find(parent: dict[str, str], value: str) -> str:
    while parent[value] != value:
        parent[value] = parent[parent[value]]
        value = parent[value]
    return value


def _union(parent: dict[str, str], left: str, right: str) -> None:
    left_root, right_root = _find(parent, left), _find(parent, right)
    if left_root != right_root:
        parent[right_root] = left_root
