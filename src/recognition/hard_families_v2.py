"""Strict, evidence-carrying hard-family construction for v2."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
import itertools
import math
from typing import Mapping, Sequence

from .hard_families import HardFamily, _jaccard, _meaningful_tokens


SMALL_MARGIN_THRESHOLD = 0.02
NAME_JACCARD_THRESHOLD = 0.75
NAME_SEQUENCE_THRESHOLD = 0.80


@dataclass(frozen=True)
class PairEvidence:
    left_slug: str
    right_slug: str
    signals: tuple[str, ...]
    image_similarity: bool
    image_phash_hamming: int | None
    image_thumbnail_mean_abs_diff: float | None
    image_center_phash_hamming: int | None
    image_center_thumbnail_mean_abs_diff: float | None
    image_aspect_ratio_delta: float | None
    image_content_aspect_ratio_delta: float | None
    name_similarity: bool
    name_token_jaccard: float
    name_sequence_ratio: float
    confusion_count: int
    confusion_rank_2_5_count: int
    target_ranks: tuple[int, ...]
    target_rank_counts: tuple[tuple[str, int], ...]
    small_margin_count: int
    min_margin: float | None
    max_margin: float | None

    @property
    def selection_reason(self) -> str:
        return "+".join(self.signals)


@dataclass(frozen=True)
class HardFamilyV2:
    family_id: str
    slugs: tuple[str, ...]
    edges: tuple[PairEvidence, ...]

    @property
    def signals(self) -> tuple[str, ...]:
        return tuple(sorted({signal for edge in self.edges for signal in edge.signals}))

    @property
    def selection_reason(self) -> str:
        return "+".join(self.signals)


def build_hard_families_v2(
    catalog_rows: Sequence[Mapping[str, str]],
    prediction_rows: Sequence[Mapping[str, str]],
    visual_evidence: Mapping[tuple[str, str], Mapping[str, object]],
    v1_families: Sequence[HardFamily],
) -> list[HardFamilyV2]:
    """Build connected families from v1 candidate pairs with >=2 strong signals.

    The candidate universe is deliberately limited to v1's existing related
    pairs. This makes v2 an auditable strict subset/refinement rather than a
    silently different catalog-wide search. Metadata alone is never a v2
    selection signal.
    """

    products = {
        row["slug"]: row
        for row in catalog_rows
        if row.get("mapping_status") == "matched" and row.get("reference_image_path")
    }
    pair_errors: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in prediction_rows:
        target = row.get("target_slug", "")
        predicted = row.get("predicted_slug", "")
        if target == predicted or target not in products or predicted not in products:
            continue
        pair_errors[tuple(sorted((target, predicted)))].append(row)

    candidate_pairs = {
        tuple(sorted(pair))
        for family in v1_families
        for pair in itertools.combinations(family.slugs, 2)
    }
    qualifying: dict[tuple[str, str], PairEvidence] = {}
    for left, right in sorted(candidate_pairs):
        evidence = _pair_evidence(
            left,
            right,
            products[left],
            products[right],
            pair_errors.get((left, right), []),
            visual_evidence.get((left, right), {}),
        )
        if len(evidence.signals) >= 2:
            qualifying[(left, right)] = evidence

    adjacency: dict[str, set[str]] = defaultdict(set)
    for left, right in qualifying:
        adjacency[left].add(right)
        adjacency[right].add(left)

    components: list[tuple[str, ...]] = []
    unseen = set(adjacency)
    while unseen:
        start = min(unseen)
        stack = [start]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current not in unseen:
                continue
            unseen.remove(current)
            component.add(current)
            stack.extend(sorted(adjacency[current] & unseen, reverse=True))
        components.append(tuple(sorted(component)))

    components.sort()
    families: list[HardFamilyV2] = []
    for number, slugs in enumerate(components, start=1):
        edges = tuple(
            qualifying[pair]
            for pair in sorted(qualifying)
            if pair[0] in slugs and pair[1] in slugs
        )
        families.append(HardFamilyV2(f"family-{number:03d}", slugs, edges))
    return families


def _pair_evidence(
    left_slug: str,
    right_slug: str,
    left: Mapping[str, str],
    right: Mapping[str, str],
    errors: Sequence[Mapping[str, str]],
    visual: Mapping[str, object],
) -> PairEvidence:
    name_jaccard, name_sequence = _name_similarity(left, right)
    name_signal = (
        name_jaccard >= NAME_JACCARD_THRESHOLD
        or name_sequence >= NAME_SEQUENCE_THRESHOLD
    )
    rank_values: list[int] = []
    rank_counter: Counter[str] = Counter()
    margins: list[float] = []
    for row in errors:
        rank = _int_or_none(row.get("target_rank", ""))
        if rank is not None:
            rank_values.append(rank)
            rank_counter[str(rank)] += 1
        try:
            margins.append(float(row.get("top1_top2_margin", "")))
        except (TypeError, ValueError):
            pass
    confusion_rank_2_5_count = sum(2 <= rank <= 5 for rank in rank_values)
    small_margin_count = sum(margin <= SMALL_MARGIN_THRESHOLD for margin in margins)
    image_signal = bool(visual.get("image_similarity"))
    signals = []
    if confusion_rank_2_5_count:
        signals.append("real_confusion_rank_2_5")
    if small_margin_count:
        signals.append("small_top1_top2_margin")
    if image_signal:
        signals.append("image_similarity")
    if name_signal:
        signals.append("normalized_name_similarity")
    return PairEvidence(
        left_slug=left_slug,
        right_slug=right_slug,
        signals=tuple(signals),
        image_similarity=image_signal,
        image_phash_hamming=_int_or_none(visual.get("phash_hamming")),
        image_thumbnail_mean_abs_diff=_float_or_none(visual.get("thumbnail_mean_abs_diff")),
        image_center_phash_hamming=_int_or_none(visual.get("center_phash_hamming")),
        image_center_thumbnail_mean_abs_diff=_float_or_none(visual.get("center_thumbnail_mean_abs_diff")),
        image_aspect_ratio_delta=_float_or_none(visual.get("aspect_ratio_delta")),
        image_content_aspect_ratio_delta=_float_or_none(visual.get("content_aspect_ratio_delta")),
        name_similarity=name_signal,
        name_token_jaccard=name_jaccard,
        name_sequence_ratio=name_sequence,
        confusion_count=len(errors),
        confusion_rank_2_5_count=confusion_rank_2_5_count,
        target_ranks=tuple(sorted(set(rank_values))),
        target_rank_counts=tuple(sorted(rank_counter.items())),
        small_margin_count=small_margin_count,
        min_margin=min(margins) if margins else None,
        max_margin=max(margins) if margins else None,
    )


def _name_similarity(left: Mapping[str, str], right: Mapping[str, str]) -> tuple[float, float]:
    left_tokens = _meaningful_tokens(left)
    right_tokens = _meaningful_tokens(right)
    jaccard = _jaccard(left_tokens, right_tokens)
    left_text = " ".join(sorted(left_tokens))
    right_text = " ".join(sorted(right_tokens))
    sequence = SequenceMatcher(None, left_text, right_text).ratio()
    return jaccard, sequence


def _int_or_none(value: object) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None
