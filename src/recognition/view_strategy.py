"""Query-side view strategies for frozen-encoder retrieval.

A strategy describes which deterministic views of one query image are
encoded and how their per-candidate similarities are aggregated into one
final score. The catalog/reference side is never affected: reference
embeddings stay one-per-image with the unchanged standard preprocessing.

Strategies are fixed for every query; neither the view set nor the
aggregation may depend on the target slug, the scenario, or any ground
truth. Only inference is involved: no training, no fine-tuning, no OCR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .crop_diagnostics import center_crop_pil


@dataclass(frozen=True)
class QueryViewStrategy:
    """One fixed query preprocessing variant."""

    name: str
    view_names: tuple[str, ...]
    aggregation: str  # "single" | "mean"

    def __post_init__(self) -> None:
        if not self.view_names:
            raise ValueError("A query view strategy needs at least one view")
        if self.aggregation not in {"single", "mean"}:
            raise ValueError(f"Unsupported aggregation: {self.aggregation}")
        if self.aggregation == "single" and len(self.view_names) != 1:
            raise ValueError("single aggregation requires exactly one view")


BASELINE_FULL = QueryViewStrategy(
    name="baseline_full",
    view_names=("full",),
    aggregation="single",
)

# preprocessing_v1: full context + two fixed center crops, mean similarity.
# Rationale (from the pilot32 crop diagnostics milestone, not reinterpreted):
# - full image keeps the overall context;
# - crop85 adds a mild zoom-in;
# - crop70 reduces background influence further;
# - crop55 is deliberately excluded: the diagnostics showed it helps
#   distance_crop but clearly degrades glare/normal retrieval;
# - mean aggregation is preferred over max as the more stable strategy.
PREPROCESSING_V1 = QueryViewStrategy(
    name="preprocessing_v1",
    view_names=("full", "center_85", "center_70"),
    aggregation="mean",
)

STRATEGIES: dict[str, QueryViewStrategy] = {
    BASELINE_FULL.name: BASELINE_FULL,
    PREPROCESSING_V1.name: PREPROCESSING_V1,
}


def _view_image(image: Any, view_name: str) -> Any:
    """Build one named view of a query image deterministically."""

    if view_name == "full":
        return image
    if view_name == "center_85":
        return center_crop_pil(image, 0.85)
    if view_name == "center_70":
        return center_crop_pil(image, 0.70)
    raise ValueError(f"Unknown view name: {view_name}")


def strategy_views(image: Any, strategy: QueryViewStrategy) -> list[Any]:
    """Return the fixed ordered view list for one query image.

    The function depends only on the image pixels and the strategy
    definition; no target label or score is consulted anywhere.
    """

    return [_view_image(image, view_name) for view_name in strategy.view_names]


def aggregate_strategy_scores(
    view_scores: Sequence[Sequence[float]], strategy: QueryViewStrategy
) -> list[float]:
    """Aggregate aligned per-view score vectors into one final score vector."""

    if len(view_scores) != len(strategy.view_names):
        raise ValueError(
            f"Strategy {strategy.name} expects {len(strategy.view_names)} view score vectors, "
            f"got {len(view_scores)}"
        )
    if strategy.aggregation == "single":
        return [float(value) for value in view_scores[0]]
    length = len(view_scores[0])
    if length == 0 or any(len(vector) != length for vector in view_scores):
        raise ValueError("All view score vectors must be non-empty and equally long")
    return [
        sum(float(vector[index]) for vector in view_scores) / len(view_scores)
        for index in range(length)
    ]


def top1_transition_matrix(
    baseline_correct: Sequence[bool], new_correct: Sequence[bool]
) -> dict[str, int]:
    """Four-way Top-1 transition counts between two methods."""

    if len(baseline_correct) != len(new_correct):
        raise ValueError("flag sequences must have the same length")
    counts = {
        "wrong_to_correct": 0,
        "correct_to_wrong": 0,
        "unchanged_correct": 0,
        "unchanged_wrong": 0,
    }
    for was_correct, is_correct in zip(baseline_correct, new_correct):
        if is_correct and not was_correct:
            counts["wrong_to_correct"] += 1
        elif was_correct and not is_correct:
            counts["correct_to_wrong"] += 1
        elif was_correct and is_correct:
            counts["unchanged_correct"] += 1
        else:
            counts["unchanged_wrong"] += 1
    return counts


def top5_transition_matrix(
    baseline_inside: Sequence[bool], new_inside: Sequence[bool]
) -> dict[str, int]:
    """Four-way Top-5 membership transition counts between two methods."""

    if len(baseline_inside) != len(new_inside):
        raise ValueError("flag sequences must have the same length")
    counts = {
        "outside_to_inside": 0,
        "inside_to_outside": 0,
        "stayed_inside": 0,
        "stayed_outside": 0,
    }
    for was_inside, is_inside in zip(baseline_inside, new_inside):
        if is_inside and not was_inside:
            counts["outside_to_inside"] += 1
        elif was_inside and not is_inside:
            counts["inside_to_outside"] += 1
        elif was_inside and is_inside:
            counts["stayed_inside"] += 1
        else:
            counts["stayed_outside"] += 1
    return counts


def rank_change_counts(
    baseline_ranks: Sequence[int | None], new_ranks: Sequence[int | None]
) -> dict[str, int]:
    """Improved/degraded/unchanged rank counts; missing ranks compare as found-only."""

    if len(baseline_ranks) != len(new_ranks):
        raise ValueError("rank sequences must have the same length")
    counts = {"improved": 0, "degraded": 0, "unchanged": 0}
    for baseline_rank, new_rank in zip(baseline_ranks, new_ranks):
        if baseline_rank is None or new_rank is None:
            if baseline_rank != new_rank:
                counts["improved" if new_rank is not None else "degraded"] += 1
            continue
        if new_rank < baseline_rank:
            counts["improved"] += 1
        elif new_rank > baseline_rank:
            counts["degraded"] += 1
        else:
            counts["unchanged"] += 1
    return counts
