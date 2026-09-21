"""Deterministic crop-diagnostics helpers for frozen-encoder analysis.

Everything here is inference-only preprocessing and ranking bookkeeping:
no training, no fine-tuning, no target-label usage inside crop or
aggregation logic. Crop boxes depend only on image size and a fixed ratio.
"""

from __future__ import annotations

from statistics import median
from typing import Any, Iterable, Mapping, Sequence


CENTER_CROP_RATIOS: dict[str, float] = {
    "center_85": 0.85,
    "center_70": 0.70,
    "center_55": 0.55,
}

VIEW_NAMES: tuple[str, ...] = ("full", "center_85", "center_70", "center_55")

MULTICROP_METHODS: tuple[str, ...] = ("multicrop_max", "multicrop_mean")

SINGLE_VIEW_METHODS: tuple[str, ...] = ("baseline_full", "center_crop_85", "center_crop_70", "center_crop_55")

RANK_BANDS: tuple[tuple[str, int, int], ...] = (
    ("1", 1, 1),
    ("2-5", 2, 5),
    ("6-10", 6, 10),
    ("11-25", 11, 25),
    ("26-100", 26, 100),
    (">100", 101, 10**9),
)


def center_crop_box(width: int, height: int, ratio: float) -> tuple[int, int, int, int]:
    """Return the deterministic centered (left, upper, right, lower) box."""

    if width < 1 or height < 1:
        raise ValueError("Image width and height must be positive")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("Crop ratio must be in (0, 1]")
    crop_width = max(1, min(width, int(round(width * ratio))))
    crop_height = max(1, min(height, int(round(height * ratio))))
    left = (width - crop_width) // 2
    upper = (height - crop_height) // 2
    return (left, upper, left + crop_width, upper + crop_height)


def center_crop_pil(image: Any, ratio: float) -> Any:
    """Crop a PIL image with :func:`center_crop_box` without copying logic."""

    return image.crop(center_crop_box(image.width, image.height, ratio))


def view_images(image: Any) -> dict[str, Any]:
    """Build the fixed view set (full + fixed center crops) for one image.

    The view set is identical for every image; no target label or score is
    consulted anywhere in this function.
    """

    views = {"full": image}
    for name, ratio in CENTER_CROP_RATIOS.items():
        views[name] = center_crop_pil(image, ratio)
    return views


def full_ranking(
    scores: Sequence[float], slugs: Sequence[str]
) -> list[tuple[str, float]]:
    """Rank all candidates deterministically: score desc, slug asc on ties.

    This mirrors the tie-break used by ``CatalogIndex.search`` so that the
    full ranking is consistent with the frozen baseline behavior.
    """

    if len(scores) != len(slugs):
        raise ValueError("scores and slugs must have the same length")
    if not slugs:
        raise ValueError("Cannot rank an empty catalog")
    ranked = sorted(zip(slugs, (float(score) for score in scores)), key=lambda pair: (-pair[1], pair[0]))
    return ranked


def target_rank_full(ranked_slugs: Sequence[str], target_slug: str) -> int | None:
    """Return the one-based full-ranking position of the target, if present."""

    try:
        return list(ranked_slugs).index(target_slug) + 1
    except ValueError:
        return None


def aggregate_view_scores(
    view_scores: Mapping[str, Sequence[float]], method: str
) -> list[float]:
    """Aggregate aligned per-view candidate scores for multi-crop inference.

    ``view_scores`` maps view name to a score vector in catalog order. The
    aggregation is fixed for all queries; no target information is used.
    """

    if method == "multicrop_max":
        selector = max
    elif method == "multicrop_mean":
        selector = None
    else:
        raise ValueError(f"Unsupported multi-crop aggregation method: {method}")
    if not view_scores:
        raise ValueError("view_scores must contain at least one view")
    lengths = {len(values) for values in view_scores.values()}
    if len(lengths) != 1:
        raise ValueError("All view score vectors must have the same length")
    length = lengths.pop()
    if length == 0:
        raise ValueError("View score vectors must not be empty")
    vectors = [list(map(float, view_scores[name])) for name in sorted(view_scores)]
    if selector is not None:
        return [selector(vector[index] for vector in vectors) for index in range(length)]
    return [
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(length)
    ]


def rank_metrics(ranks: Sequence[int | None], recall_levels: Sequence[int] = (1, 5, 10, 25)) -> dict[str, Any]:
    """Compute Top-1/Recall@k/MRR/median-rank style metrics from target ranks.

    A ``None`` rank means the target is missing from the full catalog ranking.
    ``median_target_rank_found`` covers found targets only; ``missing_count``
    records how many queries lost the target completely.
    """

    if not ranks:
        raise ValueError("ranks must not be empty")
    total = len(ranks)
    found = [rank for rank in ranks if rank is not None]
    result: dict[str, Any] = {
        "query_count": total,
        "missing_count": total - len(found),
        "mrr": sum(1.0 / rank for rank in found) / total,
        "median_target_rank_found": median(found) if found else None,
    }
    for level in recall_levels:
        if level < 1:
            raise ValueError("recall levels must be positive")
        result[f"top{level}_accuracy" if level == 1 else f"recall_at_{level}"] = (
            sum(rank is not None and rank <= level for rank in ranks) / total
        )
    return result


def rank_band(rank: int | None) -> str:
    """Map a target rank to one fixed diagnostic band."""

    if rank is None:
        return ">100"
    for label, low, high in RANK_BANDS:
        if low <= rank <= high:
            return label
    return ">100"


def rank_transition(baseline_rank: int | None, new_rank: int | None) -> str:
    """Classify one rank change between two preprocessing methods."""

    if baseline_rank is None and new_rank is None:
        return "missing_unchanged"
    if baseline_rank is None:
        return "recovered_from_missing"
    if new_rank is None:
        return "lost_to_missing"
    if new_rank < baseline_rank:
        return "rank_improved"
    if new_rank > baseline_rank:
        return "rank_degraded"
    return "rank_unchanged"


def transition_summary(baseline_ranks: Sequence[int | None], new_ranks: Sequence[int | None]) -> dict[str, Any]:
    """Aggregate rank transitions of one method against the baseline method."""

    if len(baseline_ranks) != len(new_ranks):
        raise ValueError("rank sequences must have the same length")
    if not baseline_ranks:
        raise ValueError("rank sequences must not be empty")
    counts: dict[str, int] = {
        "recovered_from_missing": 0,
        "lost_to_missing": 0,
        "rank_improved": 0,
        "rank_degraded": 0,
        "rank_unchanged": 0,
        "missing_unchanged": 0,
    }
    deltas: list[int] = []
    top5_in, top5_out = 0, 0
    for baseline_rank, new_rank in zip(baseline_ranks, new_ranks):
        transition = rank_transition(baseline_rank, new_rank)
        counts[transition] += 1
        if baseline_rank is not None and new_rank is not None:
            deltas.append(baseline_rank - new_rank)
        baseline_inside = baseline_rank is not None and baseline_rank <= 5
        new_inside = new_rank is not None and new_rank <= 5
        if new_inside and not baseline_inside:
            top5_in += 1
        if baseline_inside and not new_inside:
            top5_out += 1
    found_deltas = sorted(deltas)
    return {
        **counts,
        "top5_gained": top5_in,
        "top5_lost": top5_out,
        "median_rank_delta": median(found_deltas) if found_deltas else None,
    }


def top5_transition_counts(
    baseline_rows: Sequence[Mapping[str, Any]], new_rows: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    """Count wrong->correct and correct->wrong Top-1 flips between methods."""

    if len(baseline_rows) != len(new_rows):
        raise ValueError("prediction row sequences must have the same length")
    wrong_to_correct = 0
    correct_to_wrong = 0
    for baseline_row, new_row in zip(baseline_rows, new_rows):
        baseline_correct = bool(baseline_row.get("correct_top1"))
        new_correct = bool(new_row.get("correct_top1"))
        if new_correct and not baseline_correct:
            wrong_to_correct += 1
        if baseline_correct and not new_correct:
            correct_to_wrong += 1
    return {
        "wrong_to_correct_top1": wrong_to_correct,
        "correct_to_wrong_top1": correct_to_wrong,
    }


def family_transition_flags(
    baseline_top5: Sequence[str],
    new_top5: Sequence[str],
    target_slug: str,
    family_id: str,
    slug_to_family: Mapping[str, str],
) -> dict[str, bool]:
    """Describe family-level movement of one hard query between methods.

    All flags are computed from Top-5 candidate slugs only; the family of the
    target comes from the frozen manifest mapping, never from scores.
    """

    def has_target(top5: Sequence[str]) -> bool:
        return target_slug in top5

    def has_family(top5: Sequence[str]) -> bool:
        return any(slug_to_family.get(slug) == family_id for slug in top5)

    baseline_family_top1 = bool(baseline_top5) and slug_to_family.get(baseline_top5[0]) == family_id
    new_family_top1 = bool(new_top5) and slug_to_family.get(new_top5[0]) == family_id
    return {
        "target_back_in_top5": has_target(new_top5) and not has_target(baseline_top5),
        "family_back_in_top5": has_family(new_top5) and not has_family(baseline_top5),
        "correct_family_top1_wrong_member": new_family_top1 and bool(new_top5) and new_top5[0] != target_slug,
        "left_family_top5": has_family(baseline_top5) and not has_family(new_top5),
        "baseline_family_top1": baseline_family_top1,
        "new_family_top1": new_family_top1,
    }


def split_by(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, list[Mapping[str, Any]]]:
    """Group manifest rows by one column, preserving deterministic ordering."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "")), []).append(row)
    return grouped


def verify_manifest_alignment(
    diagnostic_rows: Sequence[Mapping[str, Any]], manifest_rows: Sequence[Mapping[str, Any]]
) -> None:
    """Fail loudly if diagnostics diverge from the frozen benchmark manifest."""

    manifest_by_id: dict[str, Mapping[str, Any]] = {}
    for row in manifest_rows:
        query_id = str(row.get("query_id", ""))
        if not query_id:
            raise ValueError("manifest contains an empty query_id")
        if query_id in manifest_by_id:
            raise ValueError(f"manifest contains duplicate query_id: {query_id}")
        manifest_by_id[query_id] = row
    diagnostics_by_id: dict[str, Mapping[str, Any]] = {}
    for row in diagnostic_rows:
        query_id = str(row.get("query_id", ""))
        if not query_id:
            raise ValueError("diagnostics contain an empty query_id")
        if query_id in diagnostics_by_id:
            raise ValueError(f"diagnostics contain duplicate query_id: {query_id}")
        diagnostics_by_id[query_id] = row
    missing = sorted(set(manifest_by_id) - set(diagnostics_by_id))
    extra = sorted(set(diagnostics_by_id) - set(manifest_by_id))
    if missing or extra:
        raise ValueError(
            f"diagnostics do not match the manifest: missing={missing[:3]} "
            f"({len(missing)} total), extra={extra[:3]} ({len(extra)} total)"
        )
    for query_id, manifest_row in manifest_by_id.items():
        diagnostic_row = diagnostics_by_id[query_id]
        if str(diagnostic_row.get("target_slug", "")) != str(manifest_row.get("target_slug", "")):
            raise ValueError(f"Target mismatch for {query_id}")


def verify_subset_scenario_split(
    manifest_rows: Sequence[Mapping[str, Any]],
    expected_subset_counts: Mapping[str, int],
    expected_scenario_counts: Mapping[str, int],
) -> None:
    """Verify that the representative/hard and scenario splits are intact."""

    subset_counts: Counter = {}
    scenario_counts: Counter = {}
    for row in manifest_rows:
        subset_counts[str(row.get("subset_role", ""))] = subset_counts.get(str(row.get("subset_role", "")), 0) + 1
        scenario_counts[str(row.get("scenario_id", ""))] = scenario_counts.get(str(row.get("scenario_id", "")), 0) + 1
    for subset, expected in expected_subset_counts.items():
        if subset_counts.get(subset, 0) != expected:
            raise ValueError(
                f"subset_role {subset!r} count changed: expected {expected}, got {subset_counts.get(subset, 0)}"
            )
    for scenario, expected in expected_scenario_counts.items():
        if scenario_counts.get(scenario, 0) != expected:
            raise ValueError(
                f"scenario {scenario!r} count changed: expected {expected}, got {scenario_counts.get(scenario, 0)}"
            )


def verify_family_mapping(manifest_rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Return the frozen slug->family mapping and verify it is consistent."""

    slug_to_family: dict[str, str] = {}
    slug_to_type: dict[str, str] = {}
    for row in manifest_rows:
        slug = str(row.get("target_slug", "")).strip()
        family_id = str(row.get("target_family_id", "")).strip()
        family_type = str(row.get("family_type", "")).strip()
        if not slug or not family_id:
            continue
        previous = slug_to_family.get(slug)
        if previous and previous != family_id:
            raise ValueError(f"Product is assigned to multiple target families: {slug}")
        if slug in slug_to_type and slug_to_type[slug] != family_type:
            raise ValueError(f"Product family_type changed between rows: {slug}")
        slug_to_family[slug] = family_id
        slug_to_type[slug] = family_type
    return slug_to_family


def count_true(rows: Iterable[Mapping[str, Any]], key: str) -> int:
    """Count truthy values of one boolean-ish column."""

    return sum(bool(row.get(key)) for row in rows)
