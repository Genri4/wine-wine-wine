"""Model-independent ranking metrics for reproducible retrieval benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class RankingRecord:
    """One query result supplied by any retrieval model."""

    query_id: str
    target_slug: str
    ranked_candidates: Sequence[str]
    scores: Sequence[float]
    latency_ms: float
    transform_type: str = ""


def evaluate_rankings(
    records: Iterable[RankingRecord | Mapping[str, Any]],
    *,
    recall_k: int = 5,
) -> dict[str, object]:
    """Calculate common retrieval metrics without depending on a model.

    ``ranked_candidates`` must be ordered best-first. A missing target has
    rank ``None`` and contributes zero to Recall@k and MRR.
    """

    if recall_k < 1:
        raise ValueError("recall_k must be positive")

    normalized = [_coerce_record(record) for record in records]
    if not normalized:
        return {
            "query_count": 0,
            "top1_accuracy": None,
            "recall_at_5": None,
            "mrr": None,
            "mean_latency_ms": None,
            "p50_latency_ms": None,
            "p95_latency_ms": None,
        }

    ranks = [_target_rank(record.target_slug, record.ranked_candidates) for record in normalized]
    top1_hits = sum(rank == 1 for rank in ranks)
    recall_hits = sum(rank is not None and rank <= recall_k for rank in ranks)
    reciprocal_ranks = [1.0 / rank if rank is not None else 0.0 for rank in ranks]
    latencies = [record.latency_ms for record in normalized]
    return {
        "query_count": len(normalized),
        "top1_accuracy": top1_hits / len(normalized),
        "recall_at_5": recall_hits / len(normalized),
        "mrr": mean(reciprocal_ranks),
        "mean_latency_ms": mean(latencies),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
    }


def _coerce_record(record: RankingRecord | Mapping[str, Any]) -> RankingRecord:
    if isinstance(record, RankingRecord):
        return record
    candidates = record.get("ranked_candidates", record.get("top5_slugs"))
    if isinstance(candidates, str):
        raise ValueError("ranked_candidates must be a sequence, not a JSON string")
    if not isinstance(candidates, Sequence):
        raise ValueError("ranking record is missing ranked_candidates")
    scores = record.get("scores", record.get("top5_scores", []))
    if isinstance(scores, str) or not isinstance(scores, Sequence):
        raise ValueError("scores must be a sequence")
    if len(scores) < len(candidates):
        # Scores are not needed for rank metrics, but a complete ranking record
        # is less error-prone for downstream error analysis.
        scores = list(scores) + [0.0] * (len(candidates) - len(scores))
    return RankingRecord(
        query_id=str(record["query_id"]),
        target_slug=str(record["target_slug"]),
        ranked_candidates=[str(candidate) for candidate in candidates],
        scores=[float(score) for score in scores[: len(candidates)]],
        latency_ms=float(record.get("latency_ms", 0.0)),
        transform_type=str(record.get("transform_type", "")),
    )


def target_rank(target_slug: str, ranked_candidates: Sequence[str]) -> int | None:
    """Return the one-based rank of a target, if it is present."""

    return _target_rank(target_slug, ranked_candidates)


def percentile(values: Sequence[float], percentage: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    if not 0.0 <= percentage <= 100.0:
        raise ValueError("percentage must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentage / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def assert_no_reference_query_leakage(
    query_paths: Iterable[str], reference_paths: Iterable[str]
) -> None:
    """Fail loudly if a generated query path is also in the reference index."""

    queries = {str(Path(path).resolve()) for path in query_paths}
    references = {str(Path(path).resolve()) for path in reference_paths}
    overlap = queries & references
    if overlap:
        raise ValueError(f"Reference/query leakage detected: {sorted(overlap)[:3]}")


def _target_rank(target_slug: str, ranked_candidates: Sequence[str]) -> int | None:
    try:
        return list(ranked_candidates).index(target_slug) + 1
    except ValueError:
        return None
