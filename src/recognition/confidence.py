"""Configurable, intentionally uncalibrated unknown-item decision."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceDecision:
    score: float | None
    threshold: float
    is_unknown: bool


def decide_unknown(score: float | None, unknown_threshold: float) -> ConfidenceDecision:
    """Apply a cosine-score threshold; this is not a probability estimate."""

    if not -1.0 <= unknown_threshold <= 1.0:
        raise ValueError("unknown_threshold must be between -1 and 1")
    is_unknown = score is None or score < unknown_threshold
    return ConfidenceDecision(score, unknown_threshold, is_unknown)
