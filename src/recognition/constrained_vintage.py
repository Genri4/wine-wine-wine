"""Candidate-constrained vintage recognition: shared helpers.

Task framing (milestone brief): for each challenge query the allowed years
are the known years of the same-family Top-5 candidates. A method must
CHOOSE one of the allowed years (or abstain) - never recognize arbitrary
text. Ground truth is never used at inference.

Crop normalization (Part 5): one fixed pipeline - grayscale, padded square,
resize to 224x224 (Lanczos). Deterministic for every crop of every method.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def normalize_crop(image: Any, size: int = 224) -> Any:
    """Grayscale -> padded square -> Lanczos resize. Deterministic."""

    from PIL import Image

    gray = image.convert("L")
    width, height = gray.size
    side = max(width, height)
    canvas = Image.new("L", (side, side), color=255)
    canvas.paste(gray, ((side - width) // 2, (side - height) // 2))
    return canvas.resize((size, size), Image.LANCZOS)


@dataclass
class ConstrainedDecision:
    chosen_year: str | None
    scores: dict[str, float]
    margin: float | None
    abstained: bool
    reason: str


def decide_from_scores(
    scores: Mapping[str, float],
    allowed_years: Sequence[str],
    min_margin: float = 0.0,
) -> ConstrainedDecision:
    """Pick the allowed year with the highest score; abstain on ties."""

    allowed_scores = {
        year: float(scores.get(year, float("-inf"))) for year in allowed_years if year in scores
    }
    if not allowed_scores:
        return ConstrainedDecision(None, dict(allowed_scores), None, True, "no_scores")
    ordered = sorted(allowed_scores.items(), key=lambda item: (-item[1], item[0]))
    best_year, best_score = ordered[0]
    if len(ordered) > 1 and best_score - ordered[1][1] <= min_margin:
        return ConstrainedDecision(None, dict(allowed_scores), 0.0, True, "tie")
    return ConstrainedDecision(
        best_year,
        dict(allowed_scores),
        best_score - ordered[1][1] if len(ordered) > 1 else None,
        False,
        "ok",
    )


def pairwise_year_accuracy(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Accuracy over queries with >= 2 distinct known competing years."""

    total = correct = 0
    for row in rows:
        if not row.get("pairwise"):
            continue
        total += 1
        correct += 1 if row.get("chosen_year") == row.get("target_year") else 0
    return {
        "pairwise_queries": total,
        "pairwise_correct": correct,
        "pairwise_accuracy": round(correct / total, 4) if total else None,
    }


def idf_placeholder() -> None:
    return None
