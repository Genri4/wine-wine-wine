"""Conservative OCR text reranking over frozen image-retrieval Top-5.

Fixed policy set (milestone brief, Part J) — no learned reranker:

- ``image_only``                  policy 0: baseline order untouched
- ``metadata_text_blend``         policy 1: text = candidate metadata score
- ``reference_ocr_blend``         policy 2: text = candidate reference OCR score
- ``combined_text_blend``         policy 3: text = best of the two above
- ``combined_vintage_blend``      policy 4: combined + explicit vintage evidence

Score fusion (Part K): per-query min-max normalization of the image cosine
scores inside the Top-5, text scores already in [0, 1], then

    final = (1 - alpha) * image_norm + alpha * text  (+ bounded vintage term)

``alpha`` is strictly global: one value per run, never per scenario, family
or product.

Conservatism (Part I): image retrieval stays the primary signal. A candidate
may only take Top-1 away from the image winner when its own text evidence
beats the image winner's text evidence by at least ``min_text_margin`` —
with no usable OCR (all-zero text scores) nothing can ever be reordered.
Reranking permutes the Top-5 only, so the Top-5 candidate set (and Recall@5)
is preserved by construction; a correct target outside Top-5 cannot be
rescued.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

POLICY_IMAGE_ONLY = "image_only"
POLICY_METADATA_BLEND = "metadata_text_blend"
POLICY_REFERENCE_BLEND = "reference_ocr_blend"
POLICY_COMBINED_BLEND = "combined_text_blend"
POLICY_COMBINED_VINTAGE = "combined_vintage_blend"

FUSION_POLICIES = (
    POLICY_IMAGE_ONLY,
    POLICY_METADATA_BLEND,
    POLICY_REFERENCE_BLEND,
    POLICY_COMBINED_BLEND,
    POLICY_COMBINED_VINTAGE,
)

TEXT_SOURCE_BY_POLICY = {
    POLICY_IMAGE_ONLY: None,
    POLICY_METADATA_BLEND: "metadata_text_score",
    POLICY_REFERENCE_BLEND: "reference_ocr_score",
    POLICY_COMBINED_BLEND: "combined",
    POLICY_COMBINED_VINTAGE: "combined",
}

DEFAULT_VINTAGE_BONUS = 0.05
DEFAULT_VINTAGE_PENALTY = 0.05


@dataclass(frozen=True)
class FusionConfig:
    """One frozen reranker configuration."""

    policy: str
    alpha: float
    vintage_bonus: float = DEFAULT_VINTAGE_BONUS
    vintage_penalty: float = DEFAULT_VINTAGE_PENALTY
    min_text_margin: float = 0.05

    def key(self) -> str:
        return (
            f"{self.policy}_alpha{self.alpha:.2f}_bonus{self.vintage_bonus:.2f}"
            f"_penalty{self.vintage_penalty:.2f}_margin{self.min_text_margin:.2f}"
        )


def policy_text_scores(policy: str, signals: Mapping[str, float | str]) -> float:
    """Reduce one candidate's raw signals to the policy's single text score."""
    if policy == POLICY_IMAGE_ONLY:
        raise ValueError("image_only has no text score")
    if policy == POLICY_METADATA_BLEND:
        return float(signals["metadata_text_score"])
    if policy == POLICY_REFERENCE_BLEND:
        return float(signals["reference_ocr_score"])
    if policy in (POLICY_COMBINED_BLEND, POLICY_COMBINED_VINTAGE):
        return max(float(signals["metadata_text_score"]), float(signals["reference_ocr_score"]))
    raise ValueError(f"Unknown policy: {policy}")


def normalize_image_scores(scores: Sequence[float]) -> list[float]:
    """Min-max normalize image scores within one query's Top-5.

    Identical scores mean no visual discrimination; they map to 1.0 so the
    text term alone decides instead of an arbitrary zero.
    """
    if not scores:
        return []
    lowest, highest = min(scores), max(scores)
    if highest - lowest <= 0.0:
        return [1.0 for _ in scores]
    return [(value - lowest) / (highest - lowest) for value in scores]


def vintage_adjustment(state: str, config: FusionConfig) -> float:
    """Bounded vintage term; ``unknown`` never changes the score."""
    if state == "exact_match":
        return config.vintage_bonus
    if state == "mismatch":
        return -config.vintage_penalty
    return 0.0


@dataclass(frozen=True)
class RerankOutcome:
    """Diagnostics of one query's conservative rerank."""

    final_order: tuple[int, ...]
    final_scores: tuple[float, ...]
    text_scores: tuple[float, ...]
    image_scores_normalized: tuple[float, ...]
    reordered: bool
    reason: str


def rerank_one_query(
    image_scores: Sequence[float],
    signals_per_candidate: Sequence[Mapping[str, float | str]],
    config: FusionConfig,
) -> RerankOutcome:
    """Fuse image and text evidence for one query's Top-5 candidates.

    Returns the final candidate order as indices into the input order. The
    input order IS the image ranking; ``image_only`` returns it untouched,
    which is what makes policy 0 an exact baseline reproduction.
    """
    count = len(image_scores)
    if count != len(signals_per_candidate):
        raise ValueError("image scores and signals must align candidate-wise")
    if config.policy == POLICY_IMAGE_ONLY:
        return RerankOutcome(
            final_order=tuple(range(count)),
            final_scores=tuple(float(score) for score in image_scores),
            text_scores=tuple(0.0 for _ in range(count)),
            image_scores_normalized=tuple(float(score) for score in image_scores),
            reordered=False,
            reason="image_only",
        )

    image_norm = normalize_image_scores(image_scores)
    text_scores = tuple(policy_text_scores(config.policy, signals) for signals in signals_per_candidate)
    final: list[float] = []
    for index in range(count):
        value = (1.0 - config.alpha) * image_norm[index] + config.alpha * text_scores[index]
        if config.policy == POLICY_COMBINED_VINTAGE:
            value += vintage_adjustment(str(signals_per_candidate[index]["vintage_match"]), config)
        final.append(value)

    order = tuple(sorted(range(count), key=lambda index: (-final[index], index)))
    image_winner = 0
    fused_winner = order[0]
    if fused_winner == image_winner:
        reordered = False
        reason = "image_top1_kept"
    elif text_scores[fused_winner] - text_scores[image_winner] >= config.min_text_margin:
        reordered = True
        reason = f"text_margin={text_scores[fused_winner] - text_scores[image_winner]:.3f}"
    else:
        # Not enough text evidence: keep the image winner at Top-1 by moving
        # it in front of the challenging candidate (stable elsewhere).
        order = (image_winner,) + tuple(index for index in order if index != image_winner)
        reordered = False
        reason = (
            f"kept_image_top1_weak_text_evidence"
            f"(margin={text_scores[fused_winner] - text_scores[image_winner]:.3f}"
            f"<{config.min_text_margin:.2f})"
        )
    return RerankOutcome(
        final_order=order,
        final_scores=tuple(final),
        text_scores=text_scores,
        image_scores_normalized=tuple(image_norm),
        reordered=reordered,
        reason=reason,
    )


# ----------------------------------------------------------------------
# Oracles (Part P) — explicitly diagnostic, never production metrics
# ----------------------------------------------------------------------


def oracle_top5_present(target_rank: int | None) -> bool:
    return target_rank is not None and target_rank <= 5


def oracle_text_top1(
    signals_per_candidate: Sequence[Mapping[str, float | str]],
    policy: str,
) -> int:
    """Index of the candidate a perfect text judge would pick.

    Stable tie-break keeps the image order, so a flat text score cannot
    invent a reorder. This is the ORACLE ceiling of the text signal, not a
    policy and not a production metric.
    """
    scores = [policy_text_scores(policy, signals) for signals in signals_per_candidate]
    if policy == POLICY_COMBINED_VINTAGE:
        scores = [
            score + vintage_adjustment(str(signals["vintage_match"]), FusionConfig(POLICY_COMBINED_VINTAGE, alpha=0.0))
            for score, signals in zip(scores, signals_per_candidate)
        ]
    return min(range(len(scores)), key=lambda index: (-scores[index], index))


# ----------------------------------------------------------------------
# Transition analysis (Part V)
# ----------------------------------------------------------------------


@dataclass
class TransitionCounts:
    wrong_to_correct: int = 0
    correct_to_wrong: int = 0
    correct_to_correct_same: int = 0
    correct_to_correct_moved: int = 0
    wrong_to_wrong: int = 0

    @property
    def rescued(self) -> int:
        return self.wrong_to_correct

    @property
    def broken(self) -> int:
        return self.correct_to_wrong


def top1_transition_counts(
    baseline_top1: Sequence[str],
    final_top1: Sequence[str],
    target_slugs: Sequence[str],
) -> TransitionCounts:
    counts = TransitionCounts(0, 0, 0, 0, 0)
    for baseline, final, target in zip(baseline_top1, final_top1, target_slugs):
        baseline_correct = baseline == target
        final_correct = final == target
        if not baseline_correct and final_correct:
            counts.wrong_to_correct += 1
        elif baseline_correct and not final_correct:
            counts.correct_to_wrong += 1
        elif baseline_correct and final_correct:
            # Both Top-1 hits equal the target, so the candidate is the same;
            # rank changes below Top-1 are tracked separately in the runner.
            counts.correct_to_correct_same += 1
        else:
            counts.wrong_to_wrong += 1
    return counts
