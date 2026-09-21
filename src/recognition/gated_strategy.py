"""Confidence-gated multi-crop fallback for the frozen SigLIP2 pipeline.

Runtime shape (production-like):

    query image
      -> full-image embedding (computed exactly once)
      -> full-catalog ranking
      -> confidence gate (top-1 score / top-1/top-2 margin only)
           confident  -> return the full ranking
           uncertain  -> encode ONLY center 85% + center 70% in one batched
                         forward, then rank by mean similarity of
                         (full, crop85, crop70) - identical to the
                         preprocessing_v1 aggregation semantics.

The gate never sees ground truth, the target slug, scenario/subset/family
labels, filenames, or any crop information: its decision depends only on the
score distribution of the full-image ranking. No OCR, no reranker, no
learning.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import median
from typing import Any, Mapping, Sequence

from .crop_diagnostics import center_crop_pil, full_ranking
from .view_strategy import PREPROCESSING_V1, aggregate_strategy_scores

FALLBACK_CROP_VIEWS: tuple[str, ...] = ("center_85", "center_70")

GATE_RULE_TYPES: tuple[str, ...] = ("margin", "score", "margin_or_score")


@dataclass(frozen=True)
class GateRule:
    """One frozen, explainable confidence rule over full-ranking scores."""

    rule_type: str
    margin_threshold: float | None = None
    score_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.rule_type not in GATE_RULE_TYPES:
            raise ValueError(f"Unsupported gate rule type: {self.rule_type}")
        if self.rule_type in {"margin", "margin_or_score"} and self.margin_threshold is None:
            raise ValueError(f"{self.rule_type} rule requires margin_threshold")
        if self.rule_type in {"score", "margin_or_score"} and self.score_threshold is None:
            raise ValueError(f"{self.rule_type} rule requires score_threshold")

    def should_fallback(self, top1_score: float, top2_score: float, top5_score: float) -> bool:
        """Decide fallback using only the full-image score distribution."""

        margin = top1_score - top2_score
        if self.rule_type == "margin":
            return margin < float(self.margin_threshold)
        if self.rule_type == "score":
            return top1_score < float(self.score_threshold)
        return (
            margin < float(self.margin_threshold)
            or top1_score < float(self.score_threshold)
        )

    def describe(self) -> str:
        if self.rule_type == "margin":
            return f"fallback if top1_top2_margin < {self.margin_threshold:.4f}"
        if self.rule_type == "score":
            return f"fallback if top1_score < {self.score_threshold:.4f}"
        return (
            f"fallback if top1_top2_margin < {self.margin_threshold:.4f} "
            f"or top1_score < {self.score_threshold:.4f}"
        )


def confidence_features(ranked_scores: Sequence[float]) -> dict[str, float]:
    """Extract gate features from the score list of one full ranking.

    Only the scores of the full-image ranking are used; nothing else enters.
    """

    if not ranked_scores:
        raise ValueError("ranked_scores must not be empty")
    top1 = float(ranked_scores[0])
    top2 = float(ranked_scores[1]) if len(ranked_scores) > 1 else top1
    top5 = float(ranked_scores[4]) if len(ranked_scores) > 4 else top2
    return {
        "top1_score": top1,
        "top2_score": top2,
        "top5_score": top5,
        "top1_top2_margin": top1 - top2,
        "top1_top5_margin": top1 - top5,
    }


@dataclass(frozen=True)
class GateDiagnostics:
    """Per-query gate diagnostics returned with every gated retrieval."""

    used_fallback: bool
    gate_reason: str
    confidence_score: float
    top1_top2_margin: float
    views_used: int

    def to_dict(self) -> dict[str, object]:
        return {
            "used_fallback": self.used_fallback,
            "gate_reason": self.gate_reason,
            "confidence_score": self.confidence_score,
            "top1_top2_margin": self.top1_top2_margin,
            "views_used": self.views_used,
        }


def _fallback_reason(rule: GateRule, features: Mapping[str, float]) -> str:
    margin_low = (
        rule.margin_threshold is not None
        and features["top1_top2_margin"] < float(rule.margin_threshold)
    )
    score_low = (
        rule.score_threshold is not None
        and features["top1_score"] < float(rule.score_threshold)
    )
    if margin_low and score_low:
        return "fallback:margin+score"
    if margin_low:
        return "fallback:margin"
    if score_low:
        return "fallback:score"
    return "confident"


def retrieve_with_confidence_gate(
    encoder: Any,
    reference_matrix: Any,
    catalog_slugs: Sequence[str],
    image: Any,
    rule: GateRule,
    device: str,
) -> tuple[list[tuple[str, float]], GateDiagnostics, dict[str, float]]:
    """Run the gated retrieval for one query image.

    Returns the full deterministic ranking (slug, score), gate diagnostics,
    and production-like latency components in milliseconds. The full-image
    embedding is computed exactly once and reused if the fallback fires.
    """

    import torch

    def _sync() -> None:
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)

    from time import perf_counter

    # 1. Full-image embedding (computed exactly once).
    _sync()
    embed_started = perf_counter()
    full_embedding = torch.tensor(
        encoder.encode_pil([image]), dtype=torch.float32, device=device
    )
    full_embedding = torch.nn.functional.normalize(full_embedding, p=2, dim=1)
    _sync()
    embed_full_ms = (perf_counter() - embed_started) * 1000.0

    # 2. Full ranking + gate decision.
    retrieval_started = perf_counter()
    full_scores_vector = (full_embedding @ reference_matrix.transpose(0, 1))[0]
    full_scores = full_scores_vector.detach().cpu().tolist()
    ranked = full_ranking(full_scores, catalog_slugs)
    features = confidence_features([score for _, score in ranked[:5]])
    needs_fallback = rule.should_fallback(
        features["top1_score"], features["top2_score"], features["top5_score"]
    )
    _sync()
    retrieval_gate_ms = (perf_counter() - retrieval_started) * 1000.0

    if not needs_fallback:
        diagnostics = GateDiagnostics(
            used_fallback=False,
            gate_reason="confident",
            confidence_score=features["top1_score"],
            top1_top2_margin=features["top1_top2_margin"],
            views_used=1,
        )
        latencies = {
            "embed_full_ms": embed_full_ms,
            "retrieval_gate_ms": retrieval_gate_ms,
            "embed_crops_ms": 0.0,
            "retrieval_fallback_ms": 0.0,
            "total_ms": embed_full_ms + retrieval_gate_ms,
        }
        return ranked, diagnostics, latencies

    # 3. Fallback: encode only the two crops (full embedding is reused).
    _sync()
    crop_started = perf_counter()
    crop_images = [center_crop_pil(image, 0.85), center_crop_pil(image, 0.70)]
    crop_embeddings = torch.tensor(
        encoder.encode_pil(crop_images), dtype=torch.float32, device=device
    )
    crop_embeddings = torch.nn.functional.normalize(crop_embeddings, p=2, dim=1)
    _sync()
    embed_crops_ms = (perf_counter() - crop_started) * 1000.0

    fallback_started = perf_counter()
    crop_scores = (crop_embeddings @ reference_matrix.transpose(0, 1)).detach().cpu().tolist()
    final_scores = aggregate_strategy_scores(
        [full_scores, crop_scores[0], crop_scores[1]], PREPROCESSING_V1
    )
    ranked = full_ranking(final_scores, catalog_slugs)
    _sync()
    retrieval_fallback_ms = (perf_counter() - fallback_started) * 1000.0

    diagnostics = GateDiagnostics(
        used_fallback=True,
        gate_reason=_fallback_reason(rule, features),
        confidence_score=features["top1_score"],
        top1_top2_margin=features["top1_top2_margin"],
        views_used=3,
    )
    latencies = {
        "embed_full_ms": embed_full_ms,
        "retrieval_gate_ms": retrieval_gate_ms,
        "embed_crops_ms": embed_crops_ms,
        "retrieval_fallback_ms": retrieval_fallback_ms,
        "total_ms": embed_full_ms + retrieval_gate_ms + embed_crops_ms + retrieval_fallback_ms,
    }
    return ranked, diagnostics, latencies


def stratified_product_split(
    manifest_rows: Sequence[Mapping[str, Any]], seed: int
) -> dict[str, str]:
    """Deterministic product-level calibration split for pilot32 manifests.

    Products are the unit: every query of one product (all four scenarios)
    lands in the same split. The split is stratified over representative
    products and hard family types (vintage, subtype): for each stratum,
    sorted product groups are deterministically shuffled with the seed and
    split in half. Returns slug -> "calibration" | "held_out".
    """

    representative: set[str] = set()
    family_groups: dict[str, set[str]] = {}
    family_type_by_family: dict[str, str] = {}
    for row in manifest_rows:
        slug = str(row.get("target_slug", "")).strip()
        if not slug:
            continue
        family_id = str(row.get("target_family_id", "")).strip()
        family_type = str(row.get("family_type", "")).strip()
        if not family_id:
            representative.add(slug)
            continue
        family_groups.setdefault(family_id, set()).add(slug)
        previous_type = family_type_by_family.get(family_id)
        if previous_type and previous_type != family_type:
            raise ValueError(f"Family {family_id} has inconsistent family_type")
        family_type_by_family[family_id] = family_type

    rng = random.Random(seed)
    split: dict[str, str] = {}

    def split_half(items: list[str]) -> None:
        ordered = sorted(items)
        rng.shuffle(ordered)
        calibration_count = len(ordered) // 2
        for index, item in enumerate(ordered):
            split[item] = "calibration" if index < calibration_count else "held_out"

    split_half(sorted(representative))
    for family_type in sorted(set(family_type_by_family.values())):
        families = sorted(
            family_id
            for family_id, type_of in family_type_by_family.items()
            if type_of == family_type
        )
        ordered = list(families)
        rng.shuffle(ordered)
        calibration_count = len(ordered) // 2
        family_split = {
            family_id: ("calibration" if index < calibration_count else "held_out")
            for index, family_id in enumerate(ordered)
        }
        for family_id, slugs in family_groups.items():
            if family_id in family_split:
                for slug in slugs:
                    split[slug] = family_split[family_id]
    return split


def seeded_product_sample(slugs: Sequence[str], sample_size: int, seed: int) -> list[str]:
    """Deterministic product-level sample for clean-benchmark calibration."""

    if sample_size > len(slugs):
        raise ValueError("sample_size cannot exceed the number of products")
    ordered = sorted(set(slugs))
    rng = random.Random(seed)
    return sorted(rng.sample(ordered, sample_size))


def roc_auc(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float | None:
    """Mann-Whitney based ROC-AUC; ties count as 0.5. Higher = more positive."""

    if not positive_scores or not negative_scores:
        return None
    labeled = [(float(score), 1) for score in positive_scores] + [
        (float(score), 0) for score in negative_scores
    ]
    labeled.sort(key=lambda pair: pair[0])
    rank_sum_positive = 0.0
    index = 0
    while index < len(labeled):
        same_score_end = index
        while same_score_end < len(labeled) and labeled[same_score_end][0] == labeled[index][0]:
            same_score_end += 1
        average_rank = (index + same_score_end + 1) / 2.0  # one-based average rank
        for position in range(index, same_score_end):
            if labeled[position][1] == 1:
                rank_sum_positive += average_rank
        index = same_score_end
    positive_count = len(positive_scores)
    negative_count = len(negative_scores)
    return (rank_sum_positive - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


def quantile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated quantile on sorted values (0 <= fraction <= 1)."""

    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be within [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def distribution_summary(positive: Sequence[float], negative: Sequence[float]) -> dict[str, Any]:
    """Median/quantile comparison of a signal for correct vs incorrect cases."""

    def stats(values: Sequence[float]) -> dict[str, float | None]:
        if not values:
            return {"median": None, "q10": None, "q90": None}
        return {
            "median": median(values),
            "q10": quantile(values, 0.10),
            "q90": quantile(values, 0.90),
        }

    return {
        "correct": stats(positive),
        "incorrect": stats(negative),
    }
