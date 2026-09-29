"""Structured tabular reranker and BGE cross-encoder for the OCR bake-off.

Structured reranker (brief Parts 11-15): one Logistic Regression over
bounded tabular features built per (query, Top-5 candidate). Features never
include the slug (identifier only) and never include ground truth; labels
are attached only for training/evaluation.

Group-aware split contract (Part 14): hard_v2 families and generated
products are split as whole units; all four scenarios of one generated
product stay together. The split itself is produced by the previous
milestone's calibration builder and consumed here read-only.

BGE (Parts 16-17): BAAI/bge-reranker-v2-m3 cross-encoder scoring
(query OCR text, candidate document) pairs; sigmoid-bounded; fused with the
min-max normalized image score using a small predefined weight grid.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from rapidfuzz import fuzz

from .text_normalization import extract_vintage_years
from .text_signals import (
    CandidateTextIndex,
    QueryTextEvidence,
    ReferenceOcrEvidence,
)


# ----------------------------------------------------------------------
# IDF over the reference-OCR corpus (Part 12)
# ----------------------------------------------------------------------

def build_idf(reference_texts: Mapping[str, str]) -> dict[str, float]:
    """Smoothed IDF over candidate reference-OCR token corpus.

    idf(token) = log((N + 1) / (df + 1)) + 1, bounded >= 1.0. Rare tokens
    such as a distinctive winery name weigh more than "вино" or "suhoe".
    """

    document_frequency: dict[str, int] = {}
    documents = [text for text in reference_texts.values() if text]
    total = max(len(documents), 1)
    for text in documents:
        for token in set(text.split(" ")):
            if token:
                document_frequency[token] = document_frequency.get(token, 0) + 1
    return {
        token: max(math.log((total + 1) / (count + 1)) + 1.0, 1.0)
        for token, count in document_frequency.items()
    }


def _numeric_tokens(tokens: Sequence[str]) -> tuple[str, ...]:
    return tuple(token for token in tokens if any(character.isdigit() for character in token))


# ----------------------------------------------------------------------
# Feature builder (Part 11)
# ----------------------------------------------------------------------

FEATURE_NAMES: tuple[str, ...] = (
    "image_score",
    "image_margin_top1",
    "image_margin_next",
    "metadata_name_score",
    "winery_score",
    "grape_score",
    "region_score",
    "metadata_text_score",
    "reference_ocr_score",
    "ocr_full_fuzzy",
    "ocr_token_overlap",
    "ocr_idf_overlap",
    "ocr_numeric_overlap",
    "ocr_partial_ratio",
    "ocr_token_set_ratio",
    "vintage_match_value",
    "ref_ocr_year_match",
    "query_token_count",
    "query_mean_confidence",
    "query_has_year",
    "query_numeric_count",
)


def build_feature_row(
    image_scores: Sequence[float],
    candidate_position: int,
    query_evidence: QueryTextEvidence,
    candidate_index: CandidateTextIndex,
    reference_evidence: ReferenceOcrEvidence | None,
    base_signals: Mapping[str, float | str],
    idf: Mapping[str, float],
) -> dict[str, float]:
    """Build one bounded feature row for a candidate at ``candidate_position``.

    Everything is derived from the query OCR evidence, the candidate's own
    metadata/reference-OCR evidence, and the image score distribution. The
    target slug is never an input; the function is deterministic.
    """

    top1 = float(image_scores[0])
    score = float(image_scores[candidate_position])
    next_score = (
        float(image_scores[candidate_position + 1])
        if candidate_position + 1 < len(image_scores)
        else 0.0
    )

    reference_text = reference_evidence.text if reference_evidence is not None else ""
    reference_translit = reference_evidence.transliterated if reference_evidence is not None else ""
    query_text, query_translit = query_evidence.keys

    full_fuzzy = 0.0
    partial = 0.0
    token_set = 0.0
    if reference_text:
        full_fuzzy = max(
            fuzz.ratio(query_text, reference_text) / 100.0,
            fuzz.ratio(query_translit, reference_translit) / 100.0,
        )
        partial = max(
            fuzz.partial_ratio(query_text, reference_text) / 100.0,
            fuzz.partial_ratio(query_translit, reference_translit) / 100.0,
        )
        token_set = max(
            fuzz.token_set_ratio(query_text, reference_text) / 100.0,
            fuzz.token_set_ratio(query_translit, reference_translit) / 100.0,
        )

    reference_tokens = tuple(reference_text.split(" ")) if reference_text else ()
    reference_token_set = set(reference_tokens)
    query_token_list = [token for token in query_evidence.tokens if token]
    overlap = (
        sum(1 for token in query_token_list if token in reference_token_set) / len(query_token_list)
        if query_token_list
        else 0.0
    )
    idf_overlap = (
        sum(
            idf.get(token, 1.0)
            for token in query_token_list
            if token in reference_token_set
        )
        / sum(idf.get(token, 1.0) for token in query_token_list)
        if query_token_list
        else 0.0
    )
    query_numeric = _numeric_tokens(query_token_list)
    numeric_overlap = (
        sum(1 for token in query_numeric if token in reference_token_set) / len(query_numeric)
        if query_numeric
        else 0.0
    )

    vintage_state = str(base_signals["vintage_match"])
    vintage_value = {"exact_match": 1.0, "mismatch": -1.0}.get(vintage_state, 0.0)

    ref_years = set(reference_evidence.vintage_years) if reference_evidence is not None else set()
    query_years = set(query_evidence.vintage_years)
    ref_year_match = 1.0 if ref_years and query_years and ref_years & query_years else 0.0

    confidences = [confidence for _, confidence in query_evidence.raw_lines]
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "image_score": score,
        "image_margin_top1": top1 - score,
        "image_margin_next": score - next_score,
        "metadata_name_score": float(base_signals["metadata_name_score"]),
        "winery_score": float(base_signals["winery_score"]),
        "grape_score": float(base_signals["grape_score"]),
        "region_score": float(base_signals["region_score"]),
        "metadata_text_score": float(base_signals["metadata_text_score"]),
        "reference_ocr_score": float(base_signals["reference_ocr_score"]),
        "ocr_full_fuzzy": round(full_fuzzy, 6),
        "ocr_token_overlap": round(overlap, 6),
        "ocr_idf_overlap": round(idf_overlap, 6),
        "ocr_numeric_overlap": round(numeric_overlap, 6),
        "ocr_partial_ratio": round(partial, 6),
        "ocr_token_set_ratio": round(token_set, 6),
        "vintage_match_value": vintage_value,
        "ref_ocr_year_match": ref_year_match,
        "query_token_count": float(len(query_token_list)),
        "query_mean_confidence": round(mean_confidence, 6),
        "query_has_year": 1.0 if query_years else 0.0,
        "query_numeric_count": float(len(query_numeric)),
    }


def candidate_document_text(
    candidate_index: CandidateTextIndex,
    reference_evidence: ReferenceOcrEvidence | None,
    include_fields: bool = True,
) -> str:
    """Candidate document text for the BGE cross-encoder (Part 16).

    Reference OCR first (closest to what the label actually shows), then the
    structured metadata fields. The slug is never included.
    """

    parts: list[str] = []
    if reference_evidence is not None and reference_evidence.text:
        parts.append(reference_evidence.text)
    if include_fields:
        for field in (candidate_index.title.original, candidate_index.winery.original,
                      candidate_index.grape.original, candidate_index.region.original):
            if field.strip():
                parts.append(field.strip())
    return " | ".join(parts)


# ----------------------------------------------------------------------
# Logistic Regression reranker (Part 13)
# ----------------------------------------------------------------------

@dataclass
class LogisticReranker:
    """A frozen, explainable logistic reranker over the fixed feature set."""

    weights: dict[str, float]
    bias: float
    means: dict[str, float]
    stds: dict[str, float]

    def score_row(self, features: Mapping[str, float]) -> float:
        standardized = 0.0
        for name in FEATURE_NAMES:
            mean = self.means.get(name, 0.0)
            std = self.stds.get(name) or 1.0
            standardized += self.weights.get(name, 0.0) * ((float(features[name]) - mean) / std)
        return 1.0 / (1.0 + math.exp(-standardized - self.bias))

    def to_json(self) -> dict[str, Any]:
        return {"weights": self.weights, "bias": self.bias, "means": self.means, "stds": self.stds}

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "LogisticReranker":
        return cls(
            weights=dict(payload["weights"]),
            bias=float(payload["bias"]),
            means=dict(payload["means"]),
            stds=dict(payload["stds"]),
        )


def fit_logistic_reranker(
    rows: Sequence[Mapping[str, float]],
    labels: Sequence[int],
    ridge: float = 1.0,
) -> LogisticReranker:
    """Fit standardized logistic regression deterministically (LBFGS)."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    matrix = [[float(row[name]) for name in FEATURE_NAMES] for row in rows]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    model = LogisticRegression(
        penalty="l2",
        C=1.0 / ridge,
        solver="lbfgs",
        max_iter=2000,
        random_state=20260920,
    )
    model.fit(scaled, list(labels))
    return LogisticReranker(
        weights={name: float(weight) for name, weight in zip(FEATURE_NAMES, model.coef_[0])},
        bias=float(model.intercept_[0]),
        means={name: float(mean) for name, mean in zip(FEATURE_NAMES, scaler.mean_)},
        stds={name: float(std) for name, std in zip(FEATURE_NAMES, scaler.scale_)},
    )


# ----------------------------------------------------------------------
# BGE cross-encoder (Part 16)
# ----------------------------------------------------------------------

BGE_MODEL_ID = "BAAI/bge-reranker-v2-m3"


class BgeReranker:
    """Local BGE cross-encoder; sigmoid-bounded relevance scores."""

    def __init__(self, device: str = "cuda", batch_size: int = 32, max_length: int = 384) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_ID)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            BGE_MODEL_ID, dtype=dtype
        ).to(device).eval()
        self.max_length = max_length

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        torch = self._torch
        scores: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start : start + self.batch_size]
                inputs = self.tokenizer(
                    [query for query, _ in batch],
                    [doc for _, doc in batch],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**inputs).logits.squeeze(-1)
                probabilities = torch.sigmoid(logits.float())
                scores.extend(probabilities.detach().cpu().tolist())
        return scores

    def release(self) -> None:
        import gc

        self.model = None
        self.tokenizer = None
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
