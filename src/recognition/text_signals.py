"""Text evidence signals for conservative OCR reranking.

For one query OCR record and one catalog candidate the module computes the
fixed signal set from the milestone brief:

1. ``metadata_name_score``   query OCR vs candidate product name
2. ``winery_score``          query OCR vs candidate winery
3. ``grape_score``           query OCR vs candidate grape
4. ``region_score``          query OCR vs candidate region
5. ``metadata_text_score``   query OCR vs the short combined catalog text
6. ``reference_ocr_score``   query OCR vs OCR of the candidate reference image
7. ``vintage_match``         exact_match / mismatch / unknown

Fuzzy scores come from RapidFuzz; every score is bounded to [0, 1]. Matching
runs on both the Cyrillic-normalized form and the project transliteration of
each side and keeps the best value, so OCR output in either script can match
Cyrillic catalog metadata.

The slug is never a text feature: it is only the candidate identifier.
Description text is intentionally excluded from the combined metadata text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from rapidfuzz import fuzz

from .text_normalization import (
    comparison_keys,
    extract_vintage_years,
    normalize_text,
    transliterated_key,
    vintage_match_state,
)

# OCR boxes below this confidence are excluded from the scoring text (the
# raw lines with confidences are still cached). Deterministic and global.
DEFAULT_MIN_OCR_CONFIDENCE = 0.5


@dataclass(frozen=True)
class TextFieldKeys:
    """Comparison keys for one text field."""

    original: str
    normalized: str
    transliterated: str


def build_field_keys(value: str) -> TextFieldKeys:
    normalized, transliterated = comparison_keys(value or "")
    return TextFieldKeys(original=value or "", normalized=normalized, transliterated=transliterated)


@dataclass(frozen=True)
class CandidateTextIndex:
    """Precomputed text evidence for one catalog product."""

    slug: str
    title: TextFieldKeys
    winery: TextFieldKeys
    grape: TextFieldKeys
    region: TextFieldKeys
    metadata_text: TextFieldKeys
    vintage_year: str | None


def build_candidate_text_index(row: Mapping[str, str]) -> CandidateTextIndex:
    """Build the text index from one catalog manifest row.

    Uses only real catalog fields (title, winery, grape, region); the slug is
    kept as the identifier and never enters any comparison text. A candidate
    vintage year is extracted from the title when the title actually contains
    a standalone in-range year; otherwise it stays unknown and is never
    invented.
    """
    title = row.get("title", "") or ""
    metadata_parts = [
        part
        for part in (
            row.get("title", "") or "",
            row.get("winery", "") or "",
            row.get("grape", "") or "",
            row.get("region", "") or "",
        )
        if part.strip()
    ]
    metadata_text = " ".join(metadata_parts)
    years = extract_vintage_years(title)
    return CandidateTextIndex(
        slug=str(row.get("slug", "")),
        title=build_field_keys(title),
        winery=build_field_keys(row.get("winery", "") or ""),
        grape=build_field_keys(row.get("grape", "") or ""),
        region=build_field_keys(row.get("region", "") or ""),
        metadata_text=build_field_keys(metadata_text),
        vintage_year=years[0] if len(years) == 1 else (years[-1] if years else None),
    )


@dataclass(frozen=True)
class QueryTextEvidence:
    """Normalized OCR evidence for one query image."""

    raw_lines: tuple[tuple[str, float], ...]
    text: str
    transliterated: str
    tokens: tuple[str, ...]
    vintage_years: tuple[str, ...]

    @property
    def keys(self) -> tuple[str, str]:
        return self.text, self.transliterated


def build_query_text_evidence(
    lines: Sequence[tuple[str, float]],
    min_confidence: float = DEFAULT_MIN_OCR_CONFIDENCE,
) -> QueryTextEvidence:
    """Build query evidence from raw ``(text, confidence)`` OCR lines.

    Lines below ``min_confidence`` are kept in ``raw_lines`` but excluded from
    the scoring text; the filter is deterministic and confidence is never used
    to scale a fuzzy score.
    """
    raw = tuple((str(text), float(score)) for text, score in lines)
    kept = " ".join(text for text, score in raw if score >= min_confidence)
    normalized = normalize_text(kept)
    return QueryTextEvidence(
        raw_lines=raw,
        text=normalized,
        transliterated=transliterated_key(kept),
        tokens=tuple(normalized.split(" ")) if normalized else (),
        vintage_years=tuple(extract_vintage_years(kept)),
    )


@dataclass(frozen=True)
class ReferenceOcrEvidence:
    """Normalized OCR evidence for one catalog reference image."""

    text: str
    transliterated: str
    vintage_years: tuple[str, ...]

    @property
    def keys(self) -> tuple[str, str]:
        return self.text, self.transliterated


def build_reference_ocr_evidence(lines: Sequence[tuple[str, float]], min_confidence: float = DEFAULT_MIN_OCR_CONFIDENCE) -> ReferenceOcrEvidence:
    kept = " ".join(text for text, score in lines if score >= min_confidence)
    return ReferenceOcrEvidence(
        text=normalize_text(kept),
        transliterated=transliterated_key(kept),
        vintage_years=tuple(extract_vintage_years(kept)),
    )


def _best_pair_score(query_text: str, query_translit: str, field: TextFieldKeys) -> float:
    """Best fuzzy score of a (usually short) field inside the query text."""
    scores: list[float] = []
    if field.normalized:
        scores.append(fuzz.partial_ratio(field.normalized, query_text) / 100.0)
        scores.append(fuzz.token_set_ratio(field.normalized, query_text) / 100.0)
    if field.transliterated:
        scores.append(fuzz.partial_ratio(field.transliterated, query_translit) / 100.0)
        scores.append(fuzz.token_set_ratio(field.transliterated, query_translit) / 100.0)
    return max(scores) if scores else 0.0


def _best_full_score(query_text: str, query_translit: str, field: TextFieldKeys) -> float:
    """Best whole-string fuzzy score between two texts of similar size."""
    scores: list[float] = []
    if field.normalized:
        scores.append(fuzz.ratio(field.normalized, query_text) / 100.0)
        scores.append(fuzz.token_set_ratio(field.normalized, query_text) / 100.0)
    if field.transliterated:
        scores.append(fuzz.ratio(field.transliterated, query_translit) / 100.0)
        scores.append(fuzz.token_set_ratio(field.transliterated, query_translit) / 100.0)
    return max(scores) if scores else 0.0


def compute_text_signals(
    query: QueryTextEvidence,
    candidate: CandidateTextIndex,
    reference_ocr: ReferenceOcrEvidence | None = None,
) -> dict[str, float | str]:
    """Compute the fixed signal set for one query-candidate pair.

    ``reference_ocr_score`` is 0.0 when no reference OCR cache entry exists
    for the candidate; the fused policy handles that as missing evidence
    rather than as a strong negative.
    """
    query_text, query_translit = query.keys
    return {
        "metadata_name_score": _best_full_score(query_text, query_translit, candidate.title),
        "winery_score": _best_pair_score(query_text, query_translit, candidate.winery),
        "grape_score": _best_pair_score(query_text, query_translit, candidate.grape),
        "region_score": _best_pair_score(query_text, query_translit, candidate.region),
        "metadata_text_score": _best_full_score(query_text, query_translit, candidate.metadata_text),
        "reference_ocr_score": (
            _best_reference_ocr_score(query_text, query_translit, reference_ocr)
            if reference_ocr is not None
            else 0.0
        ),
        "vintage_match": vintage_match_state(
            list(query.vintage_years),
            candidate.vintage_year,
        ),
    }


def _best_reference_ocr_score(query_text: str, query_translit: str, reference: ReferenceOcrEvidence) -> float:
    scores = [
        fuzz.ratio(query_text, reference.text) / 100.0,
        fuzz.token_set_ratio(query_text, reference.text) / 100.0,
        fuzz.ratio(query_translit, reference.transliterated) / 100.0,
        fuzz.token_set_ratio(query_translit, reference.transliterated) / 100.0,
    ]
    return max(scores)
