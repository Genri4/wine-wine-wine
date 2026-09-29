"""Vintage disambiguation: targeted year OCR inside the frozen Top-5.

Pipeline contract (milestone brief):

- Image retrieval (siglip2_so400m_384) and the production OCR reranker are
  frozen; this module only adds a diagnostic/targeted year stage.
- Candidate years come from catalog evidence with explicit provenance:
  catalog_metadata / product_title / reference_ocr / multiple_sources_agree.
  Conflicting sources resolve to "unknown"; years are never invented.
- Detector-box retry: re-recognize the query's own detected text boxes at
  1x/2x/4x upscale with the same frozen eslav recognizer, then extract
  year tokens with a small, logged numeric-normalization layer
  (I/l->1, O->0, S->5, B->8, Z->2 only inside year-like tokens).
- Conservative reranking: year evidence may only swap members of the SAME
  family with different known years; unrelated candidates are untouched.
  No target identity or ground truth is visible to any inference path.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .text_normalization import extract_vintage_years

YEAR_TOKEN_RE = re.compile(r"^(19|20)\d{2}$")

# Ambiguous glyph correction applied ONLY to tokens that look like a year
# once the substitutions are applied (e.g. 202I -> 2021). Every applied
# correction is logged by the caller.
_GLYPH_CORRECTIONS = str.maketrans({"I": "1", "l": "1", "|": "1", "O": "0", "o": "0", "S": "5", "B": "8", "Z": "2", "G": "6", "b": "6", "g": "9", "q": "9"})


def normalize_year_token(token: str) -> tuple[str | None, str | None]:
    """Return (year, applied_correction) for a year-like token.

    A token qualifies if it is already a 4-digit year, or becomes one after
    ambiguous-glyph substitution. Anything else returns (None, None).
    """

    token = token.strip().strip(".,;:()")
    if YEAR_TOKEN_RE.match(token):
        return token, None
    candidate = token.translate(_GLYPH_CORRECTIONS)
    if YEAR_TOKEN_RE.match(candidate):
        return candidate, token
    return None, None


def extract_years_with_corrections(
    lines: Sequence[Mapping[str, Any]], min_confidence: float = 0.5
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """Extract candidate years from OCR lines.

    Returns (years_by_value, corrections) where years_by_value maps a year
    string to the list of observations (source line, confidence, source_box
    index) and corrections lists every glyph fix applied.
    """

    years: dict[str, list[dict[str, Any]]] = {}
    corrections: list[str] = []
    for index, line in enumerate(lines):
        confidence = float(line.get("confidence", 0.0))
        if confidence < min_confidence:
            continue
        for token in str(line.get("text", "")).split():
            year, correction = normalize_year_token(token)
            if year is None:
                continue
            if correction:
                corrections.append(f"{token}->{year}")
            years.setdefault(year, []).append(
                {
                    "line_index": index,
                    "confidence": round(confidence, 4),
                    "raw_token": token,
                }
            )
    return years, corrections


# ----------------------------------------------------------------------
# Candidate year evidence with provenance (Part 4)
# ----------------------------------------------------------------------

def candidate_year_evidence(
    slug: str,
    catalog_row: Mapping[str, str],
    reference_ocr_lines: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Canonical candidate-year evidence with explicit provenance.

    Priority: multiple agreeing sources > catalog_metadata/product_title >
    reference_ocr. Conflicting years resolve to unknown/conflict; nothing
    is invented.
    """

    sources: dict[str, set[str]] = {}
    title = catalog_row.get("title", "") or ""
    product_name = catalog_row.get("product_name", "") or ""
    for field_name, text in (("product_title", title), ("catalog_metadata", product_name)):
        for year in extract_vintage_years(text):
            sources.setdefault(year, set()).add(field_name)
            sources[year].add("catalog_metadata" if field_name == "catalog_metadata" else field_name)
    ocr_years: set[str] = set()
    if reference_ocr_lines:
        extracted, _ = extract_years_with_corrections(reference_ocr_lines)
        ocr_years = set(extracted)
        for year in ocr_years:
            sources.setdefault(year, set()).add("reference_ocr")

    if not sources:
        return {"slug": slug, "year": None, "provenance": "unknown", "sources": {}}
    if len(sources) == 1:
        year = next(iter(sources))
        provenance_set = sources[year]
        if len(provenance_set) > 1:
            provenance = "multiple_sources_agree"
        else:
            provenance = next(iter(provenance_set))
        return {"slug": slug, "year": year, "provenance": provenance, "sources": {year: sorted(provenance_set)}}
    # Multiple distinct years across/inside sources: conflict unless one
    # year is supported by 2+ distinct sources while others are single-source.
    supported = {year: len(set_of_sources) for year, set_of_sources in sources.items()}
    best_count = max(supported.values())
    winners = [year for year, count in supported.items() if count == best_count and best_count >= 2]
    if len(winners) == 1:
        year = winners[0]
        provenance_set = sources[year]
        provenance = "multiple_sources_agree" if len(provenance_set) > 1 else next(iter(provenance_set))
        return {"slug": slug, "year": year, "provenance": provenance, "sources": {year: sorted(provenance_set)}, "conflict": True}
    return {
        "slug": slug,
        "year": None,
        "provenance": "unknown_conflict",
        "sources": {year: sorted(set_of_sources) for year, set_of_sources in sources.items()},
    }


# ----------------------------------------------------------------------
# Query year decision (Part 7)
# ----------------------------------------------------------------------

@dataclass
class QueryYearEvidence:
    years: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    corrections: list[str] = field(default_factory=list)
    decided_year: str | None = None
    decision: str = "no_year"  # no_year | unique | ambiguous | family_consistent
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "years": self.years,
            "corrections": self.corrections,
            "decided_year": self.decided_year,
            "decision": self.decision,
            "confidence": round(self.confidence, 4),
        }


def decide_query_year(
    years_by_value: Mapping[str, list[Mapping[str, Any]]],
    candidate_years: Mapping[str, str | None],
    min_confidence: float = 0.6,
) -> QueryYearEvidence:
    """Pick the query year using OCR confidences and family-consistency.

    Rule (deterministic, no target knowledge): among detected years, prefer
    ones that appear with higher max confidence; ties and multi-year reads
    stay ambiguous. A year that matches at least one candidate's known year
    (catalog-side information, not ground truth) may break a confidence tie
    only when its own confidence clears ``min_confidence``.
    """

    evidence = QueryYearEvidence(years=dict(years_by_value))
    if not years_by_value:
        return evidence
    best_confidence = {
        year: max(observation["confidence"] for observation in observations)
        for year, observations in years_by_value.items()
    }
    ordered = sorted(best_confidence.items(), key=lambda item: (-item[1], item[0]))
    top_year, top_confidence = ordered[0]
    evidence.confidence = top_confidence
    known_years = {year for year in candidate_years.values() if year}
    if len(ordered) == 1:
        evidence.decided_year = top_year
        evidence.decision = "unique"
        return evidence
    if top_confidence >= min_confidence and ordered[1][1] < top_confidence:
        evidence.decided_year = top_year
        evidence.decision = "unique"
        return evidence
    consistent = [year for year, confidence in ordered if confidence >= min_confidence and year in known_years]
    if len(set(consistent)) == 1:
        evidence.decided_year = consistent[0]
        evidence.decision = "family_consistent"
        return evidence
    evidence.decision = "ambiguous"
    return evidence


# ----------------------------------------------------------------------
# Conservative family-safe rerank (Part 11/23)
# ----------------------------------------------------------------------

def conservative_vintage_rerank(
    top5: Sequence[str],
    image_scores: Sequence[float],
    family_by_slug: Mapping[str, str],
    candidate_years: Mapping[str, str | None],
    query_year: str | None,
    query_year_confidence: float,
    min_confidence: float = 0.6,
) -> dict[str, Any]:
    """Family-safe vintage swap.

    Allowed only when: a confident query year exists; the Top-1 and some
    other candidate share one family with DIFFERENT known years; the query
    year exactly matches the non-top1 member's year and not the top1's.
    The permutation is then a single swap of those two members; everything
    else stays in place. Otherwise the ranking is returned untouched.
    """

    order = list(range(len(top5)))
    result = {
        "order": order,
        "action": "no_action",
        "swapped_with": None,
        "reason": "",
    }
    if query_year is None or query_year_confidence < min_confidence:
        result["reason"] = "no_confident_query_year"
        return result
    top1_slug = top5[0]
    top1_family = family_by_slug.get(top1_slug)
    if not top1_family:
        result["reason"] = "top1_family_unknown"
        return result
    top1_year = candidate_years.get(top1_slug)
    if top1_year == query_year:
        result["reason"] = "top1_year_already_matches"
        return result
    for position in range(1, len(top5)):
        slug = top5[position]
        if family_by_slug.get(slug) != top1_family:
            continue
        candidate_year = candidate_years.get(slug)
        if candidate_year and candidate_year == query_year and top1_year and top1_year != query_year:
            order[0], order[position] = order[position], order[0]
            result.update(action="family_swap", swapped_with=slug, reason=f"query_year={query_year} matches rank{position} member")
            return result
    result["reason"] = "no_family_member_matches_query_year"
    return result


# ----------------------------------------------------------------------
# Challenge slice (Part 3)
# ----------------------------------------------------------------------

def build_vintage_challenge(
    benchmark: str,
    manifest_rows: Sequence[Mapping[str, str]],
    baseline_rows: Mapping[str, Mapping[str, str]],
    family_by_slug: Mapping[str, str],
    family_types: Mapping[str, str],
    candidate_years: Mapping[str, str | None],
) -> list[dict[str, Any]]:
    """Queries where the target is in a vintage family, sits in the Top-5,
    and at least one competing same-family member with a different known
    year is also in the Top-5."""

    challenge: list[dict[str, Any]] = []
    for row in manifest_rows:
        query_id = row["query_id"]
        target = row["target_slug"]
        base_row = baseline_rows.get(query_id)
        if base_row is None:
            continue
        family_id = row.get("family_id") or row.get("target_family_id") or ""
        if not family_id or family_types.get(family_id) != "vintage":
            continue
        top5 = json.loads(base_row["top5_slugs"])
        if target not in top5:
            continue
        target_year = candidate_years.get(target)
        if not target_year:
            continue
        competitors = []
        for slug in top5:
            if slug == target:
                continue
            if family_by_slug.get(slug) != family_by_slug.get(target):
                continue
            competitor_year = candidate_years.get(slug)
            competitors.append({"slug": slug, "year": competitor_year})
        different = [entry for entry in competitors if entry["year"] and entry["year"] != target_year]
        if not different:
            continue
        challenge.append(
            {
                "query_id": query_id,
                "source_benchmark": benchmark,
                "target_slug": target,
                "family_id": family_id,
                "target_year": target_year,
                "target_in_top5_rank": top5.index(target) + 1,
                "top5_slugs": top5,
                "top5_scores": json.loads(base_row["top5_scores"]),
                "same_family_competitors": competitors,
                "distinct_competitor_years": sorted({entry["year"] for entry in different}),
                "baseline_correct_top1": base_row["correct_top1"] == "True",
                "baseline_top1": base_row["predicted_slug"],
            }
        )
    return challenge


def oracle_ceiling(
    challenge: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Perfect-query-year oracle over the challenge slice (Part 5).

    The oracle knows the target year only through the challenge definition
    (target_year is catalog-derived evidence, and the oracle is explicitly a
    diagnostic, not a policy). It picks the same-family member whose known
    year matches; ambiguity (multiple members sharing that year) counts as
    unresolved.
    """

    total = len(challenge)
    baseline_correct = sum(1 for row in challenge if row["baseline_correct_top1"])
    resolvable = 0
    resolvable_wrong = 0
    ambiguous = 0
    year_cannot_distinguish = 0
    for row in challenge:
        target_year = row["target_year"]
        # The full member_years map (built by the runner from the whole
        # family) is authoritative: a family member that is not the target
        # and not in the Top-5 still tells us whether the years distinguish
        # the family.
        member_years = dict(row.get("member_years", {}))
        if not member_years:
            member_years = {
                slug: row.get("member_years", {}).get(slug)
                for slug in [row["target_slug"]] + [entry["slug"] for entry in row["same_family_competitors"]]
            }
        distinct_years = {year for year in member_years.values() if year}
        if len(distinct_years) < 2:
            year_cannot_distinguish += 1
            continue
        matches = [slug for slug, year in member_years.items() if year == target_year]
        if len(matches) != 1:
            ambiguous += 1
            continue
        resolvable += 1
        if not row["baseline_correct_top1"]:
            resolvable_wrong += 1
    # A perfect query year answers every resolvable query correctly and
    # never breaks a correct baseline one.
    oracle_top1 = resolvable + (baseline_correct - (resolvable - resolvable_wrong))
    oracle_top1_share = round(oracle_top1 / total, 4) if total else None
    return {
        "challenge_queries": total,
        "baseline_correct_top1": baseline_correct,
        "resolvable_by_year": resolvable,
        "resolvable_and_baseline_wrong": resolvable_wrong,
        "ambiguous_multiple_members_same_year": ambiguous,
        "year_cannot_distinguish": year_cannot_distinguish,
        "oracle_top1": oracle_top1,
        "oracle_top1_share": oracle_top1_share,
        "oracle_share_note": "share over challenge queries; a perfect year judge answers all resolvable queries and keeps the rest at the baseline state",
        "rescued_potential": resolvable_wrong,
        "note": "diagnostic ceiling only; oracle picks the unique family member whose known year equals the (catalog-derived) target year",
    }


# ----------------------------------------------------------------------
# Challenge manifest I/O
# ----------------------------------------------------------------------

def write_challenge_csv(path: Path, challenge: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "query_id", "source_benchmark", "target_slug", "family_id", "target_year",
        "target_in_top5_rank", "top5_slugs", "top5_scores", "same_family_competitors",
        "distinct_competitor_years", "baseline_correct_top1", "baseline_top1",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in challenge:
            serialized = dict(row)
            serialized["top5_slugs"] = json.dumps(row["top5_slugs"], ensure_ascii=False)
            serialized["top5_scores"] = json.dumps(row["top5_scores"], ensure_ascii=False)
            serialized["same_family_competitors"] = json.dumps(row["same_family_competitors"], ensure_ascii=False)
            serialized["distinct_competitor_years"] = ",".join(row["distinct_competitor_years"])
            writer.writerow(serialized)
