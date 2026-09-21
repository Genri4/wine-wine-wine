"""Text normalization and vintage extraction for OCR/metadata matching.

Pure deterministic functions shared by the OCR reranker and its tests.
The design contract from the milestone brief:

- lowercase, unicode normalization, whitespace and punctuation normalization;
- ё/е folding;
- digits (especially 4-digit vintages) are always preserved;
- Cyrillic and Latin are both safe; a transliterated comparison key reuses
  the existing project transliteration from :mod:`recognition.catalog`;
- normalization is NOT an aggressive fuzzy soup: it never removes digits and
  never reorders text;
- vintage extraction is a deterministic range filter, not "any 4-digit
  sequence is a vintage".
"""

from __future__ import annotations

import re
import unicodedata

from .catalog import transliterate_text

# Wine vintages: a reasonable historical range, not any 4-digit run.
MIN_VINTAGE_YEAR = 1900
MAX_VINTAGE_YEAR = 2030

_YEAR_PATTERN = re.compile(r"(?<!\d)(\d{4})(?!\d)")

# Kept characters after normalization: letters, digits, whitespace.
# Everything else (punctuation, symbols, OCR noise glyphs) becomes a space.
_KEEP_PATTERN = re.compile(r"[^0-9A-Za-zА-Яа-яЁё\s&]")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    """Return a normalized comparison form of ``value``.

    Digits are always preserved (4-digit years included); punctuation and
    symbols collapse to spaces; ё folds to е; runs of whitespace collapse.
    """
    if not isinstance(value, str):
        raise TypeError("text must be a string")
    text = unicodedata.normalize("NFKC", value)
    text = text.casefold().replace("ё", "е")
    text = _KEEP_PATTERN.sub(" ", text)
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def normalize_tokens(value: str) -> list[str]:
    """Return non-empty normalized tokens of ``value``."""
    return normalize_text(value).split(" ") if normalize_text(value) else []


def transliterated_key(value: str) -> str:
    """Return a Latin comparison key after project transliteration.

    Uses the same deterministic Cyrillic transliteration the catalog mapping
    uses, followed by :func:`normalize_text`, so Cyrillic and Latin spellings
    of the same word produce one key.
    """
    return normalize_text(transliterate_text(value))


def comparison_keys(value: str) -> tuple[str, str]:
    """Return ``(normalized_cyrillic_form, transliterated_form)`` for matching."""
    return normalize_text(value), transliterated_key(value)


def extract_vintage_years(value: str) -> list[str]:
    """Extract deterministic vintage-year candidates from text.

    A year must be a standalone 4-digit run inside the wine-vintage range
    [MIN_VINTAGE_YEAR, MAX_VINTAGE_YEAR]. Runs inside longer digit sequences
    (alcohol "12.5" is not affected, but "11490" is never a year) and runs
    outside the range are not vintages. Order-preserving, de-duplicated.
    """
    if not isinstance(value, str):
        raise TypeError("text must be a string")
    years: list[str] = []
    for match in _YEAR_PATTERN.finditer(value):
        year = match.group(1)
        if not (MIN_VINTAGE_YEAR <= int(year) <= MAX_VINTAGE_YEAR):
            continue
        if year not in years:
            years.append(year)
    return years


def vintage_match_state(
    query_years: list[str] | tuple[str, ...],
    candidate_year: str | None,
) -> str:
    """Classify vintage evidence between a query and one candidate.

    Returns one of:
    - ``exact_match``: candidate has a year and the query OCR also produced
      that year;
    - ``mismatch``: both sides have years and none of the query years equals
      the candidate year;
    - ``unknown``: the candidate year is missing, or the query OCR produced
      no usable vintage. Unknown never proves a mismatch.
    """
    if not candidate_year or not query_years:
        return "unknown"
    return "exact_match" if candidate_year in query_years else "mismatch"
