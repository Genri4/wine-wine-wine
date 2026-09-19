"""Small deterministic helpers for building a canonical catalog.

This module is intentionally independent from the recognition models. It only
normalizes filenames, classifies archive variants, and reduces exact duplicate
CSV rows without changing the source data.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import re
from pathlib import Path
import unicodedata
from typing import Callable, Iterable, Mapping, Sequence


IMAGE_SUFFIXES = frozenset(
    {
        ".avif",
        ".bmp",
        ".gif",
        ".heic",
        ".heif",
        ".jfif",
        ".jpeg",
        ".jpg",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    }
)
VARIANT_PREFIXES = ("thumbnail", "small", "medium", "large")
_HASH_SUFFIX = re.compile(r"_[0-9a-f]{10}$", re.IGNORECASE)
_SEPARATORS = re.compile(r"[\s_-]+")
_VARIANT_PREFIX = re.compile(r"^(?:thumbnail|small|medium|large)_", re.IGNORECASE)
_CYRILLIC_TRANSLITERATION = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "j",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "c",
    "ч": "ch",
    "ш": "sh",
    "щ": "shch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}


@dataclass(frozen=True)
class MediaRecord:
    """Metadata for one file listed in the media archive."""

    archive_path: str
    basename: str
    extension: str
    size: int
    crc: str
    variant: str
    comparison_key: str


@dataclass(frozen=True)
class CanonicalizationResult:
    """Canonical rows plus exact-duplicate counts."""

    rows: tuple[dict[str, str], ...]
    exact_duplicate_groups: int
    exact_duplicate_rows: int


@dataclass(frozen=True)
class MappingDecision:
    """Deterministic result for one photo name and its media candidates."""

    status: str
    method: str
    candidates: tuple[MediaRecord, ...]
    selected: MediaRecord | None = None
    requires_dimension_selection: bool = False


def archive_basename(archive_path: str) -> str:
    """Return a basename for either POSIX- or Windows-style archive paths."""

    return archive_path.replace("\\", "/").rsplit("/", 1)[-1]


def normalize_archive_path(archive_path: str) -> str:
    """Use stable POSIX separators for paths stored in generated artifacts."""

    return archive_path.replace("\\", "/")


def normalize_filename(value: str) -> str:
    """Return a deterministic comparison key for CSV and archive filenames.

    The original value is never changed by this function. The comparison key
    keeps the extension, case-folds Unicode text, converts whitespace, hyphen
    and underscore runs to one underscore, removes known Strapi resize prefixes,
    and removes a terminal ten-hex-character Strapi hash suffix.
    """

    if not isinstance(value, str):
        raise TypeError("filename must be a string")

    name = unicodedata.normalize("NFKC", value).replace("\\", "/")
    name = name.rsplit("/", 1)[-1]
    suffix = Path(name).suffix.casefold()
    stem = name[: -len(suffix)] if suffix else name
    stem = unicodedata.normalize("NFKC", stem).casefold()
    stem = _VARIANT_PREFIX.sub("", stem)
    stem = _HASH_SUFFIX.sub("", stem)
    stem = _SEPARATORS.sub("_", stem).strip("_")
    return stem + suffix


def transliterate_text(value: str) -> str:
    """Transliterate Russian Cyrillic for comparison-key generation only."""

    if not isinstance(value, str):
        raise TypeError("filename must be a string")
    normalized = unicodedata.normalize("NFKC", value)
    return "".join(
        _CYRILLIC_TRANSLITERATION.get(char.casefold(), char)
        for char in normalized
    )


def normalize_filename_transliterated(value: str) -> str:
    """Normalize a filename after deterministic Russian transliteration."""

    return normalize_filename(transliterate_text(value))


def normalize_filename_transliterated_punctuation(value: str) -> str:
    """Normalize transliterated filename stems with punctuation sanitization."""

    translated = transliterate_text(value).replace("\\", "/")
    name = translated.rsplit("/", 1)[-1]
    suffix = Path(name).suffix.casefold()
    stem = name[: -len(suffix)] if suffix else name
    stem = "".join(
        "_" if unicodedata.category(char)[0] in {"P", "S"} else char
        for char in stem
    )
    return normalize_filename(stem + suffix)


def detect_variant(basename: str, extension: str | None = None) -> str:
    """Classify a basename as original, a known resize, or unknown."""

    normalized_name = unicodedata.normalize("NFKC", basename).replace("\\", "/")
    normalized_name = normalized_name.rsplit("/", 1)[-1]
    stem = Path(normalized_name).stem.casefold()
    for prefix in VARIANT_PREFIXES:
        if stem.startswith(prefix + "_"):
            return prefix

    suffix = (extension or Path(normalized_name).suffix).casefold()
    return "original" if suffix in IMAGE_SUFFIXES else "unknown"


def canonicalize_rows(
    rows: Iterable[Mapping[str, str]],
    fieldnames: Sequence[str],
    *,
    slug_field: str = "Slug",
) -> CanonicalizationResult:
    """Remove only exact duplicate rows and keep one row per slug.

    If a slug has multiple non-identical rows, the function fails rather than
    guessing which row is canonical.
    """

    materialized = [dict(row) for row in rows]
    row_counts = Counter(
        tuple(row.get(field, "") for field in fieldnames) for row in materialized
    )
    exact_duplicate_groups = sum(count > 1 for count in row_counts.values())
    exact_duplicate_rows = sum(count - 1 for count in row_counts.values() if count > 1)

    seen_rows: set[tuple[str, ...]] = set()
    deduplicated: list[dict[str, str]] = []
    for row in materialized:
        row_key = tuple(row.get(field, "") for field in fieldnames)
        if row_key in seen_rows:
            continue
        seen_rows.add(row_key)
        deduplicated.append(row)

    by_slug: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in deduplicated:
        slug = row.get(slug_field, "").strip()
        if not slug:
            raise ValueError("CSV contains an empty Slug; canonical row is undefined")
        by_slug[slug].append(row)

    conflicts = {slug: values for slug, values in by_slug.items() if len(values) > 1}
    if conflicts:
        examples = ", ".join(sorted(conflicts)[:5])
        raise ValueError(
            "A slug has multiple non-identical rows; refusing to choose a row: "
            + examples
        )

    canonical = tuple(values[0] for values in by_slug.values())
    return CanonicalizationResult(canonical, exact_duplicate_groups, exact_duplicate_rows)


def shared_photo_names(
    rows: Iterable[Mapping[str, str]],
    *,
    photo_field: str = "Название фото",
    slug_field: str = "Slug",
) -> dict[str, tuple[str, ...]]:
    """Return photo names referenced by more than one distinct slug."""

    by_photo: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        photo = row.get(photo_field, "").strip()
        slug = row.get(slug_field, "").strip()
        if photo and slug:
            by_photo[photo].add(slug)
    return {
        photo: tuple(sorted(slugs))
        for photo, slugs in sorted(by_photo.items())
        if len(slugs) > 1
    }


def resolve_media_candidates(
    photo_name: str,
    media_by_key: Mapping[str, Sequence[MediaRecord]],
) -> MappingDecision:
    """Resolve strict candidates without fuzzy matching or arbitrary selection."""

    candidates = tuple(
        sorted(
            media_by_key.get(normalize_filename(photo_name), ()),
            key=lambda item: item.archive_path,
        )
    )
    return _resolve_candidate_group(
        photo_name,
        candidates,
        method_prefix="strapi_normalized",
        exact_method=True,
    )


def resolve_media_candidates_cascade(
    photo_name: str,
    strict_media_by_key: Mapping[str, Sequence[MediaRecord]],
    transliterated_media_by_key: Mapping[str, Sequence[MediaRecord]],
    punctuation_media_by_key: Mapping[str, Sequence[MediaRecord]],
) -> MappingDecision:
    """Resolve candidates through strict, transliteration, then punctuation keys."""

    stages: tuple[tuple[str, Callable[[str], str], Mapping[str, Sequence[MediaRecord]]], ...] = (
        ("strapi_normalized", normalize_filename, strict_media_by_key),
        ("transliterated_normalized", normalize_filename_transliterated, transliterated_media_by_key),
        (
            "transliterated_punctuation_normalized",
            normalize_filename_transliterated_punctuation,
            punctuation_media_by_key,
        ),
    )
    for method_prefix, key_function, media_by_key in stages:
        candidates = tuple(
            sorted(media_by_key.get(key_function(photo_name), ()), key=lambda item: item.archive_path)
        )
        if candidates:
            return _resolve_candidate_group(
                photo_name,
                candidates,
                method_prefix=method_prefix,
                exact_method=method_prefix == "strapi_normalized",
            )
    if not candidates:
        return MappingDecision("unmatched", "no_deterministic_media_candidate", ())


def _resolve_candidate_group(
    photo_name: str,
    candidates: Sequence[MediaRecord],
    *,
    method_prefix: str,
    exact_method: bool,
) -> MappingDecision:
    """Resolve one deterministic key's candidates, including CRC-identical originals."""

    original_name = archive_basename(photo_name)
    originals = tuple(item for item in candidates if item.variant == "original")
    exact = tuple(item for item in originals if item.basename == original_name)
    if len(originals) == 1:
        if exact_method:
            method = "exact_basename_original" if len(exact) == 1 else f"{method_prefix}_original"
        else:
            method = f"{method_prefix}_original"
        return MappingDecision("matched", method, candidates, originals[0])
    if len(originals) > 1:
        signatures = {(item.size, item.crc) for item in originals}
        if all(item.crc for item in originals) and len(signatures) == 1:
            selected = min(originals, key=lambda item: item.archive_path)
            return MappingDecision(
                "matched",
                f"{method_prefix}_crc_identical_original",
                candidates,
                selected,
            )
        return MappingDecision("ambiguous", f"{method_prefix}_multiple_originals", candidates)
    if len(candidates) == 1:
        return MappingDecision("matched", f"{method_prefix}_single_candidate", candidates, candidates[0])
    return MappingDecision(
        "ambiguous",
        f"{method_prefix}_variant_requires_dimensions",
        candidates,
        requires_dimension_selection=True,
    )
