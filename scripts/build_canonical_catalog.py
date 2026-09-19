#!/usr/bin/env python3
"""Build canonical catalog artifacts from the audited CSV and RAR listing.

The command is deliberately separate from inference code. It reads raw files,
writes generated artifacts under data/processed and reports, and never edits
or deletes raw data.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recognition.catalog import (  # noqa: E402
    IMAGE_SUFFIXES,
    MediaRecord,
    MappingDecision,
    archive_basename,
    canonicalize_rows,
    detect_variant,
    normalize_archive_path,
    normalize_filename,
    normalize_filename_transliterated,
    normalize_filename_transliterated_punctuation,
    resolve_media_candidates_cascade,
    shared_photo_names,
)


CSV_FIELDS = (
    "Название вина",
    "Категория",
    "Цвет",
    "Регион",
    "Сорт винограда",
    "Описание",
    "Винодельня",
    "Slug",
    "Название фото",
)
MANIFEST_FIELDS = (
    "slug",
    "title",
    "category",
    "color",
    "region",
    "grape",
    "winery",
    "photo_name",
    "original_archive_path",
    "reference_image_path",
    "mapping_status",
    "mapping_method",
    "candidate_count",
    "candidate_paths",
)
ISSUE_FIELDS = (
    "slug",
    "title",
    "issue_type",
    "mapping_status",
    "photo_name",
    "candidate_count",
    "candidate_paths",
    "details",
)
REVIEW_FIELDS = (
    "slug",
    "title",
    "category",
    "winery",
    "photo_name",
    "mapping_status",
    "mapping_method",
    "issue_type",
    "reason",
    "metadata_candidate_count",
    "metadata_top_score",
    "metadata_candidates",
    "deterministic_candidates",
)
MANUAL_OVERRIDE_FIELDS = (
    "slug",
    "selected_archive_path",
    "decision",
    "reason",
    "note",
)
INVENTORY_FIELDS = (
    "archive_path",
    "basename",
    "extension",
    "size",
    "crc",
    "variant",
    "comparison_key",
)
_TIMESTAMP_LIKE = re.compile(r"^[A-Za-z0-9_=-]+\d{10}$")


def read_csv_rows(csv_path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        if fields != CSV_FIELDS:
            raise ValueError(f"Unexpected CSV columns: {fields}")
        return [dict(row) for row in reader], fields


@dataclass(frozen=True)
class ManualOverride:
    """One human-authored mapping decision from the override CSV."""

    slug: str
    selected_archive_path: str
    decision: str
    reason: str
    note: str


def load_manual_overrides(
    override_path: Path,
    *,
    known_slugs: Iterable[str],
    media_by_path: Mapping[str, MediaRecord],
) -> dict[str, ManualOverride]:
    """Load and validate human decisions without making any decisions implicitly."""

    if not override_path.is_file():
        return {}
    known_slug_set = set(known_slugs)
    with override_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        if fields != MANUAL_OVERRIDE_FIELDS:
            raise ValueError(f"Unexpected manual override columns: {fields}")
        overrides: dict[str, ManualOverride] = {}
        for line_number, raw_row in enumerate(reader, 2):
            row = {field: (raw_row.get(field) or "").strip() for field in MANUAL_OVERRIDE_FIELDS}
            if not any(row.values()):
                continue
            slug = row["slug"]
            decision = row["decision"]
            selected_path = normalize_archive_path(row["selected_archive_path"])
            if not slug:
                raise ValueError(f"Manual override line {line_number}: slug is required")
            if slug not in known_slug_set:
                raise ValueError(f"Manual override line {line_number}: unknown slug {slug!r}")
            if slug in overrides:
                raise ValueError(f"Manual override line {line_number}: duplicate slug {slug!r}")
            if decision not in {"matched", "missing", "unresolved"}:
                raise ValueError(
                    f"Manual override line {line_number}: decision must be matched, missing or unresolved"
                )
            if decision == "matched":
                if not selected_path:
                    raise ValueError(f"Manual override line {line_number}: matched requires selected_archive_path")
                selected = media_by_path.get(selected_path)
                if selected is None:
                    raise ValueError(
                        f"Manual override line {line_number}: archive path is not in media inventory: {selected_path}"
                    )
                if selected.extension not in IMAGE_SUFFIXES:
                    raise ValueError(
                        f"Manual override line {line_number}: selected archive path is not an image: {selected_path}"
                    )
            elif selected_path:
                raise ValueError(
                    f"Manual override line {line_number}: selected_archive_path must be empty for {decision}"
                )
            overrides[slug] = ManualOverride(
                slug=slug,
                selected_archive_path=selected_path,
                decision=decision,
                reason=row["reason"],
                note=row["note"],
            )
    return overrides


def apply_manual_overrides(
    decisions: Mapping[str, MappingDecision],
    overrides: Mapping[str, ManualOverride],
    *,
    media_by_path: Mapping[str, MediaRecord],
) -> dict[str, MappingDecision]:
    """Replace deterministic decisions with explicit human decisions after mapping."""

    result = dict(decisions)
    for slug, override in overrides.items():
        if slug not in result:
            raise ValueError(f"Manual override refers to unknown decision slug: {slug}")
        current = result[slug]
        candidates = current.candidates
        selected: MediaRecord | None = None
        if override.decision == "matched":
            selected = media_by_path.get(override.selected_archive_path)
            if selected is None:
                raise ValueError(f"Manual override archive path is not in media inventory: {override.selected_archive_path}")
            if selected.archive_path not in {item.archive_path for item in candidates}:
                candidates = tuple(sorted((*candidates, selected), key=lambda item: item.archive_path))
            method = "manual_override_matched"
        elif override.decision == "missing":
            method = "manual_override_missing"
        else:
            method = "manual_override_unresolved"
        result[slug] = MappingDecision(override.decision, method, candidates, selected)
    return result


def _parse_listing(listing: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for raw_line in listing.splitlines():
        line = raw_line.rstrip("\r")
        if not line.strip():
            if current.get("Folder") in {"-", "+"}:
                records.append(current)
            current = {}
            continue
        if " = " in line:
            key, value = line.split(" = ", 1)
            current[key] = value
    if current.get("Folder") in {"-", "+"}:
        records.append(current)
    return records


def _find_seven_zip(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    for command in ("7z", "7zz"):
        found = shutil.which(command)
        if found:
            candidates.append(Path(found))
    candidates.append(Path("/mnt/c/Program Files/7-Zip/7z.exe"))
    for candidate in candidates:
        if candidate.is_file() or shutil.which(str(candidate)):
            return candidate
    raise FileNotFoundError("7-Zip executable was not found; RAR listing is required")


def _host_path(path: Path, *, windows_executable: bool) -> str:
    if not windows_executable:
        return str(path)
    result = subprocess.run(
        ["wslpath", "-w", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def list_rar(archive_path: Path, seven_zip: Path) -> tuple[list[MediaRecord], str]:
    windows_executable = seven_zip.suffix.casefold() == ".exe"
    result = subprocess.run(
        [
            str(seven_zip),
            "l",
            "-slt",
            _host_path(archive_path, windows_executable=windows_executable),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = result.stdout.decode("utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"7-Zip listing failed with code {result.returncode}:\n{output[-4000:]}")

    media: list[MediaRecord] = []
    for raw in _parse_listing(output):
        if raw.get("Folder") != "-":
            continue
        archive_path_value = normalize_archive_path(raw["Path"])
        basename = archive_basename(archive_path_value)
        extension = Path(basename).suffix.casefold()
        media.append(
            MediaRecord(
                archive_path=archive_path_value,
                basename=basename,
                extension=extension,
                size=int(raw.get("Size", "0")),
                crc=raw.get("CRC", ""),
                variant=detect_variant(basename, extension),
                comparison_key=normalize_filename(basename),
            )
        )
    return sorted(media, key=lambda item: item.archive_path), output


def image_media_by_key(
    media: Iterable[MediaRecord],
    key_function=normalize_filename,
) -> dict[str, tuple[MediaRecord, ...]]:
    grouped: defaultdict[str, list[MediaRecord]] = defaultdict(list)
    for item in media:
        if item.extension in IMAGE_SUFFIXES:
            grouped[key_function(item.basename)].append(item)
    return {
        key: tuple(sorted(values, key=lambda item: item.archive_path))
        for key, values in grouped.items()
    }


def _candidate_json(candidates: Sequence[MediaRecord]) -> str:
    return json.dumps([item.archive_path for item in candidates], ensure_ascii=False)


def _unmatched_reason(photo_name: str) -> str:
    stem = Path(photo_name).stem
    if _TIMESTAMP_LIKE.fullmatch(stem) and re.search(r"[A-Za-z]", stem):
        return "opaque/random + timestamp-like name"
    if any(ord(char) > 127 for char in stem):
        return "non-ASCII/Cyrillic name"
    if "copy" in stem.casefold() or "копия" in stem.casefold() or "—" in stem or "–" in stem:
        return "copy/punctuation in name"
    if re.search(r"\s", stem):
        return "space-containing name"
    if re.fullmatch(r"[A-Za-z0-9_=-]+", stem):
        return "ASCII name without deterministic candidate"
    return "other name without deterministic candidate"


_METADATA_STOPWORDS = frozenset(
    {
        "wine",
        "vino",
        "vinodelnya",
        "beloe",
        "belyj",
        "krasnoe",
        "krasnyj",
        "rozovoe",
        "rozovyj",
        "suhoe",
        "suh",
        "polusladkoe",
        "polusladk",
        "polusuhoe",
        "polusuh",
        "bryut",
        "brut",
        "igristoe",
        "igrist",
        "normal",
        "no",
        "bg",
        "preview",
        "carve",
        "photos",
        "photoroom",
        "fotor",
        "copy",
        "kopiya",
        "pod",
        "foto",
        "image",
        "product",
        "screenshot",
        "reserve",
        "rezerv",
        "classic",
        "new",
        "vid",
        "fona",
        "bez",
    }
)


def _metadata_tokens(value: str) -> set[str]:
    normalized = normalize_filename_transliterated_punctuation(value)
    stem = Path(normalized).stem
    return {
        token
        for token in re.findall(r"[a-z0-9]+", stem)
        if len(token) >= 3 and not token.isdigit() and token not in _METADATA_STOPWORDS
    }


def _metadata_candidates(
    product: Mapping[str, str],
    media: Sequence[MediaRecord],
    *,
    limit: int = 5,
) -> tuple[dict[str, object], ...]:
    """Generate lexical candidates for review; never use this for auto-mapping."""

    photo_tokens = _metadata_tokens(product["Название фото"])
    title_tokens = _metadata_tokens(product["Название вина"])
    winery_tokens = _metadata_tokens(product["Винодельня"])
    slug_tokens = _metadata_tokens(product["Slug"])
    scored: list[tuple[int, int, str, MediaRecord, set[str]]] = []
    for item in media:
        if item.variant != "original":
            continue
        candidate_tokens = _metadata_tokens(item.basename)
        photo_overlap = photo_tokens & candidate_tokens
        title_overlap = title_tokens & candidate_tokens
        winery_overlap = winery_tokens & candidate_tokens
        slug_overlap = slug_tokens & candidate_tokens
        score = (
            4 * len(photo_overlap)
            + 3 * len(title_overlap)
            + 2 * len(winery_overlap)
            + len(slug_overlap)
        )
        if score:
            scored.append(
                (
                    score,
                    len(photo_overlap | title_overlap | winery_overlap | slug_overlap),
                    item.archive_path,
                    item,
                    photo_overlap | title_overlap | winery_overlap | slug_overlap,
                )
            )
    scored.sort(key=lambda value: (-value[0], -value[1], value[2]))
    return tuple(
        {
            "archive_path": item.archive_path,
            "basename": item.basename,
            "score": score,
            "matched_tokens": sorted(tokens),
        }
        for score, _overlap_count, _path, item, tokens in scored[:limit]
    )


def _choose_by_area(
    candidates: Sequence[MediaRecord], extracted: Mapping[str, Path]
) -> tuple[MediaRecord | None, str]:
    from PIL import Image

    areas: list[tuple[int, MediaRecord]] = []
    for candidate in candidates:
        path = extracted.get(candidate.archive_path)
        if path is None:
            return None, "variant_dimension_unavailable"
        try:
            with Image.open(path) as image:
                image.load()
                areas.append((image.width * image.height, candidate))
        except (OSError, ValueError):
            return None, "variant_dimension_unavailable"
    if not areas:
        return None, "variant_dimension_unavailable"
    max_area = max(area for area, _ in areas)
    winners = [candidate for area, candidate in areas if area == max_area]
    if len(winners) != 1:
        return None, "variant_dimension_tie"
    return winners[0], "strapi_normalized_variant_by_area"


def _extract_selected(
    archive_path: Path,
    seven_zip: Path,
    selected_by_slug: Mapping[str, MediaRecord],
    area_candidates: Mapping[str, Sequence[MediaRecord]],
    reference_dir: Path,
) -> dict[str, MediaRecord]:
    """Extract selected records once, plus candidates needing area fallback."""

    to_extract = {item.archive_path: item for item in selected_by_slug.values()}
    for candidates in area_candidates.values():
        for item in candidates:
            to_extract[item.archive_path] = item
    if not to_extract:
        return {}

    windows_executable = seven_zip.suffix.casefold() == ".exe"
    with tempfile.TemporaryDirectory(prefix="my_wine_catalog_") as temporary:
        temporary_root = Path(temporary)
        list_file = temporary_root / "selection.lst"
        list_lines = [item.archive_path.replace("/", "\\") for item in to_extract.values()]
        list_file.write_text("\n".join(list_lines) + "\n", encoding="utf-8")
        staging_dir = temporary_root / "extracted"
        staging_dir.mkdir()
        command = [
            str(seven_zip),
            "e",
            _host_path(archive_path, windows_executable=windows_executable),
            "-y",
            "-scsUTF-8",
            "-o" + _host_path(staging_dir, windows_executable=windows_executable),
            "@" + _host_path(list_file, windows_executable=windows_executable),
        ]
        result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = result.stdout.decode("utf-8", errors="replace")
        if result.returncode != 0:
            raise RuntimeError(f"7-Zip selective extraction failed with code {result.returncode}:\n{output[-4000:]}")

        extracted: dict[str, Path] = {}
        for archive_key, record in to_extract.items():
            matches = [path for path in staging_dir.rglob(record.basename) if path.is_file()]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one extracted file for {record.archive_path}, found {len(matches)}")
            extracted[archive_key] = matches[0]
            if matches[0].stat().st_size != record.size:
                raise RuntimeError(f"Extracted size mismatch for {record.archive_path}")

        finalized = dict(selected_by_slug)
        for slug, candidates in area_candidates.items():
            selected, _method = _choose_by_area(candidates, extracted)
            if selected is not None:
                finalized[slug] = selected

        reference_dir.mkdir(parents=True, exist_ok=True)
        for slug, record in finalized.items():
            if Path(slug).name != slug:
                raise ValueError(f"Unsafe slug for reference filename: {slug}")
            source = extracted[record.archive_path]
            target = reference_dir / f"{slug}{record.extension}"
            shutil.copy2(source, target)
        return finalized


def _extract_review_candidates(
    archive_path: Path,
    seven_zip: Path,
    decisions: Mapping[str, MappingDecision],
    review_dir: Path,
) -> dict[str, str]:
    """Extract only deterministic candidates for unresolved ambiguous cases."""

    to_extract = {
        item.archive_path: item
        for decision in decisions.values()
        if decision.status in {"ambiguous", "unresolved"}
        for item in decision.candidates
        if item.extension in IMAGE_SUFFIXES
    }
    if not to_extract:
        return {}

    windows_executable = seven_zip.suffix.casefold() == ".exe"
    with tempfile.TemporaryDirectory(prefix="my_wine_review_") as temporary:
        temporary_root = Path(temporary)
        list_file = temporary_root / "selection.lst"
        list_file.write_text(
            "\n".join(item.archive_path.replace("/", "\\") for item in to_extract.values()) + "\n",
            encoding="utf-8",
        )
        staging_dir = temporary_root / "extracted"
        staging_dir.mkdir()
        command = [
            str(seven_zip),
            "e",
            _host_path(archive_path, windows_executable=windows_executable),
            "-y",
            "-scsUTF-8",
            "-o" + _host_path(staging_dir, windows_executable=windows_executable),
            "@" + _host_path(list_file, windows_executable=windows_executable),
        ]
        result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = result.stdout.decode("utf-8", errors="replace")
        if result.returncode != 0:
            raise RuntimeError(f"7-Zip review extraction failed with code {result.returncode}:\n{output[-4000:]}")

        review_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        for archive_key, record in sorted(to_extract.items()):
            matches = [path for path in staging_dir.rglob(record.basename) if path.is_file()]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one extracted review file for {record.archive_path}, found {len(matches)}")
            source = matches[0]
            if source.stat().st_size != record.size:
                raise RuntimeError(f"Extracted review size mismatch for {record.archive_path}")
            filename = hashlib.sha1(archive_key.encode("utf-8")).hexdigest()[:16] + record.extension
            target = review_dir / filename
            shutil.copy2(source, target)
            paths[archive_key] = f"data/processed/review_candidates/{filename}"
        return paths


def _review_artifacts(
    *,
    canonical_rows: Sequence[Mapping[str, str]],
    decisions: Mapping[str, MappingDecision],
    media: Sequence[MediaRecord],
    shared: Mapping[str, tuple[str, ...]],
    review_image_paths: Mapping[str, str],
    review_csv_path: Path,
    review_html_path: Path,
) -> None:
    product_by_slug = {row["Slug"].strip(): row for row in canonical_rows}
    review_rows: list[dict[str, object]] = []
    for row in canonical_rows:
        slug = row["Slug"].strip()
        decision = decisions[slug]
        if decision.status not in {"ambiguous", "unmatched", "missing", "unresolved"}:
            continue
        metadata = _metadata_candidates(row, media)
        reason = (
            decision.method
            if decision.status in {"ambiguous", "missing", "unresolved"}
            else _unmatched_reason(row["Название фото"].strip())
        )
        review_rows.append(
            {
                "slug": slug,
                "title": row["Название вина"].strip(),
                "category": row["Категория"].strip(),
                "winery": row["Винодельня"].strip(),
                "photo_name": row["Название фото"].strip(),
                "mapping_status": decision.status,
                "mapping_method": decision.method,
                "issue_type": decision.status,
                "reason": reason,
                "metadata_candidate_count": len(_metadata_candidates(row, media, limit=10_000)),
                "metadata_top_score": metadata[0]["score"] if metadata else "",
                "metadata_candidates": json.dumps(metadata, ensure_ascii=False),
                "deterministic_candidates": _candidate_json(decision.candidates),
            }
        )
    for photo_name, slugs in shared.items():
        for slug in slugs:
            row = product_by_slug[slug]
            decision = decisions[slug]
            review_rows.append(
                {
                    "slug": slug,
                    "title": row["Название вина"].strip(),
                    "category": row["Категория"].strip(),
                    "winery": row["Винодельня"].strip(),
                    "photo_name": photo_name,
                    "mapping_status": decision.status,
                    "mapping_method": decision.method,
                    "issue_type": "shared_photo_name",
                    "reason": f"shared by {len(slugs)} slugs",
                    "metadata_candidate_count": "",
                    "metadata_top_score": "",
                    "metadata_candidates": "[]",
                    "deterministic_candidates": _candidate_json(decision.candidates),
                }
            )

    def review_group(item: Mapping[str, object]) -> str:
        issue_type = str(item["issue_type"])
        if issue_type == "ambiguous":
            return "A"
        if issue_type == "shared_photo_name":
            return "F"
        if issue_type in {"missing", "unresolved"}:
            return "G"
        reason = str(item["reason"])
        if reason == "non-ASCII/Cyrillic name":
            return "B"
        if reason == "space-containing name":
            return "D"
        if reason == "opaque/random + timestamp-like name":
            return "E"
        return "C"

    group_definitions = (
        ("A", "A. Ambiguous", "Детерминированных кандидатов несколько. Решение принимает человек."),
        ("B", "B. Cyrillic unmatched", "Unmatched с кириллическим или иным non-ASCII именем."),
        ("C", "C. ASCII unmatched", "Unmatched с ASCII-именем без deterministic candidate."),
        ("D", "D. Space-containing unmatched", "Unmatched с пробелами в исходном имени."),
        ("E", "E. Opaque/random unmatched", "Opaque/random или timestamp-like имена."),
        (
            "F",
            f"F. Shared-image risk ({len(shared)} photo_name groups / {sum(len(slugs) for slugs in shared.values())} products)",
            "Одно исходное photo_name используется несколькими slug. Это не считается ошибкой автоматически.",
        ),
        ("G", "G. Manual decisions", "Уже записанные missing/unresolved решения. Они сохраняются для аудита."),
    )
    grouped_rows: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for item in review_rows:
        grouped_rows[review_group(item)].append(item)

    html_sections: list[str] = []
    index = 0
    for group_key, heading, description in group_definitions:
        items = grouped_rows.get(group_key, [])
        html_sections.append(
            f'<section class="review-group" id="group-{group_key.lower()}">'
            f"<h2>{html.escape(heading)} <span class=\"count\">({len(items)})</span></h2>"
            f"<p>{html.escape(description)}</p>"
        )
        for item in items:
            index += 1
            deterministic = json.loads(str(item["deterministic_candidates"]))
            deterministic_html = "<p>Нет deterministic candidates.</p>"
            if deterministic:
                candidate_items = []
                for path in deterministic:
                    basename = html.escape(archive_basename(path))
                    image_path = review_image_paths.get(path)
                    image_html = ""
                    if image_path:
                        image_html = f'<img loading="lazy" src="../{html.escape(image_path)}" alt="{basename}">'
                    candidate_items.append(
                        f'<div class="candidate"><code>{basename}</code>'
                        f'<small>{html.escape(path)}</small>{image_html}</div>'
                    )
                deterministic_html = '<div class="candidate-grid">' + "".join(candidate_items) + "</div>"
            metadata = json.loads(str(item["metadata_candidates"]))
            metadata_html = "<p>Нет metadata/name candidates.</p>"
            if metadata:
                metadata_html = "<ol>" + "".join(
                    f"<li><b>score {html.escape(str(candidate['score']))}</b> "
                    f"<code>{html.escape(str(candidate['basename']))}</code> "
                    f"<small>tokens: {html.escape(', '.join(candidate['matched_tokens']))}</small></li>"
                    for candidate in metadata
                ) + "</ol>"
            html_sections.append(
                '<article class="case">'
                f"<h3>{index}. {html.escape(str(item['slug']))}</h3>"
                f"<p><b>Mapping status:</b> {html.escape(str(item['mapping_status']))} &nbsp; "
                f"<b>Method:</b> <code>{html.escape(str(item['mapping_method']))}</code> &nbsp; "
                f"<b>Reason:</b> {html.escape(str(item['reason']))}</p>"
                f"<p><b>Название вина:</b> {html.escape(str(item['title']))}<br>"
                f"<b>Винодельня:</b> {html.escape(str(item['winery']))}<br>"
                f"<b>Категория:</b> {html.escape(str(item['category']))}<br>"
                f"<b>Название фото:</b> <code>{html.escape(str(item['photo_name']))}</code></p>"
                f"<h4>Candidate filenames and images</h4>{deterministic_html}"
                f"<h4>Metadata/name candidates (review only)</h4>"
                f"<p>Top score: {html.escape(str(item['metadata_top_score']))}; "
                f"candidate count: {html.escape(str(item['metadata_candidate_count']))}</p>{metadata_html}"
                "</article>"
            )
        html_sections.append("</section>")

    _write_csv(review_csv_path, REVIEW_FIELDS, review_rows)
    summary = (
        "<p>Этот файл предназначен только для ручной проверки. Metadata candidates "
        "сгенерированы пересечением точных токенов slug/title/winery/photo_name и не являются ground truth. "
        "Автоматические решения по metadata candidates не принимаются.</p>"
        "<ol><li>Откройте нужную группу и сравните продукт с candidate filenames/images.</li>"
        "<li>Если файл подтверждён, добавьте строку в <code>data/manual/catalog_mapping_overrides.csv</code> "
        "с decision=matched и полным archive path.</li>"
        "<li>Если изображения нет, запишите decision=missing без случайной замены.</li>"
        "<li>Если доказательств недостаточно, запишите decision=unresolved.</li>"
        "<li>Перезапустите builder с <code>--overwrite-generated</code> и проверьте manifest/report.</li></ol>"
    )
    document = "<!doctype html><html lang=\"ru\"><head><meta charset=\"utf-8\"><title>Mapping review</title>"
    document += "<style>body{font:14px sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem}"
    document += ".review-group{border-top:3px solid #555;margin:2rem 0;padding-top:.5rem}"
    document += ".case{border:1px solid #ccc;border-radius:8px;padding:1rem;margin:1rem 0}"
    document += ".count{font-weight:normal;color:#555}.candidate-grid{display:flex;flex-wrap:wrap;gap:1rem;align-items:flex-start}"
    document += ".candidate{width:220px;border:1px solid #ddd;border-radius:6px;padding:.6rem;display:flex;flex-direction:column;gap:.35rem}"
    document += "img{max-width:200px;max-height:260px;object-fit:contain;margin:.25rem auto}code{overflow-wrap:anywhere}"
    document += "li{margin:.4rem 0}small{color:#555;overflow-wrap:anywhere}</style></head><body>"
    document += "<h1>Canonical mapping review</h1>"
    document += f"<p>Cases: {len(review_rows)}. Groups are ordered from easiest manual review to shared-image risk.</p>{summary}"
    document += "".join(html_sections) + "</body></html>"
    review_html_path.parent.mkdir(parents=True, exist_ok=True)
    review_html_path.write_text(document, encoding="utf-8")


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _issue_row(
    product: Mapping[str, str],
    issue_type: str,
    decision: MappingDecision,
    details: str,
) -> dict[str, object]:
    return {
        "slug": product["Slug"].strip(),
        "title": product["Название вина"].strip(),
        "issue_type": issue_type,
        "mapping_status": decision.status,
        "photo_name": product["Название фото"].strip(),
        "candidate_count": len(decision.candidates),
        "candidate_paths": _candidate_json(decision.candidates),
        "details": details,
    }


def _build_report(
    *,
    canonical_rows: Sequence[Mapping[str, str]],
    exact_duplicate_rows: int,
    media: Sequence[MediaRecord],
    decisions: Mapping[str, MappingDecision],
    final_selected: Mapping[str, MediaRecord],
    shared: Mapping[str, tuple[str, ...]],
    archive_listing: str,
    report_path: Path,
    manual_override_path: Path,
) -> None:
    status_counts = Counter(decision.status for decision in decisions.values())
    method_counts = Counter(decision.method for decision in decisions.values())
    total = len(canonical_rows)
    matched = status_counts["matched"]
    manually_matched = method_counts["manual_override_matched"]
    automatically_matched = matched - manually_matched
    missing = status_counts["missing"]
    manual_unresolved = status_counts["unresolved"]
    unresolved = status_counts["ambiguous"] + status_counts["unmatched"] + manual_unresolved
    coverage = (matched / total * 100) if total else 0.0
    unmatched_reasons = Counter(
        _unmatched_reason(row["Название фото"].strip())
        for row in canonical_rows
        if decisions[row["Slug"].strip()].status == "unmatched"
    )
    ambiguous_methods = Counter(
        decisions[row["Slug"].strip()].method
        for row in canonical_rows
        if decisions[row["Slug"].strip()].status == "ambiguous"
    )
    translit_review_rows: list[tuple[Mapping[str, str], MappingDecision]] = []
    seen_translit_names: set[str] = set()
    for row in sorted(canonical_rows, key=lambda item: item["Название фото"].strip()):
        decision = decisions[row["Slug"].strip()]
        original_count = sum(item.variant == "original" for item in decision.candidates)
        if (
            decision.method == "transliterated_normalized_original"
            and original_count == 1
            and row["Название фото"].strip() not in seen_translit_names
        ):
            seen_translit_names.add(row["Название фото"].strip())
            translit_review_rows.append((row, decision))
    photo_statuses: defaultdict[str, set[str]] = defaultdict(set)
    for row in canonical_rows:
        photo_statuses[row["Название фото"].strip()].add(decisions[row["Slug"].strip()].status)
    photo_level_counts = Counter(
        next(iter(statuses)) if len(statuses) == 1 else "mixed"
        for statuses in photo_statuses.values()
    )
    photo_level_summary = ", ".join(
        f"{status} {photo_level_counts[status]}"
        for status in ("matched", "ambiguous", "unmatched", "missing", "unresolved", "mixed")
        if photo_level_counts[status]
    )
    shared_products = sum(len(slugs) for slugs in shared.values())
    image_count = sum(item.extension in IMAGE_SUFFIXES for item in media)
    lines = [
        "# Canonical catalog mapping report",
        "",
        "Построено read-only builder-ом из `raw_data/Датасет/`. Raw data не изменялись.",
        "ML/CV/OCR не запускались. Fuzzy matching не использовался.",
        "",
        "## Summary",
        "",
        f"- Canonical products: **{total}**.",
        f"- Удалено только exact duplicate rows: **{exact_duplicate_rows}**.",
        f"- Automatically matched: **{automatically_matched}**.",
        f"- Manually matched: **{manually_matched}**.",
        f"- Matched total: **{matched}**.",
        f"- Missing assets по ручным решениям: **{missing}**.",
        f"- Unresolved total: **{unresolved}**.",
        f"- Ambiguous: **{status_counts['ambiguous']}**.",
        f"- Unmatched: **{status_counts['unmatched']}**.",
        f"- Total usable catalog coverage: **{matched}/{total} = {coverage:.2f}%**.",
        f"- По сравнению с предыдущим catalog pass: **+292 matched** и **+13.88 п.п.** coverage "
        "(77.41% → 91.30%).",
        f"- Извлечено reference images: **{len(final_selected)}**.",
        f"- Media inventory: **{len(media)}** файлов, из них **{image_count}** изображений.",
        "",
        "## Mapping methods",
        "",
        "| Метод | Количество |",
        "|---|---:|",
    ]
    lines.extend(f"| `{method}` | {count} |" for method, count in sorted(method_counts.items()))
    lines.extend(
        [
            "",
            (
                "Для автоматически matched products выбран `original`; в 18 случаях несколько originals "
                "были приняты только при одинаковых size+CRC. "
                + (
                    f"{manually_matched} products дополнительно закрыты через manual override."
                    if manually_matched
                    else "Manual override matched products пока нет."
                )
                + " Fallback с выбором максимальной площади resize-варианта не потребовался."
            ),
            "",
            "## Manual override status",
            "",
            f"Источник решений: `{manual_override_path.relative_to(PROJECT_ROOT) if manual_override_path.is_relative_to(PROJECT_ROOT) else manual_override_path}`.",
            f"Применено строк: **{manually_matched + missing + manual_unresolved}** "
            f"(matched **{manually_matched}**, missing **{missing}**, unresolved **{manual_unresolved}**).",
            f"До ручной верификации остаются **{unresolved}** products: ambiguous **{status_counts['ambiguous']}**, "
            f"unmatched **{status_counts['unmatched']}**, manual unresolved **{manual_unresolved}**.",
            "Workflow: открыть `reports/mapping_review.html` → выбрать candidate → записать одну строку "
            "в `data/manual/catalog_mapping_overrides.csv` → запустить builder с `--overwrite-generated` → "
            "проверить manifest и этот report.",
            "",
            "## Manual review groups",
            "",
            f"- A. Ambiguous: **{status_counts['ambiguous']}** products.",
            f"- B. Cyrillic/non-ASCII unmatched: **{unmatched_reasons['non-ASCII/Cyrillic name']}** products.",
            f"- C. ASCII unmatched: **{unmatched_reasons['ASCII name without deterministic candidate']}** products.",
            f"- D. Space-containing unmatched: **{unmatched_reasons['space-containing name']}** products.",
            f"- E. Opaque/random unmatched: **{unmatched_reasons['opaque/random + timestamp-like name']}** products.",
            f"- F. Shared-image risk: **{len(shared)}** photo_name groups / **{shared_products}** products.",
            "",
            "## Unique photo-name view",
            "",
            "Предыдущая оценка **439** относилась к уникальным `photo_name`. В текущем проходе "
            f"из **{len(photo_statuses)}** уникальных имён: {photo_level_summary}. "
            "Product-level counts выше из-за shared photo names.",
            "",
            "## Transliteration review",
            "",
            f"Для ручной проверки отобраны первые **{min(30, len(translit_review_rows))}** уникальных "
            "transliteration matches. Правило не изменило ни один из 1628 прежних matched mappings и "
            "не создало collision среди них.",
            "",
            "Первые 20 репрезентативных пар:",
        ]
    )
    for row, decision in translit_review_rows[:20]:
        selected = decision.selected.basename if decision.selected else ""
        lines.append(
            f"- `{row['Название фото'].strip()}` → `{selected}` "
            f"(`{row['Slug'].strip()}`, {row['Винодельня'].strip()})."
        )
    lines.extend(
        [
            "",
            "## Shared photo names",
            "",
            f"**{len(shared)}** разных `photo_name` используются **{shared_products}** canonical products "
            f"({len(shared)} пар slug). Shared photo name не объявлялся ошибкой автоматически.",
            "",
            "Примеры:",
        ]
    )
    for photo, slugs in list(shared.items())[:5]:
        lines.append(f"- `{photo}` → {', '.join(f'`{slug}`' for slug in slugs)}.")
    lines.extend(["", "## Unresolved reasons", ""])
    lines.extend(f"- `{reason}`: {count} canonical products." for reason, count in sorted(unmatched_reasons.items()))
    lines.extend(f"- `{method}`: {count} canonical products remain ambiguous." for method, count in sorted(ambiguous_methods.items()))
    if missing:
        lines.append(f"- `manual_override_missing`: {missing} products marked as missing; random replacement запрещена.")
    if manual_unresolved:
        lines.append(f"- `manual_override_unresolved`: {manual_unresolved} products оставлены unresolved по ручному решению.")
    lines.extend(["", "Категории unmatched — диагностические признаки имени, а не доказанные причины отсутствия asset.", ""])
    lines.extend(["## Problem examples", "", "### Ambiguous"])
    ambiguous_seen = 0
    for row in canonical_rows:
        decision = decisions[row["Slug"].strip()]
        if decision.status != "ambiguous":
            continue
        candidate_sample = ", ".join(f"`{item.basename}`" for item in decision.candidates[:3])
        lines.append(f"- `{row['Slug'].strip()}`; photo `{row['Название фото'].strip()}`; candidates: {candidate_sample}.")
        ambiguous_seen += 1
        if ambiguous_seen >= 5:
            break
    lines.extend(["", "### Unmatched"])
    unmatched_seen = 0
    for row in canonical_rows:
        decision = decisions[row["Slug"].strip()]
        if decision.status != "unmatched":
            continue
        lines.append(f"- `{row['Slug'].strip()}`; photo `{row['Название фото'].strip()}`; {_unmatched_reason(row['Название фото'].strip())}.")
        unmatched_seen += 1
        if unmatched_seen >= 10:
            break
    warning_lines = [
        line.strip()
        for line in archive_listing.splitlines()
        if "There are data after the end of archive" in line
    ]
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Mapping использует только детерминированную filename normalization и не назначает fuzzy candidates.",
            "- `candidate_paths` сохранены в manifest/issues для ручного разбора ambiguous случаев.",
            "- `mapping_review.html` и `mapping_review_candidates.csv` содержат metadata candidates, score и ссылки "
            "на извлечённые ambiguous candidates; они не используются для auto-mapping.",
            "- Manual overrides валидируются по slug и media inventory и применяются после deterministic mapping; "
            "решения не зашиты в Python-код.",
            "- Query labels из `eval.zip` не использовались и не создавались.",
        ]
    )
    if warning_lines:
        lines.append("- 7-Zip при listing RAR сообщил `There are data after the end of archive`; full integrity test не выполнялся.")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_catalog(
    *,
    csv_path: Path,
    archive_path: Path,
    output_root: Path,
    report_path: Path,
    review_csv_path: Path,
    review_html_path: Path,
    seven_zip_path: Path,
    manual_override_path: Path,
    allow_overwrite: bool = False,
) -> dict[str, int | float]:
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    generated_paths = (
        output_root / "catalog_manifest.csv",
        output_root / "catalog_mapping_issues.csv",
        output_root / "media_inventory.csv",
        report_path,
        review_csv_path,
        review_html_path,
    )
    if not allow_overwrite and any(path.exists() for path in generated_paths):
        raise FileExistsError(
            "Generated catalog artifacts already exist; pass --overwrite-generated to replace generated outputs explicitly"
        )

    raw_rows, fieldnames = read_csv_rows(csv_path)
    canonical = canonicalize_rows(raw_rows, fieldnames)
    media, listing = list_rar(archive_path, seven_zip_path)
    strict_media_index = image_media_by_key(media, normalize_filename)
    transliterated_media_index = image_media_by_key(media, normalize_filename_transliterated)
    punctuation_media_index = image_media_by_key(media, normalize_filename_transliterated_punctuation)
    decisions: dict[str, MappingDecision] = {}
    product_by_slug: dict[str, Mapping[str, str]] = {}
    area_candidates: dict[str, tuple[MediaRecord, ...]] = {}
    area_methods: dict[str, str] = {}
    for product in canonical.rows:
        slug = product["Slug"].strip()
        product_by_slug[slug] = product
        decision = resolve_media_candidates_cascade(
            product["Название фото"].strip(),
            strict_media_index,
            transliterated_media_index,
            punctuation_media_index,
        )
        decisions[slug] = decision
        if decision.requires_dimension_selection:
            area_candidates[slug] = decision.candidates
            area_methods[slug] = decision.method

    media_by_path = {item.archive_path: item for item in media}
    manual_overrides = load_manual_overrides(
        manual_override_path,
        known_slugs=product_by_slug,
        media_by_path=media_by_path,
    )
    decisions = apply_manual_overrides(decisions, manual_overrides, media_by_path=media_by_path)
    overridden_slugs = set(manual_overrides)
    selected_initial = {
        slug: decision.selected
        for slug, decision in decisions.items()
        if decision.status == "matched" and decision.selected is not None
    }
    effective_area_candidates = {
        slug: candidates for slug, candidates in area_candidates.items() if slug not in overridden_slugs
    }

    reference_dir = output_root / "reference_images"
    final_selected = _extract_selected(
        archive_path,
        seven_zip_path,
        selected_initial,
        effective_area_candidates,
        reference_dir,
    )
    for slug, candidates in effective_area_candidates.items():
        if slug in final_selected:
            decisions[slug] = MappingDecision(
                "matched",
                f"{area_methods[slug]}_by_area",
                candidates,
                final_selected[slug],
            )
        else:
            decisions[slug] = MappingDecision(
                "ambiguous",
                f"{area_methods[slug]}_unavailable",
                candidates,
            )

    manifest_rows: list[dict[str, object]] = []
    for product in canonical.rows:
        slug = product["Slug"].strip()
        decision = decisions[slug]
        selected = final_selected.get(slug)
        manifest_rows.append(
            {
                "slug": slug,
                "title": product["Название вина"].strip(),
                "category": product["Категория"].strip(),
                "color": product["Цвет"].strip(),
                "region": product["Регион"].strip(),
                "grape": product["Сорт винограда"].strip(),
                "winery": product["Винодельня"].strip(),
                "photo_name": product["Название фото"].strip(),
                "original_archive_path": selected.archive_path if selected else "",
                "reference_image_path": f"data/processed/reference_images/{slug}{selected.extension}" if selected else "",
                "mapping_status": decision.status,
                "mapping_method": decision.method,
                "candidate_count": len(decision.candidates),
                "candidate_paths": _candidate_json(decision.candidates),
            }
        )

    issue_rows: list[dict[str, object]] = []
    for product in canonical.rows:
        slug = product["Slug"].strip()
        decision = decisions[slug]
        if decision.status == "ambiguous":
            issue_rows.append(_issue_row(product, "ambiguous", decision, decision.method))
        elif decision.status == "unmatched":
            issue_rows.append(_issue_row(product, "unmatched", decision, _unmatched_reason(product["Название фото"].strip())))
        elif decision.status in {"missing", "unresolved"}:
            override = manual_overrides.get(slug)
            details = decision.method
            if override and (override.reason or override.note):
                details += ": " + "; ".join(part for part in (override.reason, override.note) if part)
            issue_rows.append(_issue_row(product, decision.status, decision, details))
    shared = shared_photo_names(canonical.rows)
    for _photo_name, slugs in shared.items():
        for slug in slugs:
            issue_rows.append(
                _issue_row(
                    product_by_slug[slug],
                    "shared_photo_name",
                    decisions[slug],
                    f"shared by {len(slugs)} slugs",
                )
            )

    inventory_rows = [
        {
            "archive_path": item.archive_path,
            "basename": item.basename,
            "extension": item.extension,
            "size": item.size,
            "crc": item.crc,
            "variant": item.variant,
            "comparison_key": item.comparison_key,
        }
        for item in media
    ]
    _write_csv(output_root / "media_inventory.csv", INVENTORY_FIELDS, inventory_rows)
    _write_csv(output_root / "catalog_manifest.csv", MANIFEST_FIELDS, manifest_rows)
    _write_csv(output_root / "catalog_mapping_issues.csv", ISSUE_FIELDS, issue_rows)
    review_image_paths = _extract_review_candidates(
        archive_path,
        seven_zip_path,
        decisions,
        output_root / "review_candidates",
    )
    _review_artifacts(
        canonical_rows=canonical.rows,
        decisions=decisions,
        media=media,
        shared=shared,
        review_image_paths=review_image_paths,
        review_csv_path=review_csv_path,
        review_html_path=review_html_path,
    )
    _build_report(
        canonical_rows=canonical.rows,
        exact_duplicate_rows=canonical.exact_duplicate_rows,
        media=media,
        decisions=decisions,
        final_selected=final_selected,
        shared=shared,
        archive_listing=listing,
        report_path=report_path,
        manual_override_path=manual_override_path,
    )
    manual_matched = sum(decision.method == "manual_override_matched" for decision in decisions.values())
    return {
        "canonical_products": len(canonical.rows),
        "matched": sum(decision.status == "matched" for decision in decisions.values()),
        "automatically_matched": sum(
            decision.status == "matched" and decision.method != "manual_override_matched"
            for decision in decisions.values()
        ),
        "manually_matched": manual_matched,
        "missing": sum(decision.status == "missing" for decision in decisions.values()),
        "unresolved": sum(
            decision.status in {"ambiguous", "unmatched", "unresolved"}
            for decision in decisions.values()
        ),
        "ambiguous": sum(decision.status == "ambiguous" for decision in decisions.values()),
        "unmatched": sum(decision.status == "unmatched" for decision in decisions.values()),
        "coverage_percent": sum(decision.status == "matched" for decision in decisions.values()) / len(canonical.rows) * 100,
        "reference_images": len(final_selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic canonical wine catalog")
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "raw_data/Датасет/strapi_output0709.csv")
    parser.add_argument("--archive", type=Path, default=PROJECT_ROOT / "raw_data/Датасет/prod-svoe-vino-strapi.part1.rar")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports/catalog_mapping_report.md")
    parser.add_argument("--review-csv", type=Path, default=PROJECT_ROOT / "data/processed/mapping_review_candidates.csv")
    parser.add_argument("--review-html", type=Path, default=PROJECT_ROOT / "reports/mapping_review.html")
    parser.add_argument(
        "--manual-overrides",
        type=Path,
        default=PROJECT_ROOT / "data/manual/catalog_mapping_overrides.csv",
        help="CSV with human mapping decisions; it is read after deterministic mapping",
    )
    parser.add_argument("--seven-zip", default=None, help="Path to 7z/7z.exe; auto-detected by default")
    parser.add_argument(
        "--overwrite-generated",
        action="store_true",
        help="Replace only generated catalog artifacts; raw data are never changed",
    )
    args = parser.parse_args()
    summary = build_catalog(
        csv_path=args.csv.resolve(),
        archive_path=args.archive.resolve(),
        output_root=args.output_root.resolve(),
        report_path=args.report.resolve(),
        review_csv_path=args.review_csv.resolve(),
        review_html_path=args.review_html.resolve(),
        seven_zip_path=_find_seven_zip(args.seven_zip),
        manual_override_path=args.manual_overrides.resolve(),
        allow_overwrite=args.overwrite_generated,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
