"""Data loading boundary.

The readers in this module are intentionally small local adapters. They define
only a temporary manifest format for experiments; organizer-specific parsing
must stay here and can be replaced without changing the recognition pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class CatalogItem:
    """One catalog image and its stable catalog identifier."""

    item_id: str
    image_path: Path
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationExample:
    """One query image and an optional expected catalog identifier.

    ``expected_item_id=None`` is the temporary local convention for an image
    that should be classified as unknown.
    """

    sample_id: str
    image_path: Path
    expected_item_id: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def load_catalog(
    source: str | Path, *, allow_duplicate_item_ids: bool = False
) -> list[CatalogItem]:
    """Load catalog items from a directory or a temporary JSON/JSONL manifest.

    Directory mode uses the image's path relative to the directory (without
    extension) as an identifier. A manifest is preferred when the catalog has
    an explicit product identifier.
    """

    source_path = Path(source)
    if source_path.is_dir():
        items = [
            CatalogItem(
                item_id=image_path.relative_to(source_path).with_suffix("").as_posix(),
                image_path=image_path,
            )
            for image_path in sorted(source_path.rglob("*"))
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if not items:
            raise ValueError(f"No supported images found in catalog directory: {source_path}")
        return items

    if not source_path.is_file():
        raise FileNotFoundError(f"Catalog source does not exist: {source_path}")

    return _catalog_from_records(
        _read_records(source_path),
        source_path.parent,
        allow_duplicate_item_ids=allow_duplicate_item_ids,
    )


def load_evaluation_examples(source: str | Path) -> list[EvaluationExample]:
    """Load the temporary local evaluation manifest.

    Each record must contain ``sample_id``, ``image`` and
    ``expected_item_id``. The last field may be null for an unknown example.
    """

    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(f"Evaluation manifest does not exist: {source_path}")

    examples: list[EvaluationExample] = []
    seen_ids: set[str] = set()
    for line_number, record in enumerate(_read_records(source_path), start=1):
        sample_id = _required_string(record, "sample_id", line_number)
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample_id at line {line_number}: {sample_id}")
        seen_ids.add(sample_id)

        image_path = _resolve_image_path(
            record.get("image"), source_path.parent, line_number, field_name="image"
        )
        if "expected_item_id" not in record:
            raise ValueError(
                f"Missing expected_item_id at line {line_number}; use null for unknown"
            )
        expected = record["expected_item_id"]
        if expected is not None and not isinstance(expected, str):
            raise ValueError(f"expected_item_id must be a string or null at line {line_number}")

        metadata = record.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"metadata must be an object at line {line_number}")
        examples.append(
            EvaluationExample(
                sample_id=sample_id,
                image_path=image_path,
                expected_item_id=expected,
                metadata=dict(metadata),
            )
        )

    if not examples:
        raise ValueError(f"Evaluation manifest is empty: {source_path}")
    return examples


def _catalog_from_records(
    records: Iterable[Mapping[str, Any]],
    base_dir: Path,
    *,
    allow_duplicate_item_ids: bool = False,
) -> list[CatalogItem]:
    items: list[CatalogItem] = []
    seen_ids: set[str] = set()
    for line_number, record in enumerate(records, start=1):
        item_id = _required_string(record, "item_id", line_number)
        if item_id in seen_ids and not allow_duplicate_item_ids:
            raise ValueError(f"Duplicate item_id at line {line_number}: {item_id}")
        seen_ids.add(item_id)
        image_path = _resolve_image_path(
            record.get("image"), base_dir, line_number, field_name="image"
        )
        metadata = record.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"metadata must be an object at line {line_number}")
        items.append(
            CatalogItem(item_id, image_path, dict(metadata))
        )

    if not items:
        raise ValueError("Catalog manifest is empty")
    return items


def _read_records(path: Path) -> list[Mapping[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records: list[Mapping[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            value = _parse_json(line, f"{path}:{line_number}")
            if not isinstance(value, Mapping):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            records.append(value)
        return records

    if path.suffix.lower() == ".json":
        value = _parse_json(path.read_text(encoding="utf-8"), str(path))
        if isinstance(value, Mapping):
            value = value.get("items")
        if not isinstance(value, list):
            raise ValueError(f"Expected a JSON list or an object with an 'items' list: {path}")
        if not all(isinstance(record, Mapping) for record in value):
            raise ValueError(f"Every record must be a JSON object: {path}")
        return value

    raise ValueError(f"Expected a .json or .jsonl manifest: {path}")


def _parse_json(value: str, source: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {source}: {exc}") from exc


def _required_string(record: Mapping[str, Any], field_name: str, line_number: int) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string at line {line_number}")
    return value


def _resolve_image_path(
    value: Any, base_dir: Path, line_number: int, field_name: str
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string at line {line_number}")
    image_path = Path(value)
    if not image_path.is_absolute():
        image_path = base_dir / image_path
    image_path = image_path.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist at line {line_number}: {image_path}")
    return image_path
