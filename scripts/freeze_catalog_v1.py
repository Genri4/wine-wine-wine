#!/usr/bin/env python3
"""Freeze the current usable canonical catalog metadata without copying images."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recognition.cache import sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze catalog v1 metadata")
    parser.add_argument(
        "--manifest",
        default="data/processed/catalog_manifest.csv",
        help="Canonical catalog manifest, relative to project root",
    )
    parser.add_argument(
        "--output",
        default="data/processed/catalog_v1.json",
        help="Catalog metadata output, relative to project root",
    )
    args = parser.parse_args()
    manifest_path = _project_path(args.manifest)
    output_path = _project_path(args.output)
    rows = _read_rows(manifest_path)

    usable_rows = []
    missing_reference_rows = []
    for row in rows:
        reference = row.get("reference_image_path", "").strip()
        if row.get("mapping_status") == "matched" and reference:
            if (_project_path(reference)).is_file():
                usable_rows.append(row)
            else:
                missing_reference_rows.append(row)

    metadata = {
        "catalog_version": "catalog-v1",
        "source_manifest": _relative(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "canonical_products": len(rows),
        "matched_manifest_products": sum(row.get("mapping_status") == "matched" for row in rows),
        "usable_products": len(usable_rows),
        "unresolved_products": len(rows) - len(usable_rows),
        "matched_without_existing_reference": len(missing_reference_rows),
        "source_reference_images_are_not_copied": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Catalog manifest is empty: {path}")
    required = {"slug", "reference_image_path", "mapping_status"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Catalog manifest is missing columns: {sorted(missing)}")
    return rows


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


if __name__ == "__main__":
    main()
