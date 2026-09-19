#!/usr/bin/env python3
"""Inspect the temporary catalog adapter's images without assuming organizer format."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recognition.data import load_catalog  # noqa: E402


def analyze_catalog(source: str | Path) -> dict[str, object]:
    """Return counts and file diagnostics for a temporary catalog source.

    Duplicate groups are exact byte duplicates, not perceptual duplicates.
    Product IDs and image paths come from the isolated temporary adapter.
    """

    items = load_catalog(source, allow_duplicate_item_ids=True)
    product_counts = Counter(item.item_id for item in items)
    size_counts: Counter[str] = Counter()
    digest_paths: defaultdict[str, list[str]] = defaultdict(list)
    corrupt_files: list[str] = []

    for item in items:
        image_path = item.image_path
        path_string = str(image_path)
        try:
            with Image.open(image_path) as image:
                image.load()
                size_counts[f"{image.width}x{image.height}"] += 1
        except (OSError, ValueError):
            corrupt_files.append(path_string)

        try:
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        except OSError:
            continue
        digest_paths[digest].append(path_string)

    duplicate_groups = [
        {"sha256": digest, "images": paths}
        for digest, paths in sorted(digest_paths.items())
        if len(paths) > 1
    ]
    report = {
        "source": str(Path(source).resolve()),
        "product_count": len(product_counts),
        "image_count": len(items),
        "photos_per_product": dict(sorted(product_counts.items())),
        "image_sizes": dict(sorted(size_counts.items())),
        "corrupt_files": sorted(corrupt_files),
        "duplicates": duplicate_groups,
        "class_distribution": dict(sorted(product_counts.items())),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze catalog images")
    parser.add_argument(
        "--catalog",
        required=True,
        help="Catalog directory or temporary JSON/JSONL manifest",
    )
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    args = parser.parse_args()

    report = analyze_catalog(args.catalog)
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
