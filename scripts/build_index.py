#!/usr/bin/env python3
"""Build an offline catalog embedding index."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recognition.data import load_catalog  # noqa: E402
from recognition.encoder import SUPPORTED_MODEL_NAMES, VisualEncoder  # noqa: E402
from recognition.index import CatalogIndex  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a catalog embedding index")
    parser.add_argument("--catalog", required=True, help="Catalog directory or JSON/JSONL manifest")
    parser.add_argument("--output", required=True, help="Output .pt index path")
    parser.add_argument("--model", default="resnet50", choices=SUPPORTED_MODEL_NAMES)
    parser.add_argument("--device", default=None, help="torch device, e.g. cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Optional local smoke-test limit")
    args = parser.parse_args()

    items = load_catalog(args.catalog, allow_duplicate_item_ids=True)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        items = items[: args.limit]
    if not items:
        raise ValueError("Catalog is empty after applying --limit")

    encoder = VisualEncoder(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )
    embeddings = encoder.encode([item.image_path for item in items])
    index = CatalogIndex.from_items(
        items,
        embeddings,
        encoder.encoder_name,
        model_name=encoder.model_name,
    )
    index.save(args.output)
    print(
        f"Built index: {len(index.entries)} items, "
        f"dim={index.embedding_dim}, encoder={index.encoder_name}, output={args.output}"
    )


if __name__ == "__main__":
    main()
