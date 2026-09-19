#!/usr/bin/env python3
"""Predict one image against an offline catalog index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recognition.encoder import SUPPORTED_MODEL_NAMES, VisualEncoder  # noqa: E402
from recognition.index import CatalogIndex  # noqa: E402
from recognition.pipeline import RecognitionPipeline  # noqa: E402


def _resolve_model_name(index: CatalogIndex, requested: str | None) -> str:
    model_name = requested or index.model_name
    if requested is not None and requested != index.model_name:
        raise ValueError(
            f"Model mismatch: index was built with '{index.model_name}', "
            f"but prediction requested '{requested}'"
        )
    return model_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a catalog item or unknown")
    parser.add_argument("--index", required=True, help="Catalog .pt index")
    parser.add_argument("--image", required=True, help="Query image path")
    parser.add_argument("--model", default=None, choices=SUPPORTED_MODEL_NAMES)
    parser.add_argument("--device", default=None, help="torch device, e.g. cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--unknown-threshold",
        type=float,
        default=0.5,
        help="Cosine threshold; provisional and must be calibrated on validation data",
    )
    args = parser.parse_args()

    index = CatalogIndex.load(args.index)
    model_name = _resolve_model_name(index, args.model)
    encoder = VisualEncoder(model_name, args.device, args.batch_size)
    pipeline = RecognitionPipeline(
        encoder,
        index,
        unknown_threshold=args.unknown_threshold,
        default_top_k=args.top_k,
    )
    prediction = pipeline.predict(args.image)
    print(json.dumps(prediction.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
