#!/usr/bin/env python3
"""Run the four baseline encoders sequentially and save one comparison table."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recognition.data import load_catalog, load_evaluation_examples  # noqa: E402
from recognition.encoder import SUPPORTED_MODEL_NAMES, VisualEncoder  # noqa: E402
from recognition.evaluation import run_evaluation  # noqa: E402
from recognition.index import CatalogIndex  # noqa: E402
from recognition.pipeline import RecognitionPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline visual encoders")
    parser.add_argument("--catalog", required=True, help="Catalog JSON/JSONL or directory")
    parser.add_argument("--data", required=True, help="Evaluation JSON/JSONL manifest")
    parser.add_argument("--output", required=True, help="JSON comparison table path")
    parser.add_argument("--device", default=None, help="torch device, e.g. cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--unknown-threshold", type=float, default=0.5)
    args = parser.parse_args()

    catalog = load_catalog(args.catalog, allow_duplicate_item_ids=True)
    examples = load_evaluation_examples(args.data)
    rows: list[dict[str, object]] = []

    for model_name in SUPPORTED_MODEL_NAMES:
        encoder = VisualEncoder(model_name, args.device, args.batch_size)
        catalog_embeddings = encoder.encode([item.image_path for item in catalog])
        index = CatalogIndex.from_items(
            catalog,
            catalog_embeddings,
            encoder_name=encoder.encoder_name,
            model_name=model_name,
        )
        pipeline = RecognitionPipeline(
            encoder,
            index,
            unknown_threshold=args.unknown_threshold,
            default_top_k=5,
        )
        report = run_evaluation(pipeline, examples, top_k=5, model_name=model_name)
        row = {
            "model_name": model_name,
            "top1_accuracy": report["known_top1_accuracy"],
            "top5_recall": report["top_k_recall"],
            "average_latency_ms": report["average_latency_ms"],
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

        used_cuda = encoder.device.startswith("cuda")
        del pipeline, index, encoder
        gc.collect()
        if used_cuda:
            import torch

            torch.cuda.empty_cache()

    table = {
        "models": rows,
        "samples": len(examples),
        "top_k": 5,
        "note": "Metrics are meaningful only after the organizer dataset and split are known.",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(table, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(table, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
