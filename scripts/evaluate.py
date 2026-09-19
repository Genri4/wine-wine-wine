#!/usr/bin/env python3
"""Evaluate the baseline against a temporary JSON/JSONL evaluation manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recognition.data import load_evaluation_examples  # noqa: E402
from recognition.encoder import SUPPORTED_MODEL_NAMES, VisualEncoder  # noqa: E402
from recognition.index import CatalogIndex  # noqa: E402
from recognition.pipeline import RecognitionPipeline  # noqa: E402
from recognition.evaluation import run_evaluation  # noqa: E402


def _resolve_model_name(index: CatalogIndex, requested: str | None) -> str:
    model_name = requested or index.model_name
    if requested is not None and requested != index.model_name:
        raise ValueError(
            f"Model mismatch: index was built with '{index.model_name}', "
            f"but evaluation requested '{requested}'"
        )
    return model_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate image recognition baseline")
    parser.add_argument("--index", required=True, help="Catalog .pt index")
    parser.add_argument("--data", required=True, help="Evaluation JSON/JSONL manifest")
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    parser.add_argument(
        "--errors-output",
        default=None,
        help="Optional separate JSON path for misclassified samples",
    )
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
    examples = load_evaluation_examples(args.data)
    model_name = _resolve_model_name(index, args.model)
    encoder = VisualEncoder(model_name, args.device, args.batch_size)
    pipeline = RecognitionPipeline(
        encoder,
        index,
        unknown_threshold=args.unknown_threshold,
        default_top_k=args.top_k,
    )
    report = run_evaluation(
        pipeline,
        examples,
        top_k=args.top_k,
        model_name=model_name,
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
    errors_output = args.errors_output
    if errors_output is None and args.output:
        report_path = Path(args.output)
        errors_output = str(
            report_path.with_name(f"{report_path.stem}.errors{report_path.suffix}")
        )
    if errors_output:
        errors_path = Path(errors_output)
        errors_path.parent.mkdir(parents=True, exist_ok=True)
        errors_payload = {
            "model_name": report["model_name"],
            "samples": report["samples"],
            "errors": report["errors"],
        }
        errors_path.write_text(
            json.dumps(errors_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(serialized)


if __name__ == "__main__":
    main()
