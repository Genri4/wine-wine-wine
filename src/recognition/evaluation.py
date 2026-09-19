"""Evaluation runner for known-item retrieval and temporary unknown labels."""

from __future__ import annotations

from typing import Iterable

from .data import EvaluationExample
from .pipeline import RecognitionPipeline


def run_evaluation(
    pipeline: RecognitionPipeline,
    examples: Iterable[EvaluationExample],
    top_k: int | None = None,
    model_name: str | None = None,
) -> dict[str, object]:
    """Run the same online pipeline and return metrics plus per-sample output."""

    prediction_records: list[dict[str, object]] = []
    total = 0
    correct = 0
    known_total = 0
    known_correct = 0
    top_k_hits = 0
    unknown_total = 0
    unknown_correct = 0
    latencies: list[float] = []
    errors: list[dict[str, object]] = []

    for example in examples:
        metric_top_k = top_k if top_k is not None else pipeline.default_top_k
        # Keep five candidates available for the separate error artifact even
        # when a caller asks for a smaller top-k metric.
        prediction = pipeline.predict(
            example.image_path, top_k=max(metric_top_k, 5)
        )
        candidate_ids = [
            candidate.item_id for candidate in prediction.candidates[:metric_top_k]
        ]
        expected = example.expected_item_id
        is_correct = (
            prediction.predicted_item_id is None
            if expected is None
            else prediction.predicted_item_id == expected
        )
        if is_correct:
            correct += 1

        if expected is None:
            unknown_total += 1
            if prediction.is_unknown:
                unknown_correct += 1
        else:
            known_total += 1
            if prediction.predicted_item_id == expected:
                known_correct += 1
            if expected in candidate_ids:
                top_k_hits += 1

        total += 1
        latencies.append(prediction.latency_ms)
        record = prediction.to_dict()
        record.update(
            {
                "sample_id": example.sample_id,
                "expected_item_id": expected,
                "correct": is_correct,
            }
        )
        prediction_records.append(record)
        if not is_correct:
            errors.append(
                {
                    "query_image": str(example.image_path),
                    "expected_item": expected,
                    "predicted_item": prediction.predicted_item_id,
                    "top_5_candidates": [
                        {"item_id": candidate.item_id, "score": candidate.score}
                        for candidate in prediction.candidates[:5]
                    ],
                }
            )

    unknown_predictions = sum(1 for record in prediction_records if record["unknown"])
    metrics = {
        "num_samples": total,
        "accuracy": _ratio(correct, total),
        "known_samples": known_total,
        "known_top1_accuracy": _ratio(known_correct, known_total),
        "top_k_recall": _ratio(top_k_hits, known_total),
        "unknown_samples": unknown_total,
        "unknown_detection_accuracy": _ratio(unknown_correct, unknown_total),
        "unknown_rate": _ratio(unknown_predictions, total),
        "average_latency_ms": _average(latencies),
    }
    resolved_model_name = (
        model_name
        or getattr(pipeline.encoder, "model_name", None)
        or getattr(pipeline.index, "model_name", None)
    )
    if resolved_model_name is None:
        resolved_model_name = "unknown"
    return {
        "model_name": resolved_model_name,
        "samples": total,
        "num_samples": total,
        "accuracy": metrics["accuracy"],
        "known_top1_accuracy": metrics["known_top1_accuracy"],
        "top_k_recall": metrics["top_k_recall"],
        "unknown_detection_accuracy": metrics["unknown_detection_accuracy"],
        "average_latency_ms": metrics["average_latency_ms"],
        # Keep the complete metric block and predictions for experiment analysis.
        "metrics": metrics,
        "predictions": prediction_records,
        "errors": errors,
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
