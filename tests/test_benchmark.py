from __future__ import annotations

from pathlib import Path
import random
import tempfile
import unittest
import json

from PIL import Image

from recognition.benchmark import (
    RankingRecord,
    assert_no_reference_query_leakage,
    evaluate_rankings,
)
from scripts.evaluate_benchmark import _validate_prediction_alignment
from scripts.collect_web_extra import _duplicate_check, _image_signature


class BenchmarkTests(unittest.TestCase):
    def test_evaluator_metrics_and_ranking(self) -> None:
        metrics = evaluate_rankings(
            [
                RankingRecord("q1", "a", ["a", "b"], [0.9, 0.1], 10.0),
                RankingRecord("q2", "b", ["a", "b"], [0.8, 0.7], 20.0),
                RankingRecord("q3", "c", ["a", "b"], [0.8, 0.7], 30.0),
            ]
        )
        self.assertAlmostEqual(metrics["top1_accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["recall_at_5"], 2 / 3)
        self.assertAlmostEqual(metrics["mrr"], (1 + 0.5 + 0) / 3)
        self.assertEqual(metrics["mean_latency_ms"], 20.0)
        self.assertEqual(metrics["p50_latency_ms"], 20.0)
        self.assertEqual(metrics["p95_latency_ms"], 29.0)

    def test_reference_query_leakage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "same.webp")
            with self.assertRaises(ValueError):
                assert_no_reference_query_leakage([path], [path])

    def test_web_duplicate_check_rejects_exact_reference_copy(self) -> None:
        from io import BytesIO

        image = Image.new("RGB", (32, 48), (120, 40, 20))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        signature = _image_signature(buffer.getvalue())
        check = _duplicate_check(signature, [{"slug": "wine-1", **signature}])
        self.assertTrue(check["is_reference_duplicate"])

    def test_same_seed_produces_same_query_bytes(self) -> None:
        # This is the deterministic core used by create_synthetic_dev.py.
        from scripts.create_synthetic_dev import _transform

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "reference.png"
            Image.new("RGB", (64, 48), (120, 40, 20)).save(source)
            outputs = []
            for seed in (1234, 1234):
                with Image.open(source) as opened:
                    transformed, _, _, output_format = _transform(
                        opened.convert("RGB"), random.Random(seed), 1
                    )
                target = Path(directory) / f"query-{len(outputs)}.{output_format.lower()}"
                transformed.save(target, format=output_format, quality=92)
                transformed.close()
                outputs.append(target.read_bytes())
            self.assertEqual(outputs[0], outputs[1])

    def test_evaluator_alignment_supports_web_manifest_shape(self) -> None:
        manifest = [{"query_id": "web-a", "query_path": "images/a.jpg", "target_slug": "a"}]
        predictions = [{
            "query_id": "web-a",
            "target_slug": "a",
            "predicted_slug": "a",
            "top5_slugs": json.dumps(["a"]),
            "latency_ms": "1.0",
        }]
        _validate_prediction_alignment(predictions, manifest, "web_extra_dev")

    def test_evaluator_rejects_predictions_from_another_split(self) -> None:
        manifest = [{"query_id": "web-a", "query_path": "images/a.jpg", "target_slug": "a"}]
        predictions = [{
            "query_id": "synthetic-a",
            "target_slug": "a",
            "predicted_slug": "a",
            "top5_slugs": json.dumps(["a"]),
            "latency_ms": "1.0",
        }]
        with self.assertRaises(ValueError):
            _validate_prediction_alignment(predictions, manifest, "web_extra_dev")


if __name__ == "__main__":
    unittest.main()
