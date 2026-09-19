from __future__ import annotations

from pathlib import Path
import unittest

from recognition.data import CatalogItem, EvaluationExample
from recognition.evaluation import run_evaluation
from recognition.index import CatalogIndex
from recognition.pipeline import RecognitionPipeline


class FakeEncoder:
    encoder_name = "fake"

    def encode(self, image_paths):
        path = str(image_paths[0])
        if "unknown" in path:
            return [[0, 1]]
        return [[1, 0]]


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        items = [CatalogItem("wine-1", Path("wine-1.jpg"))]
        self.index = CatalogIndex.from_items(items, [[1, 0]], "fake")
        self.pipeline = RecognitionPipeline(
            FakeEncoder(), self.index, unknown_threshold=0.8, default_top_k=1
        )

    def test_known_prediction(self) -> None:
        prediction = self.pipeline.predict(Path("known.jpg"))
        self.assertEqual(prediction.predicted_item_id, "wine-1")
        self.assertFalse(prediction.is_unknown)

    def test_unknown_prediction_keeps_candidates(self) -> None:
        prediction = self.pipeline.predict(Path("unknown.jpg"))
        self.assertIsNone(prediction.predicted_item_id)
        self.assertTrue(prediction.is_unknown)
        self.assertEqual(len(prediction.candidates), 1)

    def test_evaluation_reports_known_and_unknown_metrics(self) -> None:
        examples = [
            EvaluationExample("known", Path("known.jpg"), "wine-1"),
            EvaluationExample("unknown", Path("unknown.jpg"), None),
        ]
        report = run_evaluation(self.pipeline, examples, top_k=1)
        metrics = report["metrics"]
        self.assertEqual(report["model_name"], "fake")
        self.assertEqual(report["samples"], 2)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["top_k_recall"], 1.0)
        self.assertEqual(metrics["unknown_detection_accuracy"], 1.0)
        self.assertEqual(len(report["predictions"]), 2)

    def test_evaluation_collects_separate_error_details(self) -> None:
        examples = [EvaluationExample("wrong", Path("known.jpg"), "other-wine")]
        report = run_evaluation(self.pipeline, examples, top_k=1)
        self.assertEqual(len(report["errors"]), 1)
        error = report["errors"][0]
        self.assertEqual(error["query_image"], "known.jpg")
        self.assertEqual(error["expected_item"], "other-wine")
        self.assertEqual(error["predicted_item"], "wine-1")
        self.assertEqual(error["top_5_candidates"][0]["item_id"], "wine-1")


if __name__ == "__main__":
    unittest.main()
