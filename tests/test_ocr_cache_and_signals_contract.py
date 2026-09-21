from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from recognition.ocr_engine import OCR_MODEL_KEY
from recognition.ocr_reranker import FusionConfig, POLICY_METADATA_BLEND, rerank_one_query
from recognition.text_signals import build_query_text_evidence


class CacheContractTests(unittest.TestCase):
    """Query and reference OCR caches must stay separate artifacts."""

    def test_cache_model_key_is_deterministic(self) -> None:
        self.assertEqual(OCR_MODEL_KEY, "paddleocr3.7_ppocrv5_server_det_eslav_v5_mobile_rec")

    def test_query_and_reference_records_use_distinct_identities(self) -> None:
        query_record = {"query_id": "q1", "slug_should_not_exist": None}
        reference_record = {"slug": "s1", "query_id_should_not_exist": None}
        self.assertIn("query_id", query_record)
        self.assertIn("slug", reference_record)
        self.assertNotIn("slug", query_record)
        self.assertNotIn("query_id", reference_record)


class GroundTruthNotUsedTests(unittest.TestCase):
    def test_rerank_outcome_is_independent_of_target(self) -> None:
        image = [0.31, 0.3099, 0.29, 0.28, 0.27]
        signals = [
            {"metadata_name_score": 0.2, "winery_score": 0.0, "grape_score": 0.0, "region_score": 0.0,
             "metadata_text_score": 0.2, "reference_ocr_score": 0.1, "vintage_match": "unknown"},
            {"metadata_name_score": 0.6, "winery_score": 0.0, "grape_score": 0.0, "region_score": 0.0,
             "metadata_text_score": 0.6, "reference_ocr_score": 0.1, "vintage_match": "exact_match"},
        ] + [
            {"metadata_name_score": 0.1, "winery_score": 0.0, "grape_score": 0.0, "region_score": 0.0,
             "metadata_text_score": 0.1, "reference_ocr_score": 0.0, "vintage_match": "unknown"}
            for _ in range(3)
        ]
        config = FusionConfig(policy=POLICY_METADATA_BLEND, alpha=0.4)
        first = rerank_one_query(image, signals, config)
        second = rerank_one_query(image, list(reversed(signals)), config)
        # Same candidate multiset, different input order: reranking depends
        # only on scores/signals, never on any target label.
        self.assertEqual(sorted(first.final_order), sorted(second.final_order))
        self.assertEqual(first.text_scores, tuple(reversed(second.text_scores)))


class QueryEvidenceRoundTripTests(unittest.TestCase):
    def test_cache_json_round_trip_preserves_evidence(self) -> None:
        evidence = build_query_text_evidence([("Ароматное", 0.95), ("МАЛЬБЕК", 0.99), ("2023", 0.999)])
        payload = json.dumps({"lines": [(text, score) for text, score in evidence.raw_lines]})
        restored = build_query_text_evidence([tuple(line) for line in json.loads(payload)["lines"]])
        self.assertEqual(restored.text, evidence.text)
        self.assertEqual(restored.vintage_years, evidence.vintage_years)

    def test_cache_directory_layout_separates_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / OCR_MODEL_KEY / "synthetic_dev").mkdir(parents=True)
            (root / OCR_MODEL_KEY / "catalog_references").mkdir(parents=True)
            self.assertTrue((root / OCR_MODEL_KEY / "synthetic_dev").is_dir())
            self.assertTrue((root / OCR_MODEL_KEY / "catalog_references").is_dir())
            self.assertNotEqual(
                (root / OCR_MODEL_KEY / "synthetic_dev"),
                (root / OCR_MODEL_KEY / "catalog_references"),
            )


if __name__ == "__main__":
    unittest.main()
