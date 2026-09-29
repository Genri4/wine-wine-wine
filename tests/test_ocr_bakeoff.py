from __future__ import annotations

import unittest

from recognition.gated_strategy import seeded_product_sample
from recognition.ocr_engine import CURRENT_ESLAV_CONFIG, OCR_CONFIGS, OCR_MODEL_KEY
from recognition.structured_reranker import (
    FEATURE_NAMES,
    LogisticReranker,
    build_feature_row,
    build_idf,
    candidate_document_text,
    fit_logistic_reranker,
)
from recognition.text_signals import (
    build_candidate_text_index,
    build_query_text_evidence,
    build_reference_ocr_evidence,
    compute_text_signals,
)


def _sample_inputs():
    query = build_query_text_evidence(
        [("МАЛЬБЕК", 0.99), ("2023", 1.0), ("резерв", 0.9)]
    )
    candidate = build_candidate_text_index(
        {
            "slug": "aromatnoe-malbek-rezerv-krasnoe-suhoe-125",
            "title": "Ароматное Мальбек Резерв",
            "winery": "Ароматное",
            "grape": "Мальбек",
            "region": "Крым",
        }
    )
    reference = build_reference_ocr_evidence([("МАЛЬБЕК", 0.99), ("2023", 1.0), ("РЕЗЕРВ", 0.95)])
    return query, candidate, reference


class OcrConfigRegistryTests(unittest.TestCase):
    def test_eslav_cache_key_unchanged(self) -> None:
        self.assertEqual(CURRENT_ESLAV_CONFIG.cache_key, OCR_MODEL_KEY)
        self.assertEqual(
            OCR_CONFIGS["current_eslav"].cache_key,
            "paddleocr3.7_ppocrv5_server_det_eslav_v5_mobile_rec",
        )

    def test_all_configs_have_distinct_cache_keys(self) -> None:
        keys = [config.cache_key for config in OCR_CONFIGS.values()]
        self.assertEqual(len(keys), len(set(keys)))

    def test_cyrillic_uses_same_detector(self) -> None:
        self.assertEqual(
            OCR_CONFIGS["cyrillic"].detection_model,
            OCR_CONFIGS["current_eslav"].detection_model,
        )
        self.assertNotEqual(
            OCR_CONFIGS["cyrillic"].recognition_model,
            OCR_CONFIGS["current_eslav"].recognition_model,
        )

    def test_vl_config_kind(self) -> None:
        self.assertEqual(OCR_CONFIGS["paddleocr_vl"].kind, "vl")
        self.assertEqual(OCR_CONFIGS["current_eslav"].kind, "pipeline")


class FeatureBuilderTests(unittest.TestCase):
    def test_feature_values_bounded_and_finite(self) -> None:
        query, candidate, reference = _sample_inputs()
        idf = build_idf({"a": "МАЛЬБЕК 2023", "b": "сухое вино"})
        signals = compute_text_signals(query, candidate, reference)
        row = build_feature_row([0.9, 0.8, 0.7, 0.6, 0.5], 0, query, candidate, reference, signals, idf)
        for name in FEATURE_NAMES:
            value = row[name]
            self.assertTrue(abs(value) < 1e6, name)
            self.assertFalse(value != value, name)  # NaN check

    def test_deterministic_feature_extraction(self) -> None:
        query, candidate, reference = _sample_inputs()
        idf = build_idf({"a": "МАЛЬБЕК 2023"})
        signals = compute_text_signals(query, candidate, reference)
        first = build_feature_row([0.9, 0.8, 0.7, 0.6, 0.5], 1, query, candidate, reference, signals, idf)
        second = build_feature_row([0.9, 0.8, 0.7, 0.6, 0.5], 1, query, candidate, reference, signals, idf)
        self.assertEqual(first, second)

    def test_margins_are_position_aware(self) -> None:
        query, candidate, reference = _sample_inputs()
        idf = build_idf({"a": "МАЛЬБЕК"})
        signals = compute_text_signals(query, candidate, reference)
        top = build_feature_row([0.9, 0.8, 0.7, 0.6, 0.5], 0, query, candidate, reference, signals, idf)
        second = build_feature_row([0.9, 0.8, 0.7, 0.6, 0.5], 1, query, candidate, reference, signals, idf)
        self.assertAlmostEqual(top["image_margin_top1"], 0.0, places=6)
        self.assertAlmostEqual(second["image_margin_top1"], 0.1, places=6)

    def test_missing_reference_ocr_is_safe(self) -> None:
        query, candidate, _ = _sample_inputs()
        idf = build_idf({})
        signals = compute_text_signals(query, candidate, None)
        row = build_feature_row([0.9, 0.8, 0.7, 0.6, 0.5], 0, query, candidate, None, signals, idf)
        self.assertEqual(row["reference_ocr_score"], 0.0)
        self.assertEqual(row["ocr_token_overlap"], 0.0)

    def test_idf_rare_tokens_weigh_more(self) -> None:
        idf = build_idf(
            {
                "a": "каберне вино сухое",
                "b": "каберне вино сухое",
                "c": "каберне лермонт резерв",
            }
        )
        self.assertGreater(idf.get("лермонт", 1.0), idf.get("каберне", 1.0))


class LogisticRerankerTests(unittest.TestCase):
    def test_training_and_deterministic_scoring(self) -> None:
        query, candidate, reference = _sample_inputs()
        idf = build_idf({"a": "МАЛЬБЕК 2023"})
        signals = compute_text_signals(query, candidate, reference)
        positive = build_feature_row([0.5, 0.9, 0.85, 0.8, 0.7], 0, query, candidate, reference, signals, idf)
        negative = build_feature_row([0.5, 0.9, 0.85, 0.8, 0.7], 1, query, candidate, reference, signals, idf)
        rows = [positive, negative] * 10
        labels = [1, 0] * 10
        model = fit_logistic_reranker(rows, labels)
        score_positive = model.score_row(positive)
        score_negative = model.score_row(negative)
        self.assertGreater(score_positive, score_negative)
        self.assertGreaterEqual(score_positive, 0.0)
        self.assertLessEqual(score_positive, 1.0)
        # Determinism
        model2 = fit_logistic_reranker(rows, labels)
        self.assertEqual(model.to_json(), model2.to_json())

    def test_json_round_trip(self) -> None:
        query, candidate, reference = _sample_inputs()
        idf = build_idf({"a": "МАЛЬБЕК 2023"})
        signals = compute_text_signals(query, candidate, reference)
        rows = [
            build_feature_row([0.5, 0.9, 0.85, 0.8, 0.7], position, query, candidate, reference, signals, idf)
            for position in range(5)
        ]
        model = fit_logistic_reranker(rows, [1, 0, 0, 0, 0])
        restored = LogisticReranker.from_json(model.to_json())
        self.assertEqual(restored.score_row(rows[0]), model.score_row(rows[0]))

    def test_one_query_five_candidate_rows(self) -> None:
        # The contract: one query produces exactly 5 candidate rows.
        query, candidate, reference = _sample_inputs()
        idf = build_idf({"a": "МАЛЬБЕК"})
        signals = compute_text_signals(query, candidate, reference)
        image_scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        rows = [
            build_feature_row(image_scores, position, query, candidate, reference, signals, idf)
            for position in range(len(image_scores))
        ]
        self.assertEqual(len(rows), 5)


class BgeDocumentTextTests(unittest.TestCase):
    def test_slug_never_in_document_text(self) -> None:
        query, candidate, reference = _sample_inputs()
        doc = candidate_document_text(candidate, reference)
        self.assertNotIn("aromatnoe-malbek-rezerv-krasnoe-suhoe-125", doc)
        # Reference OCR text is normalized to lowercase; metadata keeps case.
        self.assertIn("мальбек", doc)
        self.assertIn("Мальбек", doc)

    def test_document_text_without_reference_ocr(self) -> None:
        _, candidate, _ = _sample_inputs()
        doc = candidate_document_text(candidate, None)
        self.assertTrue(doc)
        self.assertNotIn("aromatnoe-malbek", doc)


class GroupSplitTests(unittest.TestCase):
    def test_seeded_sample_is_deterministic_and_disjoint(self) -> None:
        products = [f"wine-{index:03d}" for index in range(100)]
        first = set(seeded_product_sample(products, 40, 20260920))
        second = set(seeded_product_sample(products, 40, 20260920))
        self.assertEqual(first, second)
        heldout = set(products) - first
        self.assertEqual(len(heldout), 60)
        self.assertFalse(first & heldout)


if __name__ == "__main__":
    unittest.main()
