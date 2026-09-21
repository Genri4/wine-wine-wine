from __future__ import annotations

import unittest

from recognition.text_signals import (
    DEFAULT_MIN_OCR_CONFIDENCE,
    build_candidate_text_index,
    build_query_text_evidence,
    build_reference_ocr_evidence,
    compute_text_signals,
)


def _catalog_row() -> dict[str, str]:
    return {
        "slug": "amelia-2023",
        "title": "Amelia, 2023",
        "category": "Вино",
        "color": "Красное",
        "region": "Крым",
        "grape": "Мерло",
        "winery": "Esse",
        "photo_name": "abc1234567",
    }


class CandidateTextIndexTests(unittest.TestCase):
    def test_slug_is_not_used_as_text(self) -> None:
        index = build_candidate_text_index(_catalog_row())
        self.assertNotIn("amelia-2023", index.metadata_text.normalized)
        self.assertNotIn(index.slug, index.title.normalized)

    def test_vintage_year_from_title(self) -> None:
        self.assertEqual(build_candidate_text_index(_catalog_row()).vintage_year, "2023")

    def test_vintage_year_unknown_when_title_has_no_year(self) -> None:
        row = _catalog_row() | {"title": "Amelia"}
        self.assertIsNone(build_candidate_text_index(row).vintage_year)

    def test_description_never_enters_metadata_text(self) -> None:
        row = _catalog_row() | {"description": "длинное описание вина и его вкуса"}
        index = build_candidate_text_index(row)
        self.assertNotIn("описание", index.metadata_text.normalized)


class QueryTextEvidenceTests(unittest.TestCase):
    def test_low_confidence_lines_excluded_from_scoring_text(self) -> None:
        evidence = build_query_text_evidence(
            [("AMELIA 2023", 0.9), ("мусорный текст", 0.2)],
            min_confidence=0.5,
        )
        self.assertIn("amelia 2023", evidence.text)
        self.assertNotIn("мусорный", evidence.text)
        # Raw lines are still cached for diagnostics.
        self.assertEqual(len(evidence.raw_lines), 2)

    def test_years_extracted_from_high_confidence_only(self) -> None:
        evidence = build_query_text_evidence(
            [("2023", 0.4), ("2021", 0.9)],
            min_confidence=0.5,
        )
        self.assertEqual(evidence.vintage_years, ("2021",))

    def test_transliterated_variant_present(self) -> None:
        evidence = build_query_text_evidence([("Ароматное", 0.95)])
        self.assertEqual(evidence.transliterated, "aromatnoe")

    def test_empty_ocr_is_safe(self) -> None:
        evidence = build_query_text_evidence([])
        self.assertEqual(evidence.text, "")
        self.assertEqual(evidence.vintage_years, ())

    def test_default_min_confidence_is_fixed(self) -> None:
        self.assertEqual(DEFAULT_MIN_OCR_CONFIDENCE, 0.5)


class ComputeTextSignalsTests(unittest.TestCase):
    def test_exact_label_text_scores_high_on_name(self) -> None:
        row = _catalog_row()
        index = build_candidate_text_index(row)
        query = build_query_text_evidence([("Amelia 2023", 0.95)])
        signals = compute_text_signals(query, index)
        self.assertGreaterEqual(signals["metadata_name_score"], 0.9)
        self.assertGreaterEqual(signals["metadata_text_score"], 0.7)
        self.assertEqual(signals["vintage_match"], "exact_match")

    def test_winery_partial_match_inside_ocr(self) -> None:
        index = build_candidate_text_index(_catalog_row())
        query = build_query_text_evidence([("Winery Esse presents Amelia 2023", 0.95)])
        signals = compute_text_signals(query, index)
        self.assertGreaterEqual(signals["winery_score"], 0.9)

    def test_unrelated_text_scores_low(self) -> None:
        index = build_candidate_text_index(_catalog_row())
        query = build_query_text_evidence([("Совершенно другой текст без совпадений", 0.95)])
        signals = compute_text_signals(query, index)
        self.assertLess(signals["metadata_name_score"], 0.5)
        self.assertLess(signals["winery_score"], 0.6)

    def test_reference_ocr_score_zero_without_reference(self) -> None:
        index = build_candidate_text_index(_catalog_row())
        query = build_query_text_evidence([("Amelia 2023", 0.95)])
        signals = compute_text_signals(query, index, reference_ocr=None)
        self.assertEqual(signals["reference_ocr_score"], 0.0)

    def test_reference_ocr_score_positive_with_reference(self) -> None:
        index = build_candidate_text_index(_catalog_row())
        query = build_query_text_evidence([("Amelia 2023", 0.95)])
        reference = build_reference_ocr_evidence([("AMELIA", 0.9), ("2023", 0.95)])
        signals = compute_text_signals(query, index, reference_ocr=reference)
        self.assertGreater(signals["reference_ocr_score"], 0.5)

    def test_vintage_mismatch_detected(self) -> None:
        index = build_candidate_text_index(_catalog_row())
        query = build_query_text_evidence([("Amelia 2021", 0.95)])
        signals = compute_text_signals(query, index)
        self.assertEqual(signals["vintage_match"], "mismatch")

    def test_vintage_unknown_when_query_ocr_has_no_year(self) -> None:
        index = build_candidate_text_index(_catalog_row())
        query = build_query_text_evidence([("Amelia без года", 0.95)])
        signals = compute_text_signals(query, index)
        self.assertEqual(signals["vintage_match"], "unknown")

    def test_scores_are_bounded(self) -> None:
        index = build_candidate_text_index(_catalog_row())
        query = build_query_text_evidence([("Amelia 2023 Esse Крым Мерло", 0.99)])
        signals = compute_text_signals(query, index, build_reference_ocr_evidence([("Amelia 2023", 0.99)]))
        for key in (
            "metadata_name_score",
            "winery_score",
            "grape_score",
            "region_score",
            "metadata_text_score",
            "reference_ocr_score",
        ):
            self.assertGreaterEqual(signals[key], 0.0)
            self.assertLessEqual(signals[key], 1.0)


if __name__ == "__main__":
    unittest.main()
