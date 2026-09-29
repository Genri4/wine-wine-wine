from __future__ import annotations

import unittest

from recognition.vintage_disambiguation import (
    build_vintage_challenge,
    candidate_year_evidence,
    conservative_vintage_rerank,
    decide_query_year,
    extract_years_with_corrections,
    normalize_year_token,
    oracle_ceiling,
)
from recognition.reference_guided_year import (
    crop_bounds_valid,
    normalize_year_token as rg_normalize,
    projected_crop_bounds,
)


class YearNormalizationTests(unittest.TestCase):
    def test_exact_year_token(self) -> None:
        self.assertEqual(normalize_year_token("2021"), ("2021", None))

    def test_glyph_correction_year_like_only(self) -> None:
        year, correction = normalize_year_token("202I")
        self.assertEqual(year, "2021")
        self.assertEqual(correction, "202I")
        year, correction = normalize_year_token("2O2Z")
        self.assertEqual(year, "2022")
        self.assertEqual(correction, "2O2Z")

    def test_non_year_tokens_untouched(self) -> None:
        self.assertEqual(normalize_year_token("12345"), (None, None))
        self.assertEqual(normalize_year_token("BADO"), (None, None))
        self.assertEqual(normalize_year_token("каберне"), (None, None))
        # 5 digits must not become a year
        self.assertEqual(normalize_year_token("202I1"), (None, None))

    def test_ocr_extraction_uses_confidence(self) -> None:
        lines = [
            {"text": "2021", "confidence": 0.9},
            {"text": "2019", "confidence": 0.3},
        ]
        years, corrections = extract_years_with_corrections(lines, min_confidence=0.5)
        self.assertIn("2021", years)
        self.assertNotIn("2019", years)
        self.assertEqual(corrections, [])

    def test_cyrillic_safe(self) -> None:
        lines = [{"text": "Каберне Совиньон 2021", "confidence": 0.9}]
        years, _ = extract_years_with_corrections(lines)
        self.assertIn("2021", years)


class CandidateYearEvidenceTests(unittest.TestCase):
    def test_provenance_preserved(self) -> None:
        evidence = candidate_year_evidence(
            "slug",
            {"title": "Вино 2021", "product_name": ""},
            [{"text": "2021", "confidence": 0.95}],
        )
        self.assertEqual(evidence["year"], "2021")
        self.assertEqual(evidence["provenance"], "multiple_sources_agree")

    def test_conflicting_sources_unknown(self) -> None:
        evidence = candidate_year_evidence(
            "slug",
            {"title": "Вино 2021", "product_name": ""},
            [{"text": "2022", "confidence": 0.95}],
        )
        self.assertIsNone(evidence["year"])
        self.assertEqual(evidence["provenance"], "unknown_conflict")

    def test_no_sources_unknown(self) -> None:
        evidence = candidate_year_evidence("slug", {"title": "Вино", "product_name": ""}, [])
        self.assertIsNone(evidence["year"])
        self.assertEqual(evidence["provenance"], "unknown")


class ChallengeSliceTests(unittest.TestCase):
    BASE = {
        "correct_top1": "False",
        "predicted_slug": "wine-b",
        "top5_scores": "[0.9, 0.85, 0.8, 0.7, 0.6]",
    }

    def _rows(self, top5):
        return [
            {"query_id": "q1", "target_slug": "wine-a", "family_id": "f1", **self.BASE, "top5_slugs": json_dump(top5)},
        ]

    def test_target_must_be_in_top5(self) -> None:
        manifest = [{"query_id": "q1", "target_slug": "wine-a", "family_id": "f1"}]
        baseline = {"q1": {"correct_top1": "False", "predicted_slug": "wine-b", "top5_scores": "[0.9]", "top5_slugs": '["wine-b","wine-c"]'}}
        challenge = build_vintage_challenge(
            "hard", manifest, baseline,
            {"wine-a": "f1", "wine-b": "f1", "wine-c": "f1"},
            {"f1": "vintage"},
            {"wine-a": "2021", "wine-b": "2022"},
        )
        # target wine-a NOT in top5 -> excluded
        self.assertEqual(challenge, [])

    def test_competing_family_member_required(self) -> None:
        manifest = [{"query_id": "q1", "target_slug": "wine-a", "family_id": "f1"}]
        baseline = {
            "q1": {
                "correct_top1": "True", "predicted_slug": "wine-a",
                "top5_scores": "[0.9, 0.8]", "top5_slugs": '["wine-a","wine-x"]',
            }
        }
        challenge = build_vintage_challenge(
            "hard", manifest, baseline,
            {"wine-a": "f1", "wine-x": "f-other"},
            {"f1": "vintage"},
            {"wine-a": "2021"},
        )
        # no same-family competitor in top5 -> excluded
        self.assertEqual(challenge, [])

    def test_competitor_must_have_different_known_year(self) -> None:
        manifest = [{"query_id": "q1", "target_slug": "wine-a", "family_id": "f1"}]
        baseline = {
            "q1": {
                "correct_top1": "True", "predicted_slug": "wine-a",
                "top5_scores": "[0.9, 0.8]", "top5_slugs": '["wine-a","wine-b"]',
            }
        }
        # competitor year unknown -> excluded
        challenge = build_vintage_challenge(
            "hard", manifest, baseline,
            {"wine-a": "f1", "wine-b": "f1"},
            {"f1": "vintage"},
            {"wine-a": "2021", "wine-b": None},
        )
        self.assertEqual(challenge, [])
        # competitor with different year -> included
        challenge = build_vintage_challenge(
            "hard", manifest, baseline,
            {"wine-a": "f1", "wine-b": "f1"},
            {"f1": "vintage"},
            {"wine-a": "2021", "wine-b": "2022"},
        )
        self.assertEqual(len(challenge), 1)

    def test_slice_is_deterministic(self) -> None:
        manifest = [{"query_id": "q1", "target_slug": "wine-a", "family_id": "f1"}]
        baseline = {
            "q1": {
                "correct_top1": "True", "predicted_slug": "wine-a",
                "top5_scores": "[0.9, 0.8]", "top5_slugs": '["wine-a","wine-b"]',
            }
        }
        kwargs = dict(
            family_by_slug={"wine-a": "f1", "wine-b": "f1"},
            family_types={"f1": "vintage"},
            candidate_years={"wine-a": "2021", "wine-b": "2022"},
        )
        first = build_vintage_challenge("hard", manifest, baseline, **kwargs)
        second = build_vintage_challenge("hard", manifest, baseline, **kwargs)
        self.assertEqual(first, second)


class OracleCeilingTests(unittest.TestCase):
    def test_oracle_counts(self) -> None:
        challenge = [
            {
                "baseline_correct_top1": False,
                "target_slug": "wine-a",
                "top5_slugs": ["wine-a", "wine-b"],
                "target_year": "2021",
                "same_family_competitors": [{"slug": "wine-b", "year": "2022"}],
                "member_years": {"wine-a": "2021", "wine-b": "2022"},
            },
            {
                "baseline_correct_top1": True,
                "target_slug": "wine-a2",
                "top5_slugs": ["wine-a2", "wine-c"],
                "target_year": "2020",
                "same_family_competitors": [{"slug": "wine-c", "year": "2020"}],
                # Two members share the target year and a third year exists,
                # so the years DO distinguish the family but the target year
                # itself is ambiguous (two members wear it).
                "member_years": {"wine-a2": "2020", "wine-c": "2020", "wine-d": "2019"},
            },
        ]
        result = oracle_ceiling(challenge)
        self.assertEqual(result["resolvable_by_year"], 1)
        self.assertEqual(result["ambiguous_multiple_members_same_year"], 1)
        # baseline (1) + resolvable (1) over challenge queries (2)
        self.assertEqual(result["oracle_top1_share"], 1.0)
        self.assertEqual(result["oracle_top1"], 2)


class ConservativeRerankTests(unittest.TestCase):
    def test_swap_inside_same_family_only(self) -> None:
        top5 = ["wine-b", "wine-a", "wine-x", "wine-y", "wine-z"]
        outcome = conservative_vintage_rerank(
            top5, [0.9, 0.85, 0.8, 0.7, 0.6],
            {"wine-a": "f1", "wine-b": "f1", "wine-x": "f2"},
            {"wine-a": "2021", "wine-b": "2022", "wine-x": None},
            query_year="2021", query_year_confidence=0.9,
        )
        self.assertEqual(outcome["action"], "family_swap")
        self.assertEqual([top5[i] for i in outcome["order"]][0], "wine-a")

    def test_unrelated_candidates_never_reordered(self) -> None:
        top5 = ["wine-b", "wine-x", "wine-a", "wine-y", "wine-z"]
        outcome = conservative_vintage_rerank(
            top5, [0.9, 0.85, 0.8, 0.7, 0.6],
            {"wine-a": "f1", "wine-b": "f-other", "wine-x": "f2"},
            {"wine-a": "2021", "wine-b": "2022", "wine-x": None},
            query_year="2021", query_year_confidence=0.9,
        )
        self.assertEqual(outcome["action"], "no_action")
        self.assertEqual([top5[i] for i in outcome["order"]], top5)

    def test_low_confidence_query_year_ignored(self) -> None:
        top5 = ["wine-b", "wine-a", "wine-x", "wine-y", "wine-z"]
        outcome = conservative_vintage_rerank(
            top5, [0.9, 0.85, 0.8, 0.7, 0.6],
            {"wine-a": "f1", "wine-b": "f1"},
            {"wine-a": "2021", "wine-b": "2022"},
            query_year="2021", query_year_confidence=0.3,
        )
        self.assertEqual(outcome["action"], "no_action")

    def test_query_year_matching_top1_is_noop(self) -> None:
        top5 = ["wine-b", "wine-a", "wine-x", "wine-y", "wine-z"]
        outcome = conservative_vintage_rerank(
            top5, [0.9, 0.85, 0.8, 0.7, 0.6],
            {"wine-a": "f1", "wine-b": "f1"},
            {"wine-a": "2021", "wine-b": "2022"},
            query_year="2022", query_year_confidence=0.9,
        )
        self.assertEqual(outcome["action"], "no_action")
        self.assertEqual(outcome["reason"], "top1_year_already_matches")


class QueryYearDecisionTests(unittest.TestCase):
    def test_no_year(self) -> None:
        result = decide_query_year({}, {})
        self.assertEqual(result.decision, "no_year")
        self.assertIsNone(result.decided_year)

    def test_unique_year(self) -> None:
        result = decide_query_year({"2021": [{"confidence": 0.9}]}, {"wine-a": "2021"})
        self.assertEqual(result.decided_year, "2021")
        self.assertEqual(result.decision, "unique")

    def test_ambiguous_stays_ambiguous(self) -> None:
        result = decide_query_year(
            {"2021": [{"confidence": 0.7}], "2022": [{"confidence": 0.7}]},
            {"wine-a": "2021", "wine-b": "2022"},
        )
        self.assertEqual(result.decision, "ambiguous")


class HomographyTests(unittest.TestCase):
    def test_projected_crop_bounds_clipped(self) -> None:
        import numpy as np

        projected = np.array([[-50.0, -50.0], [5000.0, -50.0], [5000.0, 5000.0], [-50.0, 5000.0]])
        bounds = projected_crop_bounds(projected, 1024, 1024)
        self.assertEqual(bounds, (0, 0, 1024, 1024))

    def test_crop_bounds_valid_area_ratio(self) -> None:
        # 100x100 reference box; 120x120 crop -> ratio 1.44, valid
        self.assertTrue(crop_bounds_valid((0, 0, 120, 120), (0, 0, 100, 100)))
        # 10x10 crop -> ratio 0.01, invalid
        self.assertFalse(crop_bounds_valid((0, 0, 10, 10), (0, 0, 100, 100)))

    def test_synthetic_homography_validation(self) -> None:
        import numpy as np

        from recognition.reference_guided_year import compute_homography

        import cv2

        # Rich textured pattern: many SIFT keypoints.
        rng = np.random.default_rng(7)
        base = rng.integers(0, 255, (500, 500, 3), dtype=np.uint8)
        base = cv2.GaussianBlur(base, (3, 3), 0)
        matrix = cv2.getRotationMatrix2D((250, 250), 5, 1.0)
        query = cv2.warpAffine(base, matrix, (500, 500))
        result = compute_homography(base, query)
        self.assertTrue(result.valid, result.reason)


import json  # noqa: E402


def json_dump(value):
    import json

    return json.dumps(value)


if __name__ == "__main__":
    unittest.main()
