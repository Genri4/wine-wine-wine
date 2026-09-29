from __future__ import annotations

import unittest

import torch

from recognition.gated_strategy import (
    FALLBACK_CROP_VIEWS,
    GateRule,
    confidence_features,
    quantile,
    retrieve_with_confidence_gate,
    roc_auc,
    seeded_product_sample,
    stratified_product_split,
)
from recognition.crop_diagnostics import full_ranking
from recognition.view_strategy import (
    BASELINE_FULL,
    PREPROCESSING_V1,
    aggregate_strategy_scores,
)


class _SizeImage:
    """Duck-typed image double: the runtime only reads width/height/crop."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def crop(self, box: tuple[int, int, int, int]) -> "_SizeImage":
        left, upper, right, lower = box
        return _SizeImage(right - left, lower - upper)


def _direction(width: int, height: int, dim: int = 8) -> torch.Tensor:
    vector = torch.zeros(dim)
    vector[0] = width / 1000.0
    vector[1] = height / 1000.0
    return vector / vector.norm()


class _RecordingEncoder:
    """Encoder double: embedding depends on image size and call index, so
    tests can verify exactly which views were encoded and in what order."""

    def __init__(self) -> None:
        self.calls: list[list[tuple[int, int]]] = []
        self.batch_size = 4
        self.device = "cpu"
        self._torch = torch

    def encode_pil(self, images: list[_SizeImage]) -> list[list[float]]:
        call_index = len(self.calls)
        self.calls.append([(image.width, image.height) for image in images])
        vectors = []
        for width, height in self.calls[-1]:
            vector = torch.zeros(8)
            vector[0] = width / 1000.0
            vector[1] = height / 1000.0
            vector[2] = call_index * 0.1
            vectors.append(vector / vector.norm())
        return torch.stack(vectors).tolist()


def _confident_matrix() -> torch.Tensor:
    # Rows follow SLUGS order. wine-b matches the full-image direction
    # exactly; the rest are orthogonal -> top1/top2 margin = 1.0.
    rows = torch.stack(
        [
            _orthogonal(1),
            _direction(100, 80),
            _orthogonal(2),
            _orthogonal(3),
        ]
    )
    return rows


def _uncertain_matrix() -> torch.Tensor:
    # wine-a is nearly collinear with the full-image direction -> tiny margin.
    rows = torch.stack(
        [
            _direction(90, 90),
            _direction(100, 80),
            _orthogonal(2),
            _orthogonal(3),
        ]
    )
    return rows


def _orthogonal(index: int, dim: int = 8) -> torch.Tensor:
    vector = torch.zeros(dim)
    vector[index + 1] = 1.0
    return vector


SLUGS = ["wine-a", "wine-b", "wine-c", "wine-d"]


class GateRuleTests(unittest.TestCase):
    def test_gate_never_uses_ground_truth(self) -> None:
        # The decision API accepts only score values; identical score inputs
        # must always produce identical decisions regardless of any target.
        rule = GateRule("margin", margin_threshold=0.05)
        self.assertFalse(rule.should_fallback(0.9, 0.7, 0.6))
        self.assertTrue(rule.should_fallback(0.9, 0.86, 0.6))
        self.assertFalse(
            GateRule("margin", margin_threshold=0.05).should_fallback(0.9, 0.7, 0.6)
        )

    def test_gate_uses_only_full_ranking_information(self) -> None:
        features = confidence_features([0.9, 0.8, 0.7, 0.6, 0.5])
        self.assertAlmostEqual(features["top1_top2_margin"], 0.1, places=6)
        self.assertAlmostEqual(features["top1_top5_margin"], 0.4, places=6)
        self.assertEqual(confidence_features([0.9, 0.8, 0.7, 0.6, 0.5]), features)
        self.assertEqual(features["top1_score"], 0.9)

    def test_rule_types(self) -> None:
        margin_rule = GateRule("margin", margin_threshold=0.1)
        score_rule = GateRule("score", score_threshold=0.5)
        combo_rule = GateRule("margin_or_score", margin_threshold=0.1, score_threshold=0.5)
        self.assertFalse(margin_rule.should_fallback(0.9, 0.75, 0.5))
        self.assertTrue(score_rule.should_fallback(0.4, 0.3, 0.2))
        self.assertFalse(score_rule.should_fallback(0.9, 0.1, 0.05))
        self.assertTrue(combo_rule.should_fallback(0.9, 0.88, 0.5))
        self.assertTrue(combo_rule.should_fallback(0.3, 0.05, 0.01))
        self.assertFalse(combo_rule.should_fallback(0.9, 0.78, 0.5))

    def test_invalid_rules_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GateRule("oracle")
        with self.assertRaises(ValueError):
            GateRule("margin")
        with self.assertRaises(ValueError):
            GateRule("score")
        with self.assertRaises(ValueError):
            GateRule("margin_or_score", margin_threshold=0.1)

    def test_fixed_threshold_reproducible(self) -> None:
        rule = GateRule("margin_or_score", margin_threshold=0.02, score_threshold=0.55)
        inputs = [(0.9, 0.87, 0.6), (0.4, 0.1, 0.05), (0.6, 0.59, 0.5)]
        first = [rule.should_fallback(*values) for values in inputs]
        second = [
            GateRule("margin_or_score", margin_threshold=0.02, score_threshold=0.55).should_fallback(*values)
            for values in inputs
        ]
        self.assertEqual(first, second)
        self.assertEqual(first, [False, True, True])


class SplitTests(unittest.TestCase):
    MANIFEST = [
        {"query_id": f"q-{slug}-{scenario}", "target_slug": slug, "target_family_id": family, "family_type": family_type}
        for slug, family, family_type in (
            ("rep-a", "", ""),
            ("rep-b", "", ""),
            ("rep-c", "", ""),
            ("rep-d", "", ""),
            ("v-a1", "vf1", "vintage"),
            ("v-a2", "vf1", "vintage"),
            ("v-b1", "vf2", "vintage"),
            ("v-b2", "vf2", "vintage"),
        )
        for scenario in ("slight_angle", "glare_bad_light", "distance_crop", "handheld")
    ]

    def test_deterministic_split(self) -> None:
        first = stratified_product_split(self.MANIFEST, 20260919)
        second = stratified_product_split(self.MANIFEST, 20260919)
        self.assertEqual(first, second)

    def test_all_scenarios_of_product_in_same_split(self) -> None:
        split = stratified_product_split(self.MANIFEST, 20260919)
        by_query = {row["query_id"]: split[row["target_slug"]] for row in self.MANIFEST}
        for slug in ("rep-a", "rep-b", "v-a1", "v-a2", "v-b1", "v-b2"):
            sides = {
                by_query[f"q-{slug}-{scenario}"]
                for scenario in ("slight_angle", "glare_bad_light", "distance_crop", "handheld")
            }
            self.assertEqual(len(sides), 1, f"product {slug} split across scenarios")

    def test_family_members_stay_together(self) -> None:
        split = stratified_product_split(self.MANIFEST, 20260919)
        self.assertEqual(split["v-a1"], split["v-a2"])
        self.assertEqual(split["v-b1"], split["v-b2"])

    def test_split_is_balanced_and_stratified(self) -> None:
        split = stratified_product_split(self.MANIFEST, 20260919)
        calibration = sorted(slug for slug, side in split.items() if side == "calibration")
        held_out = sorted(slug for slug, side in split.items() if side == "held_out")
        self.assertEqual(len(calibration), len(held_out))
        self.assertEqual(len(calibration), 4)
        self.assertEqual(len([slug for slug in calibration if slug.startswith("rep")]), 2)
        self.assertNotEqual(split["v-a1"], split["v-b1"])

    def test_seeded_product_sample_deterministic(self) -> None:
        slugs = [f"wine-{index:03d}" for index in range(50)]
        first = seeded_product_sample(slugs, 10, 20260919)
        second = seeded_product_sample(slugs, 10, 20260919)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        with self.assertRaises(ValueError):
            seeded_product_sample(slugs, 51, 20260919)


class GatedRuntimeTests(unittest.TestCase):
    def test_confident_query_skips_crop_inference(self) -> None:
        encoder = _RecordingEncoder()
        rule = GateRule("margin", margin_threshold=0.5)
        ranked, diagnostics, latencies = retrieve_with_confidence_gate(
            encoder, _confident_matrix(), SLUGS, _SizeImage(100, 80), rule, "cpu"
        )
        self.assertFalse(diagnostics.used_fallback)
        self.assertEqual(diagnostics.views_used, 1)
        self.assertEqual(diagnostics.gate_reason, "confident")
        self.assertEqual(len(encoder.calls), 1)
        self.assertEqual(encoder.calls[0], [(100, 80)])
        self.assertEqual(latencies["embed_crops_ms"], 0.0)
        self.assertEqual(latencies["retrieval_fallback_ms"], 0.0)
        self.assertEqual(ranked[0][0], "wine-b")

    def test_fallback_query_encodes_exactly_crop85_and_crop70(self) -> None:
        encoder = _RecordingEncoder()
        rule = GateRule("score", score_threshold=1.5)  # top1_score <= 1.0 -> always fallback
        ranked, diagnostics, latencies = retrieve_with_confidence_gate(
            encoder, _confident_matrix(), SLUGS, _SizeImage(100, 80), rule, "cpu"
        )
        self.assertTrue(diagnostics.used_fallback)
        self.assertEqual(diagnostics.views_used, 3)
        self.assertTrue(diagnostics.gate_reason.startswith("fallback"))
        self.assertEqual(len(encoder.calls), 2)
        # The full image is encoded exactly once; the second call carries
        # exactly the two crops (full is not recomputed).
        self.assertEqual(encoder.calls[0], [(100, 80)])
        self.assertEqual(encoder.calls[1], [(85, 68), (70, 56)])
        self.assertGreater(latencies["embed_crops_ms"], 0.0)
        self.assertGreater(latencies["retrieval_fallback_ms"], 0.0)
        self.assertEqual(ranked[0][0], "wine-b")

    def test_fallback_aggregation_matches_preprocessing_v1_semantics(self) -> None:
        encoder = _RecordingEncoder()
        rule = GateRule("score", score_threshold=1.5)
        matrix = _confident_matrix()
        ranked, _, _ = retrieve_with_confidence_gate(
            encoder, matrix, SLUGS, _SizeImage(100, 80), rule, "cpu"
        )

        # Independent recomputation with the preprocessing_v1 machinery,
        # replaying the exact view vectors the runtime produced: the full
        # view is call 0, and both crops are the single batched call 1.
        def view_vector(size: tuple[int, int], call_index: int) -> torch.Tensor:
            vector = torch.zeros(8)
            vector[0] = size[0] / 1000.0
            vector[1] = size[1] / 1000.0
            vector[2] = call_index * 0.1
            return vector / vector.norm()

        view_scores = [
            (view_vector(size, call_index) @ matrix.transpose(0, 1)).tolist()
            for size, call_index in [((100, 80), 0), ((85, 68), 1), ((70, 56), 1)]
        ]
        expected = full_ranking(aggregate_strategy_scores(view_scores, PREPROCESSING_V1), SLUGS)
        self.assertEqual([slug for slug, _ in ranked], [slug for slug, _ in expected])
        for (expected_slug, expected_score), (actual_slug, actual_score) in zip(expected, ranked):
            self.assertEqual(expected_slug, actual_slug)
            self.assertAlmostEqual(expected_score, actual_score, places=6)


    def test_gate_diagnostics_and_fallback_rate(self) -> None:
        encoder = _RecordingEncoder()
        rule = GateRule("margin", margin_threshold=0.5)
        confident_ranked, confident_diag, confident_latencies = retrieve_with_confidence_gate(
            encoder, _confident_matrix(), SLUGS, _SizeImage(100, 80), rule, "cpu"
        )
        uncertain_ranked, uncertain_diag, uncertain_latencies = retrieve_with_confidence_gate(
            encoder, _uncertain_matrix(), SLUGS, _SizeImage(100, 80), rule, "cpu"
        )
        self.assertFalse(confident_diag.used_fallback)
        self.assertTrue(uncertain_diag.used_fallback)
        fallback_rate = sum(
            1 for diagnostics in (confident_diag, uncertain_diag) if diagnostics.used_fallback
        ) / 2
        self.assertEqual(fallback_rate, 0.5)
        # Structural latency distinction: the fallback path performs extra
        # crop-encoding and re-ranking work; the confident path performs none.
        # (Wall-clock ordering is not asserted: CPU warmup noise dominates
        # these microsecond-scale timings.)
        self.assertEqual(confident_latencies["embed_crops_ms"], 0.0)
        self.assertEqual(confident_latencies["retrieval_fallback_ms"], 0.0)
        self.assertGreaterEqual(uncertain_latencies["embed_crops_ms"], 0.0)
        self.assertGreaterEqual(uncertain_latencies["retrieval_fallback_ms"], 0.0)
        self.assertEqual(confident_ranked[0][0], "wine-b")
        self.assertEqual(uncertain_ranked[0][0], "wine-b")

    def test_latency_path_distinction(self) -> None:
        encoder = _RecordingEncoder()
        rule = GateRule("margin", margin_threshold=0.5)
        _, _, latencies = retrieve_with_confidence_gate(
            encoder, _confident_matrix(), SLUGS, _SizeImage(100, 80), rule, "cpu"
        )
        self.assertEqual(latencies["retrieval_fallback_ms"], 0.0)
        self.assertEqual(latencies["embed_crops_ms"], 0.0)
        self.assertGreater(latencies["embed_full_ms"], 0.0)
        self.assertGreater(latencies["retrieval_gate_ms"], 0.0)


class DefinitionStabilityTests(unittest.TestCase):
    def test_preprocessing_v1_definition_unchanged(self) -> None:
        self.assertEqual(PREPROCESSING_V1.view_names, ("full", "center_85", "center_70"))
        self.assertEqual(PREPROCESSING_V1.aggregation, "mean")

    def test_baseline_full_definition_unchanged(self) -> None:
        self.assertEqual(BASELINE_FULL.view_names, ("full",))
        self.assertEqual(BASELINE_FULL.aggregation, "single")

    def test_fallback_views_are_crop85_and_crop70_only(self) -> None:
        self.assertEqual(FALLBACK_CROP_VIEWS, ("center_85", "center_70"))
        self.assertNotIn("center_55", FALLBACK_CROP_VIEWS)


class SignalDiagnosticsTests(unittest.TestCase):
    def test_roc_auc_known_case(self) -> None:
        self.assertEqual(roc_auc([0.9, 0.8], [0.2, 0.1]), 1.0)
        self.assertEqual(roc_auc([0.1, 0.2], [0.8, 0.9]), 0.0)
        self.assertEqual(roc_auc([0.5, 0.5], [0.5, 0.5]), 0.5)

    def test_roc_auc_handles_ties_and_empty(self) -> None:
        self.assertAlmostEqual(roc_auc([0.9, 0.5], [0.5, 0.1]), 0.875)
        self.assertIsNone(roc_auc([], [0.1]))
        self.assertIsNone(roc_auc([0.1], []))

    def test_quantile_known_values(self) -> None:
        values = [float(index) for index in range(1, 101)]
        self.assertEqual(quantile(values, 0.0), 1.0)
        self.assertEqual(quantile(values, 1.0), 100.0)
        self.assertAlmostEqual(quantile(values, 0.5), 50.5)
        self.assertAlmostEqual(quantile([0.1, 0.2, 0.9], 0.5), 0.2)


class AlignmentAndFamilyTests(unittest.TestCase):
    def test_benchmark_alignment_preserved(self) -> None:
        from recognition.crop_diagnostics import verify_manifest_alignment

        manifest = [{"query_id": "q1", "target_slug": "a"}]
        verify_manifest_alignment(manifest, manifest)
        with self.assertRaises(ValueError):
            verify_manifest_alignment([{"query_id": "q1", "target_slug": "b"}], manifest)

    def test_family_metadata_preserved(self) -> None:
        from recognition.family_metrics import family_diagnostics

        manifest = [
            {"query_id": "q1", "target_slug": "a", "target_family_id": "f1", "family_type": "vintage"},
        ]
        predictions = [{"query_id": "q1", "target_slug": "a", "top5_slugs": '["a"]'}]
        diagnostics = family_diagnostics(predictions, manifest)
        self.assertEqual(diagnostics["hard_family_count"], 1)


if __name__ == "__main__":
    unittest.main()
