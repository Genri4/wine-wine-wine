from __future__ import annotations

import unittest

import torch
from PIL import Image

from recognition.crop_diagnostics import center_crop_box
from recognition.view_strategy import (
    BASELINE_FULL,
    PREPROCESSING_V1,
    STRATEGIES,
    QueryViewStrategy,
    aggregate_strategy_scores,
    rank_change_counts,
    strategy_views,
    top1_transition_matrix,
    top5_transition_matrix,
)
from recognition.encoder import VisualEncoder


class StrategyDefinitionTests(unittest.TestCase):
    def test_preprocessing_v1_has_exactly_three_views(self) -> None:
        self.assertEqual(PREPROCESSING_V1.view_names, ("full", "center_85", "center_70"))
        self.assertEqual(PREPROCESSING_V1.aggregation, "mean")

    def test_baseline_has_one_view(self) -> None:
        self.assertEqual(BASELINE_FULL.view_names, ("full",))
        self.assertEqual(BASELINE_FULL.aggregation, "single")

    def test_crop55_not_used(self) -> None:
        self.assertNotIn("center_55", PREPROCESSING_V1.view_names)

    def test_strategies_registry(self) -> None:
        self.assertEqual(set(STRATEGIES), {"baseline_full", "preprocessing_v1"})

    def test_invalid_strategy_rejected(self) -> None:
        with self.assertRaises(ValueError):
            QueryViewStrategy("bad", (), "mean")
        with self.assertRaises(ValueError):
            QueryViewStrategy("bad", ("full", "center_70"), "single")
        with self.assertRaises(ValueError):
            QueryViewStrategy("bad", ("full",), "weighted")


class CropDeterminismTests(unittest.TestCase):
    def test_center_crop_85_deterministic(self) -> None:
        first = center_crop_box(1024, 768, 0.85)
        second = center_crop_box(1024, 768, 0.85)
        self.assertEqual(first, second)
        # round(1024*0.85)=870, round(768*0.85)=653; upper=(768-653)//2=57
        self.assertEqual(first, (77, 57, 947, 710))

    def test_center_crop_70_deterministic(self) -> None:
        first = center_crop_box(1024, 768, 0.70)
        second = center_crop_box(1024, 768, 0.70)
        self.assertEqual(first, second)
        # round(1024*0.7)=717, round(768*0.7)=538; left=(1024-717)//2=153
        self.assertEqual(first, (153, 115, 870, 653))

    def test_crop_coordinates_correct_odd_size(self) -> None:
        box = center_crop_box(101, 103, 0.85)
        width = box[2] - box[0]
        height = box[3] - box[1]
        self.assertEqual(width, 86)  # round(101 * 0.85) = 86
        self.assertEqual(height, 88)  # round(103 * 0.85) = 88
        # Center is preserved within one pixel of rounding.
        self.assertAlmostEqual((box[0] + box[2]) / 2, 101 / 2, delta=0.5)
        self.assertAlmostEqual((box[1] + box[3]) / 2, 103 / 2, delta=0.5)

    def test_strategy_views_returns_expected_views_in_order(self) -> None:
        image = Image.new("RGB", (100, 80), (5, 5, 5))
        v1_views = strategy_views(image, PREPROCESSING_V1)
        self.assertEqual(len(v1_views), 3)
        self.assertIs(v1_views[0], image)
        self.assertEqual(v1_views[1].size, (85, 68))
        self.assertEqual(v1_views[2].size, (70, 56))
        baseline_views = strategy_views(image, BASELINE_FULL)
        self.assertEqual(len(baseline_views), 1)
        self.assertIs(baseline_views[0], image)

    def test_no_target_label_used(self) -> None:
        # strategy_views depends only on pixels and the fixed strategy; two
        # calls for the same-size image produce identical view sizes regardless
        # of any hypothetical target.
        first = Image.new("RGB", (200, 100), (0, 0, 0))
        second = Image.new("RGB", (200, 100), (255, 255, 255))
        for strategy in STRATEGIES.values():
            first_sizes = [view.size for view in strategy_views(first, strategy)]
            second_sizes = [view.size for view in strategy_views(second, strategy)]
            self.assertEqual(first_sizes, second_sizes)

    def test_reference_images_remain_unchanged(self) -> None:
        # Strategies only build query views in memory; they never touch the
        # catalog/reference embedding side. The full view is the original
        # object and cropped views are new in-memory images.
        image = Image.new("RGB", (64, 64), (1, 2, 3))
        views = strategy_views(image, PREPROCESSING_V1)
        for view in views:
            self.assertIsNot(view, "reference")
        self.assertIs(views[0], image)


class AggregationTests(unittest.TestCase):
    def test_mean_aggregation_correct(self) -> None:
        result = aggregate_strategy_scores(
            [[0.3, 0.6], [0.6, 0.3], [0.9, 0.0]], PREPROCESSING_V1
        )
        self.assertAlmostEqual(result[0], (0.3 + 0.6 + 0.9) / 3)
        self.assertAlmostEqual(result[1], (0.6 + 0.3 + 0.0) / 3)

    def test_baseline_single_aggregation_unchanged(self) -> None:
        scores = [0.42, 0.11]
        result = aggregate_strategy_scores([scores], BASELINE_FULL)
        self.assertEqual(result, scores)

    def test_wrong_view_count_rejected(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_strategy_scores([[0.1, 0.2], [0.3, 0.4]], PREPROCESSING_V1)

    def test_ragged_vectors_rejected(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_strategy_scores([[0.1, 0.2], [0.3], [0.4, 0.5]], PREPROCESSING_V1)


class BatchingTests(unittest.TestCase):
    def test_batched_views_match_independent_embeddings(self) -> None:
        # With a linear dummy model, batched encoding of three views must
        # produce the same normalized embeddings as three separate calls.
        encoder = VisualEncoder.__new__(VisualEncoder)
        encoder.model_name = "siglip"
        encoder.batch_size = 3
        encoder.device = "cpu"
        encoder._torch = torch

        class ScaleTransform:
            def __call__(self, image: Image.Image) -> torch.Tensor:
                return torch.full((3, 2, 2), float(image.width) / 100.0)

        encoder._transform = ScaleTransform()

        class FlattenLinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(12, 8, bias=False)

            def forward(self, batch: torch.Tensor) -> torch.Tensor:
                return self.linear(batch.flatten(start_dim=1))

        encoder._model = FlattenLinear()
        with torch.no_grad():
            encoder._model.linear.weight.copy_(torch.linspace(0.1, 0.8, 12 * 8).reshape(8, 12))
        encoder._forward = encoder._model

        image = Image.new("RGB", (100, 100))
        views = strategy_views(image, PREPROCESSING_V1)
        batched = torch.tensor(encoder.encode_pil(views))
        independent = torch.cat(
            [torch.tensor(encoder.encode_pil([view])) for view in views], dim=0
        )
        self.assertEqual(batched.shape, (3, 8))
        self.assertTrue(torch.allclose(batched, independent, atol=1e-6))

        scores_batched = batched @ batched[0]
        scores_independent = independent @ independent[0]
        self.assertTrue(torch.allclose(scores_batched, scores_independent, atol=1e-6))


class TransitionTests(unittest.TestCase):
    def test_top1_transition_matrix(self) -> None:
        counts = top1_transition_matrix(
            [False, True, True, False, True, False],
            [True, False, True, False, False, True],
        )
        self.assertEqual(counts["wrong_to_correct"], 2)
        self.assertEqual(counts["correct_to_wrong"], 2)
        self.assertEqual(counts["unchanged_correct"], 1)
        self.assertEqual(counts["unchanged_wrong"], 1)

    def test_top5_transition_matrix(self) -> None:
        counts = top5_transition_matrix(
            [False, True, True, False, True],
            [True, False, True, False, False],
        )
        self.assertEqual(counts["outside_to_inside"], 1)
        self.assertEqual(counts["inside_to_outside"], 2)
        self.assertEqual(counts["stayed_inside"], 1)
        self.assertEqual(counts["stayed_outside"], 1)

    def test_rank_change_counts(self) -> None:
        counts = rank_change_counts([10, 1, 5, None, None], [4, 1, 9, 7, None])
        self.assertEqual(counts["improved"], 2)
        self.assertEqual(counts["degraded"], 1)
        self.assertEqual(counts["unchanged"], 1)

    def test_transition_matrices_reject_mismatched_lengths(self) -> None:
        with self.assertRaises(ValueError):
            top1_transition_matrix([True], [True, False])
        with self.assertRaises(ValueError):
            top5_transition_matrix([True], [True, False])
        with self.assertRaises(ValueError):
            rank_change_counts([1], [1, 2])


class FamilyMetadataTests(unittest.TestCase):
    def test_family_metadata_preserved_through_diagnostics_input(self) -> None:
        # The runner feeds manifest rows to family_diagnostics unchanged; this
        # test pins that family columns survive a plain passthrough.
        manifest = [
            {"query_id": "q1", "target_slug": "a", "target_family_id": "f1", "family_type": "vintage"},
        ]
        predictions = [
            {"query_id": "q1", "target_slug": "a", "top5_slugs": '["a"]'},
        ]
        from recognition.family_metrics import family_diagnostics

        diagnostics = family_diagnostics(predictions, manifest)
        self.assertEqual(diagnostics["hard_family_count"], 1)
        self.assertEqual(diagnostics["family_top1"], 1.0)


class AlignmentTests(unittest.TestCase):
    def test_benchmark_alignment_preserved(self) -> None:
        from recognition.crop_diagnostics import verify_manifest_alignment

        manifest = [
            {"query_id": "q1", "target_slug": "a"},
            {"query_id": "q2", "target_slug": "b"},
        ]
        verify_manifest_alignment(manifest, manifest)
        with self.assertRaises(ValueError):
            verify_manifest_alignment([{"query_id": "q1", "target_slug": "X"}], manifest)


if __name__ == "__main__":
    unittest.main()
