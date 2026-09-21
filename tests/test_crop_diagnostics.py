from __future__ import annotations

import unittest

import torch
from PIL import Image

from recognition.crop_diagnostics import (
    CENTER_CROP_RATIOS,
    RANK_BANDS,
    aggregate_view_scores,
    center_crop_box,
    family_transition_flags,
    full_ranking,
    rank_band,
    rank_metrics,
    rank_transition,
    split_by,
    target_rank_full,
    transition_summary,
    top5_transition_counts,
    verify_family_mapping,
    verify_manifest_alignment,
    verify_subset_scenario_split,
    view_images,
)
from recognition.encoder import VisualEncoder


class FullRankTests(unittest.TestCase):
    def test_full_rank_computation_correct_with_ties(self) -> None:
        # Tie on score must be broken by slug ascending.
        ranked = full_ranking([0.9, 0.8, 0.8, 0.7], ["b", "a", "target", "c"])
        self.assertEqual([slug for slug, _ in ranked], ["b", "a", "target", "c"])
        self.assertEqual(target_rank_full([slug for slug, _ in ranked], "target"), 3)

    def test_full_rank_deterministic(self) -> None:
        scores = [0.5, 0.9, 0.1, 0.9]
        slugs = ["a", "b", "c", "d"]
        first = full_ranking(scores, slugs)
        second = full_ranking(list(scores), list(slugs))
        self.assertEqual(first, second)
        self.assertEqual(target_rank_full([slug for slug, _ in first], "c"), 4)

    def test_full_rank_missing_target(self) -> None:
        ranked = full_ranking([0.9, 0.8], ["a", "b"])
        self.assertIsNone(target_rank_full([slug for slug, _ in ranked], "missing"))

    def test_full_rank_rejects_mismatched_lengths(self) -> None:
        with self.assertRaises(ValueError):
            full_ranking([0.1, 0.2], ["a"])


class CropTests(unittest.TestCase):
    def test_crop_ratios_are_fixed(self) -> None:
        self.assertEqual(
            CENTER_CROP_RATIOS, {"center_85": 0.85, "center_70": 0.70, "center_55": 0.55}
        )

    def test_crop_dimensions_correct_even_size(self) -> None:
        self.assertEqual(center_crop_box(200, 100, 0.5), (50, 25, 150, 75))

    def test_crop_dimensions_correct_odd_size(self) -> None:
        box = center_crop_box(101, 103, 0.7)
        width = box[2] - box[0]
        height = box[3] - box[1]
        self.assertEqual(width, 71)  # round(101 * 0.7) = 71
        self.assertEqual(height, 72)  # round(103 * 0.7) = 72
        self.assertLessEqual(box[2], 101)
        self.assertLessEqual(box[3], 103)

    def test_crop_box_stays_inside_image(self) -> None:
        for ratio in (0.05, 0.5, 0.85, 1.0):
            for size in ((1, 1), (2, 3), (37, 53), (224, 224), (1024, 768)):
                left, upper, right, lower = center_crop_box(size[0], size[1], ratio)
                self.assertGreaterEqual(left, 0)
                self.assertGreaterEqual(upper, 0)
                self.assertLessEqual(right, size[0])
                self.assertLessEqual(lower, size[1])
                self.assertLess(left, right)
                self.assertLess(upper, lower)

    def test_crop_ratio_one_keeps_full_image(self) -> None:
        self.assertEqual(center_crop_box(64, 48, 1.0), (0, 0, 64, 48))

    def test_crop_does_not_use_target_label(self) -> None:
        # The view set is identical for any pixel content of the same size:
        # it cannot depend on a target label because none is an input.
        first = Image.new("RGB", (100, 80), (10, 20, 30))
        second = Image.new("RGB", (100, 80), (200, 200, 200))
        first_views = view_images(first)
        second_views = view_images(second)
        self.assertEqual(sorted(first_views), sorted(second_views))
        for name, image in first_views.items():
            self.assertEqual(image.size, second_views[name].size)
        self.assertEqual(sorted(first_views), ["center_55", "center_70", "center_85", "full"])

    def test_crop_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            center_crop_box(0, 10, 0.5)
        with self.assertRaises(ValueError):
            center_crop_box(10, 10, 1.5)


class AggregationTests(unittest.TestCase):
    def test_multicrop_max_aggregation_correct(self) -> None:
        result = aggregate_view_scores(
            {"full": [0.1, 0.6], "center_85": [0.5, 0.2], "center_70": [0.3, 0.9]},
            "multicrop_max",
        )
        self.assertEqual(result, [0.5, 0.9])

    def test_multicrop_mean_aggregation_correct(self) -> None:
        result = aggregate_view_scores(
            {"full": [0.1, 0.6], "center_85": [0.5, 0.2], "center_70": [0.3, 0.9]},
            "multicrop_mean",
        )
        self.assertAlmostEqual(result[0], (0.1 + 0.5 + 0.3) / 3)
        self.assertAlmostEqual(result[1], (0.6 + 0.2 + 0.9) / 3)

    def test_aggregation_is_deterministic_regardless_of_view_order(self) -> None:
        first = aggregate_view_scores({"full": [0.2], "center_85": [0.8]}, "multicrop_max")
        second = aggregate_view_scores({"center_85": [0.8], "full": [0.2]}, "multicrop_max")
        self.assertEqual(first, second)

    def test_aggregation_rejects_unknown_method_and_ragged_views(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_view_scores({"full": [0.1]}, "oracle_max")
        with self.assertRaises(ValueError):
            aggregate_view_scores({"full": [0.1, 0.2], "center_85": [0.1]}, "multicrop_max")


class MetricsTests(unittest.TestCase):
    def test_rank_metrics(self) -> None:
        metrics = rank_metrics([1, 3, 6, 12, None])
        self.assertEqual(metrics["query_count"], 5)
        self.assertAlmostEqual(metrics["top1_accuracy"], 0.2)
        self.assertAlmostEqual(metrics["recall_at_5"], 0.4)
        self.assertAlmostEqual(metrics["recall_at_10"], 0.6)
        self.assertAlmostEqual(metrics["recall_at_25"], 0.8)
        self.assertAlmostEqual(metrics["mrr"], (1 + 1 / 3 + 1 / 6 + 1 / 12) / 5)
        self.assertEqual(metrics["median_target_rank_found"], 4.5)  # median([1, 3, 6, 12])
        self.assertEqual(metrics["missing_count"], 1)

    def test_rank_metrics_all_missing(self) -> None:
        metrics = rank_metrics([None, None])
        self.assertEqual(metrics["top1_accuracy"], 0.0)
        self.assertIsNone(metrics["median_target_rank_found"])
        self.assertEqual(metrics["missing_count"], 2)

    def test_rank_band_boundaries(self) -> None:
        expected = {1: "1", 2: "2-5", 5: "2-5", 6: "6-10", 10: "6-10",
                    11: "11-25", 25: "11-25", 26: "26-100", 100: "26-100",
                    101: ">100", None: ">100"}
        for rank, band in expected.items():
            self.assertEqual(rank_band(rank), band)
        labels = [label for label, _, _ in RANK_BANDS]
        self.assertEqual(labels, ["1", "2-5", "6-10", "11-25", "26-100", ">100"])


class TransitionTests(unittest.TestCase):
    def test_rank_transition_categories(self) -> None:
        self.assertEqual(rank_transition(10, 4), "rank_improved")
        self.assertEqual(rank_transition(4, 10), "rank_degraded")
        self.assertEqual(rank_transition(4, 4), "rank_unchanged")
        self.assertEqual(rank_transition(None, 5), "recovered_from_missing")
        self.assertEqual(rank_transition(5, None), "lost_to_missing")
        self.assertEqual(rank_transition(None, None), "missing_unchanged")

    def test_transition_summary_counts(self) -> None:
        summary = transition_summary([10, 1, None, 3, 40], [4, 1, 8, None, 30])
        self.assertEqual(summary["rank_improved"], 2)  # 10->4 and 40->30
        self.assertEqual(summary["rank_degraded"], 0)
        self.assertEqual(summary["rank_unchanged"], 1)
        self.assertEqual(summary["recovered_from_missing"], 1)
        self.assertEqual(summary["lost_to_missing"], 1)
        self.assertEqual(summary["top5_gained"], 1)  # only 10->4; None->8 stays outside Top5
        self.assertEqual(summary["top5_lost"], 1)
        self.assertEqual(summary["median_rank_delta"], 6)  # deltas: 0, +6, +10

    def test_top5_transition_counts(self) -> None:
        baseline = [
            {"correct_top1": False},
            {"correct_top1": True},
            {"correct_top1": True},
            {"correct_top1": False},
        ]
        new = [
            {"correct_top1": True},
            {"correct_top1": False},
            {"correct_top1": True},
            {"correct_top1": False},
        ]
        counts = top5_transition_counts(baseline, new)
        self.assertEqual(counts["wrong_to_correct_top1"], 1)
        self.assertEqual(counts["correct_to_wrong_top1"], 1)


class FamilyTransitionTests(unittest.TestCase):
    SLUG_FAMILY = {
        "target-a": "f1",
        "sibling-a": "f1",
        "other": "f2",
    }

    def test_family_transition_flags(self) -> None:
        flags = family_transition_flags(
            baseline_top5=["other", "sibling-a"],
            new_top5=["other", "target-a", "sibling-a"],
            target_slug="target-a",
            family_id="f1",
            slug_to_family=self.SLUG_FAMILY,
        )
        self.assertTrue(flags["target_back_in_top5"])
        self.assertFalse(flags["family_back_in_top5"])
        self.assertFalse(flags["correct_family_top1_wrong_member"])
        self.assertFalse(flags["left_family_top5"])

    def test_family_left_top5(self) -> None:
        flags = family_transition_flags(
            baseline_top5=["sibling-a"],
            new_top5=["other"],
            target_slug="target-a",
            family_id="f1",
            slug_to_family=self.SLUG_FAMILY,
        )
        self.assertTrue(flags["left_family_top5"])
        self.assertFalse(flags["target_back_in_top5"])

    def test_correct_family_top1_wrong_member(self) -> None:
        flags = family_transition_flags(
            baseline_top5=["other"],
            new_top5=["sibling-a"],
            target_slug="target-a",
            family_id="f1",
            slug_to_family=self.SLUG_FAMILY,
        )
        self.assertTrue(flags["correct_family_top1_wrong_member"])
        self.assertTrue(flags["family_back_in_top5"])


class AlignmentTests(unittest.TestCase):
    MANIFEST = [
        {"query_id": "q1", "target_slug": "a", "subset_role": "representative", "scenario_id": "s1", "target_family_id": "", "family_type": ""},
        {"query_id": "q2", "target_slug": "b", "subset_role": "hard", "scenario_id": "s1", "target_family_id": "f1", "family_type": "vintage"},
        {"query_id": "q3", "target_slug": "c", "subset_role": "hard", "scenario_id": "s2", "target_family_id": "f1", "family_type": "vintage"},
    ]

    def test_manifest_alignment_preserved(self) -> None:
        rows = [
            {"query_id": "q1", "target_slug": "a"},
            {"query_id": "q2", "target_slug": "b"},
            {"query_id": "q3", "target_slug": "c"},
        ]
        verify_manifest_alignment(rows, self.MANIFEST)

    def test_manifest_alignment_detects_drift(self) -> None:
        rows = [{"query_id": "q1", "target_slug": "a"}, {"query_id": "q2", "target_slug": "X"}]
        with self.assertRaises(ValueError):
            verify_manifest_alignment(rows, self.MANIFEST)

    def test_manifest_alignment_detects_missing_query(self) -> None:
        rows = [{"query_id": "q1", "target_slug": "a"}]
        with self.assertRaises(ValueError):
            verify_manifest_alignment(rows, self.MANIFEST)

    def test_subset_and_scenario_split_preserved(self) -> None:
        verify_subset_scenario_split(
            self.MANIFEST, {"representative": 1, "hard": 2}, {"s1": 2, "s2": 1}
        )
        with self.assertRaises(ValueError):
            verify_subset_scenario_split(
                self.MANIFEST, {"representative": 3, "hard": 0}, {"s1": 2, "s2": 1}
            )

    def test_family_mapping_preserved(self) -> None:
        mapping = verify_family_mapping(self.MANIFEST)
        self.assertEqual(mapping, {"b": "f1", "c": "f1"})
        conflicting = self.MANIFEST + [
            {"query_id": "q2-dup", "target_slug": "b", "subset_role": "hard",
             "scenario_id": "s2", "target_family_id": "f2", "family_type": "vintage"},
        ]
        with self.assertRaises(ValueError):
            verify_family_mapping(conflicting)

    def test_split_by_groups_preserve_order(self) -> None:
        grouped = split_by(self.MANIFEST, "scenario_id")
        self.assertEqual(sorted(grouped), ["s1", "s2"])
        self.assertEqual([row["query_id"] for row in grouped["s1"]], ["q1", "q2"])


class _DummyTransform:
    def __call__(self, image: Image.Image) -> torch.Tensor:
        return torch.zeros(3, 4, 4)


class _DummyModel(torch.nn.Module):
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return batch + 1.0


class EncodePilTests(unittest.TestCase):
    def test_encode_pil_batches_and_normalizes(self) -> None:
        encoder = VisualEncoder.__new__(VisualEncoder)
        encoder.model_name = "siglip"
        encoder.batch_size = 2
        encoder.device = "cpu"
        encoder._torch = torch
        encoder._transform = _DummyTransform()
        encoder._model = _DummyModel()
        encoder._forward = encoder._model.forward

        images = [Image.new("RGB", (8, 8)) for _ in range(3)]
        embeddings = encoder.encode_pil(images)
        self.assertEqual(len(embeddings), 3)
        for vector in embeddings:
            self.assertEqual(len(vector), 48)
            norm = sum(value * value for value in vector) ** 0.5
            self.assertAlmostEqual(norm, 1.0, places=5)
        self.assertEqual(embeddings[0], embeddings[1])

    def test_encode_pil_empty(self) -> None:
        encoder = VisualEncoder.__new__(VisualEncoder)
        encoder.batch_size = 2
        self.assertEqual(encoder.encode_pil([]), [])


if __name__ == "__main__":
    unittest.main()
