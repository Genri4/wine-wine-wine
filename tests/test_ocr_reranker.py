from __future__ import annotations

import unittest

from recognition.ocr_reranker import (
    POLICY_COMBINED_BLEND,
    POLICY_COMBINED_VINTAGE,
    POLICY_IMAGE_ONLY,
    POLICY_METADATA_BLEND,
    POLICY_REFERENCE_BLEND,
    FusionConfig,
    normalize_image_scores,
    oracle_text_top1,
    oracle_top5_present,
    policy_text_scores,
    rerank_one_query,
    top1_transition_counts,
    vintage_adjustment,
)


def _signals(name=0.0, reference=0.0, vintage="unknown") -> dict:
    return {
        "metadata_name_score": name,
        "winery_score": 0.0,
        "grape_score": 0.0,
        "region_score": 0.0,
        "metadata_text_score": name,
        "reference_ocr_score": reference,
        "vintage_match": vintage,
    }


class NormalizeImageScoresTests(unittest.TestCase):
    def test_min_max_mapping(self) -> None:
        scores = normalize_image_scores([0.10, 0.20, 0.30])
        self.assertAlmostEqual(scores[0], 0.0)
        self.assertAlmostEqual(scores[1], 0.5)
        self.assertAlmostEqual(scores[2], 1.0)

    def test_identical_scores_map_to_one(self) -> None:
        self.assertEqual(normalize_image_scores([0.5, 0.5, 0.5]), [1.0, 1.0, 1.0])

    def test_empty(self) -> None:
        self.assertEqual(normalize_image_scores([]), [])


class RerankOneQueryTests(unittest.TestCase):
    IMAGE = [0.31, 0.30, 0.29, 0.28, 0.27]

    def test_image_only_is_exact_passthrough(self) -> None:
        config = FusionConfig(policy=POLICY_IMAGE_ONLY, alpha=0.3)
        outcome = rerank_one_query(self.IMAGE, [_signals() for _ in range(5)], config)
        self.assertEqual(outcome.final_order, (0, 1, 2, 3, 4))
        self.assertFalse(outcome.reordered)
        self.assertEqual(outcome.final_scores, tuple(self.IMAGE))

    def test_strong_text_evidence_reorders_top1(self) -> None:
        # Candidate 1 sits at image-norm 0.75 (near-tie), so alpha=0.3 with a
        # 0.9 text score flips it; candidates far below cannot be flipped —
        # that is the intended conservative behavior of min-max fusion.
        signals = [_signals() for _ in range(5)]
        signals[1] = _signals(name=0.9)
        config = FusionConfig(policy=POLICY_METADATA_BLEND, alpha=0.3)
        outcome = rerank_one_query(self.IMAGE, signals, config)
        self.assertEqual(outcome.final_order[0], 1)
        self.assertTrue(outcome.reordered)

    def test_bottom_rank_candidate_cannot_flip_with_single_candidate_text(self) -> None:
        signals = [_signals() for _ in range(5)]
        signals[4] = _signals(name=0.9)
        config = FusionConfig(policy=POLICY_METADATA_BLEND, alpha=0.3)
        outcome = rerank_one_query(self.IMAGE, signals, config)
        self.assertEqual(outcome.final_order[0], 0)

    def test_weak_text_evidence_keeps_image_top1(self) -> None:
        signals = [_signals() for _ in range(5)]
        signals[1] = _signals(name=0.06)  # 0.06 text diff cannot beat a near-tie image gap
        config = FusionConfig(policy=POLICY_METADATA_BLEND, alpha=0.4)
        outcome = rerank_one_query(self.IMAGE, signals, config)
        self.assertEqual(outcome.final_order[0], 0)
        self.assertFalse(outcome.reordered)

    def test_margin_guard_blocks_small_text_advantage(self) -> None:
        # Candidate 1 is a near-tie by image (norm 0.9975); fusion would move
        # it to Top-1 with a tiny text advantage, but 0.01 < min_text_margin
        # 0.05, so the guard keeps the image winner.
        image = [0.31, 0.3099, 0.29, 0.28, 0.27]
        signals = [_signals(name=0.20) for _ in range(5)]
        signals[1] = _signals(name=0.21)
        config = FusionConfig(policy=POLICY_METADATA_BLEND, alpha=0.4, min_text_margin=0.05)
        outcome = rerank_one_query(image, signals, config)
        self.assertEqual(outcome.final_order[0], 0)
        self.assertFalse(outcome.reordered)
        self.assertIn("kept_image_top1", outcome.reason)

    def test_margin_guard_allows_large_text_advantage(self) -> None:
        image = [0.31, 0.3099, 0.29, 0.28, 0.27]
        signals = [_signals(name=0.20) for _ in range(5)]
        signals[1] = _signals(name=0.60)
        config = FusionConfig(policy=POLICY_METADATA_BLEND, alpha=0.4, min_text_margin=0.05)
        outcome = rerank_one_query(image, signals, config)
        self.assertEqual(outcome.final_order[0], 1)
        self.assertTrue(outcome.reordered)

    def test_empty_ocr_can_never_reorder(self) -> None:
        signals = [_signals() for _ in range(5)]
        config = FusionConfig(policy=POLICY_METADATA_BLEND, alpha=0.4)
        outcome = rerank_one_query(self.IMAGE, signals, config)
        self.assertEqual(outcome.final_order[0], 0)

    def test_reference_policy_uses_reference_score(self) -> None:
        signals = [_signals() for _ in range(5)]
        signals[1] = _signals(reference=0.9)
        config = FusionConfig(policy=POLICY_REFERENCE_BLEND, alpha=0.3)
        outcome = rerank_one_query(self.IMAGE, signals, config)
        self.assertEqual(outcome.final_order[0], 1)

    def test_combined_policy_takes_best_of_both(self) -> None:
        signals = [_signals(name=0.1) for _ in range(5)]
        signals[1] = _signals(name=0.0, reference=0.9)
        config = FusionConfig(policy=POLICY_COMBINED_BLEND, alpha=0.3)
        outcome = rerank_one_query(self.IMAGE, signals, config)
        self.assertEqual(outcome.final_order[0], 1)
        self.assertEqual(policy_text_scores(POLICY_COMBINED_BLEND, signals[1]), 0.9)

    def test_vintage_bonus_and_penalty_apply(self) -> None:
        signals = [_signals(name=0.5) for _ in range(5)]
        signals[1] = _signals(name=0.9, vintage="exact_match")
        signals[2] = _signals(name=0.5, vintage="mismatch")
        config = FusionConfig(policy=POLICY_COMBINED_VINTAGE, alpha=0.3, vintage_bonus=0.10, vintage_penalty=0.10)
        outcome = rerank_one_query(self.IMAGE, signals, config)
        self.assertEqual(outcome.final_order[0], 1)
        self.assertAlmostEqual(outcome.final_scores[outcome.final_order[0]], 0.7 * 0.75 + 0.3 * 0.9 + 0.10)
        penalized_index = outcome.final_order[2]
        self.assertAlmostEqual(outcome.final_scores[penalized_index], 0.7 * 0.5 + 0.3 * 0.5 - 0.10)

    def test_vintage_unknown_never_changes_score(self) -> None:
        config = FusionConfig(policy=POLICY_COMBINED_VINTAGE, alpha=0.0)
        self.assertEqual(vintage_adjustment("unknown", config), 0.0)

    def test_top5_candidate_set_preserved(self) -> None:
        signals = [_signals() for _ in range(5)]
        signals[1] = _signals(name=0.95)
        config = FusionConfig(policy=POLICY_METADATA_BLEND, alpha=0.4)
        outcome = rerank_one_query(self.IMAGE, signals, config)
        self.assertEqual(sorted(outcome.final_order), [0, 1, 2, 3, 4])

    def test_reranker_is_deterministic(self) -> None:
        signals = [_signals(name=0.3, reference=0.4) for _ in range(5)]
        signals[1] = _signals(name=0.7, vintage="exact_match")
        config = FusionConfig(policy=POLICY_COMBINED_VINTAGE, alpha=0.2)
        first = rerank_one_query(self.IMAGE, signals, config)
        second = rerank_one_query(self.IMAGE, signals, config)
        self.assertEqual(first.final_order, second.final_order)
        self.assertEqual(first.final_scores, second.final_scores)

    def test_ties_keep_image_order(self) -> None:
        signals = [_signals(name=0.5) for _ in range(5)]
        config = FusionConfig(policy=POLICY_METADATA_BLEND, alpha=0.2)
        outcome = rerank_one_query(self.IMAGE, signals, config)
        self.assertEqual(outcome.final_order, (0, 1, 2, 3, 4))

    def test_misaligned_inputs_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rerank_one_query(self.IMAGE, [_signals() for _ in range(3)], FusionConfig(POLICY_METADATA_BLEND, 0.2))


class OracleTests(unittest.TestCase):
    def test_oracle_top5(self) -> None:
        self.assertTrue(oracle_top5_present(1))
        self.assertTrue(oracle_top5_present(5))
        self.assertFalse(oracle_top5_present(6))
        self.assertFalse(oracle_top5_present(None))

    def test_text_oracle_picks_best_and_ties_keep_image_order(self) -> None:
        signals = [_signals(name=0.2) for _ in range(5)]
        signals[3] = _signals(name=0.8)
        self.assertEqual(oracle_text_top1(signals, POLICY_METADATA_BLEND), 3)
        flat = [_signals(name=0.5) for _ in range(5)]
        self.assertEqual(oracle_text_top1(flat, POLICY_METADATA_BLEND), 0)

    def test_oracle_vintage_policy_uses_vintage_evidence(self) -> None:
        signals = [_signals(name=0.5) for _ in range(5)]
        signals[2] = _signals(name=0.5, vintage="exact_match")
        winner = oracle_text_top1(signals, POLICY_COMBINED_VINTAGE)
        self.assertEqual(winner, 2)


class TransitionTests(unittest.TestCase):
    def test_counts_all_cases(self) -> None:
        counts = top1_transition_counts(
            baseline_top1=["a", "b", "c", "x", "e"],
            final_top1=["b", "x", "c", "d", "e"],
            target_slugs=["a", "b", "c", "d", "e"],
        )
        self.assertEqual(counts.wrong_to_correct, 1)  # x -> d
        self.assertEqual(counts.correct_to_wrong, 2)  # a->b, b->x
        self.assertEqual(counts.correct_to_correct_same, 2)  # c->c, e->e
        self.assertEqual(counts.wrong_to_wrong, 0)
        self.assertEqual(counts.rescued, 1)
        self.assertEqual(counts.broken, 2)

    def test_rescued_broken_ratio_property(self) -> None:
        counts = top1_transition_counts(["a"], ["a"], ["a"])
        self.assertEqual(counts.broken, 0)


if __name__ == "__main__":
    unittest.main()
