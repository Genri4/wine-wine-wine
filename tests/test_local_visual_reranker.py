from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

from recognition.local_visual_reranker import (
    fuse_local_signals,
    local_feature_cache_fingerprint,
    normalize_candidate_scores,
    rank_top5,
    signal_oracle_wins,
    validate_feature_alignment,
    validate_five_candidate_pairs,
    validate_normalized_embeddings,
    warp_aligned_overlap,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_final_ml_sanity_check as frozen_runner  # noqa: E402
import run_strong_local_visual_reranker as experiment_runner  # noqa: E402


def test_local_feature_cache_fingerprint_covers_matcher_config_and_reference_bytes() -> None:
    refs = [("a", "sha-a"), ("b", "sha-b")]
    first = local_feature_cache_fingerprint("aliked+lightglue", "v1", {"max_kp": 2048}, refs)
    reordered = local_feature_cache_fingerprint("aliked+lightglue", "v1", {"max_kp": 2048}, list(reversed(refs)))
    assert first == reordered
    assert first != local_feature_cache_fingerprint("aliked+lightglue", "v1", {"max_kp": 4096}, refs)
    assert first != local_feature_cache_fingerprint("aliked+lightglue", "v1", {"max_kp": 2048}, [("a", "changed"), ("b", "sha-b")])


def test_feature_cache_keypoint_descriptor_alignment_is_validated() -> None:
    points = np.asarray([[1, 2], [3, 4]], dtype=np.float32)
    descriptors = np.ones((2, 128), dtype=np.float32)
    validate_feature_alignment(points, descriptors)
    with pytest.raises(ValueError, match="descriptor row"):
        validate_feature_alignment(points, descriptors[:1])


def test_homography_warps_query_to_reference_and_masks_only_overlap() -> None:
    rng = np.random.default_rng(24)
    reference = rng.integers(0, 255, size=(120, 120, 3), dtype=np.uint8)
    query = np.zeros((180, 200, 3), dtype=np.uint8)
    query[25:145, 40:160] = reference
    h_ref_to_query = np.asarray([[1, 0, 40], [0, 1, 25], [0, 0, 1]], dtype=np.float64)
    points = np.asarray([[20, 20], [90, 20], [90, 90], [20, 90], [60, 60]], dtype=np.float32)
    aligned = warp_aligned_overlap(query, reference, h_ref_to_query, points)
    assert aligned is not None
    assert aligned.query_patch.size == aligned.reference_patch.size
    assert aligned.overlap_mask.shape == (aligned.query_patch.height, aligned.query_patch.width)
    assert aligned.overlap_fraction == pytest.approx(1.0)
    np.testing.assert_array_equal(np.asarray(aligned.query_patch), np.asarray(aligned.reference_patch))


def test_invalid_or_low_overlap_warp_is_rejected() -> None:
    image = np.full((100, 100, 3), 120, dtype=np.uint8)
    assert warp_aligned_overlap(image, image, np.zeros((3, 3))) is None
    far_away = np.asarray([[1, 0, 500], [0, 1, 500], [0, 0, 1]], dtype=np.float64)
    assert warp_aligned_overlap(image, image, far_away) is None


def test_aligned_embeddings_must_be_unit_normalized() -> None:
    validate_normalized_embeddings(np.asarray([[0.6, 0.8], [1.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="L2 normalized"):
        validate_normalized_embeddings(np.asarray([[2.0, 0.0]], dtype=np.float32))


def test_local_score_normalization_and_fixed_fusion_are_deterministic() -> None:
    signal = [0.2, 0.4, None, 0.3, 0.1]
    base = [0.8, 0.5, 0.4, 0.2, 0.1]
    first = fuse_local_signals(base, [signal], 0.2)
    assert first == fuse_local_signals(base, [signal], 0.2)
    assert normalize_candidate_scores([None, None]) == [0.0, 0.0]
    assert rank_top5(["a", "b", "c", "d", "e"], first)[0] in {"a", "b", "c", "d", "e"}


def test_fixed_reranking_never_changes_the_top5_candidate_set() -> None:
    slugs = ["a", "b", "c", "d", "e"]
    ranking = rank_top5(slugs, fuse_local_signals([1, .8, .6, .4, .2], [[0, 1, 0, 0, 0]], .4))
    assert set(ranking) == set(slugs)
    assert len(ranking) == 5
    with pytest.raises(ValueError, match="exactly five"):
        rank_top5(slugs[:4], [1, .8, .6, .4])


def test_multiview_and_combined_fusion_only_rerank_the_frozen_top5() -> None:
    slugs = ["a", "b", "c", "d", "e"]
    benchmark_data = {}
    pair_rows = []
    local_scores = {}
    multiview_scores = {}
    for benchmark in experiment_runner.BENCHMARKS:
        query_id = "q"
        benchmark_data[benchmark] = {
            query_id: {
                "candidate_slugs": slugs,
                "target_slug": "a",
                "current_scores": [0.9, 0.8, 0.7, 0.6, 0.5],
                "sift_ranking_recomputed": list(slugs),
                "sift_base_scores": [0.9, 0.8, 0.7, 0.6, 0.5],
            }
        }
        for index, slug in enumerate(slugs):
            pair_rows.append({
                "benchmark": benchmark,
                "query_id": query_id,
                "candidate_slug": slug,
                "homography_valid": True,
                "geometric_score": 0.5 - index * 0.1,
            })
            local_scores[(benchmark, query_id, slug)] = {"so400m": 0.8 - index * 0.05, "pe_core": 0.7 - index * 0.04}
            multiview_scores[(benchmark, query_id, slug)] = {
                "multiview_mean": 0.75 - index * 0.05,
                "multiview_max": 0.85 - index * 0.05,
            }

    orders, _ = experiment_runner.build_method_orders(benchmark_data, pair_rows, local_scores, multiview_scores)
    assert "current_plus_sift_plus_multiview_mean_w0.20" in orders
    assert "current_plus_sift_plus_multiview_max_w0.20" in orders
    assert "combined_local_signals_w0.10" in orders
    for rankings in orders.values():
        ranking = rankings[(experiment_runner.BENCHMARKS[0], "q")]
        assert len(ranking) == 5
        assert set(ranking) == set(slugs)


def test_five_candidate_latency_guard_requires_all_unique_pairs() -> None:
    validate_five_candidate_pairs(["a", "b", "c", "d", "e"], 5)
    with pytest.raises(ValueError, match="all five"):
        validate_five_candidate_pairs(["a", "b", "c", "d", "e"], 4)


def test_oracle_analysis_is_separate_from_inference_api() -> None:
    assert "target_slug" not in inspect.signature(warp_aligned_overlap).parameters
    assert "ground_truth" not in inspect.signature(fuse_local_signals).parameters
    assert signal_oracle_wins(["a", "b", "c", "d", "e"], [.5, .8, .2, .1, .0], "b", "a")
    assert not signal_oracle_wins(["a", "b", "c", "d", "e"], [.8, .5, .2, .1, .0], "b", "a")


def test_top5_input_and_full_frozen_baseline_reproduction() -> None:
    reproduction = frozen_runner.verify_frozen_current_pipeline(ROOT)
    assert reproduction["benchmarks"]["hard_near_duplicate_dev_v2"]["agreement"] == 1.0
    assert reproduction["benchmarks"]["generated_stress_dev_pilot32"]["agreement"] == 1.0
    benchmark_data = experiment_runner.load_benchmark_inputs(ROOT)
    for data in benchmark_data.values():
        for query in data.values():
            validate_five_candidate_pairs(query["candidate_slugs"], len(query["candidate_slugs"]))


def test_transition_counts_are_exact() -> None:
    counts = experiment_runner.transition(
        ["wrong1", "a", "wrong2", "a"], ["a", "wrong", "wrong2", "a"], ["a", "a", "a", "a"]
    )
    assert counts == {"rescued": 1, "broken": 1, "wrong_to_wrong": 1, "correct_to_correct": 1}
