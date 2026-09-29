from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from recognition.geometric_reranker import (
    DEFAULT_CONFIG,
    GeometryScore,
    SIFTConfig,
    SIFTFeatures,
    cache_filename,
    conservative_geometry_order,
    fuse_scores,
    load_feature_cache,
    match_sift_pair,
    normalize_geometry_scores,
    rank_candidates,
    save_feature_cache,
    sift_cache_fingerprint,
    transition_counts,
    valid_geometry_scores,
    validate_homography,
    validate_top5_candidates,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_final_ml_sanity_check as runner  # noqa: E402


def corresponding_features() -> tuple[SIFTFeatures, SIFTFeatures]:
    points = np.asarray(
        [[25, 25], [60, 25], [95, 25], [130, 25], [25, 60], [60, 60], [95, 60], [130, 60],
         [25, 95], [60, 95], [95, 95], [130, 95]], dtype=np.float32,
    )
    query_points = points * 1.15 + np.asarray([18, 24], dtype=np.float32)
    descriptors = np.eye(12, 128, dtype=np.float32)
    return (
        SIFTFeatures(query_points, descriptors.copy(), 240, 240),
        SIFTFeatures(points, descriptors.copy(), 180, 160),
    )


def test_reference_sift_cache_preserves_keypoint_descriptor_alignment(tmp_path: Path) -> None:
    query, _ = corresponding_features()
    cache_path = tmp_path / cache_filename("product-a")
    save_feature_cache(cache_path, query, "image-hash", "extractor-fingerprint")
    loaded = load_feature_cache(cache_path, "image-hash", "extractor-fingerprint")
    assert loaded is not None
    assert loaded.keypoint_count == loaded.descriptors.shape[0] == 12
    np.testing.assert_array_equal(loaded.points_xy, query.points_xy)
    np.testing.assert_array_equal(loaded.descriptors, query.descriptors)


def test_descriptor_cache_fingerprint_and_image_hash_must_match(tmp_path: Path) -> None:
    image = tmp_path / "reference.webp"
    image.write_bytes(b"reference-image-v1")
    fingerprint = sift_cache_fingerprint([("slug-a", str(image))], DEFAULT_CONFIG)
    cache_path = tmp_path / "cache.npz"
    query, _ = corresponding_features()
    save_feature_cache(cache_path, query, "sha-a", fingerprint)
    assert load_feature_cache(cache_path, "sha-a", fingerprint) is not None
    assert load_feature_cache(cache_path, "sha-b", fingerprint) is None
    assert load_feature_cache(cache_path, "sha-a", "other-fingerprint") is None
    image.write_bytes(b"reference-image-v2")
    changed = sift_cache_fingerprint([("slug-a", str(image))], DEFAULT_CONFIG)
    assert changed != fingerprint
    assert sift_cache_fingerprint([("slug-a", str(image))], SIFTConfig(nfeatures=64)) != changed


def test_exactly_five_unique_candidates_are_required() -> None:
    assert validate_top5_candidates(["a", "b", "c", "d", "e"]) == ["a", "b", "c", "d", "e"]
    with pytest.raises(ValueError, match="exactly five"):
        validate_top5_candidates(["a", "b", "c", "d"])
    with pytest.raises(ValueError, match="exactly five"):
        validate_top5_candidates(["a", "b", "c", "d", "d"])


def test_inference_functions_do_not_accept_ground_truth() -> None:
    import inspect

    assert "target_slug" not in inspect.signature(match_sift_pair).parameters
    assert "ground_truth" not in inspect.signature(rank_candidates).parameters
    assert "target_slug" not in inspect.signature(fuse_scores).parameters


def test_valid_homography_is_accepted_with_projected_area() -> None:
    query, reference = corresponding_features()
    result = match_sift_pair(query, reference)
    assert result.homography_valid
    assert result.ransac_inliers == 12
    assert result.inlier_ratio == pytest.approx(1.0)
    assert result.projected_area_ratio == pytest.approx(1.15**2, rel=0.04)
    assert result.geometric_score > 0
    assert result.reprojection_error == pytest.approx(0, abs=1e-3)


@pytest.mark.parametrize(
    "matrix,reason",
    [
        (None, "homography_missing"),
        (np.zeros((3, 3)), "homography_singular"),
        (np.full((3, 3), np.nan), "homography_non_finite_or_bad_shape"),
        (np.diag([0.001, 0.001, 1.0]), "projected_area_out_of_range"),
    ],
)
def test_homography_validation_rejects_invalid_geometry(matrix, reason: str) -> None:
    valid, actual_reason, _ = validate_homography(matrix, 500, 500, 500, 500)
    assert not valid
    assert actual_reason == reason


def test_invalid_geometry_cannot_dominate_fused_or_conservative_order() -> None:
    invalid = GeometryScore(30, 30, 30, 20, 20, 1.0, False, "projection_outside_query", 0.0, 1.0, 0.8, 1000.0)
    valid = GeometryScore(30, 30, 30, 18, 16, 0.88, True, "valid", 1.0, 0.9, 0.5, 2.0)
    scores = valid_geometry_scores([invalid, valid])
    assert scores == [0.0, 2.0]
    assert rank_candidates(["invalid", "valid"], scores)[0] == "valid"
    order = conservative_geometry_order(["current", "bad_geometry"], [valid, invalid], [1.0, 0.0])
    assert order[0] == "current"


def test_score_normalization_is_deterministic_and_flat_geometry_is_zero() -> None:
    assert normalize_geometry_scores([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]
    assert normalize_geometry_scores([3.0, 3.0, 3.0]) == [0.0, 0.0, 0.0]
    assert normalize_geometry_scores([2.0, 4.0, 6.0]) == normalize_geometry_scores([2.0, 4.0, 6.0])


def test_fusion_changes_order_only_within_the_existing_candidate_set() -> None:
    slugs = ["a", "b", "c", "d", "e"]
    fused = fuse_scores([1, 0.8, 0.7, 0.6, 0.5], [0, 2, 0, 0, 0], 0.2)
    ordered = rank_candidates(slugs, fused)
    assert set(ordered) == set(slugs)
    assert len(ordered) == 5


def test_transition_counts_classify_all_four_outcomes() -> None:
    actual = transition_counts(
        ["x", "a", "x", "a"], ["a", "x", "x", "a"], ["a", "a", "a", "a"]
    )
    assert actual == {
        "wrong_to_correct": 1,
        "correct_to_wrong": 1,
        "wrong_to_wrong": 1,
        "correct_to_correct": 1,
    }


def test_current_pipeline_reproduces_frozen_hard_and_generated_outputs_exactly() -> None:
    result = runner.verify_frozen_current_pipeline(ROOT)
    assert result["benchmarks"]["hard_near_duplicate_dev_v2"]["agreement"] == 1.0
    assert result["benchmarks"]["generated_stress_dev_pilot32"]["agreement"] == 1.0


def test_saved_reference_cache_is_reused_without_reextracting(tmp_path: Path) -> None:
    query, _ = corresponding_features()
    path = tmp_path / cache_filename("same-reference")
    save_feature_cache(path, query, "image-sha", "fingerprint")
    first = load_feature_cache(path, "image-sha", "fingerprint")
    second = load_feature_cache(path, "image-sha", "fingerprint")
    assert first is not None and second is not None
    np.testing.assert_array_equal(first.descriptors, second.descriptors)


def test_geometry_runner_matches_all_five_candidates(monkeypatch, tmp_path: Path) -> None:
    query, reference = corresponding_features()
    top5 = ["a", "b", "c", "d", "e"]
    current = {"query": {"top5_slugs": json.dumps(top5)}}
    manifests = {"query": {"query_path": "query.jpg"}}
    inputs = ({}, current, manifests, {}, {})
    query_extractions = []

    class FakeReader:
        def get(self, slug):
            return reference

    monkeypatch.setattr(runner, "read_rgb", lambda path: np.zeros((20, 20, 3), dtype=np.uint8))

    def extract_query_once(image, config):
        query_extractions.append(image)
        return query

    monkeypatch.setattr(runner, "extract_sift", extract_query_once)
    monkeypatch.setattr(runner, "match_sift_pair", lambda q, ref, config: match_sift_pair(q, ref, config))
    pair_rows, per_query, latency = runner.geometry_for_benchmark(
        tmp_path, "fixture", inputs, FakeReader(), {slug: tmp_path / f"{slug}.webp" for slug in top5}
    )
    assert len(pair_rows) == 5
    assert {row["candidate_slug"] for row in pair_rows} == set(top5)
    assert len(per_query["query"]) == 5
    assert len(query_extractions) == 1
    assert latency["query"]["five_candidate_matching_ms"] >= 0


def test_changing_ground_truth_does_not_change_inference_order() -> None:
    slugs = ["a", "b", "c", "d", "e"]
    scores = [1.0, 0.9, 0.8, 0.7, 0.6]
    geo = [GeometryScore(0, 0, 0, 0, 0, 0, False, "no_descriptors", None, None, 0, 0) for _ in slugs]
    baseline = {"query": {"top5_slugs": json.dumps(slugs), "top5_scores": json.dumps(scores), "target_slug": "a"}}
    current = {"query": {"top5_slugs": json.dumps(slugs), "top5_scores": json.dumps(scores), "target_slug": "a"}}
    inputs = (baseline, current, {}, {}, {})
    first = runner.get_method_orders("fixture", inputs, {"query": geo})
    baseline["query"]["target_slug"] = "not-a-candidate"
    current["query"]["target_slug"] = "different-ground-truth"
    second = runner.get_method_orders("fixture", inputs, {"query": geo})
    assert first == second
