import pytest

from recognition.encoder_followup_protocol import (
    assert_catalog_only_capture_manifest,
    assert_immutable_query_ids,
    assert_last2_trainable_scope,
    assert_rank_only_change,
    assert_same_manifests,
    assert_training_data_access_scope,
    fixed_half_fusion,
    make_encoder_fingerprint,
    original_error_id_fingerprint,
    require_canonical_query_batch_size,
    require_exact_baseline_reproduction,
    require_frozen_before_external,
    require_model_identity,
    transition_counts,
)


def test_canonical_query_batch_is_exactly_one():
    require_canonical_query_batch_size(1)
    for invalid in (0, 2, 16, True):
        with pytest.raises(ValueError, match="query_batch_size=1"):
            require_canonical_query_batch_size(invalid)


def test_rank_comparison_requires_only_capacity_to_change():
    r8 = {"rank": 8, "alpha": 16, "dropout": 0.05, "seed": 9, "loss": "listwise"}
    r16 = {"rank": 16, "alpha": 32, "dropout": 0.05, "seed": 9, "loss": "listwise"}
    assert_rank_only_change(r8, r16)
    with pytest.raises(ValueError, match="outside rank/alpha"):
        assert_rank_only_change(r8, {**r16, "seed": 10})


def test_r8_r16_train_and_validation_manifests_must_match():
    train = [{"slug": "sku-a", "augmentation_seed": 1}]
    validation = [{"slug": "sku-a", "augmentation_seed": 2}]
    assert_same_manifests(train, list(train), validation, list(validation))
    with pytest.raises(ValueError, match="byte-identical"):
        assert_same_manifests(train, [{**train[0], "augmentation_seed": 3}], validation, validation)


def test_capture_training_manifest_rejects_benchmark_assets_and_labels():
    assert_catalog_only_capture_manifest([{"slug": "sku-a", "reference_image_path": "data/processed/reference_images/a.webp"}])
    for invalid in (
        {"slug": "a", "reference_image_path": "data/benchmarks/generated/a.webp"},
        {"slug": "a", "reference_image_path": "data/processed/reference_images/a.webp", "target_slug": "a"},
    ):
        with pytest.raises(ValueError):
            assert_catalog_only_capture_manifest([invalid])


def test_training_access_guard_keeps_benchmark_data_closed_and_selection_internal():
    assert_training_data_access_scope({
        "benchmark_images_opened": False,
        "benchmark_labels_opened": False,
        "benchmark_error_mining": False,
        "selection_source": "internal validation manifests only",
    })
    for field in ("benchmark_images_opened", "benchmark_labels_opened", "benchmark_error_mining"):
        with pytest.raises(ValueError, match="benchmark assets and labels closed"):
            assert_training_data_access_scope({
                "benchmark_images_opened": False,
                "benchmark_labels_opened": False,
                "benchmark_error_mining": False,
                "selection_source": "internal validation manifests only",
                field: True,
            })
    with pytest.raises(ValueError, match="internal validation only"):
        assert_training_data_access_scope({
            "benchmark_images_opened": False,
            "benchmark_labels_opened": False,
            "benchmark_error_mining": False,
            "selection_source": "external benchmark",
        })


def test_last2_scope_requires_every_and_only_final_vision_block_parameter():
    names = [
        f"vision_model.encoder.layers.{block}.weight" for block in range(4)
    ] + ["text_model.encoder.weight"]
    assert_last2_trainable_scope(names, names[2:4], vision_block_count=4)
    with pytest.raises(ValueError, match="exactly all parameters"):
        assert_last2_trainable_scope(names, [names[3]], vision_block_count=4)
    with pytest.raises(ValueError, match="exactly all parameters"):
        assert_last2_trainable_scope(names, names[2:5], vision_block_count=4)


def test_canonical_query_batch_participates_in_model_fingerprint():
    kwargs = dict(base_model_id="siglip", base_revision="abc", checkpoint_sha256="lora", preprocessing={"size": 384}, dtype="fp16", code_sha256="code", lora_rank=8, lora_alpha=16, target_modules=("q_proj", "k_proj", "v_proj", "out_proj"), last_vision_blocks=4)
    first = make_encoder_fingerprint(**kwargs, query_batch_size=1)
    assert first == make_encoder_fingerprint(**kwargs, query_batch_size=1)
    with pytest.raises(ValueError):
        make_encoder_fingerprint(**kwargs, query_batch_size=16)


def test_canonical_baseline_must_reproduce_exactly_before_external_evaluation():
    expected = {"hard": 1150, "generated": 128, "synthetic": 4084}
    exact = {
        name: {
            "queries": count,
            "top5_exact_order": count,
            "same_top5_set": count,
            "max_score_abs_diff_on_historical_top5": 0.0,
        }
        for name, count in expected.items()
    }
    require_exact_baseline_reproduction(exact, expected)
    exact["hard"]["top5_exact_order"] = 1149
    with pytest.raises(RuntimeError, match="external evaluation is blocked"):
        require_exact_baseline_reproduction(exact, expected)


def test_dynamic_baseline_allows_only_float32_score_roundoff_not_ranking_changes():
    expected = {"hard": 2}
    rows = {"hard": {"queries": 2, "top5_exact_order": 2, "same_top5_set": 2,
                      "max_score_abs_diff_on_historical_top5": 2.3e-6}}
    require_exact_baseline_reproduction(rows, expected, max_score_abs_diff_tolerance=3e-6)
    rows["hard"]["max_score_abs_diff_on_historical_top5"] = 3.1e-6
    with pytest.raises(RuntimeError, match="external evaluation is blocked"):
        require_exact_baseline_reproduction(rows, expected, max_score_abs_diff_tolerance=3e-6)


def test_reference_and_query_must_use_the_same_checkpoint():
    require_model_identity("checkpoint-sha", "checkpoint-sha")
    with pytest.raises(ValueError, match="same frozen encoder"):
        require_model_identity("reference-sha", "query-sha")


def test_external_evaluation_requires_a_frozen_selected_checkpoint():
    require_frozen_before_external({"stage": "frozen", "checkpoint_sha256": "sha"})
    with pytest.raises(RuntimeError, match="blocked"):
        require_frozen_before_external({"stage": "training", "checkpoint_sha256": "sha"})


def test_fixed_half_fusion_normalizes_both_models_and_has_no_weight_argument():
    assert fixed_half_fusion([0.2, 0.4], [10.0, 30.0]) == pytest.approx([0.0, 1.0])
    with pytest.raises(ValueError, match="aligned"):
        fixed_half_fusion([0.2], [0.2, 0.3])


def test_transition_counts_and_original_error_id_fingerprint_are_deterministic():
    assert transition_counts(["wrong", "right", "wrong"], ["right", "wrong", "wrong"], ["right", "right", "right"]) == {
        "rescued": 1, "broken": 1, "correct_to_correct": 0, "wrong_to_wrong": 1, "net_gain": 0,
    }
    assert original_error_id_fingerprint(["q2", "q1"]) == original_error_id_fingerprint(["q1", "q2"])
    assert_immutable_query_ids(["q1", "q2"], ["q2", "q1"])
    with pytest.raises(ValueError, match="set changed"):
        assert_immutable_query_ids(["q1", "q3"], ["q1", "q2"])
