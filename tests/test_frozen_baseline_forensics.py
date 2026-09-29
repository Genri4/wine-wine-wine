import hashlib

import numpy as np
import pytest

from recognition.frozen_baseline_forensics import (
    cache_fingerprint_matches,
    canonical_fingerprint,
    checkpoint_sha_matches,
    assert_eval_mode,
    deterministic_ranking,
    deterministic_topk,
    file_sha256,
    file_manifest_matches,
    model_fingerprint,
    preprocessing_fingerprint,
    repeated_inference_check,
    require_valid_baseline,
    score_delta,
    top5_comparison,
)


def test_top5_mismatch_classification_distinguishes_order_set_top1_and_target():
    target = "target"
    assert top5_comparison(["a", "b", "c", "d", "target"], ["a", "b", "c", "d", "target"], target)["type"] == "EXACT"
    order = top5_comparison(["a", "b", "c", "d", "target"], ["a", "c", "b", "d", "target"], target)
    assert order["type"] == "A_SET_IDENTICAL_ORDER_DIFF"
    assert order["same_set"] and not order["top1_changed"]
    top1 = top5_comparison(["a", "b", "c", "d", "target"], ["b", "a", "c", "d", "target"], "outside")
    assert top1["type"] == "C_TOP1_CHANGED_SAME_SET"
    changed = top5_comparison(["a", "b", "c", "d", "target"], ["a", "b", "c", "d", "outside"], target)
    assert changed["type"] == "D_TARGET_TOP5_CROSSING"
    set_change = top5_comparison(["a", "b", "c", "d", "target"], ["a", "b", "c", "d", "outside"], "another")
    assert set_change["type"] == "B_TOP5_SET_DIFF"


def test_ranking_uses_slug_secondary_key_for_exact_ties():
    scores = [0.5, 0.5, 0.2]
    slugs = ["zeta", "alpha", "middle"]
    order = deterministic_ranking(scores, slugs)
    assert [slugs[i] for i in order] == ["alpha", "zeta", "middle"]
    assert [slugs[i] for i in deterministic_topk(scores, slugs, 2)] == ["alpha", "zeta"]


def test_score_comparison_reports_max_and_mean_absolute_deltas():
    assert score_delta([1.0, 0.5], [0.9, 0.7]) == {"max_abs_diff": pytest.approx(0.2), "mean_abs_diff": pytest.approx(0.15)}


def test_baseline_gate_rejects_mismatch_before_lora_evaluation():
    with pytest.raises(RuntimeError, match="LoRA evaluation is blocked"):
        require_valid_baseline({"hard": {"top5_exact_order": 1134, "queries": 1150}})
    require_valid_baseline({"hard": {"top5_exact_order": 1150, "queries": 1150}})


def test_baseline_gate_rejects_empty_result():
    with pytest.raises(RuntimeError, match="result is empty"):
        require_valid_baseline({})


def test_repeated_inference_detects_deterministic_output():
    checked = repeated_inference_check(lambda: np.array([0.1, 0.2], dtype=np.float32), repeats=10)
    assert checked == {"repeats": 10, "all_bitwise_equal": True, "max_abs_diff": 0.0}


def test_repeated_inference_detects_output_drift():
    values = iter((np.array([0.0]), np.array([1.0]), np.array([0.0])))
    checked = repeated_inference_check(lambda: next(values), repeats=3)
    assert not checked["all_bitwise_equal"]
    assert checked["max_abs_diff"] == 1.0


def test_canonical_fingerprint_is_key_order_independent():
    assert canonical_fingerprint({"a": 1, "b": 2}) == canonical_fingerprint({"b": 2, "a": 1})


def test_cache_fingerprint_must_match_expected_value():
    assert cache_fingerprint_matches({"fingerprint": "abc"}, "abc")
    assert not cache_fingerprint_matches({"fingerprint": "abc"}, "different")
    assert not cache_fingerprint_matches({}, "abc")


def test_file_and_checkpoint_sha256_are_verified(tmp_path):
    path = tmp_path / "checkpoint.bin"
    path.write_bytes(b"frozen checkpoint")
    expected = hashlib.sha256(b"frozen checkpoint").hexdigest()
    assert file_sha256(path) == expected
    assert checkpoint_sha_matches(path, expected)
    assert not checkpoint_sha_matches(path, "0" * 64)
    assert file_manifest_matches(tmp_path, [{"path": "checkpoint.bin", "sha256": expected}])
    assert not file_manifest_matches(tmp_path, [{"path": "checkpoint.bin", "sha256": "0" * 64}])


def test_model_and_preprocessing_fingerprints_change_with_identity():
    model_a = model_fingerprint("siglip", "rev-a", {"shape": [2, 3]}, {"vision": "abc"})
    model_b = model_fingerprint("siglip", "rev-b", {"shape": [2, 3]}, {"vision": "abc"})
    assert model_a != model_b
    assert preprocessing_fingerprint({"size": 384, "mean": [0.5]}) == preprocessing_fingerprint({"mean": [0.5], "size": 384})
    assert preprocessing_fingerprint({"size": 384}) != preprocessing_fingerprint({"size": 224})


def test_eval_mode_guard_rejects_training_modules():
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
    model.eval()
    assert_eval_mode(model)
    model[1].train()
    with pytest.raises(RuntimeError, match="eval mode"):
        assert_eval_mode(model)
