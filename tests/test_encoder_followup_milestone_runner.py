from scripts.run_encoder_followup_common_external import _select_best_lora


def _internal(same_family, full, margin):
    return {
        "validation_frozen": {"full_catalog_top1": 0.60},
        "validation_adapted": {
            "same_family_top1": same_family,
            "full_catalog_top1": full,
            "same_family_margin_mean": margin,
        },
        "external_benchmark_results": {"generated_top1": 1.0},
    }


def test_rank_selection_uses_only_internal_metrics_and_keeps_r8_on_practical_tie():
    r8 = _internal(0.92, 0.80, 0.18)
    r16 = _internal(0.921, 0.81, 0.19)
    winner_with_external = _select_best_lora(r8, r16)
    r8["external_benchmark_results"] = {"generated_top1": 0.1}
    r16["external_benchmark_results"] = {"generated_top1": 1.0}
    assert _select_best_lora(r8, r16) == winner_with_external == "r8"


def test_rank_selection_chooses_r16_only_for_clear_internal_family_gain():
    assert _select_best_lora(
        _internal(0.92, 0.80, 0.18),
        _internal(0.925, 0.79, 0.20),
    ) == "r16"
