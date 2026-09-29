"""Protocol guards for the controlled SO400M capacity follow-up."""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence


CANONICAL_QUERY_BATCH_SIZE = 1
RANK_ONLY_FIELDS = {"rank", "alpha"}


def require_canonical_query_batch_size(query_batch_size: int) -> None:
    if isinstance(query_batch_size, bool) or int(query_batch_size) != CANONICAL_QUERY_BATCH_SIZE:
        raise ValueError(
            "canonical external benchmarks require query_batch_size=1; "
            "reference precompute batch size is a separate setting"
        )


def require_exact_baseline_reproduction(
    reproduction: Mapping[str, Mapping[str, Any]],
    expected_queries: Mapping[str, int],
    *,
    max_score_abs_diff_tolerance: float = 0.0,
) -> None:
    if max_score_abs_diff_tolerance < 0:
        raise ValueError("score-difference tolerance must be non-negative")
    if set(reproduction) != set(expected_queries):
        raise RuntimeError("canonical baseline reproduction does not cover the three required benchmark splits")
    for split, expected_count in expected_queries.items():
        row = reproduction[split]
        if (int(row.get("queries", -1)) != expected_count
                or int(row.get("top5_exact_order", -1)) != expected_count
                or int(row.get("same_top5_set", -1)) != expected_count
                or float(row.get("max_score_abs_diff_on_historical_top5", float("inf"))) > max_score_abs_diff_tolerance):
            raise RuntimeError(f"canonical frozen baseline is not exact on {split}; external evaluation is blocked")


def assert_rank_only_change(r8: Mapping[str, Any], r16: Mapping[str, Any]) -> None:
    keys = set(r8) | set(r16)
    changed = {key for key in keys if r8.get(key) != r16.get(key)}
    if changed - RANK_ONLY_FIELDS:
        raise ValueError(f"R8/R16 configs differ outside rank/alpha: {sorted(changed - RANK_ONLY_FIELDS)}")
    if r8.get("rank") != 8 or r8.get("alpha") != 16 or r16.get("rank") != 16 or r16.get("alpha") != 32:
        raise ValueError("capacity comparison requires r8/alpha16 versus r16/alpha32")


def assert_same_manifests(
    r8_train: Sequence[Mapping[str, Any]],
    r16_train: Sequence[Mapping[str, Any]],
    r8_validation: Sequence[Mapping[str, Any]],
    r16_validation: Sequence[Mapping[str, Any]],
) -> None:
    if list(r8_train) != list(r16_train) or list(r8_validation) != list(r16_validation):
        raise ValueError("R8 and R16 must use byte-identical train and validation manifest rows")


def assert_training_data_access_scope(access: Mapping[str, Any]) -> None:
    forbidden = ("benchmark_images_opened", "benchmark_labels_opened", "benchmark_error_mining")
    if any(access.get(key) is not False for key in forbidden):
        raise ValueError("encoder training must keep benchmark assets and labels closed")
    if access.get("selection_source") != "internal validation manifests only":
        raise ValueError("checkpoint selection must use internal validation only")


def assert_last2_trainable_scope(
    all_parameter_names: Sequence[str],
    trainable_parameter_names: Sequence[str],
    *,
    vision_block_count: int,
) -> None:
    """Require all and only full parameters in the final two vision blocks."""
    first_block = vision_block_count - 2
    if first_block < 0:
        raise ValueError("vision encoder must contain at least two blocks")
    expected = {
        name for name in all_parameter_names
        if name.startswith("vision_model.encoder.layers.")
        and name.split(".")[3].isdigit()
        and int(name.split(".")[3]) >= first_block
    }
    actual = set(trainable_parameter_names)
    if not expected or actual != expected:
        raise ValueError("partial fine-tuning must train exactly all parameters in the last two vision blocks")
    if any(name.startswith("text_model.") or ".lora_" in name for name in actual):
        raise ValueError("last2 experiment requires a frozen text tower and no concurrent LoRA parameters")


def assert_catalog_only_capture_manifest(rows: Sequence[Mapping[str, Any]]) -> None:
    forbidden_keys = ("benchmark", "query", "target", "ground_truth", "prediction", "rank")
    for row in rows:
        if any(token in str(key).casefold() for key in row for token in forbidden_keys):
            raise ValueError("capture training manifest exposes benchmark/query labels")
        path = PurePosixPath(str(row.get("reference_image_path", "")))
        if path.is_absolute() or path.parts[:3] != ("data", "processed", "reference_images"):
            raise ValueError("capture training may use catalog reference images only")


def fingerprint(payload: Mapping[str, Any]) -> str:
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def make_encoder_fingerprint(
    *,
    base_model_id: str,
    base_revision: str,
    checkpoint_sha256: str,
    preprocessing: Mapping[str, Any],
    dtype: str,
    code_sha256: str,
    lora_rank: int,
    lora_alpha: int,
    target_modules: Sequence[str],
    last_vision_blocks: int,
    query_batch_size: int = CANONICAL_QUERY_BATCH_SIZE,
) -> str:
    require_canonical_query_batch_size(query_batch_size)
    return fingerprint({
        "base_model_id": base_model_id,
        "base_revision": base_revision,
        "checkpoint_sha256": checkpoint_sha256,
        "preprocessing": dict(preprocessing),
        "dtype": dtype,
        "code_sha256": code_sha256,
        "lora_architecture": {
            "rank": int(lora_rank),
            "alpha": int(lora_alpha),
            "target_modules": list(target_modules),
            "last_vision_blocks": int(last_vision_blocks),
        },
        "query_batch_size": CANONICAL_QUERY_BATCH_SIZE,
    })


def require_model_identity(reference_sha256: str, query_sha256: str) -> None:
    if not reference_sha256 or reference_sha256 != query_sha256:
        raise ValueError("reference and query embeddings must use the same frozen encoder checkpoint")


def require_frozen_before_external(selected_model: Mapping[str, Any]) -> None:
    if selected_model.get("stage") != "frozen" or not selected_model.get("checkpoint_sha256"):
        raise RuntimeError("external evaluation is blocked until the selected checkpoint is frozen")


def fixed_half_fusion(frozen_scores: Sequence[float], adapted_scores: Sequence[float]) -> list[float]:
    if len(frozen_scores) != len(adapted_scores) or not frozen_scores:
        raise ValueError("fusion score lists must be non-empty and aligned")

    def normalize(values: Sequence[float]) -> list[float]:
        low, high = min(values), max(values)
        if high == low:
            return [0.0] * len(values)
        return [(float(value) - low) / (high - low) for value in values]

    frozen = normalize(frozen_scores)
    adapted = normalize(adapted_scores)
    return [0.5 * left + 0.5 * right for left, right in zip(frozen, adapted)]


def transition_counts(
    baseline_predictions: Sequence[str],
    adapted_predictions: Sequence[str],
    targets: Sequence[str],
) -> dict[str, int]:
    if not (len(baseline_predictions) == len(adapted_predictions) == len(targets)):
        raise ValueError("prediction and target arrays must align")
    counts = {"rescued": 0, "broken": 0, "correct_to_correct": 0, "wrong_to_wrong": 0}
    for base, new, target in zip(baseline_predictions, adapted_predictions, targets):
        before, after = base == target, new == target
        if not before and after:
            counts["rescued"] += 1
        elif before and not after:
            counts["broken"] += 1
        elif before:
            counts["correct_to_correct"] += 1
        else:
            counts["wrong_to_wrong"] += 1
    counts["net_gain"] = counts["rescued"] - counts["broken"]
    return counts


def original_error_id_fingerprint(query_ids: Sequence[str]) -> str:
    return fingerprint({"query_ids": sorted(str(value) for value in query_ids)})


def assert_immutable_query_ids(current: Sequence[str], expected: Sequence[str]) -> None:
    current_ids, expected_ids = list(map(str, current)), list(map(str, expected))
    if len(current_ids) != len(set(current_ids)) or len(expected_ids) != len(set(expected_ids)):
        raise ValueError("immutable query ID sets cannot contain duplicates")
    if set(current_ids) != set(expected_ids):
        raise ValueError("canonical original error query ID set changed")
