"""Small deterministic checks shared by the frozen-baseline forensic runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


def canonical_fingerprint(payload: Any) -> str:
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def top5_comparison(old: Sequence[str], new: Sequence[str], target: str) -> dict[str, Any]:
    if len(old) != 5 or len(new) != 5 or len(set(old)) != 5 or len(set(new)) != 5:
        raise ValueError("old and new rankings must each contain five unique slugs")
    same_set = set(old) == set(new)
    same_order = list(old) == list(new)
    top1_changed = old[0] != new[0]
    target_crossed = (target in old) != (target in new)
    if same_order:
        kind = "EXACT"
    elif same_set:
        kind = "C_TOP1_CHANGED_SAME_SET" if top1_changed else "A_SET_IDENTICAL_ORDER_DIFF"
    else:
        kind = "D_TARGET_TOP5_CROSSING" if target_crossed else "B_TOP5_SET_DIFF"
    return {
        "same_set": same_set,
        "same_order": same_order,
        "top1_changed": top1_changed,
        "target_crossed": target_crossed,
        "type": kind,
    }


def deterministic_ranking(scores: Sequence[float], slugs: Sequence[str]) -> list[int]:
    if len(scores) != len(slugs):
        raise ValueError("scores and slugs must have the same length")
    return sorted(range(len(slugs)), key=lambda i: (-float(scores[i]), slugs[i]))


def deterministic_topk(scores: Sequence[float], slugs: Sequence[str], k: int) -> list[int]:
    if k < 1 or k > len(slugs):
        raise ValueError("k must be between one and the number of candidates")
    return deterministic_ranking(scores, slugs)[:k]


def model_fingerprint(model_id: str, revision: str, config: Any, weight_hashes: Any = None) -> str:
    return canonical_fingerprint({"model_id": model_id, "revision": revision, "config": config, "weight_hashes": weight_hashes})


def preprocessing_fingerprint(config: Any) -> str:
    return canonical_fingerprint(config)


def assert_eval_mode(model: Any) -> None:
    training = [type(module).__name__ for module in model.modules() if module.training]
    if training:
        raise RuntimeError("frozen inference requires eval mode for every module")


def file_manifest_matches(root: str | Path, records: Sequence[dict[str, str]]) -> bool:
    base = Path(root)
    return all(file_sha256(base / row["path"]) == row["sha256"] for row in records)


def score_delta(old: Sequence[float], new: Sequence[float]) -> dict[str, float]:
    if len(old) != len(new) or not old:
        raise ValueError("score vectors must be non-empty and have equal length")
    diff = np.abs(np.asarray(old, dtype=np.float64) - np.asarray(new, dtype=np.float64))
    return {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean())}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_sha_matches(path: str | Path, expected_sha256: str) -> bool:
    return bool(expected_sha256) and file_sha256(path) == expected_sha256


def cache_fingerprint_matches(metadata: dict[str, Any], expected: str) -> bool:
    return bool(expected) and metadata.get("fingerprint") == expected


def require_valid_baseline(summary: dict[str, dict[str, Any]]) -> None:
    if not summary:
        raise RuntimeError("LoRA evaluation is blocked: frozen baseline result is empty")
    invalid = [
        f"{name} {row.get('top5_exact_order', row.get('exact_top5_order', 0))}/{row.get('queries', 0)}"
        for name, row in summary.items()
        if row.get("top5_exact_order", row.get("exact_top5_order", 0)) != row.get("queries", -1)
    ]
    if invalid:
        raise RuntimeError("LoRA evaluation is blocked until frozen Top-5 baseline is valid: " + ", ".join(invalid))


def repeated_inference_check(infer: Callable[[], Any], repeats: int = 10) -> dict[str, Any]:
    if repeats < 2:
        raise ValueError("repeats must be at least two")
    outputs = [np.asarray(infer()) for _ in range(repeats)]
    first = outputs[0]
    max_diff = max(float(np.max(np.abs(first.astype(np.float64) - x.astype(np.float64)))) for x in outputs[1:])
    return {"repeats": repeats, "all_bitwise_equal": all(np.array_equal(first, x) for x in outputs[1:]), "max_abs_diff": max_diff}
