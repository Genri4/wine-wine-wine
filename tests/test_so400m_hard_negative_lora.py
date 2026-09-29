from __future__ import annotations

import random
import copy

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from recognition.so400m_lora import (
    LoRALinear,
    adapted_reference_fingerprint,
    assert_same_lora_checkpoint,
    build_capture_manifest,
    build_hard_negative_map,
    capture_rng_state,
    inject_last_vision_lora,
    lora_sha256,
    lora_state_dict,
    load_lora_state_dict,
    make_capture_view_v2,
    pooled_image_features,
    restore_rng_state,
)
from scripts.run_so400m_hard_negative_lora import (
    _checkpoint_payload,
    _top1_transitions,
    build_candidate_id_matrix,
    exact_top5_reproduction,
    sift_reference_cache_aligned,
)
from scripts.run_strong_local_visual_reranker import reference_sift_cache_manifest_aligned


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)
        self.out_proj = nn.Linear(8, 8)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = nn.Linear(8, 8)


class _Vision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([_Block() for _ in range(6)])

    def forward(self, pixel_values: torch.Tensor, return_dict: bool = True):
        vector = pixel_values.mean(dim=(2, 3))
        vector = torch.nn.functional.pad(vector, (0, 1149))
        return type("VisionOutput", (), {"pooler_output": vector})()


class _FakeSiglip(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_model = _Vision()
        self.text_model = nn.Linear(3, 3)


def _catalog() -> list[dict[str, str]]:
    return [
        {"slug": f"wine-{index}", "reference_image_path": f"data/processed/reference_images/wine-{index}.webp"}
        for index in range(24)
    ]


def test_train_manifest_is_catalog_only_and_split_seeds_differ() -> None:
    rows = _catalog()
    hashes = {row["slug"]: f"hash-{row['slug']}" for row in rows}
    train, validation = build_capture_manifest(rows, hashes, 42, 2, 2)
    assert all(row["reference_image_path"].startswith("data/processed/reference_images/") for row in train)
    assert not any("benchmark" in str(key).lower() or "query" in str(key).lower() or "target" in str(key).lower()
                   for row in train + validation for key in row)
    assert {row["augmentation_seed"] for row in train}.isdisjoint({row["augmentation_seed"] for row in validation})


def test_capture_augmentation_is_deterministic_for_seed() -> None:
    image = Image.fromarray(np.full((80, 56, 3), 180, dtype=np.uint8), mode="RGB")
    first = np.asarray(make_capture_view_v2(image, 12345, "train", "perspective"))
    second = np.asarray(make_capture_view_v2(image, 12345, "train", "perspective"))
    other = np.asarray(make_capture_view_v2(image, 12346, "train", "perspective"))
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert first.shape == (512, 512, 3)


def test_catalog_negative_map_has_aligned_positive_and_only_catalog_sources() -> None:
    slugs = [f"sku-{i}" for i in range(24)]
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(len(slugs), 16)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    families = {slug: f"fam-{i // 3}" for i, slug in enumerate(slugs)}
    mapping, rows = build_hard_negative_map(slugs, vectors, families, seed=3,
                                             family_count=2, visual_count=4,
                                             random_count=2, neighbor_pool=12)
    assert len(rows) == len(slugs) * 8
    for index, slug in enumerate(slugs):
        selected = mapping[slug]["all"]
        assert index not in selected
        assert len(selected) == len(set(selected)) == 8
        assert all(slugs[negative] != slug for negative in selected)
        assert all(families[slugs[negative]] == families[slug] for negative in mapping[slug]["same_family"])
    assert {row["negative_source"] for row in rows} <= {
        "catalog_metadata_same_family", "frozen_so400m_reference_top20", "uniform_catalog_random",
        "frozen_so400m_reference_top20_fallback",
    }
    assert not any("benchmark" in key or "query_id" in key or "target" in key for row in rows for key in row)


def test_variable_negative_counts_pad_with_an_explicit_mask_for_training() -> None:
    rows = [{"slug": "a"}, {"slug": "b"}, {"slug": "c"}]
    negative_map = {
        "a": {"all": [1, 2, 3]},
        "b": {"all": [0]},
        "c": {"all": [0, 1]},
    }
    ids, valid = build_candidate_id_matrix(rows, {"a": 0, "b": 1, "c": 2}, negative_map)
    assert ids.shape == valid.shape == (3, 4)
    assert valid.tolist() == [
        [True, True, True, True],
        [True, True, False, False],
        [True, True, True, False],
    ]
    # Padded catalog IDs can never participate in the loss after this mask.
    scores = torch.zeros((3, 4)).masked_fill(~torch.as_tensor(valid), float("-inf"))
    assert torch.isfinite(scores[0]).all()
    assert torch.isneginf(scores[1, 2:]).all()
    assert torch.isneginf(scores[2, 3:]).all()


def test_lora_attaches_only_to_last_vision_blocks_and_freezes_text_and_base() -> None:
    model = _FakeSiglip()
    names = inject_last_vision_lora(model, rank=2, alpha=4, dropout=0.05, last_blocks=4)
    assert len(names) == 16
    assert {int(name.split(".")[3]) for name in names} == {2, 3, 4, 5}
    assert all(name.endswith(("q_proj", "k_proj", "v_proj", "out_proj")) for name in names)
    assert all(not parameter.requires_grad for parameter in model.text_model.parameters())
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert len(trainable_names) == 32
    assert all(".lora_A" in name or ".lora_B" in name for name in trainable_names)
    assert not any("mlp" in name and parameter.requires_grad for name, parameter in model.named_parameters())


def test_lora_identity_and_checkpoint_state_round_trip() -> None:
    model = _FakeSiglip()
    inject_last_vision_lora(model, rank=2, alpha=4, last_blocks=4)
    layer = model.vision_model.encoder.layers[2].self_attn.q_proj
    assert isinstance(layer, LoRALinear)
    sample = torch.randn(2, 5, 8)
    layer.eval()
    before = layer.base(sample)
    after = layer(sample)
    assert torch.equal(before, after)
    state = lora_state_dict(model)
    digest = lora_sha256(state)
    with torch.no_grad():
        for name, tensor in state.items():
            tensor.add_(0.25)
    load_lora_state_dict(model, state)
    assert lora_sha256(model) == lora_sha256(state)
    assert digest != lora_sha256(model)


def test_adapted_reference_cache_fingerprint_includes_checkpoint_hash() -> None:
    slugs = ["a", "b"]
    image_hashes = {"a": "image-a", "b": "image-b"}
    first = adapted_reference_fingerprint("base", "lora-a", slugs, image_hashes)
    second = adapted_reference_fingerprint("base", "lora-b", slugs, image_hashes)
    assert first != second
    with pytest.raises(ValueError, match="query_batch_size=1"):
        adapted_reference_fingerprint("base", "lora-a", slugs, image_hashes, query_batch_size=16)
    assert_same_lora_checkpoint("lora-a", "lora-a")
    try:
        assert_same_lora_checkpoint("lora-a", "lora-b")
    except ValueError:
        pass
    else:
        raise AssertionError("query/reference checkpoint mismatch must fail closed")


def test_image_features_are_l2_normalized() -> None:
    model = _FakeSiglip()
    pixels = torch.rand(3, 3, 4, 4)
    vectors = pooled_image_features(model, pixels)
    norms = torch.linalg.vector_norm(vectors, dim=-1)
    assert vectors.shape == (3, 1152)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_frozen_query_top5_reproduction_is_order_sensitive() -> None:
    baseline = ["a", "b", "c", "d", "e"]
    assert exact_top5_reproduction(baseline, baseline)
    assert not exact_top5_reproduction(["b", "a", "c", "d", "e"], baseline)
    assert not exact_top5_reproduction(["a", "b", "c", "d"], baseline)


def test_sift_reference_cache_alignment_requires_exact_catalog_slug_set() -> None:
    slugs = ["a", "b"]
    assert sift_reference_cache_aligned(slugs, {"a": __import__("pathlib").Path("a"), "b": __import__("pathlib").Path("b")})
    assert not sift_reference_cache_aligned(slugs, {"a": __import__("pathlib").Path("a")})


def test_frozen_sift_cache_can_reuse_same_content_under_another_mount_root() -> None:
    import cv2
    from recognition.geometric_reranker import DEFAULT_CONFIG

    slugs = ["a", "b"]
    hashes = {"a": "sha-a", "b": "sha-b"}
    index = {
        "cache_fingerprint": "path-specific-fingerprint",
        "opencv_version": cv2.__version__,
        "sift_config": DEFAULT_CONFIG.__dict__,
        "slugs_in_order": slugs,
        "reference_count": len(slugs),
        "reference_image_sha256": hashes,
    }
    assert reference_sift_cache_manifest_aligned(index, slugs, hashes)

    changed_hashes = {**hashes, "b": "changed-image"}
    assert not reference_sift_cache_manifest_aligned(index, slugs, changed_hashes)
    assert not reference_sift_cache_manifest_aligned(index, list(reversed(slugs)), hashes)


def test_production_transition_counts_are_correct() -> None:
    rows = [
        {"frozen_production_top1": "wrong-1", "adapted_production_top1": "t1", "target_slug": "t1"},
        {"frozen_production_top1": "t2", "adapted_production_top1": "wrong-2", "target_slug": "t2"},
        {"frozen_production_top1": "t3", "adapted_production_top1": "t3", "target_slug": "t3"},
        {"frozen_production_top1": "wrong-4", "adapted_production_top1": "wrong-4", "target_slug": "t4"},
    ]
    counts = _top1_transitions(rows)
    assert counts == {"rescued": 1, "broken": 1, "correct_to_correct": 1, "wrong_to_wrong": 1, "net_gain": 0}


def test_production_transition_counts_include_zero_categories() -> None:
    rows = [
        {"frozen_production_top1": "t1", "adapted_production_top1": "t1", "target_slug": "t1"},
        {"frozen_production_top1": "t2", "adapted_production_top1": "t2", "target_slug": "t2"},
    ]
    assert _top1_transitions(rows) == {
        "rescued": 0, "broken": 0, "correct_to_correct": 2, "wrong_to_wrong": 0, "net_gain": 0,
    }


def test_checkpoint_rng_restore_replays_python_numpy_and_torch() -> None:
    random.seed(991)
    np.random.seed(991)
    torch.manual_seed(991)
    saved = capture_rng_state()
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))
    restore_rng_state(saved)
    actual = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert actual == expected


def test_resume_checkpoint_replays_next_optimizer_step(tmp_path) -> None:
    random.seed(73)
    np.random.seed(73)
    torch.manual_seed(73)
    model = nn.Module()
    model.proj = LoRALinear(nn.Linear(4, 4), rank=2, alpha=4, dropout=0.2)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    def step(target_model, target_optimizer, target_scheduler):
        target_model.train()
        target_optimizer.zero_grad(set_to_none=True)
        sample = torch.randn(3, 4)
        loss = target_model.proj(sample).square().mean()
        loss.backward()
        target_optimizer.step()
        target_scheduler.step()

    step(model, optimizer, scheduler)
    payload = _checkpoint_payload(model, optimizer, scheduler, scaler, 1, "fingerprint", {}, {}, 1, 0)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(payload, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected = copy.deepcopy(model)
    step(model, optimizer, scheduler)

    resumed = copy.deepcopy(expected)
    resumed_optimizer = torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad], lr=0.01)
    resumed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(resumed_optimizer, T_max=4)
    resumed_optimizer.load_state_dict(loaded["optimizer_state"])
    resumed_scheduler.load_state_dict(loaded["scheduler_state"])
    restore_rng_state(loaded["rng_state"])
    step(resumed, resumed_optimizer, resumed_scheduler)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, resumed.state_dict()[name]), name
