from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from PIL import Image

from recognition.metric_adapter import (
    ResidualMetricAdapter,
    assert_training_manifest_catalog_only,
    build_hard_negative_map,
    build_view_manifest,
    candidate_index_matrix,
    derive_augmentation_seed,
    make_catalog_view,
    rank_fixed_top5,
    reference_cache_fingerprint,
    transition_name,
)


def test_training_manifests_are_catalog_only_and_split_views_are_distinct():
    catalog = [
        {"slug": "sku-a", "reference_image_path": "data/processed/reference_images/a.webp"},
        {"slug": "sku-b", "reference_image_path": "data/processed/reference_images/b.webp"},
    ]
    hashes = {"sku-a": "a" * 64, "sku-b": "b" * 64}
    train, validation = build_view_manifest(catalog, hashes, 17, train_views_per_sku=3,
                                            validation_views_per_sku=2)
    assert len(train) == 6
    assert len(validation) == 4
    assert {row["split"] for row in train} == {"train"}
    assert {row["split"] for row in validation} == {"validation"}
    assert not ({row["augmentation_seed"] for row in train}
                & {row["augmentation_seed"] for row in validation})
    assert {row["reference_image_path"] for row in train + validation} <= {
        "data/processed/reference_images/a.webp", "data/processed/reference_images/b.webp"
    }
    assert all("target_slug" not in row and "benchmark" not in row and "query_path" not in row
               for row in train + validation)
    assert derive_augmentation_seed(17, "sku-a", "train", 0) != derive_augmentation_seed(17, "sku-a", "validation", 0)
    with pytest.raises(ValueError, match="ground-truth"):
        assert_training_manifest_catalog_only([{
            "reference_image_path": "data/processed/reference_images/a.webp", "target_slug": "sku-a"
        }])
    with pytest.raises(ValueError, match="catalog reference"):
        build_view_manifest([{"slug": "leak", "reference_image_path": "data/benchmarks/test/queries/q.jpg"}],
                            {"leak": "0"}, 17)


def test_augmentation_is_repeatable_and_train_validation_views_differ():
    y, x = np.mgrid[0:160, 0:128]
    image = np.stack(((x * 2) % 255, (y * 2) % 255, ((x + y) * 3) % 255), axis=-1).astype(np.uint8)
    source = Image.fromarray(image, "RGB")
    train_seed = derive_augmentation_seed(9, "sku-x", "train", 0)
    val_seed = derive_augmentation_seed(9, "sku-x", "validation", 0)
    first = make_catalog_view(source, train_seed)
    repeated = make_catalog_view(source, train_seed)
    validation = make_catalog_view(source, val_seed)
    assert first.tobytes() == repeated.tobytes()
    assert first.tobytes() != validation.tobytes()


def test_hard_negatives_are_catalog_derived_unique_and_positive_first():
    slugs = [f"sku-{index:02d}" for index in range(30)]
    vectors = np.eye(30, dtype=np.float32)
    # Create two close catalog-only reference neighbors and one metadata family.
    vectors[1] = vectors[0] * 0.8 + vectors[1] * 0.6
    vectors[1] /= np.linalg.norm(vectors[1])
    family_keys = {slug: ("line-a" if index < 4 else None) for index, slug in enumerate(slugs)}
    negatives, family, rows = build_hard_negative_map(slugs, vectors, family_keys, 29)
    assert family[slugs[0]] == [1, 2, 3]
    for slug in slugs:
        anchor = slugs.index(slug)
        assert len(negatives[slug]) == 10
        assert anchor not in negatives[slug]
        assert len(set(negatives[slug])) == 10
    assert all(row["anchor_slug"] != row["negative_slug"] for row in rows)
    assert {row["negative_source"] for row in rows} <= {
        "catalog_metadata_same_family", "frozen_so400m_reference_top20", "uniform_catalog_random",
        "frozen_so400m_reference_top20_fallback",
    }
    candidates = candidate_index_matrix(slugs, negatives)
    assert candidates.shape == (30, 11)
    assert np.array_equal(candidates[:, 0], np.arange(30))


def test_residual_adapter_is_identity_initialized_and_normalizes_embeddings():
    adapter = ResidualMetricAdapter(input_dim=8, hidden_dim=4, residual_scale=0.1)
    inputs = torch.randn(6, 8)
    outputs = adapter(inputs)
    expected = torch.nn.functional.normalize(inputs, dim=-1)
    assert torch.allclose(outputs, expected, atol=1e-7)
    assert torch.allclose(outputs.norm(dim=-1), torch.ones(6), atol=1e-6)
    assert sum(parameter.numel() for parameter in adapter.parameters()) == 8 * 4 + 4 + 4 * 8 + 8


def test_adapted_reference_cache_fingerprint_changes_with_weights_or_reference():
    slugs = ["a", "b"]
    hashes = {"a": "a1", "b": "b1"}
    base = reference_cache_fingerprint("base", "adapter-1", slugs, hashes)
    assert base != reference_cache_fingerprint("base", "adapter-2", slugs, hashes)
    assert base != reference_cache_fingerprint("base", "adapter-1", slugs, {"a": "a2", "b": "b1"})


def test_exact_frozen_top5_only_target_blind_scoring_and_deterministic_transitions():
    from scripts.run_hard_negative_metric_adapter import score_query_without_target, train_adapter
    from recognition.local_visual_reranker import fuse_local_signals

    candidates = ["a", "b", "c", "d", "e"]
    refs = np.eye(5, dtype=np.float32)
    query = np.eye(5, dtype=np.float32)[2]
    before = rank_fixed_top5(candidates, [0.2, 0.1, 1.0, 0.3, 0.0])
    after = rank_fixed_top5(candidates, score_query_without_target(query, candidates, [0, 1, 2, 3, 4], refs))
    assert before == ["c", "d", "a", "b", "e"]
    assert after == ["c", "a", "b", "d", "e"]
    assert set(before) == set(after) == set(candidates)
    assert list(inspect.signature(score_query_without_target).parameters) == [
        "query_embedding", "candidate_slugs", "candidate_ids", "adapted_reference_embeddings"
    ]
    assert transition_name(False, True) == "wrong_to_correct"
    assert transition_name(True, False) == "correct_to_wrong"
    assert transition_name(False, False) == "wrong_to_wrong"
    assert transition_name(True, True) == "correct_to_correct"
    fused1 = fuse_local_signals([0.8, 0.7, 0.6, 0.2, 0.1], [[0.3, 0.9, 0.1, 0.2, 0.4]], 0.2)
    fused2 = fuse_local_signals([0.8, 0.7, 0.6, 0.2, 0.1], [[0.3, 0.9, 0.1, 0.2, 0.4]], 0.2)
    assert fused1 == fused2
    assert not any("target" in name.casefold() or "benchmark" in name.casefold()
                   for name in inspect.signature(train_adapter).parameters)
    with pytest.raises(ValueError, match="exactly five"):
        rank_fixed_top5(candidates[:4], [0.1] * 4)


def test_frozen_current_plus_sift_top5_reproduces_exactly():
    from recognition.geometric_reranker import fuse_scores, rank_candidates
    from scripts.run_hard_negative_metric_adapter import (
        ROOT,
        load_postfreeze_family_slices,
        load_synthetic_inputs,
        verify_frozen_sift_baseline,
    )

    result = verify_frozen_sift_baseline(ROOT)
    assert result["training_started_after_exact_reproduction"]
    assert result["current_plus_sift_w0.40"]["hard_near_duplicate_dev_v2"]["exact_top5_order"] == 1150
    assert result["current_plus_sift_w0.40"]["generated_stress_dev_pilot32"]["exact_top5_order"] == 128
    synthetic = load_synthetic_inputs(ROOT)["synthetic_dev"]
    exact = 0
    correct = 0
    for query_id, query in synthetic.items():
        geo = [float(query["old_geometry"][(query_id, slug)]["geometric_score"])
               if str(query["old_geometry"][(query_id, slug)]["homography_valid"]).lower() == "true" else 0.0
               for slug in query["candidate_slugs"]]
        ranking = rank_candidates(query["candidate_slugs"], fuse_scores(query["current_scores"], geo, 0.40))
        exact += ranking == query["sift_selected_ranking"]
        correct += ranking[0] == query["target_slug"]
    assert exact == len(synthetic) == 4084
    assert correct / len(synthetic) == pytest.approx(0.9714, abs=0.001)
    family_slices = load_postfreeze_family_slices(ROOT)
    assert family_slices["hard_near_duplicate_dev_v2"]
    assert family_slices["generated_stress_dev_pilot32"]
    assert {item["family_type"] for item in family_slices["generated_stress_dev_pilot32"].values()} <= {
        "vintage", "subtype", "other"
    }


def test_view_precompute_resumes_after_last_completed_batch(tmp_path):
    from scripts.run_hard_negative_metric_adapter import npy_has_shape, precompute_catalog_view_split

    root = tmp_path / "project"
    run_dir = tmp_path / "run"
    rows = []
    for index in range(3):
        relative = f"data/processed/reference_images/{index}.png"
        image_path = root / relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (96, 128), (60 + index * 30, 100, 150)).save(image_path)
        rows.append({"slug": f"sku-{index}", "reference_image_path": relative,
                     "augmentation_seed": index + 10})

    class InterruptingEncoder:
        calls = 0

        def encode_images(self, images):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated stop between batches")
            vector = np.zeros((len(images), 1152), dtype=np.float32)
            vector[:, 0] = 1.0
            return vector.tolist()

    with pytest.raises(RuntimeError, match="simulated stop"):
        precompute_catalog_view_split(root, run_dir, "train", rows, InterruptingEncoder(), "fingerprint", batch_size=1)
    progress = __import__("json").loads((run_dir / "train_view_precompute_progress.json").read_text())
    assert progress["completed_rows"] == 1

    class FinishingEncoder:
        def __init__(self):
            self.calls = 0

        def encode_images(self, images):
            self.calls += 1
            vector = np.zeros((len(images), 1152), dtype=np.float32)
            vector[:, 1] = 1.0
            return vector.tolist()

    encoder = FinishingEncoder()
    output = precompute_catalog_view_split(root, run_dir, "train", rows, encoder, "fingerprint", batch_size=1)
    assert encoder.calls == 2  # the first completed row was not recomputed
    assert npy_has_shape(output, (3, 1152), np.float16)
    assert __import__("json").loads((run_dir / "train_view_precompute_progress.json").read_text())["completed_rows"] == 3
