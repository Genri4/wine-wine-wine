from __future__ import annotations

import unittest
from pathlib import Path

import torch
from PIL import Image

from recognition.encoder_adapters import (
    ADAPTER_REGISTRY,
    CurrentSigLIP2Adapter,
    ImageEncoderAdapter,
    ReferenceCacheMismatch,
    create_adapter,
    load_reference_cache,
    reference_cache_dir,
    save_reference_cache,
)


class AdapterInterfaceTests(unittest.TestCase):
    def test_model_adapter_interface_exists(self) -> None:
        expected = {
            "current_siglip2",
            "siglip2_so400m_384",
            "dinov2_vitl14_reg",
            "pe_core_l14_336",
            "dfn5b_h14_378",
        }
        self.assertEqual(set(ADAPTER_REGISTRY), expected)

    def test_unknown_model_reported_correctly(self) -> None:
        with self.assertRaises(ValueError) as context:
            create_adapter("nonexistent_model")
        self.assertIn("nonexistent_model", str(context.exception))
        self.assertIn("current_siglip2", str(context.exception))

    def test_normalized_embeddings(self) -> None:
        adapter = CurrentSigLIP2Adapter.__new__(CurrentSigLIP2Adapter)
        tensor = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
        normalized = ImageEncoderAdapter._normalized(adapter, tensor)
        self.assertAlmostEqual(sum(value * value for value in normalized[0]), 1.0, places=6)
        self.assertEqual(normalized[1], [0.0, 0.0])

    def test_reference_and_query_dimensions_align(self) -> None:
        # The frozen SigLIP2 reference cache declares dim 768; the current
        # adapter must declare the same dimension.
        self.assertEqual(CurrentSigLIP2Adapter.embedding_dim, 768)

    def test_metadata_contains_preprocessing_contract(self) -> None:
        class _Dummy(ImageEncoderAdapter):
            model_key = "dummy"
            hf_model_id = "dummy/id"

            def _encode_batch(self, images):
                return [[1.0, 0.0]]

        dummy = _Dummy()
        metadata = dummy.metadata()
        for key in ("model_key", "hf_model_id", "preprocessing_config", "global_representation", "native_resolution"):
            self.assertIn(key, metadata)


class ReferenceCacheTests(unittest.TestCase):
    def test_cache_roundtrip_and_fingerprint_guard(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            cache_dir = reference_cache_dir(directory, "dummy_model")
            save_reference_cache(
                cache_dir,
                [[1.0, 0.0], [0.0, 1.0]],
                ["slug-a", "slug-b"],
                {"model_key": "dummy_model"},
                fingerprint="fingerprint-1",
            )
            embeddings = load_reference_cache(cache_dir, "fingerprint-1", ["slug-a", "slug-b"])
            self.assertEqual(len(embeddings), 2)
            with self.assertRaises(ReferenceCacheMismatch):
                load_reference_cache(cache_dir, "fingerprint-2", ["slug-a", "slug-b"])
            with self.assertRaises(ReferenceCacheMismatch):
                load_reference_cache(cache_dir, "fingerprint-1", ["slug-b", "slug-a"])

    def test_cannot_use_cache_of_another_model(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            save_reference_cache(
                reference_cache_dir(directory, "model_x"),
                [[1.0]],
                ["slug-a"],
                {"model_key": "model_x"},
                fingerprint="fp-x",
            )
            with self.assertRaises(FileNotFoundError):
                load_reference_cache(
                    reference_cache_dir(directory, "model_y"),
                    "fp-x",
                    ["slug-a"],
                )

    def test_fingerprint_includes_model_and_catalog(self) -> None:
        adapter = CurrentSigLIP2Adapter.__new__(CurrentSigLIP2Adapter)
        adapter.model_key = CurrentSigLIP2Adapter.model_key
        adapter.hf_model_id = CurrentSigLIP2Adapter.hf_model_id
        adapter.checkpoint_revision = "rev"
        adapter.embedding_dim = 768
        adapter.dtype = "float32"
        adapter.preprocessing_config = {"size": 224}
        first = adapter.fingerprint("catalog-sha", 2042)
        second = adapter.fingerprint("catalog-sha", 2042)
        third = adapter.fingerprint("other-catalog-sha", 2042)
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)


class RankingDeterminismTests(unittest.TestCase):
    def test_deterministic_ranking_on_fixed_embeddings(self) -> None:
        from recognition.crop_diagnostics import full_ranking

        scores = [0.5, 0.9, 0.9]
        slugs = ["b", "a", "c"]
        first = full_ranking(scores, slugs)
        second = full_ranking(list(scores), list(slugs))
        self.assertEqual(first, second)
        # Tie broken by slug ascending: "a" before "c".
        self.assertEqual([slug for slug, _ in first], ["a", "c", "b"])

    def test_target_rank_calculation(self) -> None:
        from recognition.crop_diagnostics import target_rank_full

        self.assertEqual(target_rank_full(["x", "y", "z"], "y"), 2)
        self.assertIsNone(target_rank_full(["x", "y"], "missing"))


class AlignmentTests(unittest.TestCase):
    def test_catalog_slug_alignment(self) -> None:
        from recognition.crop_diagnostics import verify_manifest_alignment

        manifest = [{"query_id": "q1", "target_slug": "a"}]
        verify_manifest_alignment(manifest, manifest)
        with self.assertRaises(ValueError):
            verify_manifest_alignment([{"query_id": "q1", "target_slug": "b"}], manifest)

    def test_family_metrics_alignment(self) -> None:
        from recognition.family_metrics import family_diagnostics

        manifest = [
            {"query_id": "q1", "target_slug": "a", "target_family_id": "f1", "family_type": "vintage"},
        ]
        predictions = [{"query_id": "q1", "target_slug": "a", "top5_slugs": '["a"]'}]
        diagnostics = family_diagnostics(predictions, manifest)
        self.assertEqual(diagnostics["hard_family_count"], 1)

    def test_scenario_subset_splits(self) -> None:
        from recognition.crop_diagnostics import verify_subset_scenario_split

        manifest = [
            {"query_id": "q1", "target_slug": "a", "subset_role": "representative", "scenario_id": "s1"},
            {"query_id": "q2", "target_slug": "b", "subset_role": "hard", "scenario_id": "s1"},
            {"query_id": "q3", "target_slug": "c", "subset_role": "hard", "scenario_id": "s2"},
        ]
        verify_subset_scenario_split(manifest, {"representative": 1, "hard": 2}, {"s1": 2, "s2": 1})
        with self.assertRaises(ValueError):
            verify_subset_scenario_split(manifest, {"representative": 2, "hard": 1}, {"s1": 2, "s2": 1})


class BakeoffFailureSafetyTests(unittest.TestCase):
    def test_failed_model_does_not_break_bakeoff(self) -> None:
        from scripts.run_encoder_bakeoff import _build_summaries, _pairwise_counts, _oracle_counts

        # _pairwise_counts and _oracle_counts work on plain row dicts.
        baseline = [
            {"correct_top1": "True", "correct_top5": "True", "target_rank": "1"},
            {"correct_top1": "False", "correct_top5": "False", "target_rank": ""},
        ]
        other = [
            {"correct_top1": "False", "correct_top5": "True", "target_rank": "3"},
            {"correct_top1": "True", "correct_top5": "True", "target_rank": "1"},
        ]
        counts = _pairwise_counts(baseline, other)
        self.assertEqual(counts["top1_siglip_wrong_model_correct"], 1)
        self.assertEqual(counts["top1_siglip_correct_model_wrong"], 1)
        self.assertEqual(counts["rank_improved"], 1)  # missing -> rank 3
        self.assertEqual(counts["rank_degraded"], 1)  # rank 1 -> missing... other rank 1 vs base missing
        self.assertEqual(counts["rank_same"], 0)
        oracle = _oracle_counts(baseline, other)
        self.assertEqual(oracle["oracle_top1_coverage"], 1.0)

        # _build_summaries tolerates a missing (failed) model directory.
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            _build_summaries(Path(directory), ("synthetic_dev",))
            self.assertFalse((Path(directory) / "benchmark_summary.csv").is_file())

    def test_missing_model_directory_is_reported_not_crashed(self) -> None:
        from scripts.run_encoder_bakeoff import _load_predictions

        self.assertIsNone(_load_predictions(Path("/nonexistent"), "model", "synthetic_dev"))


class BaselineReproductionTests(unittest.TestCase):
    def test_current_adapter_matches_frozen_cache_contract(self) -> None:
        """The current SigLIP2 adapter must reproduce the frozen baseline:
        same model id/version, same preprocessing config, same dim, so the
        validated cache loads instead of being recomputed."""

        import json

        project_root = Path(__file__).resolve().parents[1]
        cache_payload = torch.load(
            project_root / "artifacts" / "catalog_embeddings" / "siglip2_catalog.pt",
            map_location="cpu",
            weights_only=True,
        )
        frozen_metadata = cache_payload["metadata"]

        from recognition.encoder import VisualEncoder

        encoder = VisualEncoder(model_name="siglip", device="cpu", batch_size=2)
        self.assertEqual(encoder.model_id, frozen_metadata["model_id"])
        self.assertEqual(encoder.model_version, frozen_metadata["model_version"])
        self.assertEqual(encoder.preprocessing_config, frozen_metadata["preprocessing_config"])
        self.assertEqual(encoder.embedding_dim, frozen_metadata["embedding_dim"])
        self.assertEqual(CurrentSigLIP2Adapter.embedding_dim, 768)
        self.assertEqual(CurrentSigLIP2Adapter.dtype, "float32")

    def test_current_adapter_encoding_matches_cache_values(self) -> None:
        """Direct numerical reproduction: encoding the first catalog image
        with the frozen pipeline must equal the cached embedding."""

        import csv as csv_module

        project_root = Path(__file__).resolve().parents[1]
        cache_payload = torch.load(
            project_root / "artifacts" / "catalog_embeddings" / "siglip2_catalog.pt",
            map_location="cpu",
            weights_only=True,
        )
        with (project_root / "data" / "processed" / "catalog_manifest.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as stream:
            catalog_rows = list(csv_module.DictReader(stream))
        usable = sorted(
            (row for row in catalog_rows if row["mapping_status"] == "matched" and row["reference_image_path"]),
            key=lambda row: row["slug"],
        )
        cached_embeddings = cache_payload["embeddings"]
        # The frozen loader skipped missing files; both lists stay aligned
        # because all 2042 references exist locally.
        self.assertEqual(cached_embeddings.shape[0], 2042)
        from recognition.preprocessing import load_rgb_image

        from recognition.encoder import VisualEncoder

        encoder = VisualEncoder(model_name="siglip", device="cpu", batch_size=2)
        image_path = project_root / usable[0]["reference_image_path"]
        encoded = encoder.encode_pil([load_rgb_image(image_path)])[0]
        cached = cached_embeddings[0].tolist()
        max_diff = max(abs(a - b) for a, b in zip(encoded, cached))
        self.assertLess(max_diff, 1e-5)


if __name__ == "__main__":
    unittest.main()
