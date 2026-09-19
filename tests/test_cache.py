from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from recognition.cache import EmbeddingCacheMismatch, load_embedding_cache, save_embedding_cache


class EmbeddingCacheTests(unittest.TestCase):
    def test_cache_metadata_is_validated(self) -> None:
        metadata = {
            "model_id": "model",
            "model_version": "v1",
            "catalog_version": "catalog-v1",
            "catalog_manifest_sha256": "abc",
            "preprocessing_config": {"size": 224},
            "catalog_count": 2,
            "embedding_dim": 2,
            "catalog_item_ids": ["a", "b"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pt"
            save_embedding_cache(path, [[1, 0], [0, 1]], metadata)
            self.assertEqual(load_embedding_cache(path, metadata), [[1.0, 0.0], [0.0, 1.0]])
            changed = dict(metadata, model_version="v2")
            with self.assertRaises(EmbeddingCacheMismatch):
                load_embedding_cache(path, changed)


if __name__ == "__main__":
    unittest.main()
