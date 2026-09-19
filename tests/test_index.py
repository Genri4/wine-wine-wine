from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from recognition.data import CatalogItem
from recognition.index import CatalogIndex, cosine_similarity


class CatalogIndexTests(unittest.TestCase):
    def test_cosine_similarity(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 1]), 2**-0.5)
        self.assertEqual(cosine_similarity([0, 0], [1, 0]), 0.0)

    def test_search_returns_cosine_order(self) -> None:
        items = [
            CatalogItem("red", Path("red.jpg")),
            CatalogItem("blue", Path("blue.jpg")),
        ]
        index = CatalogIndex.from_items(items, [[1, 0], [0, 1]], "fake")
        results = index.search([0.9, 0.1], top_k=2)
        self.assertEqual([result.item_id for result in results], ["red", "blue"])
        self.assertEqual([result.rank for result in results], [1, 2])

    def test_save_and_load(self) -> None:
        item = CatalogItem("red", Path("red.jpg"))
        index = CatalogIndex.from_items([item], [[3, 4]], "fake")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.pt"
            index.save(path)
            loaded = CatalogIndex.load(path)
            self.assertEqual(loaded.encoder_name, "fake")
            self.assertEqual(loaded.model_name, "fake")
            self.assertEqual(loaded.embedding_dim, 2)
            self.assertEqual(loaded.search([3, 4], 1)[0].item_id, "red")


if __name__ == "__main__":
    unittest.main()
