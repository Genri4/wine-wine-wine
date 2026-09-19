from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from recognition.data import load_catalog, load_evaluation_examples


class DataLoadingTests(unittest.TestCase):
    def test_manifest_paths_are_resolved_relative_to_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "images" / "wine.jpg").touch()
            (root / "catalog.jsonl").write_text(
                '{"item_id":"wine-1","image":"images/wine.jpg"}\n',
                encoding="utf-8",
            )
            items = load_catalog(root / "catalog.jsonl")
            self.assertEqual(items[0].item_id, "wine-1")
            self.assertEqual(items[0].image_path, (root / "images/wine.jpg").resolve())

    def test_evaluation_manifest_keeps_unknown_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "query.jpg").touch()
            (root / "eval.jsonl").write_text(
                '{"sample_id":"q-1","image":"query.jpg","expected_item_id":null}\n',
                encoding="utf-8",
            )
            examples = load_evaluation_examples(root / "eval.jsonl")
            self.assertIsNone(examples[0].expected_item_id)

    def test_analysis_mode_allows_multiple_images_per_product(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "one.jpg").touch()
            (root / "two.jpg").touch()
            (root / "catalog.jsonl").write_text(
                '{"item_id":"wine-1","image":"one.jpg"}\n'
                '{"item_id":"wine-1","image":"two.jpg"}\n',
                encoding="utf-8",
            )
            items = load_catalog(root / "catalog.jsonl", allow_duplicate_item_ids=True)
            self.assertEqual(len(items), 2)


if __name__ == "__main__":
    unittest.main()
