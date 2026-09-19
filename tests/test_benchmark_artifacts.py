from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import unittest


class BenchmarkArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]

    def test_web_scored_set_has_no_exact_reference_copy(self) -> None:
        manifest_path = self.root / "data/benchmarks/web_extra_dev/manifest.csv"
        catalog_path = self.root / "data/processed/catalog_manifest.csv"
        if not manifest_path.is_file() or not catalog_path.is_file():
            self.skipTest("generated benchmark artifacts are not available")
        with manifest_path.open(encoding="utf-8", newline="") as stream:
            manifest = list(csv.DictReader(stream))
        with catalog_path.open(encoding="utf-8", newline="") as stream:
            catalog = list(csv.DictReader(stream))
        reference_hashes = {
            hashlib.sha256((self.root / row["reference_image_path"]).read_bytes()).hexdigest()
            for row in catalog
            if row.get("mapping_status") == "matched"
            and row.get("reference_image_path")
            and (self.root / row["reference_image_path"]).is_file()
        }
        accepted_hashes = {
            hashlib.sha256((self.root / row["query_path"]).read_bytes()).hexdigest()
            for row in manifest
        }
        self.assertFalse(accepted_hashes & reference_hashes)
        self.assertEqual(len(manifest), len({row["query_id"] for row in manifest}))
        self.assertEqual(len(manifest), len({row["target_slug"] for row in manifest}))
        for row in manifest:
            self.assertFalse(json.loads(row["reference_duplicate_check"])["is_reference_duplicate"])

    def test_web_rejected_and_review_rows_are_not_scored(self) -> None:
        base = self.root / "data/benchmarks/web_extra_dev"
        paths = [base / name for name in ("manifest.csv", "rejected.csv", "review_candidates.csv")]
        if not all(path.is_file() for path in paths):
            self.skipTest("generated benchmark artifacts are not available")
        rows = []
        for path in paths:
            with path.open(encoding="utf-8", newline="") as stream:
                rows.append(list(csv.DictReader(stream)))
        scored_ids = {row["query_id"] for row in rows[0]}
        auxiliary_ids = {row["query_id"] for row in rows[1]} | {row["query_id"] for row in rows[2]}
        self.assertFalse(scored_ids & auxiliary_ids)

    def test_hard_v2_manifest_and_evidence_are_aligned(self) -> None:
        base = self.root / "data/benchmarks/hard_near_duplicate_dev_v2"
        paths = [base / name for name in ("families.csv", "family_summary.csv", "manifest.csv", "predictions.csv", "metadata.json")]
        if not all(path.is_file() for path in paths):
            self.skipTest("generated hard v2 artifacts are not available")
        with paths[0].open(encoding="utf-8", newline="") as stream:
            families = list(csv.DictReader(stream))
        with paths[1].open(encoding="utf-8", newline="") as stream:
            family_summary = list(csv.DictReader(stream))
        with paths[2].open(encoding="utf-8", newline="") as stream:
            manifest = list(csv.DictReader(stream))
        with paths[3].open(encoding="utf-8", newline="") as stream:
            predictions = list(csv.DictReader(stream))
        self.assertEqual(len(family_summary), len({row["family_id"] for row in families}))
        self.assertEqual(len(family_summary), len({row["family_id"] for row in family_summary}))
        self.assertEqual(len(manifest), 2 * len({row["slug"] for row in families}))
        self.assertEqual(len(manifest), len({row["query_id"] for row in manifest}))
        self.assertEqual({row["query_id"] for row in manifest}, {row["query_id"] for row in predictions})
        self.assertTrue(all(len(row["selection_reason"].split("+")) >= 2 for row in families))
        for row in families:
            evidence = json.loads(row["selection_evidence_json"])
            self.assertTrue(evidence)
            self.assertTrue(all(len(item["signals"]) >= 2 for item in evidence))


if __name__ == "__main__":
    unittest.main()
