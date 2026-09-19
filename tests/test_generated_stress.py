from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from recognition.benchmark import RankingRecord
from recognition.family_metrics import family_diagnostics
from scripts.build_generated_stress_plan import SCENARIOS, SCENARIO_IDS, build_generation_rows, make_generation_id, select_products
from scripts.build_generated_stress_benchmark import build_benchmark
from scripts.build_generated_stress_pilot32 import select_hard_families, select_representatives
from scripts.import_generated_stress import import_generated_samples
from scripts.evaluate_benchmark import _generated_stress_diagnostics


class GeneratedStressTests(unittest.TestCase):
    def test_pilot32_artifacts_have_expected_shape_and_canonical_references(self) -> None:
        root = Path(__file__).resolve().parents[1]
        base = root / "data/benchmarks/generated_stress_dev_pilot32"
        with (base / "selected_products.csv").open(encoding="utf-8", newline="") as stream:
            selected = list(csv.DictReader(stream))
        with (base / "generation_manifest.csv").open(encoding="utf-8", newline="") as stream:
            plan = list(csv.DictReader(stream))
        with (base / "hard_families.csv").open(encoding="utf-8", newline="") as stream:
            families = list(csv.DictReader(stream))
        with (root / "data/processed/catalog_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
            catalog = list(csv.DictReader(stream))

        usable = {
            row["slug"]: row for row in catalog
            if row.get("mapping_status") == "matched" and row.get("reference_image_path")
        }
        self.assertEqual(len(selected), 32)
        self.assertEqual(len({row["slug"] for row in selected}), 32)
        self.assertEqual(Counter(row["subset_role"] for row in selected), {"representative": 16, "hard": 16})
        self.assertEqual(len(families), 8)
        self.assertEqual(Counter(row["family_type"] for row in families), {"vintage": 4, "subtype": 4})
        self.assertEqual(
            sorted(len(json.loads(row["member_slugs"])) for row in families),
            [2] * 8,
        )
        self.assertTrue({row["slug"] for row in selected} <= set(usable))
        for row in selected:
            reference = Path(row["reference_image"])
            if not reference.is_absolute():
                reference = root / reference
            self.assertTrue(reference.is_file(), row["slug"])
            if row["subset_role"] == "hard":
                self.assertTrue(row["target_family_id"])
                self.assertIn(row["family_type"], {"vintage", "subtype"})
            else:
                self.assertEqual(row["target_family_id"], "")

        self.assertEqual(len(plan), 128)
        self.assertEqual(len({row["generation_id"] for row in plan}), 128)
        self.assertEqual(len({row["output_filename"] for row in plan}), 128)
        self.assertEqual({row["scenario_id"] for row in plan}, set(SCENARIO_IDS))
        self.assertEqual(Counter(row["target_slug"] for row in plan), {row["slug"]: 4 for row in selected})
        self.assertTrue(all(row["generation_status"] == "pending" for row in plan))

    def test_pilot32_selection_is_deterministic(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with (root / "data/processed/catalog_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
            catalog = list(csv.DictReader(stream))
        with (root / "data/benchmarks/generated_stress_dev/selected_products.csv").open(encoding="utf-8", newline="") as stream:
            existing = list(csv.DictReader(stream))
        with (root / "data/benchmarks/hard_near_duplicate_dev_v2/families.csv").open(encoding="utf-8", newline="") as stream:
            hard_rows = list(csv.DictReader(stream))
        with (root / "data/benchmarks/hard_near_duplicate_dev_v2/family_summary.csv").open(encoding="utf-8", newline="") as stream:
            summaries = list(csv.DictReader(stream))
        usable = [row for row in catalog if row.get("mapping_status") == "matched" and row.get("reference_image_path")]
        catalog_by_slug = {row["slug"]: row for row in usable}
        excluded = {row["slug"] for row in hard_rows}
        first_reps = select_representatives(usable, existing, excluded, 16, 20260916)[0]
        second_reps = select_representatives(usable, existing, excluded, 16, 20260916)[0]
        first_families = select_hard_families(hard_rows, summaries, catalog_by_slug, vintage_count=4, subtype_count=4)
        second_families = select_hard_families(hard_rows, summaries, catalog_by_slug, vintage_count=4, subtype_count=4)
        self.assertEqual(first_reps, second_reps)
        self.assertEqual(
            [(family.source_family_id, family.family_type, [member["slug"] for member in family.members]) for family in first_families],
            [(family.source_family_id, family.family_type, [member["slug"] for member in family.members]) for family in second_families],
        )

    def test_product_selection_is_deterministic_and_diversity_aware(self) -> None:
        rows = []
        for category in ("Белое", "Красное"):
            for region in ("Кубань", "Крым"):
                for index in range(8):
                    rows.append({
                        "slug": f"{category}-{region}-{index}",
                        "title": f"Wine {index}",
                        "reference_image_path": f"/tmp/{index}.webp",
                        "winery": f"Winery {index % 4}",
                        "category": category,
                        "region": region,
                        "color": "",
                        "grape": "",
                    })
        first, first_quotas = select_products(rows, 12, 99)
        second, second_quotas = select_products(rows, 12, 99)
        self.assertEqual(first, second)
        self.assertEqual(first_quotas, second_quotas)
        self.assertEqual(len({row["selection_stratum"] for row in first}), 4)
        self.assertLessEqual(max(sum(row["winery"] == winery for row in first) for winery in {row["winery"] for row in first}), 4)

    def test_generation_ids_and_four_scenarios_are_deterministic(self) -> None:
        selected = [{"slug": "wine-alpha", "reference_image": "reference.webp"}, {"slug": "wine-beta", "reference_image": "reference2.webp"}]
        first = build_generation_rows(selected, SCENARIOS)
        second = build_generation_rows(selected, SCENARIOS)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertEqual({row["scenario_id"] for row in first}, set(SCENARIO_IDS))
        self.assertEqual(len({row["generation_id"] for row in first}), len(first))
        self.assertEqual(make_generation_id("wine-alpha", "handheld"), first[3]["generation_id"])
        self.assertTrue(all(row["generation_status"] == "pending" for row in first))

    def test_checked_in_plan_uses_only_usable_canonical_products(self) -> None:
        root = Path(__file__).resolve().parents[1]
        base = root / "data/benchmarks/generated_stress_dev"
        with (base / "selected_products.csv").open(encoding="utf-8", newline="") as stream:
            selected = list(csv.DictReader(stream))
        with (base / "generation_manifest.csv").open(encoding="utf-8", newline="") as stream:
            plan = list(csv.DictReader(stream))
        with (root / "data/processed/catalog_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
            catalog = list(csv.DictReader(stream))
        usable = {row["slug"] for row in catalog if row.get("mapping_status") == "matched" and row.get("reference_image_path")}
        self.assertEqual(len(selected), 150)
        self.assertTrue({row["slug"] for row in selected} <= usable)
        self.assertTrue(all((root / row["reference_image"]).is_file() for row in selected))
        self.assertEqual(len(plan), 600)
        self.assertEqual(len({row["generation_id"] for row in plan}), len(plan))
        self.assertEqual(len({row["output_filename"] for row in plan}), len(plan))
        self.assertEqual({row["scenario_id"] for row in plan}, set(SCENARIO_IDS))
        self.assertTrue(all(sum(row["target_slug"] == slug for row in plan) == 4 for slug in {row["slug"] for row in selected}))

    def test_import_missing_and_corrupt_images_never_become_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.png"
            Image.new("RGB", (160, 240), "white").save(reference)
            generated_dir = root / "generated_raw"
            generated_dir.mkdir()
            (generated_dir / "corrupt.png").write_bytes(b"not an image")
            manifest = root / "generation_manifest.csv"
            _write_csv(manifest, [
                {"generation_id": "g-missing", "target_slug": "wine", "reference_image": str(reference), "scenario_id": "handheld", "output_filename": "missing.png", "prompt": "", "generation_status": "pending"},
                {"generation_id": "g-corrupt", "target_slug": "wine", "reference_image": str(reference), "scenario_id": "slight_angle", "output_filename": "corrupt.png", "prompt": "", "generation_status": "pending"},
            ])
            selected = root / "selected_products.csv"
            _write_csv(selected, [{"slug": "wine", "reference_image": str(reference), "product_name": "Wine", "winery": "Winery", "category": "Белое"}])
            review = root / "review.csv"
            result = import_generated_samples(manifest, selected, generated_dir, review, root / "review.html")
            with review.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            by_id = {row["generation_id"]: row for row in rows}
            self.assertEqual(result["review_status"], {"pending": 1, "rejected": 1})
            self.assertEqual(by_id["g-missing"]["review_status"], "pending")
            self.assertEqual(by_id["g-corrupt"]["review_status"], "rejected")
            self.assertEqual(by_id["g-corrupt"]["validation_status"], "corrupted_image")
            html = (root / "review.html").read_text(encoding="utf-8")
            self.assertIn("Сохранить review.csv", html)
            self.assertIn("data-action='accept'", html)
            self.assertIn("function reviewCsv()", html)

    def test_exact_reference_copy_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.png"
            Image.new("RGB", (160, 240), "white").save(reference)
            generated_dir = root / "generated_raw"
            generated_dir.mkdir()
            (generated_dir / "copy.png").write_bytes(reference.read_bytes())
            manifest = root / "generation_manifest.csv"
            _write_csv(manifest, [{"generation_id": "g-copy", "target_slug": "wine", "reference_image": str(reference), "scenario_id": "handheld", "output_filename": "copy.png"}])
            selected = root / "selected_products.csv"
            _write_csv(selected, [{"slug": "wine", "reference_image": str(reference), "product_name": "Wine", "winery": "Winery", "category": "Белое"}])
            review = root / "review.csv"
            import_generated_samples(manifest, selected, generated_dir, review, root / "review.html")
            with review.open(encoding="utf-8", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["validation_status"], "reference_exact_duplicate")
            self.assertEqual(row["review_status"], "rejected")
            self.assertEqual(row["reject_reason"], "too_easy_reference_copy")

    def test_pending_and_rejected_samples_are_excluded_from_scored_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.png"
            accepted_image = root / "accepted.png"
            Image.new("RGB", (160, 240), "white").save(reference)
            Image.new("RGB", (160, 240), "gray").save(accepted_image)
            catalog = root / "catalog.csv"
            _write_csv(catalog, [{"slug": "wine", "title": "Wine", "reference_image_path": str(reference), "mapping_status": "matched"}])
            plan = root / "generation_manifest.csv"
            _write_csv(plan, [
                {"generation_id": "g-accepted", "target_slug": "wine", "reference_image": str(reference), "scenario_id": "handheld", "output_filename": "accepted.png"},
                {"generation_id": "g-pending", "target_slug": "wine", "reference_image": str(reference), "scenario_id": "handheld", "output_filename": "pending.png"},
                {"generation_id": "g-rejected", "target_slug": "wine", "reference_image": str(reference), "scenario_id": "handheld", "output_filename": "rejected.png"},
            ])
            review = root / "review.csv"
            _write_csv(review, [
                {"generation_id": "g-accepted", "target_slug": "wine", "reference_image": str(reference), "generated_path": str(accepted_image), "review_status": "accepted"},
                {"generation_id": "g-pending", "target_slug": "wine", "reference_image": str(reference), "generated_path": str(root / "pending.png"), "review_status": "pending"},
                {"generation_id": "g-rejected", "target_slug": "wine", "reference_image": str(reference), "generated_path": str(root / "rejected.png"), "review_status": "rejected"},
            ])
            result = build_benchmark(plan, review, catalog, root / "output")
            self.assertEqual(result["accepted"], 1)
            with (root / "output/manifest.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["generation_id"] for row in rows], ["g-accepted"])

    def test_accepted_pilot_rows_preserve_family_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.png"
            accepted_image = root / "accepted.png"
            Image.new("RGB", (160, 240), "white").save(reference)
            Image.new("RGB", (160, 240), "gray").save(accepted_image)
            catalog = root / "catalog.csv"
            _write_csv(catalog, [{"slug": "wine", "title": "Wine", "reference_image_path": str(reference), "mapping_status": "matched"}])
            plan = root / "generation_manifest.csv"
            _write_csv(plan, [{
                "generation_id": "g-accepted", "target_slug": "wine", "reference_image": str(reference),
                "scenario_id": "handheld", "output_filename": "accepted.png", "subset_role": "hard",
                "target_family_id": "family-1", "family_type": "vintage", "family_size": "2",
                "family_member_order": "1",
            }])
            review = root / "review.csv"
            _write_csv(review, [{
                "generation_id": "g-accepted", "target_slug": "wine", "reference_image": str(reference),
                "generated_path": str(accepted_image), "review_status": "accepted",
            }])
            build_benchmark(plan, review, catalog, root / "output")
            with (root / "output/manifest.csv").open(encoding="utf-8", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["target_family_id"], "family-1")
            self.assertEqual(row["family_type"], "vintage")

    def test_family_metrics_distinguish_family_retrieval_from_exact_disambiguation(self) -> None:
        manifest = [
            {"query_id": "q1", "target_slug": "a", "target_family_id": "f1", "family_type": "vintage"},
            {"query_id": "q2", "target_slug": "b", "target_family_id": "f1", "family_type": "vintage"},
            {"query_id": "q3", "target_slug": "c", "target_family_id": "f2", "family_type": "subtype"},
            {"query_id": "q4", "target_slug": "r", "target_family_id": "", "family_type": ""},
        ]
        predictions = [
            {"query_id": "q1", "target_slug": "a", "top5_slugs": '["b","a","c"]'},
            {"query_id": "q2", "target_slug": "b", "top5_slugs": '["b","a","c"]'},
            {"query_id": "q3", "target_slug": "c", "top5_slugs": '["x","c","a"]'},
            {"query_id": "q4", "target_slug": "r", "top5_slugs": '["r","a"]'},
        ]
        result = family_diagnostics(predictions, manifest)
        self.assertEqual(result["hard_query_count"], 3)
        self.assertEqual(result["hard_family_count"], 2)
        self.assertAlmostEqual(result["family_top1"], 2 / 3)
        self.assertEqual(result["family_recall_at_5"], 1.0)
        self.assertAlmostEqual(result["within_family_disambiguation_top1"], 0.5)
        self.assertEqual(result["confusion_within_family"][0]["target_slug"], "a")

    def test_evaluator_reports_per_scenario_and_alignment(self) -> None:
        manifest = [
            {"query_id": "q1", "target_slug": "a", "scenario_id": "handheld"},
            {"query_id": "q2", "target_slug": "b", "scenario_id": "glare_bad_light"},
        ]
        predictions = [
            {"query_id": "q1", "target_slug": "a", "predicted_slug": "a", "top5_slugs": '["a","b"]', "top5_scores": "[0.9,0.1]", "latency_ms": "1"},
            {"query_id": "q2", "target_slug": "b", "predicted_slug": "a", "top5_slugs": '["a","b"]', "top5_scores": "[0.9,0.89]", "latency_ms": "1"},
        ]
        result = _generated_stress_diagnostics(predictions, manifest, {"top1_accuracy": 0.8, "recall_at_5": 0.9, "mrr": 0.85})
        self.assertEqual(result["per_scenario"]["handheld"]["top1_accuracy"], 1.0)
        self.assertEqual(result["per_scenario"]["glare_bad_light"]["top1_accuracy"], 0.0)
        self.assertAlmostEqual(result["degradation_vs_synthetic_dev"]["top1_accuracy"]["delta_generated_minus_synthetic"], -0.3)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
