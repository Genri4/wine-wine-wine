from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from recognition.catalog import (
    MediaRecord,
    MappingDecision,
    canonicalize_rows,
    detect_variant,
    normalize_filename_transliterated,
    normalize_filename_transliterated_punctuation,
    normalize_filename,
    resolve_media_candidates,
    resolve_media_candidates_cascade,
)

from scripts.build_canonical_catalog import (
    apply_manual_overrides,
    image_media_by_key,
    load_manual_overrides,
)


class CatalogBuilderTests(unittest.TestCase):
    def test_normalization_matches_real_archive_example(self) -> None:
        csv_name = "Amelia 22_png.webp"
        archive_name = "thumbnail_Amelia_22_png_e439aa8564.webp"
        self.assertEqual(normalize_filename(csv_name), normalize_filename(archive_name))

    def test_normalization_handles_hyphen_and_hash_suffix(self) -> None:
        csv_name = "fanagoriya-100-ottenkov-krasnogo-kaberne-kaberne-sovinon-krasnoe-suhoe-135.webp"
        archive_name = "fanagoriya_100_ottenkov_krasnogo_kaberne_kaberne_sovinon_krasnoe_suhoe_135_ad8495191f.webp"
        self.assertEqual(normalize_filename(csv_name), normalize_filename(archive_name))

    def test_transliteration_matches_real_cyrillic_archive_example(self) -> None:
        csv_name = "Агора_Блэк Стоун.webp"
        archive_name = "Agora_Blek_Stoun_e98339ae1b.webp"
        self.assertEqual(normalize_filename_transliterated(csv_name), normalize_filename_transliterated(archive_name))

    def test_punctuation_normalization_matches_real_archive_example(self) -> None:
        csv_name = "(-)_Аристов Кюве Розе_Пино_2023_0,75л_брют Normal-Photoroom.webp"
        archive_name = "Aristov_Kyuve_Roze_Pino_2023_0_75l_bryut_Normal_Photoroom_cc598880d5.webp"
        self.assertEqual(
            normalize_filename_transliterated_punctuation(csv_name),
            normalize_filename_transliterated_punctuation(archive_name),
        )

    def test_variant_detection_uses_real_archive_names(self) -> None:
        self.assertEqual(detect_variant("001_1_4171b0b864.webp"), "original")
        self.assertEqual(detect_variant("thumbnail_001_1_4171b0b864.webp"), "thumbnail")
        self.assertEqual(detect_variant("large_001_1_4171b0b864.webp"), "large")
        self.assertEqual(detect_variant("wines_sitemap_d778a8e06a.xml"), "unknown")

    def test_canonicalization_removes_only_exact_duplicate_rows(self) -> None:
        fields = ("Slug", "Название фото", "Название вина")
        rows = [
            {"Slug": "wine-1", "Название фото": "one.webp", "Название вина": "One"},
            {"Slug": "wine-1", "Название фото": "one.webp", "Название вина": "One"},
            {"Slug": "wine-2", "Название фото": "two.webp", "Название вина": "Two"},
        ]
        result = canonicalize_rows(rows, fields)
        self.assertEqual(len(result.rows), 2)
        self.assertEqual(result.exact_duplicate_groups, 1)
        self.assertEqual(result.exact_duplicate_rows, 1)

    def test_mapping_prefers_original_over_resize_variants(self) -> None:
        original = MediaRecord(
            "uploads/Amelia_22_png_e439aa8564.webp",
            "Amelia_22_png_e439aa8564.webp",
            ".webp",
            100,
            "AAAA",
            "original",
            "amelia_22_png.webp",
        )
        thumbnail = MediaRecord(
            "uploads/thumbnail_Amelia_22_png_e439aa8564.webp",
            "thumbnail_Amelia_22_png_e439aa8564.webp",
            ".webp",
            10,
            "BBBB",
            "thumbnail",
            "amelia_22_png.webp",
        )
        decision = resolve_media_candidates(
            "Amelia 22_png.webp",
            {"amelia_22_png.webp": (thumbnail, original)},
        )
        self.assertEqual(decision.status, "matched")
        self.assertEqual(decision.method, "strapi_normalized_original")
        self.assertEqual(decision.selected, original)

    def test_cascade_uses_unique_transliterated_original(self) -> None:
        original = MediaRecord(
            "uploads/Agora_Blek_Stoun_e98339ae1b.webp",
            "Agora_Blek_Stoun_e98339ae1b.webp",
            ".webp",
            100,
            "AAAA",
            "original",
            "agora_blek_stoun.webp",
        )
        decision = resolve_media_candidates_cascade(
            "Агора_Блэк Стоун.webp",
            {},
            {"agora_blek_stoun.webp": (original,)},
            {},
        )
        self.assertEqual(decision.status, "matched")
        self.assertEqual(decision.method, "transliterated_normalized_original")
        self.assertEqual(decision.selected, original)

    def test_mapping_resolves_crc_identical_originals(self) -> None:
        first = MediaRecord(
            "uploads/product_a_aaaaaaaaaa.webp",
            "product_a_aaaaaaaaaa.webp",
            ".webp",
            100,
            "AAAA",
            "original",
            "product_a.webp",
        )
        second = MediaRecord(
            "uploads/product_a_bbbbbbbbbb.webp",
            "product_a_bbbbbbbbbb.webp",
            ".webp",
            100,
            "AAAA",
            "original",
            "product_a.webp",
        )
        decision = resolve_media_candidates(
            "product_a.webp",
            {"product_a.webp": (second, first)},
        )
        self.assertEqual(decision.status, "matched")
        self.assertEqual(decision.method, "strapi_normalized_crc_identical_original")
        self.assertEqual(decision.selected, first)

    def test_new_rules_preserve_existing_matched_catalog(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "data/processed/catalog_manifest.csv"
        inventory_path = root / "data/processed/media_inventory.csv"
        if not manifest_path.is_file() or not inventory_path.is_file():
            self.skipTest("generated catalog artifacts are not available")
        with manifest_path.open(encoding="utf-8", newline="") as handle:
            manifest = list(csv.DictReader(handle))
        with inventory_path.open(encoding="utf-8", newline="") as handle:
            inventory = list(csv.DictReader(handle))
        media = tuple(
            MediaRecord(
                row["archive_path"],
                row["basename"],
                row["extension"],
                int(row["size"]),
                row["crc"],
                row["variant"],
                row["comparison_key"],
            )
            for row in inventory
        )
        indexes = (
            image_media_by_key(media, normalize_filename),
            image_media_by_key(media, normalize_filename_transliterated),
            image_media_by_key(media, normalize_filename_transliterated_punctuation),
        )
        previous_matched = [row for row in manifest if row["mapping_method"] == "strapi_normalized_original"]
        self.assertEqual(len(previous_matched), 1628)
        for row in previous_matched:
            decision = resolve_media_candidates_cascade(row["photo_name"], *indexes)
            self.assertEqual(decision.status, "matched", row["slug"])
            self.assertIsNotNone(decision.selected)
            self.assertEqual(decision.selected.archive_path, row["original_archive_path"])

    def test_manual_overrides_are_loaded_and_applied_after_deterministic_mapping(self) -> None:
        selected = MediaRecord(
            "uploads/manual_selected.webp",
            "manual_selected.webp",
            ".webp",
            100,
            "AAAA",
            "original",
            "manual_selected.webp",
        )
        with tempfile.TemporaryDirectory() as temporary:
            override_path = Path(temporary) / "overrides.csv"
            override_path.write_text(
                "slug,selected_archive_path,decision,reason,note\n"
                "wine-1,uploads/manual_selected.webp,matched,visual check,confirmed\n"
                "wine-2,,missing,not in archive,\n"
                "wine-3,,unresolved,needs human review,\n",
                encoding="utf-8",
            )
            overrides = load_manual_overrides(
                override_path,
                known_slugs={"wine-1", "wine-2", "wine-3"},
                media_by_path={selected.archive_path: selected},
            )
        decisions = {
            "wine-1": resolve_media_candidates("not-found.webp", {}),
            "wine-2": MappingDecision("matched", "strapi_normalized_original", (selected,), selected),
            "wine-3": MappingDecision("ambiguous", "strapi_normalized_multiple_originals", (selected,)),
        }
        applied = apply_manual_overrides(decisions, overrides, media_by_path={selected.archive_path: selected})
        self.assertEqual(applied["wine-1"].status, "matched")
        self.assertEqual(applied["wine-1"].method, "manual_override_matched")
        self.assertEqual(applied["wine-1"].selected, selected)
        self.assertEqual(applied["wine-2"].status, "missing")
        self.assertIsNone(applied["wine-2"].selected)
        self.assertEqual(applied["wine-3"].status, "unresolved")
        self.assertEqual(applied["wine-3"].candidates, (selected,))

    def test_manual_override_rejects_unknown_asset_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            override_path = Path(temporary) / "overrides.csv"
            override_path.write_text(
                "slug,selected_archive_path,decision,reason,note\n"
                "wine-1,uploads/missing.webp,matched,visual check,\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not in media inventory"):
                load_manual_overrides(
                    override_path,
                    known_slugs={"wine-1"},
                    media_by_path={},
                )


if __name__ == "__main__":
    unittest.main()
