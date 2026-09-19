from __future__ import annotations

import unittest

from recognition.hard_families import build_hard_families
from recognition.hard_families import HardFamily
from recognition.hard_families_v2 import build_hard_families_v2


class HardFamilyTests(unittest.TestCase):
    def test_family_construction_is_reproducible_and_uses_confusion_signal(self) -> None:
        catalog = [
            {"slug": "line-cabernet-2021", "title": "Line Cabernet, 2021", "winery": "Winery", "category": "Красное", "color": "Красный", "region": "Кубань", "grape": "Каберне", "mapping_status": "matched", "reference_image_path": "a.webp"},
            {"slug": "line-cabernet-2022", "title": "Line Cabernet, 2022", "winery": "Winery", "category": "Красное", "color": "Красный", "region": "Кубань", "grape": "Каберне", "mapping_status": "matched", "reference_image_path": "b.webp"},
            {"slug": "other-white", "title": "Other White", "winery": "Other Winery", "category": "Белое", "color": "Белый", "region": "Крым", "grape": "Рислинг", "mapping_status": "matched", "reference_image_path": "c.webp"},
        ]
        predictions = [
            {"target_slug": "line-cabernet-2021", "predicted_slug": "line-cabernet-2022", "target_rank": "2", "top1_top2_margin": "0.01"}
        ]
        first = build_hard_families(catalog, predictions)
        second = build_hard_families(catalog, predictions)
        self.assertEqual(first, second)
        self.assertEqual(first[0].slugs, ("line-cabernet-2021", "line-cabernet-2022"))
        self.assertIn("observed_confusion_count=1", first[0].reason)

    def test_v2_requires_two_strong_signals_and_carries_evidence(self) -> None:
        catalog = [
            {"slug": "line-cabernet-2021", "title": "Line Cabernet 2021", "winery": "Winery", "category": "Красное", "color": "Красный", "region": "Кубань", "grape": "Каберне", "mapping_status": "matched", "reference_image_path": "a.webp"},
            {"slug": "line-cabernet-2022", "title": "Line Cabernet 2022", "winery": "Winery", "category": "Красное", "color": "Красный", "region": "Кубань", "grape": "Каберне", "mapping_status": "matched", "reference_image_path": "b.webp"},
            {"slug": "line-merlot-2022", "title": "Line Merlot 2022", "winery": "Winery", "category": "Красное", "color": "Красный", "region": "Кубань", "grape": "Мерло", "mapping_status": "matched", "reference_image_path": "c.webp"},
        ]
        v1 = [HardFamily("family-001", tuple(row["slug"] for row in catalog), "same_winery;shared_name_and_wine_attributes", ())]
        predictions = [{
            "target_slug": "line-cabernet-2021", "predicted_slug": "line-cabernet-2022",
            "target_rank": "2", "top1_top2_margin": "0.01",
        }]
        visual = {
            ("line-cabernet-2021", "line-cabernet-2022"): {"image_similarity": True, "phash_hamming": 12},
            ("line-cabernet-2021", "line-merlot-2022"): {"image_similarity": False},
            ("line-cabernet-2022", "line-merlot-2022"): {"image_similarity": False},
        }
        families = build_hard_families_v2(catalog, predictions, visual, v1)
        self.assertEqual([family.slugs for family in families], [("line-cabernet-2021", "line-cabernet-2022")])
        self.assertIn("real_confusion_rank_2_5", families[0].selection_reason)
        self.assertIn("image_similarity", families[0].selection_reason)
        self.assertEqual(families[0].edges[0].image_phash_hamming, 12)

    def test_v2_does_not_select_metadata_only_pair(self) -> None:
        catalog = [
            {"slug": "same-line-a", "title": "Same Line A", "winery": "Winery", "category": "Красное", "mapping_status": "matched", "reference_image_path": "a.webp"},
            {"slug": "same-line-b", "title": "Same Line B", "winery": "Winery", "category": "Красное", "mapping_status": "matched", "reference_image_path": "b.webp"},
        ]
        v1 = [HardFamily("family-001", ("same-line-a", "same-line-b"), "same_winery;shared_name_and_wine_attributes", ())]
        self.assertEqual(build_hard_families_v2(catalog, [], {}, v1), [])


if __name__ == "__main__":
    unittest.main()
