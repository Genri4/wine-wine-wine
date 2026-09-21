from __future__ import annotations

import unittest

from recognition.text_normalization import (
    MAX_VINTAGE_YEAR,
    MIN_VINTAGE_YEAR,
    comparison_keys,
    extract_vintage_years,
    normalize_text,
    vintage_match_state,
)


class NormalizeTextTests(unittest.TestCase):
    def test_lowercases_and_folds_yo(self) -> None:
        self.assertEqual(normalize_text("Ароматное Ёлки"), "ароматное елки")

    def test_removes_punctuation_but_keeps_digits(self) -> None:
        self.assertEqual(normalize_text("Amelia, 2023! (dry)"), "amelia 2023 dry")

    def test_collapses_whitespace(self) -> None:
        self.assertEqual(normalize_text("  a\t\nb   c "), "a b c")

    def test_never_removes_years(self) -> None:
        self.assertIn("2021", normalize_text("Вино (2021) — выдержка"))
        self.assertIn("1900", normalize_text("1900"))
        self.assertIn("2030", normalize_text("2030"))

    def test_cyrillic_and_latin_both_safe(self) -> None:
        self.assertEqual(normalize_text("Каберне Совиньон"), "каберне совиньон")
        self.assertEqual(normalize_text("Cabernet Sauvignon"), "cabernet sauvignon")

    def test_unicode_normalization(self) -> None:
        # NFKC folds fullwidth digits into ASCII digits, preserving the year.
        self.assertIn("2023", normalize_text("２０２３"))

    def test_non_string_rejected(self) -> None:
        with self.assertRaises(TypeError):
            normalize_text(123)  # type: ignore[arg-type]


class ComparisonKeysTests(unittest.TestCase):
    def test_transliterated_key(self) -> None:
        self.assertEqual(comparison_keys("Ароматное")[1], "aromatnoe")

    def test_keys_pair(self) -> None:
        normalized, translit = comparison_keys("Ароматное Малбек")
        self.assertEqual(normalized, "ароматное малбек")
        self.assertEqual(translit, "aromatnoe malbek")


class ExtractVintageYearsTests(unittest.TestCase):
    def test_extracts_single_year(self) -> None:
        self.assertEqual(extract_vintage_years("Amelia, 2023"), ["2023"])

    def test_extracts_multiple_years_in_order(self) -> None:
        self.assertEqual(extract_vintage_years("1998 урожай, розлив 2023"), ["1998", "2023"])

    def test_deduplicates(self) -> None:
        self.assertEqual(extract_vintage_years("2023 2023"), ["2023"])

    def test_rejects_out_of_range(self) -> None:
        self.assertEqual(extract_vintage_years("1490"), [])
        self.assertEqual(extract_vintage_years("2200"), [])

    def test_rejects_digits_inside_longer_runs(self) -> None:
        self.assertEqual(extract_vintage_years("11490"), [])
        self.assertEqual(extract_vintage_years("20235"), [])

    def test_decimal_alcohol_is_not_a_year(self) -> None:
        self.assertEqual(extract_vintage_years("алкоголь 12.5% об."), [])
        self.assertEqual(extract_vintage_years("13.5"), [])

    def test_range_constants_are_sane(self) -> None:
        self.assertEqual(MIN_VINTAGE_YEAR, 1900)
        self.assertGreaterEqual(MAX_VINTAGE_YEAR, 2026)


class VintageMatchStateTests(unittest.TestCase):
    def test_exact_match(self) -> None:
        self.assertEqual(vintage_match_state(["2023"], "2023"), "exact_match")

    def test_mismatch(self) -> None:
        self.assertEqual(vintage_match_state(["2022"], "2023"), "mismatch")

    def test_unknown_when_candidate_year_missing(self) -> None:
        self.assertEqual(vintage_match_state(["2022"], None), "unknown")

    def test_unknown_when_query_has_no_year(self) -> None:
        self.assertEqual(vintage_match_state([], "2023"), "unknown")

    def test_unknown_never_proves_mismatch(self) -> None:
        self.assertNotEqual(vintage_match_state([], "2023"), "mismatch")
        self.assertNotEqual(vintage_match_state(["2022"], None), "mismatch")


if __name__ == "__main__":
    unittest.main()
