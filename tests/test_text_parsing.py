import unittest

from AI_georef_plugin.georef_core.location_hints import (
    _address_is_specific,
    _extract_project_city,
    _extract_road_codes,
    extract_structured_location_hints,
)
from AI_georef_plugin.georef_core.text_parsing import (
    _extract_best_scale,
    _merge_text_sources,
    _normalize_text_token,
    parse_coordinates,
)


class TextParsingTests(unittest.TestCase):
    def test_extract_best_scale_prefers_title_block_context(self):
        text = "\n".join(["Detail A", "1:100", "Massstab", "1 : 5000"])
        self.assertEqual(_extract_best_scale(text), 5000)

    def test_merge_text_sources_avoids_near_duplicate_lines(self):
        merged = _merge_text_sources(
            "Projekt Musterstrasse 1\nStandort Kamen",
            "Projekt Musterstrasse 1\nStandort Kamen\nB55 Ausbau",
        )
        self.assertEqual(merged.splitlines(), ["Projekt Musterstrasse 1", "Standort Kamen", "B55 Ausbau"])

    def test_parse_coordinates_extracts_pairs_scale_and_hints(self):
        parsed = parse_coordinates("E 395000 N 5723000\nUTM 32\nMassstab 1:2500")
        self.assertEqual(parsed["pairs"], [(395000.0, 5723000.0)])
        self.assertEqual(parsed["scale"], 2500)
        self.assertIn("UTM", parsed["crs_hints"])

    def test_location_helpers_keep_site_and_office_separate(self):
        text = "\n".join(
            [
                "Regionalniederlassung Ruhr",
                "Harpener Hellweg 1",
                "44791 Bochum",
                "Ausbau Dortmunder Allee in Kamen",
                "B 55 bei Kamen",
            ]
        )
        hints = extract_structured_location_hints(text)
        self.assertEqual(hints["site_city"], "Kamen")
        self.assertIn("B55", hints["road_codes"])
        self.assertEqual(hints["client_address"], "Harpener Hellweg 1, 44791 Bochum")

    def test_address_specificity_and_city_extraction(self):
        self.assertTrue(_address_is_specific("Musterstrasse 12, 44791 Bochum"))
        self.assertFalse(_address_is_specific("44791 Bochum"))
        self.assertEqual(_extract_project_city("Musterstrasse 12, 44791 Bochum"), "Bochum")

    def test_extract_road_codes_skips_paper_sizes(self):
        self.assertEqual(_extract_road_codes("DIN A1 Planblatt\nL 663 Westabschnitt\nFormat A3"), ["L663"])

    def test_normalize_text_token_handles_umlauts(self):
        self.assertEqual(_normalize_text_token("Straße Köln-Süd"), "strasse koln sud")


if __name__ == "__main__":
    unittest.main()
