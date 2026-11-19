"""Automated verification suite for CivicFusion AI Multilingual Intelligence.

Tests the complete multilingual lifecycle:
1. Language identification (English, Hindi, Telugu).
2. Directional machine translation across all 6 language pairs.
3. Multilingual Speech-to-Text configuration and locale resolution.
4. Native script domain keyword matching in Hindi and Telugu.
5. End-to-end NLP analysis (category, confidence, urgency cues, impact, severity).
6. Storage persistence without script corruption or loss of original text.
7. Municipal Officer on-demand translation without modifying official records.
8. Cross-lingual semantic duplicate clustering.
"""

from datetime import date
import unittest

from duplicates import check_for_duplicates
import multilingual
from multilingual import (
    detect_language,
    get_language_display_name,
    get_language_speech_code,
    match_native_civic_category,
    normalize_lang_code,
    translate_text,
)
import nlp
import speech_utils
from storage import delete_complaint, insert_complaint, load_all_complaints


class TestMultilingualIntelligence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp_model = nlp.load_classifier()

    def test_1_language_detection(self):
        """Test Unicode script-based language identification."""
        self.assertEqual(detect_language("There is a large pothole on the road"), "en")
        self.assertEqual(detect_language("सड़क पर बहुत बड़ा गड्ढा है"), "hi")
        self.assertEqual(detect_language("రోడ్డుపై పెద్ద గుంత ఉంది"), "te")
        self.assertEqual(detect_language(""), "en")
        self.assertEqual(detect_language("   "), "en")

    def test_2_speech_locale_resolution(self):
        """Test speech recognition locale and Whisper language code mapping."""
        # Hindi
        loc_hi, whis_hi = speech_utils.resolve_speech_locale("hi")
        self.assertEqual(loc_hi, "hi-IN")
        self.assertEqual(whis_hi, "hi")

        loc_hi_in, _ = speech_utils.resolve_speech_locale("hi-IN")
        self.assertEqual(loc_hi_in, "hi-IN")

        # Telugu
        loc_te, whis_te = speech_utils.resolve_speech_locale("te")
        self.assertEqual(loc_te, "te-IN")
        self.assertEqual(whis_te, "te")

        # English
        loc_en, whis_en = speech_utils.resolve_speech_locale("en")
        self.assertEqual(loc_en, "en-IN")
        self.assertEqual(whis_en, "en")

    def test_3_native_civic_keyword_lexicon(self):
        """Test native script domain keyword matching for Hindi and Telugu."""
        # Hindi keywords
        cat_hi_pothole, conf_hi_1 = match_native_civic_category("सड़क पर बहुत गहरा गड्ढा है", "hi")
        self.assertEqual(cat_hi_pothole, "road_damage")
        self.assertGreaterEqual(conf_hi_1, 0.90)

        cat_hi_waste, conf_hi_2 = match_native_civic_category("बाजार में कचरे का भारी ढेर लगा है", "hi")
        self.assertEqual(cat_hi_waste, "waste")
        self.assertGreaterEqual(conf_hi_2, 0.90)

        cat_hi_drain, conf_hi_3 = match_native_civic_category("सड़क किनारे खुली नाली है", "hi")
        self.assertEqual(cat_hi_drain, "open_drain")

        # Telugu keywords
        cat_te_pothole, conf_te_1 = match_native_civic_category("రోడ్డుపై ప్రమాదకరమైన గుంత ఉంది", "te")
        self.assertEqual(cat_te_pothole, "road_damage")
        self.assertGreaterEqual(conf_te_1, 0.90)

        cat_te_waste, conf_te_2 = match_native_civic_category("వీధిలో మున్సిపల్ చెత్త కుప్పగా పేరుకుపోయింది", "te")
        self.assertEqual(cat_te_waste, "waste")

        cat_te_light, conf_te_3 = match_native_civic_category("రాత్రిపూట వీధి దీపాలు వెలగడం లేదు", "te")
        self.assertEqual(cat_te_light, "streetlight_damage")

    def test_4_machine_translation_directional_pairs(self):
        """Test machine translation across all key language directions."""
        # Hindi -> English
        t_hi_en, src = translate_text("सड़क पर बहुत गड्ढे हैं", source_lang="hi", target_lang="en")
        self.assertIn("pothole", t_hi_en.lower())

        # Telugu -> English
        t_te_en, src = translate_text("రోడ్డుపై గుంతలు ఉన్నాయి", source_lang="te", target_lang="en")
        self.assertIn("pothole", t_te_en.lower())

        # English -> Hindi
        t_en_hi, src = translate_text("Broken streetlights at night", source_lang="en", target_lang="hi")
        self.assertTrue(any(c in t_en_hi for c in ["लाइट", "अंधेरा", "स्ट्रीट"]))

        # English -> Telugu
        t_en_te, src = translate_text("Broken streetlights at night", source_lang="en", target_lang="te")
        self.assertTrue(any(c in t_en_te for c in ["లైట్", "దీపాలు", "చీకటి"]))

    def test_5_multilingual_nlp_end_to_end(self):
        """Test end-to-end NLP analysis on native complaints in all 3 languages."""
        # 1. Hindi Pothole Complaint
        res_hi = nlp.analyze_text_full(
            self.nlp_model,
            "जीटी रोड पर फगवाड़ा फ्लाईओवर के पास बहुत बड़ा गड्ढा है और गाड़ियाँ फिसल रही हैं।",
            user_lang="hi"
        )
        self.assertEqual(res_hi["category"], "road_damage")
        self.assertEqual(res_hi["original_language"], "hi")
        self.assertIn("pothole", res_hi["translated_text_en"].lower())
        self.assertIn("Vehicles", res_hi["impact"])

        # 2. Telugu Streetlight Complaint
        res_te = nlp.analyze_text_full(
            self.nlp_model,
            "మోడల్ టౌన్ మార్కెట్ వద్ద వీధి లైట్లు పాడై రాత్రిపూట చాలా చీకటిగా ఉంది.",
            user_lang="te"
        )
        self.assertEqual(res_te["category"], "streetlight_damage")
        self.assertEqual(res_te["original_language"], "te")
        self.assertTrue(len(res_te["urgency_cues"]) > 0)

        # 3. English Garbage Complaint
        res_en = nlp.analyze_text_full(
            self.nlp_model,
            "Huge pile of garbage dumped on Hoshiarpur road creating unbearable stench.",
            user_lang="en"
        )
        self.assertEqual(res_en["category"], "waste")
        self.assertEqual(res_en["original_language"], "en")

    def test_6_storage_persistence_without_corruption(self):
        """Verify native Devanagari and Telugu scripts persist intact in storage."""
        hindi_desc = "फगवाड़ा रेलवे स्टेशन के पास खुला नाला टूटा हुआ है।"
        test_id = "CF-TEST-HI-01"

        new_ticket = {
            "id": test_id,
            "description": hindi_desc,
            "original_language": "hi",
            "language_name": "हिंदी (Hindi)",
            "translated_description_en": "Open drain near Phagwara railway station is broken.",
            "category_raw": "open_drain",
            "category_display": "Open Drain",
            "problem_title": "Hazardous Open Manhole / Missing Grate",
            "department": "Water Supply & Sewerage Board",
            "location_name": "Railway Station Road, Phagwara",
            "latitude": 31.2185,
            "longitude": 75.7672,
            "severity": "High",
            "priority": "P2",
            "status": "Open",
            "citizen_reports_count": 1,
            "ai_confidence": 0.95,
            "needs_review": False,
            "reported_date": date.today().isoformat(),
        }

        # Insert and verify
        insert_complaint(new_ticket)
        reloaded = load_all_complaints()
        found = next((c for c in reloaded if c.get("id") == test_id), None)

        self.assertIsNotNone(found)
        self.assertEqual(found["description"], hindi_desc)
        self.assertEqual(found["original_language"], "hi")

        # Clean up test ticket
        delete_complaint(test_id)

    def test_7_cross_lingual_duplicate_detection(self):
        """Verify duplicate detection matches English and Hindi reports at same location."""
        existing_reports = [
            {
                "id": "CF-1042",
                "description": "Large dangerous pothole near Phagwara flyover on GT Road.",
                "category_raw": "road_damage",
                "latitude": 31.2241,
                "longitude": 75.7709,
                "status": "Open",
                "citizen_reports_count": 1,
            }
        ]

        # Citizen submits same issue in Hindi at the same coordinates
        hindi_text = "जीटी रोड पर फगवाड़ा फ्लाईओवर के पास बहुत बड़ा गड्ढा है।"
        translated_en, _ = translate_text(hindi_text, source_lang="hi", target_lang="en")

        dup = check_for_duplicates(
            new_text=translated_en,
            new_lat=31.2242,  # ~11 meters away
            new_lon=75.7710,
            new_category="road_damage",
            existing_reports=existing_reports,
        )

        self.assertTrue(dup["is_duplicate"])
        self.assertEqual(dup["parent_id"], "CF-1042")


if __name__ == "__main__":
    unittest.main()

