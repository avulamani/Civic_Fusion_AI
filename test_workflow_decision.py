"""Automated test suite verifying the improved CivicFusion AI complaint decision-making workflow.

Verifies:
1. Citizen Ground Truth: Citizen description & NLP determine final category.
2. Compound Defect Resolution:
   - Pothole with rainwater -> Primary: road_damage, Secondary: waterlogging
   - Pothole with garbage -> Primary: road_damage, Secondary: waste
   - Blocked drain with flooding -> Primary: drain_blockage, Secondary: waterlogging
3. Multilingual Support: Hindi (Devanagari), Telugu, Romanized Hindi (Hinglish).
4. Auto-Registration: Immediate registration with status 'Registered', no blocking 'Pending Review'.
5. Explanations: Transparent decision reasoning and modality alignment.
6. Audit Logging: Disagreements logged to cv_nlp_mismatches.json without pseudo-labels.
"""

from io import BytesIO
import json
from pathlib import Path
import unittest

from fusion import (
    DEPARTMENTS,
    DISPLAY_NAMES,
    fuse_and_recommend,
)
import nlp
from nlp import (
    analyze_text_full,
    detect_all_civic_issues,
    load_classifier,
    predict_category,
    resolve_compound_issues,
)
import storage
from storage import (
    load_all_mismatches,
    log_cv_nlp_mismatch,
    MISMATCH_FEEDBACK_FILE,
)


class TestWorkflowDecisionMaking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp_model = load_classifier()

    def test_case_1_cv_wrong_nlp_correct_citizen_authority(self):
        """Case 1: CV predicts Open Manhole (94%), citizen describes deep pothole near bakery."""
        text = "Deep pothole near bakery causing two-wheelers to fall"
        nlp_res = analyze_text_full(self.nlp_model, text, user_lang="en")

        self.assertEqual(nlp_res["category"], "road_damage")

        # Simulate YOLO detecting Open Manhole
        fake_detections = [
            {
                "class": "open_manhole",
                "category": "Open Manhole",
                "confidence": 0.94,
                "confidence_percent": 94.0,
                "box": [100, 150, 400, 450],
            }
        ]

        fusion = fuse_and_recommend(
            text_category=nlp_res["category"],
            nlp_confidence=nlp_res["confidence"],
            urgency_cues=nlp_res["urgency_cues"],
            detections=fake_detections,
            location_name="Bakery, Phagwara",
            latitude=31.2295,
            longitude=75.77999,
            impact_text=nlp_res["impact"],
            duplicate_info={"is_duplicate": False},
            secondary_categories=nlp_res.get("secondary_categories", []),
            resolution_note=nlp_res.get("resolution_note", ""),
        )

        # Ground truth check
        self.assertEqual(fusion["category_raw"], "road_damage")
        self.assertEqual(fusion["department"], "Road Maintenance Department")
        self.assertEqual(fusion["registration_status"], "Registered")
        self.assertFalse(fusion["needs_review"])
        self.assertFalse(fusion["agreement"])
        self.assertIn("Road", fusion["department"])
        self.assertIn("open_manhole", fusion["visual_categories"])
        self.assertTrue(
            "supplementary visual evidence" in fusion["decision_reason"].lower()
            or "supporting visual evidence" in fusion["decision_reason"].lower()
        )

    def test_case_2_pothole_with_rainwater_compound_issue(self):
        """Case 2: Rainwater accumulated in big potholes on main road.
        Must classify as road_damage primary, waterlogging secondary.
        Never classify as purely waterlogging merely because water is inside a pothole cavity.
        """
        text = "Rainwater accumulated in big potholes on main road"
        nlp_res = analyze_text_full(self.nlp_model, text, user_lang="en")

        self.assertEqual(nlp_res["category"], "road_damage")
        self.assertIn("waterlogging", nlp_res["secondary_categories"])

        fusion = fuse_and_recommend(
            text_category=nlp_res["category"],
            nlp_confidence=nlp_res["confidence"],
            urgency_cues=nlp_res["urgency_cues"],
            detections=[],
            location_name="Main Road, Jalandhar",
            latitude=31.326,
            longitude=75.576,
            impact_text=nlp_res["impact"],
            duplicate_info={"is_duplicate": False},
            secondary_categories=nlp_res["secondary_categories"],
            resolution_note=nlp_res.get("resolution_note", ""),
        )

        self.assertEqual(fusion["category_raw"], "road_damage")
        self.assertEqual(fusion["department"], "Road Maintenance Department")
        self.assertEqual(fusion["registration_status"], "Registered")
        self.assertIn("waterlogging", fusion["secondary_categories"])

    def test_case_3_genuine_garbage_complaint(self):
        """Case 3: Huge pile of garbage dumped on roadside not cleared for days."""
        text = "Huge pile of garbage dumped on the roadside not cleared for days"
        nlp_res = analyze_text_full(self.nlp_model, text, user_lang="en")

        self.assertEqual(nlp_res["category"], "waste")

        fake_detections = [
            {
                "class": "garbage",
                "category": "Garbage / Waste",
                "confidence": 0.88,
                "confidence_percent": 88.0,
                "box": [50, 50, 300, 300],
            }
        ]

        fusion = fuse_and_recommend(
            text_category=nlp_res["category"],
            nlp_confidence=nlp_res["confidence"],
            urgency_cues=nlp_res["urgency_cues"],
            detections=fake_detections,
            location_name="Model Town, Jalandhar",
            latitude=31.31,
            longitude=75.58,
            impact_text=nlp_res["impact"],
            duplicate_info={"is_duplicate": False},
        )

        self.assertEqual(fusion["category_raw"], "waste")
        self.assertEqual(fusion["department"], "Sanitation & Waste Management Department")
        self.assertTrue(fusion["agreement"])
        self.assertFalse(fusion["needs_review"])

    def test_case_4_open_drainage_canal_vs_manhole(self):
        """Case 4: Open drainage canal overflowing onto the street -> open_drain / Drainage."""
        text = "Open drainage canal overflowing onto the street with dirty sewage water"
        nlp_res = analyze_text_full(self.nlp_model, text, user_lang="en")

        # Must route to Drainage & Sewage Division
        self.assertIn(nlp_res["category"], ["open_drain", "drain_blockage"])

        fusion = fuse_and_recommend(
            text_category=nlp_res["category"],
            nlp_confidence=nlp_res["confidence"],
            urgency_cues=nlp_res["urgency_cues"],
            detections=[],
            location_name="Patel Chowk, Jalandhar",
            latitude=31.33,
            longitude=75.57,
            impact_text=nlp_res["impact"],
            duplicate_info={"is_duplicate": False},
        )

        self.assertEqual(fusion["department"], "Stormwater Drainage & Sewerage Division")
        self.assertEqual(fusion["registration_status"], "Registered")

    def test_case_5_hindi_complaint_pothole_rainwater(self):
        """Case 5: Hindi text: हमारी सड़क पर बहुत सारे गड्ढे हैं और बारिश का पानी भरा हुआ है."""
        hindi_text = "हमारी सड़क पर बहुत सारे गड्ढे हैं और बारिश का पानी भरा हुआ है"
        nlp_res = analyze_text_full(self.nlp_model, hindi_text, user_lang="hi")

        self.assertEqual(nlp_res["category"], "road_damage")
        self.assertIn("waterlogging", nlp_res["secondary_categories"])

        fusion = fuse_and_recommend(
            text_category=nlp_res["category"],
            nlp_confidence=nlp_res["confidence"],
            urgency_cues=nlp_res["urgency_cues"],
            detections=[],
            location_name="Phagwara Bypass",
            latitude=31.22,
            longitude=75.77,
            impact_text=nlp_res["impact"],
            duplicate_info={"is_duplicate": False},
            secondary_categories=nlp_res["secondary_categories"],
        )

        self.assertEqual(fusion["category_raw"], "road_damage")
        self.assertEqual(fusion["department"], "Road Maintenance Department")
        self.assertEqual(fusion["registration_status"], "Registered")

    def test_case_6_telugu_complaint_road_damage(self):
        """Case 6: Telugu text: మా ప్రాంతంలో రోడ్లపై చాలా గుంతలు ఉన్నాయి వాహనాలు నడపడం చాలా కష్టంగా ఉంది."""
        telugu_text = "మా ప్రాంతంలో రోడ్లపై చాలా గుంతలు ఉన్నాయి వాహనాలు నడపడం చాలా కష్టంగా ఉంది"
        nlp_res = analyze_text_full(self.nlp_model, telugu_text, user_lang="te")

        self.assertEqual(nlp_res["category"], "road_damage")

        fusion = fuse_and_recommend(
            text_category=nlp_res["category"],
            nlp_confidence=nlp_res["confidence"],
            urgency_cues=nlp_res["urgency_cues"],
            detections=[],
            location_name="Phagwara Local",
            latitude=31.22,
            longitude=75.77,
            impact_text=nlp_res["impact"],
            duplicate_info={"is_duplicate": False},
        )

        self.assertEqual(fusion["category_raw"], "road_damage")
        self.assertEqual(fusion["department"], "Road Maintenance Department")
        self.assertEqual(fusion["registration_status"], "Registered")

    def test_case_7_multiple_problems_compound(self):
        """Case 7: Multiple problems: potholes, garbage, and blocked drain."""
        text = "Deep road potholes with heaps of smelly garbage and a blocked drainage ditch"
        nlp_res = analyze_text_full(self.nlp_model, text, user_lang="en")

        # Primary is road_damage (infrastructure safety)
        self.assertEqual(nlp_res["category"], "road_damage")
        # Secondary captures waste and drainage
        self.assertIn("waste", nlp_res["secondary_categories"])
        self.assertTrue(
            "drain_blockage" in nlp_res["secondary_categories"]
            or "open_drain" in nlp_res["secondary_categories"]
        )

    def test_hinglish_road_damage(self):
        """Bonus: Hinglish (Romanized Hindi) complaint: sadak par bahut bade gaddhe hain aur paani bhara hai."""
        hinglish_text = "sadak par bahut bade gaddhe hain aur paani bhara hai"
        nlp_res = analyze_text_full(self.nlp_model, hinglish_text, user_lang="en")

        self.assertEqual(nlp_res["category"], "road_damage")
        self.assertIn("waterlogging", nlp_res["secondary_categories"])

    def test_explicit_category_override(self):
        """Citizen explicitly selects category in dropdown."""
        text = "The area is completely messy and needs attention"
        nlp_res = analyze_text_full(
            self.nlp_model,
            text,
            user_lang="en",
            explicit_category="waste",
        )
        self.assertEqual(nlp_res["category"], "waste")
        self.assertIn("explicitly", nlp_res.get("resolution_note", "").lower())

    def test_mismatch_logging_audit_trail(self):
        """Verify log_cv_nlp_mismatch creates a structured JSON record without pseudo-labels."""
        ticket_id = "CF-TEST-9999"
        log_cv_nlp_mismatch(
            ticket_id=ticket_id,
            citizen_description="Deep pothole near bakery",
            nlp_category="road_damage",
            nlp_confidence=0.92,
            cv_categories=["open_manhole"],
            cv_confidence=0.94,
            final_category="road_damage",
            photo_path="data/uploads/CF-TEST-9999.jpg",
            model_version="YOLO11n-Municipal-v2",
            decision_reason="Classified as Pothole based on citizen ground-truth description.",
            original_language="en",
        )

        all_mismatches = load_all_mismatches()
        matching = [m for m in all_mismatches if m.get("ticket_id") == ticket_id]
        self.assertTrue(len(matching) >= 1)
        record = matching[-1]
        self.assertEqual(record["nlp_category"], "road_damage")
        self.assertEqual(record["cv_categories"], ["open_manhole"])
        self.assertEqual(record["final_category"], "road_damage")
        self.assertTrue(record.get("requires_human_reannotation", True))
        self.assertIn("decision_reason", record)


if __name__ == "__main__":
    unittest.main()
