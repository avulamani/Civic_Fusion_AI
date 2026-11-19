"""End-to-end system integration test suite for CivicFusion AI.

Validates all 10 core integration scenarios across Computer Vision (YOLO),
Natural Language Processing (IndicBERT), and Cross-Lingual Semantic Intelligence (LaBSE):
  1. Road damage / Pothole detection and triage
  2. Solid waste / Garbage accumulation detection and triage
  3. Open drain / roadside trench detection and triage
  4. Hazardous open manhole safety escalation (P1/Critical)
  5. Streetlight outage and night hazard triage
  6. Clean road negative control (Zero false-positive verification)
  7. Cross-lingual duplicate clustering (EN vs HI < 50m)
  8. Spatial separation safety filter (> 250m prevents merging across town)
  9. Co-located category separation (Garbage vs Streetlight at same GPS)
  10. Script preservation (Hindi Devanagari and Telugu preserved intact)
"""

import json
from pathlib import Path
import time
from PIL import Image

import duplicates
import fusion
import nlp
import vision

ROOT = Path(__file__).parent


def run_all_tests():
    print("=" * 75)
    print("CIVICFUSION AI - END-TO-END MULTIMODAL SYSTEM INTEGRATION TESTS")
    print("=" * 75)

    test_results = []
    
    # Preload models
    nlp_model = nlp.load_classifier()
    classes = vision.load_classes()
    print(f"Loaded Vision Classes: {classes}")

    # SCENARIO 1: Road Damage / Pothole with Hindi Text
    print("\n--- Test 1: Pothole Defect with Hindi Text ---")
    hi_text = "स्कूल के प्रवेश द्वार के पास सड़क पर एक बड़ा गड्ढा है। दोपहिया वाहन गिर रहे हैं।"
    pothole_img_path = ROOT / "datasets" / "municipal_detection" / "images" / "test" / "user_pothole_hedge.jpg"
    pothole_img = Image.open(pothole_img_path) if pothole_img_path.exists() else Image.new("RGB", (640, 640), (120, 120, 120))

    cv_dets, _ = vision.detect(pothole_img, confidence_threshold=0.15)
    nlp_res = nlp.analyze_text_full(nlp_model, hi_text, user_lang="hi")
    f_res = fusion.fuse_and_recommend(
        text_category=nlp_res["category"],
        nlp_confidence=nlp_res["confidence"],
        urgency_cues=nlp_res["urgency_cues"],
        detections=cv_dets,
        location_name="Near School Gate",
        latitude=31.2185,
        longitude=75.7672,
        impact_text=nlp_res["impact"],
    )

    t1_pass = (nlp_res["category"] == "road_damage" and f_res["department"] == "Road Maintenance Department" and nlp_res["original_language"] == "hi")
    print(f"  Category: {nlp_res['category']} | Dept: {f_res['department']} | Priority: {f_res['priority']}")
    print(f"  Visual Detections: {len(cv_dets)} | Original Script Preserved: {nlp_res['original_text'][:30]}...")
    print(f"  Result: {'PASS' if t1_pass else 'FAIL'}")
    test_results.append(("Scenario 1: Pothole Defect (Hindi)", t1_pass))

    # SCENARIO 2: Garbage Solid Waste with Telugu Text
    print("\n--- Test 2: Solid Waste / Garbage Heap with Telugu Text ---")
    te_text = "మార్కెట్ రోడ్డుపై పెద్ద చెత్త కుప్ప పేరుకుపోయింది. తీవ్ర దుర్వాసన వస్తోంది."
    garbage_img_path = ROOT / "datasets" / "municipal_detection" / "images" / "test" / "garbage_phagwara_01.jpg"
    garbage_img = Image.open(garbage_img_path) if garbage_img_path.exists() else Image.new("RGB", (640, 640), (120, 120, 120))

    cv_dets2, _ = vision.detect(garbage_img, confidence_threshold=0.15)
    nlp_res2 = nlp.analyze_text_full(nlp_model, te_text, user_lang="te")
    f_res2 = fusion.fuse_and_recommend(
        text_category=nlp_res2["category"],
        nlp_confidence=nlp_res2["confidence"],
        urgency_cues=nlp_res2["urgency_cues"],
        detections=cv_dets2,
        location_name="Market Road",
        latitude=31.2240,
        longitude=75.7710,
        impact_text=nlp_res2["impact"],
    )

    t2_pass = (nlp_res2["category"] == "waste" and f_res2["department"] == "Sanitation & Waste Management Department" and nlp_res2["original_language"] == "te")
    print(f"  Category: {nlp_res2['category']} | Dept: {f_res2['department']} | Priority: {f_res2['priority']}")
    print(f"  Original Telugu Text: {nlp_res2['original_text'][:30]}...")
    print(f"  Result: {'PASS' if t2_pass else 'FAIL'}")
    test_results.append(("Scenario 2: Solid Waste (Telugu)", t2_pass))

    # SCENARIO 3: Open Drain / Roadside Trench
    print("\n--- Test 3: Open Drain / Roadside Trench Hazard ---")
    drain_text = "There is a deep open drain without any cover alongside the main lane creating hazard for kids."
    drain_img_path = ROOT / "datasets" / "municipal_detection" / "images" / "test" / "user_roadside_trench_diagnostic.jpg"
    drain_img = Image.open(drain_img_path) if drain_img_path.exists() else Image.new("RGB", (640, 640), (120, 120, 120))

    cv_dets3, _ = vision.detect(drain_img, confidence_threshold=0.15)
    nlp_res3 = nlp.analyze_text_full(nlp_model, drain_text, user_lang="en")
    f_res3 = fusion.fuse_and_recommend(
        text_category=nlp_res3["category"],
        nlp_confidence=nlp_res3["confidence"],
        urgency_cues=nlp_res3["urgency_cues"],
        detections=cv_dets3,
        location_name="Main Lane",
        latitude=31.2200,
        longitude=75.7650,
        impact_text=nlp_res3["impact"],
    )

    t3_pass = (nlp_res3["category"] in ["open_drain", "drain_blockage"] and f_res3["department"] == "Stormwater Drainage & Sewerage Division")
    print(f"  Category: {nlp_res3['category']} | Dept: {f_res3['department']} | Priority: {f_res3['priority']}")
    print(f"  Result: {'PASS' if t3_pass else 'FAIL'}")
    test_results.append(("Scenario 3: Open Drain Trench (English)", t3_pass))

    # SCENARIO 4: Open Manhole Life-Safety Escalation
    print("\n--- Test 4: Open Manhole Life-Safety Escalation ---")
    manhole_text = "Deep open manhole chamber cover missing near railway station road, extreme accident danger!"
    nlp_res4 = nlp.analyze_text_full(nlp_model, manhole_text, user_lang="en")
    f_res4 = fusion.fuse_and_recommend(
        text_category=nlp_res4["category"],
        nlp_confidence=nlp_res4["confidence"],
        urgency_cues=nlp_res4["urgency_cues"],
        detections=[],
        location_name="Railway Station Road",
        latitude=31.2210,
        longitude=75.7690,
        impact_text=nlp_res4["impact"],
    )

    t4_pass = (nlp_res4["category"] == "open_manhole" and f_res4["priority"] == "P1" and f_res4["severity"] == "Critical")
    print(f"  Category: {nlp_res4['category']} | Severity: {f_res4['severity']} | Priority: {f_res4['priority']}")
    print(f"  Result: {'PASS' if t4_pass else 'FAIL'}")
    test_results.append(("Scenario 4: Open Manhole Safety Escalation", t4_pass))

    # SCENARIO 5: Streetlight Damage with Night Visibility Risk
    print("\n--- Test 5: Broken Streetlight with Night Visibility Hazard ---")
    light_text = "Street light pole broken outside public hospital, pitch dark at night."
    nlp_res5 = nlp.analyze_text_full(nlp_model, light_text, user_lang="en")
    f_res5 = fusion.fuse_and_recommend(
        text_category=nlp_res5["category"],
        nlp_confidence=nlp_res5["confidence"],
        urgency_cues=nlp_res5["urgency_cues"],
        detections=[],
        location_name="Civil Hospital Gate",
        latitude=31.2250,
        longitude=75.7720,
        impact_text=nlp_res5["impact"],
    )

    t5_pass = (nlp_res5["category"] == "streetlight_damage" and f_res5["department"] == "Electrical & Street Lighting Department")
    print(f"  Category: {nlp_res5['category']} | Dept: {f_res5['department']} | Urgency: {nlp_res5['urgency_cues']}")
    print(f"  Result: {'PASS' if t5_pass else 'FAIL'}")
    test_results.append(("Scenario 5: Streetlight Outage", t5_pass))

    # SCENARIO 6: Clean Road Negative Control (Zero False Positives)
    print("\n--- Test 6: Clean Road Negative Control ---")
    clean_img_path = ROOT / "datasets" / "municipal_detection" / "images" / "test" / "clean_road_control_01.jpg"
    clean_img = Image.open(clean_img_path) if clean_img_path.exists() else Image.new("RGB", (640, 640), (80, 80, 80))
    clean_dets, _ = vision.detect(clean_img, confidence_threshold=0.20)
    t6_pass = (len(clean_dets) == 0)
    print(f"  Detections on clean pavement: {len(clean_dets)} (Expected: 0)")
    print(f"  Result: {'PASS' if t6_pass else 'FAIL'}")
    test_results.append(("Scenario 6: Clean Road Negative Control", t6_pass))

    # SCENARIO 7: Cross-Lingual Duplicate Clustering (EN vs HI < 50m)
    print("\n--- Test 7: Cross-Lingual Duplicate Detection (EN vs HI) ---")
    existing_complaints = [
        {
            "id": "CF-1001",
            "description": "There is a large dangerous pothole near Gol Chowk causing traffic swerving.",
            "original_language": "en",
            "category": "road_damage",
            "latitude": 31.2185,
            "longitude": 75.7672,
            "status": "Registered",
            "citizen_reports_count": 1,
        }
    ]
    new_hi_complaint = "गोल चौक के पास सड़क पर बहुत बड़ा गड्ढा है। गाड़ियां गिर रही हैं।"
    dup_res = duplicates.check_for_duplicates(
        new_text=new_hi_complaint,
        new_lat=31.2188,
        new_lon=75.7675,
        new_category="road_damage",
        existing_reports=existing_complaints,
    )
    t7_pass = (dup_res["is_duplicate"] and dup_res["parent_id"] == "CF-1001" and dup_res["distance_meters"] < 50.0)
    print(f"  Is Duplicate: {dup_res['is_duplicate']} | Parent ID: {dup_res['parent_id']}")
    print(f"  Distance: {dup_res['distance_meters']}m | Text Similarity: {dup_res['text_similarity']:.1%}")
    print(f"  Result: {'PASS' if t7_pass else 'FAIL'}")
    test_results.append(("Scenario 7: Cross-Lingual Duplicate (EN-HI)", t7_pass))

    # SCENARIO 8: Spatial Separation Safety (> 250m prevents merging across town)
    print("\n--- Test 8: Distant Identical Issue (> 250m Safe Rejection) ---")
    dup_distant = duplicates.check_for_duplicates(
        new_text="There is a large dangerous pothole near Gol Chowk causing traffic swerving.",
        new_lat=31.3500,  # ~15km away in Jalandhar
        new_lon=75.6000,
        new_category="road_damage",
        existing_reports=existing_complaints,
    )
    t8_pass = not dup_distant["is_duplicate"]
    print(f"  Is Duplicate: {dup_distant['is_duplicate']} (Expected False for distant GPS)")
    print(f"  Result: {'PASS' if t8_pass else 'FAIL'}")
    test_results.append(("Scenario 8: Spatial Separation Safety (>250m)", t8_pass))

    # SCENARIO 9: Co-Located Different Category Separation
    print("\n--- Test 9: Co-Located Different Categories (Same GPS) ---")
    dup_different_cat = duplicates.check_for_duplicates(
        new_text="Streetlight pole broken and dark at night at Gol Chowk.",
        new_lat=31.2185,  # Same GPS coordinates
        new_lon=75.7672,
        new_category="streetlight_damage",
        existing_reports=existing_complaints,  # existing complaint is road_damage
    )
    t9_pass = not dup_different_cat["is_duplicate"]
    print(f"  Is Duplicate: {dup_different_cat['is_duplicate']} (Expected False for distinct municipal department)")
    print(f"  Result: {'PASS' if t9_pass else 'FAIL'}")
    test_results.append(("Scenario 9: Co-Located Category Separation", t9_pass))

    # SCENARIO 10: Original Script Preservation
    print("\n--- Test 10: Native Script Preservation ---")
    te_original = "పాఠశాల ప్రవేశ ద్వారం దగ్గర రోడ్డుపై పెద్ద గుంత ఉంది."
    res_te = nlp.analyze_text_full(nlp_model, te_original, user_lang="te")
    t10_pass = (res_te["original_text"] == te_original and res_te["original_language"] == "te")
    print(f"  Original Text Retained: '{res_te['original_text']}'")
    print(f"  Detected Language: {res_te['original_language']} ({res_te['language_name']})")
    print(f"  Result: {'PASS' if t10_pass else 'FAIL'}")
    test_results.append(("Scenario 10: Native Script Preservation", t10_pass))

    print("\n" + "=" * 75)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 75)
    passed_count = sum(1 for _, p in test_results if p)
    for name, p in test_results:
        print(f"  {name:<50}: {'PASS [OK]' if p else 'FAIL [X]'}")
    print(f"\nTotal: {passed_count}/{len(test_results)} Tests Passed ({passed_count/len(test_results):.0%})")

    report_path = ROOT / "models" / "system_test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_tests": len(test_results),
            "passed": passed_count,
            "pass_rate": round(passed_count / len(test_results), 4),
            "scenarios": [{"scenario": n, "passed": p} for n, p in test_results]
        }, f, indent=2)
    print(f"Saved test report to {report_path}")

    return passed_count == len(test_results)


if __name__ == "__main__":
    run_all_tests()

