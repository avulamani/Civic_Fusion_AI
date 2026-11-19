"""Explainable multimodal fusion, priority engine, and structured ticket generator.

Combines Computer Vision (YOLO), Natural Language Processing (text/speech),
Geographic Location, and Duplicate Detection to generate actionable
Municipal Work Orders.

ARCHITECTURAL PRINCIPLE:
The citizen's description and NLP interpretation serve as the primary source of truth
for final complaint classification and departmental dispatch. Computer Vision detections
provide supplementary visual evidence and bounding-box defect localization.
Mandatory pre-registration human verification is eliminated in favor of immediate
automated ticket registration.
"""

from typing import Any


DEPARTMENTS = {
    "road_damage": "Road Maintenance Department",
    "streetlight_damage": "Electrical & Street Lighting Department",
    "waste": "Sanitation & Waste Management Department",
    "open_manhole": "Water Supply & Sewerage Board",
    "water_leakage": "Water Supply & Sewerage Board",
    "damaged_sign": "Traffic & Road Signage Division",
    "drain_blockage": "Stormwater Drainage & Sewerage Division",
    "traffic_signal_damage": "Traffic Police & Infrastructure Wing",
    "fallen_tree": "Horticulture & Emergency Clearing Department",
    "waterlogging": "Flood Control & Stormwater Management",
    "open_drain": "Stormwater Drainage & Sewerage Division",
}

DEPARTMENT_SECTORS = {
    "road_damage": "Roads & Pavement",
    "streetlight_damage": "Electrical Infrastructure",
    "waste": "Sanitation & Public Health",
    "open_manhole": "Public Safety & Sewerage",
    "water_leakage": "Water Infrastructure",
    "damaged_sign": "Traffic Safety",
    "drain_blockage": "Drainage Infrastructure",
    "traffic_signal_damage": "Traffic Control",
    "fallen_tree": "Urban Forestry & Obstruction",
    "waterlogging": "Drainage & Flood Management",
    "open_drain": "Sewerage & Stormwater Safety",
}

PROBLEM_TITLES = {
    "road_damage": "Pothole / Road Surface Damage",
    "streetlight_damage": "Broken / Inoperative Streetlight",
    "waste": "Garbage Accumulation / Overflowing Bin",
    "open_manhole": "Hazardous Open Manhole / Missing Grate",
    "water_leakage": "Burst Pipe / Street Water Leakage",
    "damaged_sign": "Damaged / Missing Traffic Signboard",
    "drain_blockage": "Blocked Storm Drain / Rubbish Jam",
    "traffic_signal_damage": "Damaged / Dark Traffic Signal",
    "fallen_tree": "Fallen Tree / Branch Roadway Hazard",
    "waterlogging": "Severe Road Waterlogging",
    "open_drain": "Open Drain / Roadside Trench Hazard",
}

CV_TO_CATEGORY = {
    "pothole": "road_damage",
    "road_damage": "road_damage",
    "road_crack": "road_damage",
    "waste_pile": "waste",
    "garbage": "waste",
    "waste": "waste",
    "open_manhole": "open_manhole",
    "open_drain": "open_drain",
    "streetlight_damage": "streetlight_damage",
    "water_leakage": "water_leakage",
    "damaged_sign": "damaged_sign",
    "traffic_signal": "traffic_signal_damage",
    "traffic_signal_damage": "traffic_signal_damage",
    "fallen_tree": "fallen_tree",
    "tree": "fallen_tree",
    "drain_blockage": "drain_blockage",
    "blocked_drain": "drain_blockage",
    "waterlogging": "waterlogging",
}

DISPLAY_NAMES = {
    "road_damage": "Road Damage",
    "streetlight_damage": "Streetlight Damage",
    "waste": "Garbage / Waste",
    "open_manhole": "Open Manhole",
    "open_drain": "Open Drain",
    "water_leakage": "Water Leakage",
    "damaged_sign": "Damaged Sign",
    "drain_blockage": "Blocked Drain",
    "traffic_signal_damage": "Damaged Traffic Signal",
    "fallen_tree": "Fallen Tree / Branch",
    "waterlogging": "Waterlogging",
}


def build_complaint_card(
    category_name: str,
    problem_title: str,
    severity: str,
    priority: str,
    location_name: str,
    confidence_pct: float,
    summary_text: str,
) -> str:
    """Generate the structured municipal complaint card in ASCII format."""
    lines = [
        "╔══════════════════════════════════════════════════════╗",
        "║              MUNICIPAL COMPLAINT TICKET              ║",
        "╠══════════════════════════════════════════════════════╣",
        f"║ Category: {category_name:<42} ║",
        f"║ Problem:  {problem_title:<42} ║",
        f"║ Severity: {severity:<42} ║",
        f"║ Priority: {priority:<42} ║",
        f"║ Location: {location_name:<42} ║",
        "║                                                      ║",
        f"║ AI Confidence: {confidence_pct:>3.0f}%                                ║",
        "║                                                      ║",
        "║ Description:                                         ║",
    ]
    # Word-wrap summary to fit within box width of ~50 chars
    words = summary_text.split()
    current_line = "║ "
    for word in words:
        if len(current_line) + len(word) + 1 <= 53:
            current_line += word + " "
        else:
            lines.append(f"{current_line:<54} ║")
            current_line = "║ " + word + " "
    if current_line.strip() != "║":
        lines.append(f"{current_line:<54} ║")
    lines.append("╚══════════════════════════════════════════════════════╝")
    return "\n".join(lines)


def fuse_and_recommend(
    text_category: str,
    nlp_confidence: float,
    urgency_cues: list[str],
    detections: list[dict[str, Any]],
    location_name: str,
    latitude: float,
    longitude: float,
    impact_text: str = "",
    duplicate_info: dict[str, Any] | None = None,
    secondary_categories: list[str] | None = None,
    resolution_note: str = "",
    citizen_text: str = "",
) -> dict[str, Any]:
    """Multimodal decision engine: merges Vision, NLP, GPS, and Duplicates.
    
    PRIMARY DECISION AUTHORITY:
    The final complaint classification and municipal routing are determined by the citizen's
    description as interpreted by NLP. Computer Vision acts as supplementary visual evidence
    and defect localization without overriding the citizen's ground-truth report.
    """
    final_category = text_category if text_category in DEPARTMENTS else "road_damage"
    assigned_department = DEPARTMENTS.get(final_category, "Municipal Review Desk")
    display_category = DISPLAY_NAMES.get(final_category, final_category.replace("_", " ").title())
    problem_name = PROBLEM_TITLES.get(final_category, final_category.replace("_", " ").title())
    sector_name = DEPARTMENT_SECTORS.get(final_category, "Municipal Infrastructure")
    loc_display = location_name.strip() if location_name and location_name.strip() else (f"GPS ({latitude:.4f}, {longitude:.4f})" if (latitude != 0.0 or longitude != 0.0) else "Incident Location")

    # Computer Vision visual detection processing
    visual_categories = sorted({
        CV_TO_CATEGORY.get(str(item["class"]).lower().replace(" ", "_"))
        for item in detections
        if CV_TO_CATEGORY.get(str(item["class"]).lower().replace(" ", "_"))
    })
    best_cv_conf = max([d["confidence"] for d in detections], default=0.0)
    best_cv_class = detections[0]["class"] if detections else "None"

    # Modality agreement check
    agreement = (final_category in visual_categories) if visual_categories else None

    # Calibrate composite confidence (NLP is the authority on category)
    if detections and nlp_confidence > 0:
        if agreement:
            composite_confidence = 0.5 * best_cv_conf + 0.5 * nlp_confidence + 0.05
        else:
            # Citizen text remains authority; slight confidence calibration without invalidating complaint
            composite_confidence = max(0.75, nlp_confidence)
    elif detections:
        composite_confidence = best_cv_conf
    else:
        composite_confidence = nlp_confidence
    composite_confidence = min(0.99, max(0.70, composite_confidence))

    # Base severity and priority decision using final NLP category and citizen urgency cues
    reasons: list[str] = []
    
    if final_category == "open_manhole" or "Safety risk" in urgency_cues:
        severity, priority = "Critical", "P1"
        reasons.append("Safety risk or hazardous open manhole defect identified from citizen report.")
    elif (
        len(urgency_cues) >= 2
        or "Urgent obstruction" in urgency_cues
        or final_category in ["open_drain", "drain_blockage"]
        or (final_category == "road_damage" and secondary_categories and "waterlogging" in secondary_categories)
    ):
        severity, priority = "High", "P2"
        reasons.append("High-priority hazard (water-filled defect, drainage obstruction, or multi-factor risk) impacting commuter safety.")
    elif urgency_cues or final_category == "road_damage":
        severity, priority = "Medium", "P3"
        reasons.append("Road surface defect or active urgency cues recorded in complaint report.")
    else:
        severity, priority = "Low", "P4"
        reasons.append("Standard civic maintenance report; no critical safety risk detected.")

    # Visual evidence documentation
    if detections:
        reasons.append(f"Visual model detected {len(detections)} defect bounding box(es).")
    
    if agreement is True:
        reasons.append(f"Computer Vision corroborated the reported complaint class ({best_cv_class} at {best_cv_conf:.0%}).")
        decision_reason = (
            f"Final category '{display_category}' confirmed from citizen description via NLP and "
            f"corroborated by YOLO visual object detection ({best_cv_conf:.0%})."
        )
        visual_evidence_summary = f"Corroborated by YOLO Vision: Detected {best_cv_class} ({best_cv_conf:.0%})"
    elif agreement is False:
        cv_names = ", ".join(DISPLAY_NAMES.get(v, v.replace("_", " ").title()) for v in visual_categories) or "Other defect"
        reasons.append(
            f"Visual model detected '{cv_names}' ({best_cv_conf:.0%}). "
            f"Citizen description explicitly specifies '{display_category}', which governs final registration."
        )
        decision_reason = (
            f"Final category '{display_category}' determined from citizen description via NLP. "
            f"Visual model's prediction ('{cv_names}' {best_cv_conf:.0%}) is retained as supplementary visual evidence "
            f"and does not override the citizen's reported context."
        )
        visual_evidence_summary = f"Supplementary Visual Evidence: YOLO detected {cv_names} ({best_cv_conf:.0%})"
    else:
        reasons.append("Visual confirmation pending (no image uploaded or defect below detection threshold).")
        decision_reason = f"Final category '{display_category}' determined from citizen description via NLP."
        visual_evidence_summary = "Visual evidence pending / no objects exceeded detection threshold."

    # Priority escalation if duplicate reports exist from other citizens
    is_duplicate = duplicate_info and duplicate_info.get("is_duplicate", False)
    if is_duplicate:
        reasons.append(
            f"Clustered with existing report {duplicate_info.get('parent_id')}; "
            "multiple citizen reports received for this location."
        )
        if priority in ["P3", "P4"]:
            priority = "P2"
            severity = "High"
            reasons.append("Escalated priority to P2 due to multiple citizen reports.")

    # Ticket registration status: Automatic Registration (Never blocked by human review)
    registration_status = "Registered"
    needs_review = False

    summary = (
        f"{problem_name} reported at {loc_display}. "
        f"{impact_text}. "
        f"AI assessed priority as {priority} ({severity} severity). "
        f"Status: Automatically Registered & Dispatched to {assigned_department}."
    )

    card_str = build_complaint_card(
        category_name=sector_name,
        problem_title=problem_name,
        severity=severity.upper(),
        priority=priority,
        location_name=loc_display,
        confidence_pct=composite_confidence * 100,
        summary_text=summary,
    )

    return {
        "category_raw": final_category,
        "category_display": display_category,
        "sector": sector_name,
        "problem_title": problem_name,
        "department": assigned_department,
        "severity": severity,
        "priority": priority,
        "composite_confidence": composite_confidence,
        "nlp_confidence": nlp_confidence,
        "visual_categories": visual_categories,
        "best_cv_conf": best_cv_conf,
        "agreement": agreement,
        "needs_review": needs_review,
        "registration_status": registration_status,
        "decision_reason": decision_reason,
        "visual_evidence_summary": visual_evidence_summary,
        "secondary_categories": secondary_categories or [],
        "resolution_note": resolution_note,
        "reasons": reasons,
        "summary": summary,
        "ascii_card": card_str,
        "duplicate_info": duplicate_info,
    }


def recommend(text_category: str, urgency_cues: list[str], detections: list[dict[str, Any]]) -> dict[str, Any]:
    """Backward compatibility wrapper for legacy callers."""
    return fuse_and_recommend(
        text_category=text_category,
        nlp_confidence=0.85,
        urgency_cues=urgency_cues,
        detections=detections,
        location_name="Reported Location",
        latitude=31.6340,
        longitude=74.8723,
    )
