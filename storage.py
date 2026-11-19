"""Persistence and database storage manager for CivicFusion AI.

Handles complaint data storage across both local execution and cloud deployments.
Uses JSON-backed disk storage with automatic fallback to memory.
Supports permanent persistence across app restarts.
"""

from datetime import date, timedelta
import json
from pathlib import Path
from typing import Any


DB_FILE = Path(__file__).parent / "data" / "complaints_store.json"
MISMATCH_FEEDBACK_FILE = Path(__file__).parent / "data" / "cv_nlp_mismatches.json"

__all__ = [
    "load_all_complaints",
    "save_all_complaints",
    "insert_complaint",
    "delete_complaint",
    "update_ticket_status",
    "increment_cluster_count",
    "get_default_seed_complaints",
    "log_cv_nlp_mismatch",
    "load_all_mismatches",
]


def get_default_seed_complaints() -> list[dict[str, Any]]:
    """Return default complaints store initialized from tested complaints (CF-1052 to CF-1058)."""
    today = date.today()
    return [
        {
            "id": "CF-1058",
            "city": "City Zone",
            "description": "there is garbage left here, this should be cleaned immediately.",
            "original_language": "en",
            "language_name": "English",
            "translated_description_en": "there is garbage left here, this should be cleaned immediately.",
            "category_raw": "waste",
            "category_display": "Garbage / Waste",
            "secondary_categories": [],
            "decision_reason": "Final category 'Garbage / Waste' confirmed from citizen description via NLP and corroborated by YOLO visual object detection (78%).",
            "visual_evidence_summary": "Corroborated by YOLO Vision: Detected garbage (78%)",
            "problem_title": "Garbage Accumulation / Overflowing Bin",
            "department": "Sanitation & Waste Management Department",
            "location_name": "GPS (31.3000, 75.6998)",
            "latitude": 31.3,
            "longitude": 75.69977,
            "severity": "Low",
            "priority": "P4",
            "status": "Registered",
            "citizen_reports_count": 1,
            "ai_confidence": 0.93,
            "needs_review": False,
            "reported_date": today.isoformat(),
            "photo_path": "data/uploads/CF-1058.jpg",
            "annotated_photo_path": "data/uploads/CF-1058_annotated.jpg",
            "detections": [{"class_name": "garbage", "category": "waste", "confidence": 0.7785, "confidence_percent": 77.8, "bbox": [147.5, 722.2, 737.1, 1191.7]}, {"class_name": "garbage", "category": "waste", "confidence": 0.6762, "confidence_percent": 67.6, "bbox": [199.6, 511.9, 797.1, 967.8]}],
            "model_version": "YOLO11n-municipal_detection-v1.0",
        },
        {
            "id": "CF-1057",
            "city": "City Zone",
            "description": "there is so much garbage on the road as well as beside the road near the railway station. People are facing difficulty over passing through it, please clean it immediately.",
            "original_language": "en",
            "language_name": "English",
            "translated_description_en": "there is so much garbage on the road as well as beside the road near the railway station. People are facing difficulty over passing through it, please clean it immediately.",
            "category_raw": "waste",
            "category_display": "Garbage / Waste",
            "secondary_categories": [],
            "decision_reason": "Final category 'Garbage / Waste' confirmed from citizen description via NLP and corroborated by YOLO visual object detection (72%).",
            "visual_evidence_summary": "Corroborated by YOLO Vision: Detected garbage (72%)",
            "problem_title": "Garbage Accumulation / Overflowing Bin",
            "department": "Sanitation & Waste Management Department",
            "location_name": "railway station",
            "latitude": 31.25,
            "longitude": 75.8,
            "severity": "Low",
            "priority": "P4",
            "status": "Registered",
            "citizen_reports_count": 1,
            "ai_confidence": 0.89,
            "needs_review": False,
            "reported_date": today.isoformat(),
            "photo_path": "data/uploads/CF-1057.jpg",
            "annotated_photo_path": "data/uploads/CF-1057_annotated.jpg",
            "detections": [{"class_name": "garbage", "category": "waste", "confidence": 0.7213, "confidence_percent": 72.1, "bbox": [124.9, 210.0, 700.4, 448.4]}],
            "model_version": "YOLO11n-municipal_detection-v1.0",
        },
        {
            "id": "CF-1056",
            "city": "City Zone",
            "description": "the garbage should be removed immediately near the bakery, they are in excessive amount, bad smell is coming from those packages",
            "original_language": "en",
            "language_name": "English",
            "translated_description_en": "the garbage should be removed immediately near the bakery, they are in excessive amount, bad smell is coming from those packages",
            "category_raw": "waste",
            "category_display": "Garbage / Waste",
            "secondary_categories": [],
            "decision_reason": "Final category 'Garbage / Waste' confirmed from citizen description via NLP and corroborated by YOLO visual object detection (86%).",
            "visual_evidence_summary": "Corroborated by YOLO Vision: Detected garbage (86%)",
            "problem_title": "Garbage Accumulation / Overflowing Bin",
            "department": "Sanitation & Waste Management Department",
            "location_name": "bakery",
            "latitude": 31.33,
            "longitude": 75.77,
            "severity": "Low",
            "priority": "P4",
            "status": "Registered",
            "citizen_reports_count": 1,
            "ai_confidence": 0.95,
            "needs_review": False,
            "reported_date": today.isoformat(),
            "photo_path": "data/uploads/CF-1056.jpg",
            "annotated_photo_path": "data/uploads/CF-1056_annotated.jpg",
            "detections": [{"class_name": "garbage", "category": "waste", "confidence": 0.8599, "confidence_percent": 86.0, "bbox": [37.8, 59.9, 523.3, 269.0]}],
            "model_version": "YOLO11n-municipal_detection-v1.0",
        },
        {
            "id": "CF-1054",
            "city": "City Zone",
            "description": "there is a pot hole on the road near the railway station, vehicles facing difficulty over passing through it.",
            "original_language": "en",
            "language_name": "English",
            "translated_description_en": "there is a pot hole on the road near the railway station, vehicles facing difficulty over passing through it.",
            "category_raw": "road_damage",
            "category_display": "Road Damage",
            "secondary_categories": [],
            "decision_reason": "Final category 'Road Damage' determined from citizen description via NLP.",
            "visual_evidence_summary": "Visual evidence pending / no objects exceeded detection threshold.",
            "problem_title": "Pothole / Road Surface Damage",
            "department": "Road Maintenance Department",
            "location_name": "Device GPS (31.2511, 75.6998)",
            "latitude": 31.29,
            "longitude": 75.64,
            "severity": "Medium",
            "priority": "P3",
            "status": "Registered",
            "citizen_reports_count": 1,
            "ai_confidence": 0.98,
            "needs_review": False,
            "reported_date": today.isoformat(),
            "photo_path": "data/uploads/CF-1054.jpg",
            "annotated_photo_path": "data/uploads/CF-1054_annotated.jpg",
            "detections": [{"class_name": "damaged_road", "category": "road_damage", "confidence": 0.7531, "confidence_percent": 75.3, "bbox": [108.3, 362.3, 1530.9, 756.8]}, {"class_name": "damaged_road", "category": "road_damage", "confidence": 0.1993, "confidence_percent": 19.9, "bbox": [183.5, 455.2, 1066.6, 828.9]}, {"class_name": "damaged_road", "category": "road_damage", "confidence": 0.1964, "confidence_percent": 19.6, "bbox": [712.9, 480.8, 1404.3, 827.3]}],
            "model_version": "YOLO11n-municipal_detection-v1.0",
        },
        {
            "id": "CF-1053",
            "city": "City Zone",
            "description": "some part of the road is damaged near the school gate",
            "original_language": "en",
            "language_name": "English",
            "translated_description_en": "some part of the road is damaged near the school gate",
            "category_raw": "road_damage",
            "category_display": "Road Damage",
            "secondary_categories": [],
            "decision_reason": "Final category 'Road Damage' confirmed from citizen description via NLP and corroborated by YOLO visual object detection (44%).",
            "visual_evidence_summary": "Corroborated by YOLO Vision: Detected pothole (44%)",
            "problem_title": "Pothole / Road Surface Damage",
            "department": "Road Maintenance Department",
            "location_name": "Near School Gate",
            "latitude": 31.19,
            "longitude": 75.74,
            "severity": "Medium",
            "priority": "P3",
            "status": "Registered",
            "citizen_reports_count": 1,
            "ai_confidence": 0.74,
            "needs_review": False,
            "reported_date": today.isoformat(),
            "photo_path": "data/uploads/CF-1053.jpg",
            "annotated_photo_path": "data/uploads/CF-1053_annotated.jpg",
            "detections": [{"class_name": "pothole", "category": "road_damage", "confidence": 0.438, "confidence_percent": 43.8, "bbox": [158.5, 152.9, 680.1, 469.3]}, {"class_name": "pothole", "category": "road_damage", "confidence": 0.3004, "confidence_percent": 30.0, "bbox": [157.6, 40.8, 689.1, 312.5]}, {"class_name": "pothole", "category": "road_damage", "confidence": 0.2741, "confidence_percent": 27.4, "bbox": [161.1, 255.7, 608.4, 578.0]}, {"class_name": "pothole", "category": "road_damage", "confidence": 0.2654, "confidence_percent": 26.5, "bbox": [286.9, 17.7, 825.0, 425.0]}, {"class_name": "pothole", "category": "road_damage", "confidence": 0.2103, "confidence_percent": 21.0, "bbox": [178.5, 423.3, 609.0, 679.2]}],
            "model_version": "YOLO11n-municipal_detection-v1.0",
        },
        {
            "id": "CF-1052",
            "city": "City Zone",
            "description": "There are multiple deep potholes near the bakery.",
            "original_language": "en",
            "language_name": "English",
            "translated_description_en": "There are multiple deep potholes near the bakery.",
            "category_raw": "road_damage",
            "category_display": "Road Damage",
            "secondary_categories": [],
            "decision_reason": "Final category 'Road Damage' confirmed from citizen description via NLP and corroborated by YOLO visual object detection (62%).",
            "visual_evidence_summary": "Corroborated by YOLO Vision: Detected damaged_road (62%)",
            "problem_title": "Pothole / Road Surface Damage",
            "department": "Road Maintenance Department",
            "location_name": "Device GPS (31.2510, 75.6998)",
            "latitude": 31.29,
            "longitude": 75.74,
            "severity": "Medium",
            "priority": "P3",
            "status": "Registered",
            "citizen_reports_count": 1,
            "ai_confidence": 0.86,
            "needs_review": False,
            "reported_date": today.isoformat(),
            "photo_path": "data/uploads/CF-1052.jpg",
            "annotated_photo_path": "data/uploads/CF-1052_annotated.jpg",
            "detections": [{"class_name": "damaged_road", "category": "road_damage", "confidence": 0.6199, "confidence_percent": 62.0, "bbox": [4.8, 19.2, 573.7, 330.6]}, {"class_name": "pothole", "category": "road_damage", "confidence": 0.2792, "confidence_percent": 27.9, "bbox": [185.2, 204.6, 389.0, 312.2]}, {"class_name": "pothole", "category": "road_damage", "confidence": 0.168, "confidence_percent": 16.8, "bbox": [214.9, 89.3, 391.3, 167.9]}],
            "model_version": "YOLO11n-municipal_detection-v1.0",
        },
    ]


def load_all_complaints() -> list[dict[str, Any]]:
    """Load complaints from persistent JSON storage on disk."""
    if DB_FILE.exists():
        try:
            content = DB_FILE.read_text(encoding="utf-8")
            data = json.loads(content)
            if isinstance(data, list) and len(data) > 0:
                return data
        except Exception:
            pass

    # Initialize default seed data
    defaults = get_default_seed_complaints()
    save_all_complaints(defaults)
    return defaults


def save_all_complaints(complaints: list[dict[str, Any]]) -> None:
    """Save full complaint list to disk and synchronize root complaints.json."""
    try:
        DB_FILE.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(complaints, indent=2, ensure_ascii=False)
        DB_FILE.write_text(content, encoding="utf-8")
        root_file = Path(__file__).parent / "complaints.json"
        root_file.write_text(content, encoding="utf-8")
    except Exception as exc:
        print(f"Warning: Could not save complaints to disk: {exc}")


def insert_complaint(new_complaint: dict[str, Any]) -> list[dict[str, Any]]:
    """Insert a newly analyzed complaint at the top of the database and persist."""
    current = load_all_complaints()
    current.insert(0, new_complaint)
    save_all_complaints(current)
    return current


def delete_complaint(ticket_id: str) -> list[dict[str, Any]]:
    """Delete a complaint by ID from persistent storage."""
    current = load_all_complaints()
    clean_id = str(ticket_id).strip().upper()
    current = [c for c in current if str(c.get("id", "")).strip().upper() != clean_id]
    save_all_complaints(current)
    return current


def update_ticket_status(
    ticket_id: str,
    new_status: str,
    remarks: str | None = None,
    assigned_officer: str | None = None,
    department: str | None = None,
) -> list[dict[str, Any]]:
    """Update status, remarks, assignment, and last updated timestamp of a ticket in persistent storage."""
    current = load_all_complaints()
    for c in current:
        if str(c.get("id", "")).strip().upper() == str(ticket_id).strip().upper():
            c["status"] = new_status
            c["last_updated"] = date.today().isoformat()
            if remarks is not None:
                c["officer_remarks"] = remarks.strip()
            if assigned_officer is not None:
                c["assigned_officer"] = assigned_officer.strip()
            if department is not None:
                c["department"] = department.strip()
            break
    save_all_complaints(current)
    return current


def increment_cluster_count(parent_id: str, priority_escalation: str | None = None) -> list[dict[str, Any]]:
    """Increment citizen count on duplicate cluster and optionally escalate priority."""
    current = load_all_complaints()
    for c in current:
        if c.get("id") == parent_id:
            c["citizen_reports_count"] = c.get("citizen_reports_count", 1) + 1
            if priority_escalation:
                c["priority"] = priority_escalation
                if priority_escalation == "P1":
                    c["severity"] = "Critical"
                elif priority_escalation == "P2":
                    c["severity"] = "High"
            break
    save_all_complaints(current)
    return current


def log_cv_nlp_mismatch(
    ticket_id: str,
    citizen_description: str = "",
    nlp_category: str = "",
    nlp_confidence: float = 0.0,
    cv_categories: list[str] | None = None,
    cv_confidence: float = 0.0,
    final_category: str = "",
    photo_path: str | None = None,
    model_version: str = "",
    decision_reason: str = "",
    original_language: str = "en",
    **kwargs: Any,
) -> None:
    """Record a CV vs NLP disagreement to the audit log for future human re-annotation.
    
    CRITICAL TECHNICAL SAFEGUARD: Does NOT generate pseudo-bounding boxes or trigger
    automatic re-training. Queues the image for certified human annotators to review
    actual defect coordinates before dataset inclusion.
    """
    # Accept parameter aliases for maximum backwards and forwards compatibility
    if not citizen_description:
        citizen_description = kwargs.get("text_content", "") or kwargs.get("description", "")
    if not cv_confidence:
        cv_confidence = float(kwargs.get("best_cv_confidence", 0.0))
    if original_language == "en" and "language" in kwargs:
        original_language = str(kwargs.get("language"))
    if cv_categories is None:
        cv_categories = kwargs.get("cv_categories", [])
    try:
        MISMATCH_FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
        records = []
        if MISMATCH_FEEDBACK_FILE.exists():
            try:
                records = json.loads(MISMATCH_FEEDBACK_FILE.read_text(encoding="utf-8"))
            except Exception:
                records = []
        
        # Avoid duplicate log entries for the same ticket
        if any(r.get("ticket_id") == ticket_id for r in records):
            return

        new_record = {
            "ticket_id": ticket_id,
            "timestamp": date.today().isoformat(),
            "citizen_description": citizen_description,
            "original_language": original_language,
            "nlp_category": nlp_category,
            "nlp_confidence": round(float(nlp_confidence), 3),
            "cv_categories": cv_categories,
            "cv_confidence": round(float(cv_confidence), 3),
            "final_category": final_category,
            "photo_path": photo_path,
            "model_version": model_version,
            "decision_reason": decision_reason,
            "annotation_status": "Queued for Manual Ground-Truth Bounding-Box Annotation",
            "curation_note": "Awaiting manual verification of bounding-box coordinates before dataset ingestion.",
        }
        records.append(new_record)
        MISMATCH_FEEDBACK_FILE.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        print(f"Warning: Could not log CV/NLP mismatch: {exc}")


def load_all_mismatches() -> list[dict[str, Any]]:
    """Load the mismatch feedback log for CV engineering review."""
    if MISMATCH_FEEDBACK_FILE.exists():
        try:
            return json.loads(MISMATCH_FEEDBACK_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


