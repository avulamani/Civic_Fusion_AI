"""Geospatial and LaBSE-based semantic duplicate complaint detection engine for CivicFusion AI.

Identifies when multiple citizens report the same civic issue within close
geographic proximity (e.g., within 250m) and with cross-lingual semantic similarity.
Supports English, Hindi (Devanagari), and Telugu (Telugu script).
Prevents duplicate work orders, groups related complaints, and escalates
priority when multiple citizens report the same hazard.
"""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from typing import Any

from semantic_engine import SemanticIntelligenceEngine, haversine_distance_meters


def compute_text_similarity(text1: str, text2: str) -> tuple[float, str]:
    """Compute cross-lingual semantic similarity between two complaint descriptions.
    
    Returns:
        (similarity_score: float, engine_name: str)
    """
    engine = SemanticIntelligenceEngine.get_instance()
    return engine.compute_similarity(text1, text2)


def check_for_duplicates(
    new_text: str,
    new_lat: float,
    new_lon: float,
    new_category: str,
    existing_reports: list[dict[str, Any]],
    distance_threshold_m: float = 250.0,
    similarity_threshold: float = 0.58,
) -> dict[str, Any]:
    """Scan existing reports to determine if a newly submitted complaint is a duplicate.
    
    Rules for valid duplicate identification:
    1. Must match municipal category (e.g. road_damage == road_damage).
       A streetlight outage and a pothole at the same spot are distinct tickets.
    2. Must be within spatial proximity (<= 250 meters).
    3. If very close (<= 50 meters), automatically grouped under same category.
    4. If between 50m and 250m, requires cross-lingual semantic similarity >= threshold.
    """
    best_match = None
    highest_score = -1.0
    matched_engine = "None"

    if new_lat == 0.0 and new_lon == 0.0:
        return {
            "is_duplicate": False,
            "parent_id": None,
            "distance_meters": None,
            "text_similarity": 0.0,
            "matched_by": "None",
            "action": "register_new",
            "explanation": "No GPS coordinates provided; treated as standalone incident.",
        }

    # Normalize incoming category
    norm_new_cat = str(new_category).strip().lower().replace(" ", "_")
    category_synonyms = {
        "damaged_road": "road_damage",
        "pothole": "road_damage",
        "waste_pile": "waste",
        "garbage": "waste",
        "blocked_drain": "drain_blockage",
        "drain": "open_drain",
    }
    norm_new_cat = category_synonyms.get(norm_new_cat, norm_new_cat)

    for report in existing_reports:
        # Only compare against active, unresolved complaints
        if report.get("status") in ["Resolved", "Closed"]:
            continue

        report_cat = str(report.get("category_raw") or report.get("category", "")).strip().lower().replace(" ", "_")
        report_cat = category_synonyms.get(report_cat, report_cat)
        if report_cat != norm_new_cat:
            continue

        r_lat = float(report.get("latitude", 0.0))
        r_lon = float(report.get("longitude", 0.0))
        if r_lat == 0.0 and r_lon == 0.0:
            continue

        distance = haversine_distance_meters(new_lat, new_lon, r_lat, r_lon)
        if distance > distance_threshold_m:
            continue

        # Compute cross-lingual semantic similarity (using original text directly)
        existing_text = report.get("description", "") or report.get("original_text", "")
        similarity, engine_name = compute_text_similarity(new_text, existing_text)
        matched_engine = engine_name

        # Proximity score (1.0 at 0m down to 0.0 at distance_threshold_m)
        proximity_factor = max(0.0, 1.0 - (distance / distance_threshold_m))
        combined_score = (0.4 * proximity_factor) + (0.6 * similarity)

        is_close_duplicate = distance <= 50.0
        is_semantic_duplicate = (distance <= distance_threshold_m and similarity >= similarity_threshold)

        if (is_close_duplicate or is_semantic_duplicate) and combined_score > highest_score:
            highest_score = combined_score
            best_match = {
                "parent_id": report.get("id"),
                "parent_description": existing_text,
                "distance_meters": round(distance, 1),
                "text_similarity": round(similarity, 3),
                "score": round(combined_score, 3),
                "existing_cluster_count": report.get("citizen_reports_count", 1),
                "engine": engine_name,
            }

    if best_match:
        return {
            "is_duplicate": True,
            "parent_id": best_match["parent_id"],
            "distance_meters": best_match["distance_meters"],
            "text_similarity": best_match["text_similarity"],
            "match_details": best_match,
            "engine": best_match["engine"],
            "action": "cluster_and_escalate",
            "message": (
                f"Duplicate detected via {best_match['engine']}: Matches active ticket {best_match['parent_id']} "
                f"({best_match['distance_meters']}m away, {best_match['text_similarity']:.0%} semantic similarity). "
                f"Clustered into parent ticket; citizen count incremented."
            ),
        }

    return {
        "is_duplicate": False,
        "parent_id": None,
        "distance_meters": None,
        "text_similarity": 0.0,
        "match_details": None,
        "engine": matched_engine,
        "action": "create_new",
        "message": "Unique issue: No matching active complaints found nearby.",
    }
