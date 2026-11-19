"""Complaint details formatting and presentation view for CivicFusion AI.

Provides unified, secure rendering of complaint tickets for both
Citizen and Municipal Officer dashboards.
"""

from datetime import date
from html import escape
from pathlib import Path
import textwrap
from typing import Any

import pandas as pd
from PIL import Image
import streamlit as st

import multilingual
from multilingual import (
    detect_language,
    get_language_display_name,
    LANGUAGE_CHOICES,
    normalize_lang_code,
    translate_text,
)
import vision


def clean_html(html_str: str) -> str:
    """Strip leading and trailing whitespace from each line so Streamlit markdown doesn't treat it as code."""
    return "\n".join(line.strip() for line in html_str.strip().splitlines() if line.strip())


def get_status_theme(status: str) -> dict[str, str]:
    """Return styling colors and icons for complaint lifecycle statuses."""
    themes = {
        "Registered": {
            "color": "#0284c7",
            "bg": "rgba(2, 132, 199, 0.12)",
            "border": "#0284c7",
            "icon": "📋",
            "explanation": "Your complaint has been automatically registered and dispatched to the assigned municipal department.",
        },
        "Open": {
            "color": "#0284c7",
            "bg": "rgba(2, 132, 199, 0.12)",
            "border": "#0284c7",
            "icon": "🔵",
            "explanation": "Your complaint has been received and is queued for municipal inspection.",
        },
        "Pending Review": {
            "color": "#b91c1c",
            "bg": "rgba(220, 38, 38, 0.12)",
            "border": "#dc2626",
            "icon": "🔴",
            "explanation": "The AI flagged a modality mismatch. An officer will manually verify this ticket.",
        },
        "In Progress": {
            "color": "#c2410c",
            "bg": "rgba(234, 88, 12, 0.12)",
            "border": "#ea580c",
            "icon": "🟠",
            "explanation": "A municipal field crew has been dispatched and is actively resolving the issue.",
        },
        "Assigned": {
            "color": "#7e22ce",
            "bg": "rgba(147, 51, 234, 0.12)",
            "border": "#9333ea",
            "icon": "🟣",
            "explanation": "This ticket has been assigned to a department division / supervisor for action.",
        },
        "Resolved": {
            "color": "#047857",
            "bg": "rgba(5, 150, 105, 0.12)",
            "border": "#059669",
            "icon": "🟢",
            "explanation": "This civic issue has been resolved. Thank you for contributing to city upkeep!",
        },
    }
    return themes.get(
        status.strip(),
        {
            "color": "#475569",
            "bg": "rgba(100, 116, 139, 0.12)",
            "border": "#64748b",
            "icon": "⚪",
            "explanation": "Ticket is recorded in municipal database.",
        },
    )


def prepare_complaint_data(complaint: dict[str, Any]) -> dict[str, Any]:
    """Extract, sanitize, and normalize structured complaint attributes."""
    raw_id = str(complaint.get("id", "N/A")).strip()
    reported_date = str(complaint.get("reported_date", "N/A")).strip()
    last_updated = complaint.get("last_updated") or complaint.get("updated_at")
    last_updated_str = str(last_updated).strip() if last_updated else None

    cat = str(
        complaint.get("category_display")
        or complaint.get("category_raw")
        or "General Civic Issue"
    ).strip()
    problem_title = str(complaint.get("problem_title") or cat).strip()
    description = str(complaint.get("description", "")).strip() or "No description provided."
    location_name = str(complaint.get("location_name", "Unspecified Location")).strip()
    city = complaint.get("city") or "City Zone"

    lat = complaint.get("latitude")
    lon = complaint.get("longitude")
    try:
        lat_f = float(lat) if lat is not None else None
        lon_f = float(lon) if lon is not None else None
    except (ValueError, TypeError):
        lat_f, lon_f = None, None

    status = str(complaint.get("status", "Open")).strip()
    department = str(complaint.get("department", "Pending Department Assignment")).strip()
    assigned_officer = str(complaint.get("assigned_officer", "")).strip() or None
    remarks = str(complaint.get("officer_remarks") or complaint.get("remarks") or "").strip() or None

    severity = str(complaint.get("severity", "Medium")).strip()
    priority = str(complaint.get("priority", "P3")).strip()
    try:
        reports_count = int(complaint.get("citizen_reports_count", 1))
    except (ValueError, TypeError):
        reports_count = 1

    photo_path = complaint.get("photo_path")
    if not photo_path:
        check_p = Path("data/uploads") / f"{raw_id}.jpg"
        if check_p.exists():
            photo_path = str(check_p)

    annotated_path = complaint.get("annotated_photo_path")
    if not annotated_path:
        check_ann = Path("data/uploads") / f"{raw_id}_annotated.jpg"
        if check_ann.exists():
            annotated_path = str(check_ann)

    detections = complaint.get("detections")
    if detections is None:
        detections = []

    model_version = complaint.get("model_version", vision.ACTIVE_MODEL_VERSION)

    orig_lang = complaint.get("original_language")
    if not orig_lang:
        orig_lang = detect_language(description)
    lang_name = complaint.get("language_name")
    if not lang_name:
        lang_name = get_language_display_name(orig_lang)
    translated_desc_en = complaint.get("translated_description_en")
    secondary_cats = complaint.get("secondary_categories", [])
    decision_reason = complaint.get("decision_reason")
    visual_evidence_summary = complaint.get("visual_evidence_summary")

    return {
        "id": raw_id,
        "reported_date": reported_date,
        "last_updated": last_updated_str,
        "category": cat,
        "problem_title": problem_title,
        "description": description,
        "original_language": orig_lang,
        "language_name": lang_name,
        "translated_description_en": translated_desc_en,
        "secondary_categories": secondary_cats,
        "decision_reason": decision_reason,
        "visual_evidence_summary": visual_evidence_summary,
        "location_name": location_name,
        "city": city,
        "latitude": lat_f,
        "longitude": lon_f,
        "status": status,
        "department": department,
        "assigned_officer": assigned_officer,
        "officer_remarks": remarks,
        "severity": severity,
        "priority": priority,
        "citizen_reports_count": reports_count,
        "photo_path": photo_path,
        "annotated_photo_path": annotated_path,
        "detections": detections,
        "model_version": model_version,
        "needs_review": bool(complaint.get("needs_review", False)),
    }


def format_complaint_card_html(complaint: dict[str, Any], is_officer: bool = False) -> str:
    """Generate safe, professional card HTML.

    Guarantees:
    - Every dynamic value is HTML-escaped to prevent code injection / XSS.
    - Zero leading line-indentation ensures CommonMark does not convert lines into code blocks.
    - Matches CivicFusion AI's dark theme palette.
    """
    data = prepare_complaint_data(complaint)
    st_theme = get_status_theme(data["status"])

    cid_esc = escape(data["id"])
    reported_esc = escape(data["reported_date"])
    updated_esc = escape(data["last_updated"]) if data["last_updated"] else None
    cat_esc = escape(data["category"])
    problem_esc = escape(data["problem_title"])
    desc_esc = escape(data["description"])
    loc_esc = escape(data["location_name"])
    city_esc = escape(data.get("city") or "City Zone")
    dept_esc = escape(data["department"])
    officer_esc = escape(data["assigned_officer"]) if data["assigned_officer"] else None
    remarks_esc = escape(data["officer_remarks"]) if data["officer_remarks"] else None
    status_esc = escape(data["status"])
    severity_esc = escape(data["severity"])
    priority_esc = escape(data["priority"])

    coords_str = None
    gmaps_link = None
    if data["latitude"] is not None and data["longitude"] is not None:
        coords_str = f"{data['latitude']:.5f}° N, {data['longitude']:.5f}° E"
        gmaps_link = f"https://www.google.com/maps/search/?api=1&query={data['latitude']},{data['longitude']}"

    prio_colors = {
        "P1": "#dc2626",
        "P2": "#ea580c",
        "P3": "#d97706",
        "P4": "#059669",
    }
    prio_color = prio_colors.get(data["priority"], "#64748b")

    # Build card HTML without indentation on ANY line
    card_lines: list[str] = [
        '<div class="civic-complaint-card" style="background:#ffffff;border:1.5px solid rgba(37,99,235,0.22);border-radius:14px;padding:22px 26px;margin-bottom:18px;font-family:system-ui,-apple-system,sans-serif;box-sizing:border-box;box-shadow:0 4px 20px rgba(0,0,0,0.05);">',
        # Header Row
        '<div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;margin-bottom:16px;border-bottom:1px solid rgba(148,163,184,0.22);padding-bottom:14px;">',
        '<div>',
        f'<div style="font-size:1.4rem;font-weight:700;color:#0f172a;letter-spacing:-0.01em;">🎫 Ticket {cid_esc}</div>',
        f'<div style="font-size:0.86rem;color:#475569;margin-top:4px;">📅 Submitted on <b>{reported_esc}</b>'
        + (f' &nbsp;•&nbsp; 🕒 Last updated <b>{updated_esc}</b>' if updated_esc and updated_esc != reported_esc else '')
        + '</div>',
        '</div>',
        f'<div style="background:{st_theme["bg"]};color:{st_theme["color"]};border:1.5px solid {st_theme["border"]};padding:6px 14px;border-radius:20px;font-size:0.92rem;font-weight:700;display:inline-flex;align-items:center;gap:6px;">{st_theme["icon"]} {status_esc}</div>',
        '</div>',

        # Structured Details Grid
        '<div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(240px, 1fr));gap:14px 20px;margin-bottom:16px;font-size:0.92rem;">',

        '<div>',
        '<div style="color:#64748b;font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;">🏷️ Category & Problem</div>',
        f'<div style="color:#0f172a;font-weight:700;font-size:1.02rem;margin-top:2px;">{cat_esc}</div>',
        (f'<div style="color:#475569;font-size:0.86rem;margin-top:2px;">{problem_esc}</div>' if problem_esc != cat_esc else ''),
        (f'<div style="color:#0284c7;font-size:0.80rem;font-weight:600;margin-top:3px;">➕ Secondary Condition: {", ".join([escape(s.replace("_", " ").title()) for s in data["secondary_categories"]])}</div>' if data.get("secondary_categories") else ''),
        '</div>',

        '<div>',
        '<div style="color:#64748b;font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;">⚠️ Severity & Priority</div>',
        f'<div style="margin-top:2px;"><span style="color:{prio_color};font-weight:700;font-size:1.02rem;">{priority_esc} ({severity_esc})</span> <span style="color:#475569;font-size:0.84rem;">• {data["citizen_reports_count"]} report(s) in cluster</span></div>',
        '</div>',

        '<div>',
        '<div style="color:#64748b;font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;">📍 Location & City</div>',
        f'<div style="color:#0f172a;font-weight:700;font-size:1.02rem;margin-top:2px;">{loc_esc} <span style="background:rgba(37,99,235,0.10);color:#1d4ed8;border:1px solid rgba(37,99,235,0.22);padding:2px 8px;border-radius:10px;font-size:0.75rem;font-weight:700;margin-left:4px;">🏛️ {city_esc}</span></div>',
        (f'<div style="color:#475569;font-size:0.84rem;font-family:monospace;margin-top:3px;">🌐 {escape(coords_str)}</div>' if coords_str else ''),
        '</div>',

        '<div>',
        '<div style="color:#64748b;font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;">🏛️ Department</div>',
        f'<div style="color:#0f172a;font-weight:700;font-size:1.02rem;margin-top:2px;">{dept_esc}</div>',
        (f'<div style="color:#475569;font-size:0.86rem;margin-top:2px;">👤 {officer_esc}</div>' if officer_esc else ''),
        '</div>',

        '</div>',

        # Description Box
        '<div style="background:#f8fafc;border:1.5px solid rgba(37,99,235,0.20);border-radius:10px;padding:14px 18px;margin-bottom:14px;">',
        f'<div style="color:#64748b;font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:6px;">💬 Citizen Description <span style="background:rgba(37,99,235,0.10);color:#1d4ed8;border:1px solid rgba(37,99,235,0.22);padding:2px 8px;border-radius:10px;font-size:0.75rem;font-weight:700;margin-left:6px;">🌐 {escape(str(data.get("language_name", "English")))}</span></div>',
        f'<div style="color:#0f172a;font-size:0.95rem;font-weight:500;line-height:1.55;word-break:break-word;">"{desc_esc}"</div>',
        '</div>',
    ]

    # Remarks Box (only displayed if remarks exist)
    if remarks_esc:
        card_lines.extend([
            '<div style="background:rgba(37,99,235,0.06);border:1.5px solid rgba(37,99,235,0.25);border-left:4px solid #1d4ed8;border-radius:10px;padding:14px 18px;margin-bottom:14px;">',
            '<div style="color:#1d4ed8;font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:6px;">📝 Resolution Updates</div>',
            f'<div style="color:var(--civic-text-body, #1e293b);font-size:0.92rem;line-height:1.5;word-break:break-word;">{remarks_esc}</div>',
            '</div>',
        ])

    # Officer Map Link
    if gmaps_link:
        card_lines.append(
            f'<div style="margin-top:6px;font-size:0.86rem;"><a href="{gmaps_link}" target="_blank" style="color:#1d4ed8;font-weight:600;text-decoration:none;">🌐 Open Incident Coordinates in Google Maps ↗</a></div>'
        )

    card_lines.append('</div>')
    return "\n".join(card_lines)


def render_complaint_card(complaint: dict[str, Any], is_officer: bool = False) -> None:
    """Render the structured complaint card in Streamlit."""
    html_content = format_complaint_card_html(complaint, is_officer=is_officer)
    if hasattr(st, "html"):
        st.html(html_content)
    else:
        st.markdown(textwrap.dedent(html_content), unsafe_allow_html=True)


def render_officer_translation_tool(complaint: dict[str, Any]) -> None:
    """Render an interactive, on-demand translation tool for municipal officers.
    
    Allows officers to translate citizen complaints in Hindi or Telugu to English
    (or English to Hindi/Telugu) on-demand without modifying the official original
    complaint record stored in the municipal database.
    """
    data = prepare_complaint_data(complaint)
    cid = data["id"]
    desc = data["description"]
    orig_lang = data.get("original_language") or "en"
    orig_lang_name = data.get("language_name") or "English"

    with st.expander("🌐 On-Demand Multilingual Translation Tool (Municipal Officer)", expanded=True):
        st.caption(
            "Translate the citizen's complaint description into another official language on-demand. "
            "The official original record in municipal storage is never modified."
        )

        col_t_sel, col_t_btn = st.columns([2.5, 1.2])

        # Default target language:
        # If citizen spoke Hindi or Telugu -> default target is English
        # If citizen spoke English -> default target is Hindi
        default_idx = 0 if orig_lang != "en" else 1

        with col_t_sel:
            target_lang_choice = st.selectbox(
                "Translate into target language:",
                LANGUAGE_CHOICES,
                index=default_idx,
                key=f"trans_target_lang_{cid}",
            )

        target_code = normalize_lang_code(target_lang_choice)
        cache_key = f"trans_result_{cid}_{target_code}"

        with col_t_btn:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            translate_clicked = st.button(
                "🔄 Translate",
                key=f"btn_translate_{cid}",
                type="secondary",
                use_container_width=True,
            )

        # Trigger translation when button clicked
        if translate_clicked:
            with st.spinner(f"Translating to {target_lang_choice}..."):
                trans_text, _ = translate_text(desc, source_lang=orig_lang, target_lang=target_code)
                st.session_state[cache_key] = trans_text
                st.toast(f"Translated to {target_lang_choice}!", icon="🌐")

        active_translation = st.session_state.get(cache_key)

        # Show original vs translated side-by-side
        col_orig, col_trans = st.columns(2)
        with col_orig:
            st.markdown(
                clean_html(f"""
                <div style="background:#f8fafc;border:1.5px solid rgba(37,99,235,0.20);border-radius:10px;padding:14px 16px;min-height:95px;box-shadow:0 2px 8px rgba(0,0,0,0.03);">
                  <div style="color:#64748b;font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;">
                    📄 Original Citizen Text ({escape(orig_lang_name)})
                  </div>
                  <div style="color:#0f172a;font-size:0.94rem;font-weight:500;margin-top:6px;line-height:1.5;word-break:break-word;">
                    "{escape(desc)}"
                  </div>
                </div>
                """),
                unsafe_allow_html=True,
            )

        with col_trans:
            if active_translation:
                st.markdown(
                    clean_html(f"""
                    <div style="background:rgba(5,150,105,0.06);border:1.5px solid rgba(5,150,105,0.30);border-left:4px solid #059669;border-radius:10px;padding:14px 16px;min-height:95px;box-shadow:0 2px 8px rgba(0,0,0,0.03);">
                      <div style="color:#059669;font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;">
                        🌐 Translated Text ({escape(target_lang_choice)})
                      </div>
                      <div style="color:#0f172a;font-size:0.94rem;font-weight:500;margin-top:6px;line-height:1.5;word-break:break-word;">
                        "{escape(active_translation)}"
                      </div>
                    </div>
                    """),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    clean_html(f"""
                    <div style="background:#f8fafc;border:1.5px dashed rgba(148,163,184,0.40);border-radius:10px;padding:14px 16px;min-height:95px;display:flex;flex-direction:column;justify-content:center;align-items:center;">
                      <div style="color:#475569;font-size:0.86rem;font-weight:500;">Click <b>🔄 Translate</b> to generate on-demand translation</div>
                    </div>
                    """),
                    unsafe_allow_html=True,
                )


def render_complaint_media_and_detections(complaint: dict[str, Any], is_officer: bool = False) -> None:
    """Render original and YOLO-annotated images with detection details.

    Features:
    - Side-by-side presentation of original and YOLO-annotated photographs.
    - Displays bounding boxes, predicted classes, and confidence percentages.
    - Caches results to prevent redundant CPU inference on page refreshes.
    - Displays polite, informative feedback when no objects exceed the threshold.
    """
    data = prepare_complaint_data(complaint)
    photo_path = data["photo_path"]
    annotated_path = data["annotated_photo_path"]
    detections = data["detections"]
    cid = data["id"]

    if not photo_path or not Path(photo_path).exists():
        if is_officer:
            st.info("ℹ️ No photographic evidence attached to this ticket.")
        return

    st.markdown("#### 📷 Photographic Evidence & YOLO AI Vision Analysis")

    # If annotated image is missing or detections not cached, perform lazy inference once
    if (not annotated_path or not Path(annotated_path).exists()) and not detections:
        try:
            raw_img = Image.open(photo_path)
            new_dets, _ = vision.detect(raw_img)
            detections = [
                {
                    "class_name": d["class"],
                    "category": d["category"],
                    "confidence": d["confidence"],
                    "confidence_percent": d["confidence_percent"],
                    "bbox": d["box"],
                }
                for d in new_dets
            ]
            complaint["detections"] = detections
            data["detections"] = detections
            if detections:
                ann_file = Path("data/uploads") / f"{cid}_annotated.jpg"
                vision.save_annotated_image(raw_img, new_dets, ann_file)
                annotated_path = str(ann_file)
                complaint["annotated_photo_path"] = annotated_path
                data["annotated_photo_path"] = annotated_path
        except Exception as exc:
            st.warning(f"Note: Could not run vision analysis on image: {exc}")

    # Layout images
    col_orig, col_ann = st.columns(2, gap="medium")

    with col_orig:
        try:
            orig_img = Image.open(photo_path)
            st.image(orig_img, caption=f"Original Photographic Evidence (Ticket {cid})", use_container_width=True)
        except Exception:
            st.error(f"Could not load original image from {photo_path}")

    with col_ann:
        if annotated_path and Path(annotated_path).exists():
            try:
                ann_img = Image.open(annotated_path)
                st.image(ann_img, caption=f"YOLO Detected Objects & Bounding Boxes ({data['model_version']})", use_container_width=True)
            except Exception:
                st.info("ℹ️ Annotated image preview unavailable.")
        else:
            st.markdown(
                '<div style="background:#f8fafc;border:1.5px dashed rgba(37,99,235,0.25);border-radius:10px;padding:24px 18px;text-align:center;color:#1e293b;margin-top:10px;">'
                '<b style="color:#0f172a;">🔍 YOLO Vision Assessment:</b><br/>'
                'No municipal hazards detected above the <b>50% confidence threshold</b> in this photograph.<br/>'
                '<span style="font-size:0.85rem;color:#475569;">NLP text analysis serves as the primary classification signal.</span>'
                '</div>',
                unsafe_allow_html=True,
            )

    # Detailed Detection Table
    if detections:
        st.markdown(f"##### 🎯 Detected Civic Infrastructure Objects ({len(detections)})")
        rows = []
        for idx, d in enumerate(detections, 1):
            c_name = d.get("class_name") or d.get("class", "Issue")
            cat_name = d.get("category", "General")
            conf_pct = d.get("confidence_percent", d.get("confidence", 0) * 100)
            bbox = d.get("bbox") or d.get("box", [])
            bbox_str = f"[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]" if len(bbox) == 4 else str(bbox)
            
            row = {
                "#": idx,
                "Detected Hazard": c_name.replace("_", " ").title(),
                "Civic Category": cat_name.replace("_", " ").title(),
                "AI Confidence": f"{conf_pct:.1f}%",
            }
            if is_officer:
                row["Bounding Box [x1, y1, x2, y2]"] = bbox_str
                row["Model Architecture"] = data.get("model_version", "YOLO11n")
            rows.append(row)
        
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.caption("ℹ️ *Photograph attached. No object detections exceeded the 50% confidence threshold.*")

