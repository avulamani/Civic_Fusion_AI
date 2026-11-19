"""CivicFusion AI: Multimodal Municipal Complaint Intelligence System.

Combines:
- 📷 Computer Vision (YOLO-World v2 / YOLO onnx)
- 📝 Natural Language Processing (Category, urgency cues, impact extraction)
- 🎤 Voice Input & Speech-to-Text
- 📍 Geospatial Location Intelligence & Severity-Mapped Visualization
- 🔄 Semantic & Spatial Duplicate Complaint Detection
- 🚨 Automated Severity & Priority Engine (P1 to P4)
- 🏛️ Municipal Official Command Center & Work-Order Queue
"""

from datetime import date, timedelta
from html import escape
from io import BytesIO
import json
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
from PIL import Image
import urllib.request
import pydeck as pdk
import streamlit as st
import streamlit.components.v1 as components

try:
    from streamlit_js_eval import get_geolocation
except Exception:
    get_geolocation = None

# Declare custom GPS button component for bidirectional browser GPS communication
GPS_COMP_DIR = Path(__file__).parent / "components" / "gps_button"
gps_location_button = components.declare_component(
    "gps_location_button",
    path=str(GPS_COMP_DIR.resolve())
)


def clean_html(html_str: str) -> str:
    """Strip all leading and trailing whitespace from every line to ensure Markdown never treats HTML as an indented code block."""
    return "\n".join(line.strip() for line in html_str.strip().splitlines() if line.strip())



def fetch_ip_location() -> tuple[float, float, str] | None:
    """Detect real device/network location via fast IP geolocation (no browser permissions required)."""
    endpoints = [
        ("http://ip-api.com/json", lambda d: (float(d["lat"]), float(d["lon"]), f"{d.get('city', 'Local')}, {d.get('regionName', '')}".strip(", "))),
        ("https://ipapi.co/json/", lambda d: (float(d["latitude"]), float(d["longitude"]), f"{d.get('city', 'Local')}, {d.get('region', '')}".strip(", ")))
    ]
    for url, parser in endpoints:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "CivicFusion/1.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                return parser(data)
        except Exception:
            continue
    return None

import importlib
import sys

# Streamlit in-memory process reload guard: ensure hot-reloading recognizes newly added methods across modules
for _mod_name in ["complaint_view", "multilingual", "speech_utils", "nlp", "storage", "vision", "duplicates", "fusion", "auth"]:
    if _mod_name in sys.modules:
        try:
            importlib.reload(sys.modules[_mod_name])
        except Exception:
            pass

from auth import is_officer, login, logout
import complaint_view
try:
    importlib.reload(complaint_view)
except Exception:
    pass

get_status_theme = complaint_view.get_status_theme
render_complaint_card = complaint_view.render_complaint_card
render_complaint_media_and_detections = complaint_view.render_complaint_media_and_detections
render_officer_translation_tool = getattr(
    complaint_view,
    "render_officer_translation_tool",
    lambda *args, **kwargs: None
)

from duplicates import check_for_duplicates, haversine_distance_meters
from fusion import (
    CV_TO_CATEGORY,
    DEPARTMENTS,
    DISPLAY_NAMES,
    PROBLEM_TITLES,
    fuse_and_recommend,
)
import multilingual
try:
    importlib.reload(multilingual)
except Exception:
    pass
from multilingual import (
    detect_language,
    get_language_display_name,
    get_language_speech_code,
    LANGUAGE_CHOICES,
    normalize_lang_code,
    translate_text,
)
import nlp
try:
    importlib.reload(nlp)
except Exception:
    pass
from nlp import analyze_text_full, load_classifier
import speech_utils
try:
    importlib.reload(speech_utils)
except Exception:
    pass
from speech_utils import (
    get_sample_voice_complaints,
    SAMPLE_VOICE_COMPLAINTS,
    transcribe_audio,
)
import storage

if not hasattr(storage, "delete_complaint"):
    try:
        importlib.reload(storage)
    except Exception:
        pass

if hasattr(storage, "delete_complaint"):
    delete_complaint = storage.delete_complaint
else:
    def delete_complaint(ticket_id: str):
        current = storage.load_all_complaints()
        clean_id = str(ticket_id).strip().upper()
        current = [c for c in current if str(c.get("id", "")).strip().upper() != clean_id]
        storage.save_all_complaints(current)
        return current

increment_cluster_count = storage.increment_cluster_count
insert_complaint = storage.insert_complaint
load_all_complaints = storage.load_all_complaints
update_ticket_status = storage.update_ticket_status
log_cv_nlp_mismatch = getattr(storage, "log_cv_nlp_mismatch", lambda *args, **kwargs: None)
load_all_mismatches = getattr(storage, "load_all_mismatches", lambda *args, **kwargs: [])
import vision
try:
    importlib.reload(vision)
except Exception:
    pass
from vision import ACTIVE_MODEL_VERSION, DEFAULT_CONFIDENCE_THRESHOLD, MODEL_PATH, detect, draw_detections
from vision_config import CV_CLASSES


st.set_page_config(
    page_title="CivicFusion AI | Municipal Complaint Intelligence",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS with full dark/light theme adaptivity, White/Green/Blue Municipal Theme
st.markdown(clean_html("""
<style>
  :root {
      --civic-blue: #1d4ed8;
      --civic-blue-hover: #1e40af;
      --civic-blue-light: rgba(37, 99, 235, 0.08);
      --civic-green: #059669;
      --civic-green-hover: #047857;
      --civic-green-light: rgba(16, 185, 129, 0.10);
      --civic-white: #ffffff;
      --civic-card-bg: #ffffff;
      --civic-card-border: rgba(37, 99, 235, 0.20);
      --civic-card-shadow: 0 4px 18px rgba(29, 78, 216, 0.05);
      --civic-text-title: #0f172a;
      --civic-text-body: #1e293b;
      --civic-text-sub: #475569;
      --civic-text-label: #64748b;
      --civic-desc-bg: #f8fafc;
      --civic-desc-border: rgba(37, 99, 235, 0.18);
      --civic-desc-text: #0f172a;
      --civic-divider: rgba(148, 163, 184, 0.22);
      --civic-step-pending-bg: rgba(241, 245, 249, 0.85);
      --civic-step-pending-text: #475569;
      --civic-review-bg: linear-gradient(135deg, rgba(29, 78, 216, 0.05) 0%, rgba(5, 150, 105, 0.05) 100%);
  }

  html {
      scroll-behavior: smooth !important;
  }

  .block-container {
      padding-top: 1.2rem;
      max-width: 1400px;
  }
  
  /* Isolate map scroll so mouse wheel zooms the map without scrolling the webpage */
  .stDeckGlJsonContainer, .deckgl-overlay, div[data-testid="stDeckGlJsonContainer"] canvas {
      overscroll-behavior: contain !important;
  }

  /* Primary Gradient Buttons (Civic Blue to Green) */
  button[kind="primary"] {
      background: linear-gradient(135deg, #1d4ed8 0%, #059669 100%) !important;
      color: #ffffff !important;
      border: none !important;
      font-weight: 600 !important;
      border-radius: 10px !important;
      padding: 0.55rem 1.25rem !important;
      box-shadow: 0 4px 14px rgba(29, 78, 216, 0.28) !important;
      transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
  }
  button[kind="primary"] p {
      color: #ffffff !important;
  }
  button[kind="primary"]:hover {
      background: linear-gradient(135deg, #1e40af 0%, #047857 100%) !important;
      transform: translateY(-2px) !important;
      box-shadow: 0 6px 20px rgba(5, 150, 105, 0.35) !important;
  }

  /* Secondary & Action Buttons: Clean White with Civic Blue border & text */
  button:not([kind="primary"]) {
      background: #ffffff !important;
      border: 1.5px solid rgba(37, 99, 235, 0.30) !important;
      color: #1d4ed8 !important;
      font-weight: 600 !important;
      border-radius: 10px !important;
      padding: 0.55rem 1.25rem !important;
      transition: all 0.2s ease !important;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
  }
  button:not([kind="primary"]) p {
      color: #1d4ed8 !important;
      font-weight: 600 !important;
  }
  button:not([kind="primary"]):hover {
      border-color: #1d4ed8 !important;
      background: #eff6ff !important;
      color: #1d4ed8 !important;
      transform: translateY(-2px) !important;
      box-shadow: 0 4px 14px rgba(29, 78, 216, 0.18) !important;
  }
  button:not([kind="primary"]):hover p {
      color: #1d4ed8 !important;
      font-weight: 700 !important;
  }

  /* Left Sidebar: Clean Municipal Styling */
  [data-testid="stSidebar"] {
      background-color: #f8fafc !important;
  }
  [data-testid="stSidebar"] [data-testid="stRadio"] > div {
      gap: 12px;
  }
  [data-testid="stSidebar"] [data-testid="stRadio"] label {
      background: #ffffff !important;
      border: 1.5px solid rgba(37, 99, 235, 0.20) !important;
      border-radius: 12px !important;
      padding: 12px 16px !important;
      color: #0f172a !important;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
      transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
      cursor: pointer !important;
      width: 100% !important;
      margin: 0 !important;
  }
  [data-testid="stSidebar"] [data-testid="stRadio"] label p,
  [data-testid="stSidebar"] [data-testid="stRadio"] label span,
  [data-testid="stSidebar"] [data-testid="stRadio"] label div {
      color: #0f172a !important;
      font-weight: 600 !important;
  }
  /* Left Sidebar Navigation HOVER: Ice Blue background with Deep Civic Blue text */
  [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
      background: #eff6ff !important;
      border-color: #1d4ed8 !important;
      color: #1d4ed8 !important;
      transform: translateX(4px) !important;
      box-shadow: 0 4px 16px rgba(29, 78, 216, 0.18) !important;
  }
  [data-testid="stSidebar"] [data-testid="stRadio"] label:hover p,
  [data-testid="stSidebar"] [data-testid="stRadio"] label:hover span,
  [data-testid="stSidebar"] [data-testid="stRadio"] label:hover div {
      color: #1d4ed8 !important;
      font-weight: 700 !important;
  }
  /* Left Sidebar Navigation ACTIVE: Civic gradient tint with bold civic blue border & text */
  [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
      background: linear-gradient(135deg, rgba(29, 78, 216, 0.10) 0%, rgba(5, 150, 105, 0.08) 100%) !important;
      border: 2px solid #1d4ed8 !important;
      box-shadow: 0 4px 14px rgba(29, 78, 216, 0.15) !important;
  }
  [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p,
  [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) span {
      color: #1d4ed8 !important;
      font-weight: 700 !important;
  }

  /* Sidebar Action / Logout Button */
  [data-testid="stSidebar"] button {
      background: #ffffff !important;
      border: 1.5px solid rgba(220, 38, 38, 0.35) !important;
      color: #dc2626 !important;
      font-weight: 600 !important;
      border-radius: 10px !important;
      transition: all 0.2s ease !important;
      box-shadow: 0 2px 6px rgba(0,0,0,0.03) !important;
  }
  [data-testid="stSidebar"] button p {
      color: #dc2626 !important;
      font-weight: 600 !important;
  }
  [data-testid="stSidebar"] button:hover {
      background: #fef2f2 !important;
      border-color: #dc2626 !important;
      color: #b91c1c !important;
      transform: translateY(-1px) !important;
      box-shadow: 0 4px 12px rgba(220, 38, 38, 0.15) !important;
  }
  [data-testid="stSidebar"] button:hover p {
      color: #b91c1c !important;
  }

  /* Metric KPI Cards with Top Civic Blue/Green Accent */
  [data-testid="stMetric"] {
      background: #ffffff !important;
      border: 1.5px solid rgba(37, 99, 235, 0.20) !important;
      border-top: 4px solid #2563eb !important;
      padding: 16px 20px;
      border-radius: 14px;
      box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04) !important;
      transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
  }
  [data-testid="stMetric"]:hover {
      transform: translateY(-4px) !important;
      border-top-color: #059669 !important;
      box-shadow: 0 10px 24px -4px rgba(5, 150, 105, 0.20) !important;
  }
  [data-testid="stMetric"] [data-testid="stMetricLabel"] {
      color: #475569 !important;
      font-weight: 600 !important;
  }
  [data-testid="stMetric"] [data-testid="stMetricValue"] {
      color: #0f172a !important;
      font-weight: 700 !important;
  }

  /* Workflow Stepper Bar */
  .workflow-stepper {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-bottom: 24px;
      background: linear-gradient(135deg, rgba(29, 78, 216, 0.06) 0%, rgba(5, 150, 105, 0.06) 100%);
      border: 1.5px solid rgba(37, 99, 235, 0.25);
      border-radius: 16px;
      padding: 14px;
  }
  .step-card {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 14px;
      border-radius: 12px;
      font-size: 0.90rem;
      font-weight: 600;
      transition: all 0.25s ease;
  }
  .step-card.active {
      background: #1d4ed8;
      color: #ffffff;
      border: 1.5px solid #3b82f6;
      box-shadow: 0 4px 16px rgba(29, 78, 216, 0.35);
  }
  .step-card.completed {
      background: #059669;
      color: #ffffff;
      border: 1.5px solid #10b981;
      box-shadow: 0 4px 14px rgba(5, 150, 105, 0.25);
  }
  .step-card.pending {
      background: #f8fafc;
      color: #475569;
      border: 1.5px dashed rgba(148, 163, 184, 0.45);
  }
  .step-card.pending .step-num {
      background: rgba(100, 116, 139, 0.15);
      color: #475569;
  }
  .step-num {
      width: 26px;
      height: 26px;
      border-radius: 50%;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 0.82rem;
      font-weight: 700;
      background: rgba(255, 255, 255, 0.25);
  }

  /* Preset Location Buttons as Interactive Badges */
  div[data-testid="stHorizontalBlock"] button {
      border-radius: 10px;
      border: 1.5px solid rgba(37, 99, 235, 0.25) !important;
      background: #ffffff !important;
      color: #1d4ed8 !important;
      transition: all 0.2s ease !important;
      font-weight: 500;
  }
  div[data-testid="stHorizontalBlock"] button p {
      color: #1d4ed8 !important;
  }
  div[data-testid="stHorizontalBlock"] button:hover {
      border-color: #059669 !important;
      background: #ecfdf5 !important;
      color: #059669 !important;
      transform: translateY(-2px);
      box-shadow: 0 6px 16px rgba(5, 150, 105, 0.20);
  }
  div[data-testid="stHorizontalBlock"] button:hover p {
      color: #059669 !important;
  }

  /* Officer Section Switcher horizontal radio pills */
  div[data-testid="stRadio"]:has(input[name="officer_section_view"]) label {
      background: #ffffff !important;
      border: 1.5px solid rgba(37, 99, 235, 0.22) !important;
      border-radius: 20px;
      padding: 8px 18px;
      font-weight: 600;
      color: #0f172a !important;
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
  }
  div[data-testid="stRadio"]:has(input[name="officer_section_view"]) label p {
      color: #0f172a !important;
      font-weight: 600 !important;
  }
  div[data-testid="stRadio"]:has(input[name="officer_section_view"]) label:hover {
      border-color: #059669 !important;
      background: #ecfdf5 !important;
      color: #059669 !important;
  }
  div[data-testid="stRadio"]:has(input[name="officer_section_view"]) label:hover p {
      color: #059669 !important;
  }
  div[data-testid="stRadio"]:has(input[name="officer_section_view"]) label:has(input:checked) {
      background: linear-gradient(135deg, rgba(29, 78, 216, 0.10), rgba(5, 150, 105, 0.08)) !important;
      border: 2px solid #059669 !important;
  }
  div[data-testid="stRadio"]:has(input[name="officer_section_view"]) label:has(input:checked) p {
      color: #059669 !important;
      font-weight: 700 !important;
  }

  /* Pre-submission review summary card */
  .review-summary-card {
      background: linear-gradient(135deg, rgba(29, 78, 216, 0.05) 0%, rgba(5, 150, 105, 0.05) 100%);
      border: 1.5px solid rgba(37, 99, 235, 0.25);
      border-left: 5px solid #059669;
      border-radius: 14px;
      padding: 18px 22px;
      margin: 16px 0;
      color: #1e293b;
      box-shadow: 0 4px 16px rgba(0,0,0,0.03);
  }

  /* Card boxes for intake steps */
  .intake-box {
      background: #ffffff !important;
      border: 1.5px solid rgba(37, 99, 235, 0.20) !important;
      border-radius: 14px;
      padding: 20px 22px;
      margin-bottom: 20px;
      box-shadow: 0 4px 18px rgba(0, 0, 0, 0.04);
      transition: all 0.25s ease;
  }
  .intake-box:hover {
      border-color: #2563eb;
      box-shadow: 0 6px 20px rgba(37, 99, 235, 0.12);
  }
  
  .ticket-box {
      background-color: #0f172a;
      color: #38bdf8;
      font-family: 'Courier New', Courier, monospace;
      padding: 16px 20px;
      border-radius: 12px;
      font-size: 0.88rem;
      line-height: 1.35;
      white-space: pre-wrap;
      border: 1.5px solid rgba(56, 189, 248, 0.3);
      box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
  }
  .duplicate-pill {
      background-color: rgba(245, 158, 11, 0.15);
      border: 1.5px solid #f59e0b;
      color: #f59e0b;
      padding: 12px 16px;
      border-radius: 10px;
      font-weight: 500;
      margin-bottom: 14px;
  }
  .priority-p1 { color: #ef4444; font-weight: bold; }
  .priority-p2 { color: #f97316; font-weight: bold; }
  .priority-p3 { color: #eab308; font-weight: bold; }
  .priority-p4 { color: #10b981; font-weight: bold; }

  /* Role badge in sidebar */
  .role-badge-officer {
      background: linear-gradient(135deg, rgba(16,185,129,0.18), rgba(5,150,105,0.12));
      border: 1.5px solid #10b981;
      color: #10b981;
      border-radius: 20px;
      padding: 6px 16px;
      font-size: 0.85rem;
      font-weight: 600;
      display: inline-block;
      margin-bottom: 8px;
  }
  .role-badge-citizen {
      background: linear-gradient(135deg, rgba(37,99,235,0.15), rgba(2,132,199,0.10));
      border: 1.5px solid #3b82f6;
      color: #3b82f6;
      border-radius: 20px;
      padding: 6px 16px;
      font-size: 0.85rem;
      font-weight: 600;
      display: inline-block;
      margin-bottom: 8px;
  }
  /* Track-ticket status badge */
  .status-open       { color: #3b82f6; font-weight: 600; }
  .status-pending    { color: #ef4444; font-weight: 600; }
  .status-inprogress { color: #f97316; font-weight: 600; }
  .status-assigned   { color: #a855f7; font-weight: 600; }
  .status-resolved   { color: #10b981; font-weight: 600; }
</style>
"""), unsafe_allow_html=True)


def auto_scroll_to(element_id: str):
    """Smoothly scroll the browser window to a target element ID."""
    js_code = f"""
    <script>
      (function() {{
        function doScroll() {{
          try {{
            const doc = window.parent.document;
            const el = doc.getElementById('{element_id}') || doc.querySelector('[id="{element_id}"]');
            if (el) {{
              el.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
            }}
          }} catch(e) {{
            console.log('Scroll error:', e);
          }}
        }}
        setTimeout(doScroll, 80);
        setTimeout(doScroll, 250);
      }})();
    </script>
    """
    components.html(js_code, height=0, width=0)


def clean_html(html_str: str) -> str:
    """Strip all leading and trailing whitespace from every line to ensure Markdown never treats HTML as an indented code block."""
    return "\n".join(line.strip() for line in html_str.strip().splitlines() if line.strip())



@st.cache_resource
def get_nlp_model():
    return load_classifier()


# Demo presets based on tested complaints (CF-1052 to CF-1058)
DEMO_PRESETS = {
    "CF-1052: Potholes Near Bakery": {
        "id": "CF-1052",
        "title": "Potholes Near Bakery",
        "cat": "road_damage",
        "lat": 31.2900,
        "lon": 75.7400,
        "loc": "Near Bakery",
        "sample_text": "There are multiple deep potholes near the bakery.",
        "sample_photo": "data/uploads/CF-1052.jpg",
    },
    "CF-1053: Road Damage (School Gate)": {
        "id": "CF-1053",
        "title": "Road Damage (School Gate)",
        "cat": "road_damage",
        "lat": 31.1900,
        "lon": 75.7400,
        "loc": "Near School Gate",
        "sample_text": "some part of the road is damaged near the school gate",
        "sample_photo": "data/uploads/CF-1053.jpg",
    },
    "CF-1054: Pothole (Railway Station Road)": {
        "id": "CF-1054",
        "title": "Pothole (Railway Station Road)",
        "cat": "road_damage",
        "lat": 31.2900,
        "lon": 75.6400,
        "loc": "Railway Station Road",
        "sample_text": "there is a pot hole on the road near the railway station, vehicles facing difficulty over passing through it.",
        "sample_photo": "data/uploads/CF-1054.jpg",
    },
    "CF-1056: Garbage Bags Near Bakery": {
        "id": "CF-1056",
        "title": "Garbage Bags Near Bakery",
        "cat": "waste",
        "lat": 31.3300,
        "lon": 75.7700,
        "loc": "Near Bakery",
        "sample_text": "the garbage should be removed immediately near the bakery, they are in excessive amount, bad smell is coming from those packages",
        "sample_photo": "data/uploads/CF-1056.jpg",
    },
    "CF-1057: Roadside Garbage Dump": {
        "id": "CF-1057",
        "title": "Roadside Garbage Dump",
        "cat": "waste",
        "lat": 31.2500,
        "lon": 75.8000,
        "loc": "Railway Station Approach",
        "sample_text": "there is so much garbage on the road as well as beside the road near the railway station. People are facing difficulty over passing through it, please clean it immediately.",
        "sample_photo": "data/uploads/CF-1057.jpg",
    },
    "CF-1058: Roadside Garbage Accumulation": {
        "id": "CF-1058",
        "title": "Roadside Garbage Accumulation",
        "cat": "waste",
        "lat": 31.3000,
        "lon": 75.69977,
        "loc": "Market Corridor",
        "sample_text": "there is garbage left here, this should be cleaned immediately.",
        "sample_photo": "data/uploads/CF-1058.jpg",
    },
}
MUNICIPAL_LOCATIONS = DEMO_PRESETS
CITY_PRESETS = DEMO_PRESETS

SEVERITY_COLORS = {
    "Critical": [239, 68, 68, 220],     # Red
    "High": [249, 115, 22, 220],         # Orange-Red
    "Medium": [234, 179, 8, 220],        # Amber-Yellow
    "Low": [16, 185, 129, 220],          # Emerald-Green
}

SEVERITY_RADIUS = {
    "Critical": 110,
    "High": 85,
    "Medium": 65,
    "Low": 50,
}


def extract_exif_gps(image: Image.Image) -> tuple[float | None, float | None]:
    """Extract decimal latitude and longitude from mobile camera EXIF metadata."""
    try:
        exif = image.getexif()
        if not exif:
            return None, None
        gps_info = exif.get_ifd(0x8825)
        if not gps_info:
            return None, None

        lat_data = gps_info.get(2)
        lat_ref = gps_info.get(1, "N")
        lon_data = gps_info.get(4)
        lon_ref = gps_info.get(3, "E")

        if not lat_data or not lon_data:
            return None, None

        def _convert(dms, ref):
            try:
                d = float(dms[0])
                m = float(dms[1])
                s = float(dms[2])
                dec = d + (m / 60.0) + (s / 3600.0)
                if ref in ["S", "W"]:
                    dec = -dec
                return round(dec, 5)
            except Exception:
                return None

        lat = _convert(lat_data, lat_ref)
        lon = _convert(lon_data, lon_ref)
        return lat, lon
    except Exception:
        return None, None


# Persistent Complaints Storage (Disk-backed, shared across sessions)
if "complaints" not in st.session_state:
    st.session_state["complaints"] = load_all_complaints()

if "ticket_counter" not in st.session_state:
    existing_ids = [
        int(c["id"].replace("CF-", ""))
        for c in st.session_state["complaints"]
        if "CF-" in str(c.get("id", "")) and c["id"].replace("CF-", "").isdigit()
    ]
    st.session_state["ticket_counter"] = (max(existing_ids) + 1) if existing_ids else 1043


def render_map(
    complaints: list[dict[str, Any]],
    focus_ticket_id: str | None = None,
    officer_loc: tuple[float, float] | None = None,
):
    """Render high-resolution interactive map with Google Maps Roadmap tiles and teardrop pin markers."""
    if not complaints:
        st.info("No complaint locations to display on map.")
        return

    # If an individual ticket is focused, ONLY display that ticket!
    if focus_ticket_id:
        target_complaint = next((c for c in complaints if str(c.get("id")).upper() == str(focus_ticket_id).upper()), None)
        active_complaints = [target_complaint] if target_complaint else complaints
    else:
        target_complaint = None
        active_complaints = complaints

    map_items = []
    for c in active_complaints:
        sev = c.get("severity", "Low")
        prio = c.get("priority", "P3")
        map_items.append({
            "id": c.get("id"),
            "problem": c.get("problem_title", c.get("category_display", "Civic Issue")),
            "department": c.get("department", "Municipal"),
            "severity": sev,
            "priority": prio,
            "location": c.get("location_name", "Reported Area"),
            "status": c.get("status", "Open"),
            "reports": c.get("citizen_reports_count", 1),
            "lat": float(c.get("latitude", 31.6340)),
            "lon": float(c.get("longitude", 74.8723)),
        })

    complaints_json = json.dumps(map_items)
    is_focused = bool(focus_ticket_id and target_complaint)
    focus_lat = float(target_complaint.get("latitude", 31.6340)) if target_complaint else 31.6340
    focus_lon = float(target_complaint.get("longitude", 74.8723)) if target_complaint else 74.8723
    officer_json = json.dumps(list(officer_loc)) if (officer_loc and (officer_loc[0] != 0.0 or officer_loc[1] != 0.0)) else "null"

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                background-color: #0f172a;
            }}
            #map-wrapper {{
                position: relative;
                width: 100%;
                height: 520px;
                border-radius: 14px;
                overflow: hidden;
                border: 1px solid rgba(255, 255, 255, 0.12);
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
            }}
            #map {{
                width: 100%;
                height: 100%;
            }}
            .map-live-bar {{
                position: absolute;
                top: 12px;
                left: 60px;
                z-index: 1000;
                display: flex;
                align-items: center;
                gap: 10px;
                background: rgba(15, 23, 42, 0.88);
                backdrop-filter: blur(8px);
                border: 1px solid rgba(59, 130, 246, 0.35);
                padding: 7px 14px;
                border-radius: 10px;
                color: #f8fafc;
                font-size: 13px;
                box-shadow: 0 6px 18px rgba(0,0,0,0.35);
            }}
            .live-gps-btn {{
                background: linear-gradient(135deg, #0284c7, #0369a1);
                color: white;
                border: 1px solid rgba(255,255,255,0.25);
                border-radius: 7px;
                padding: 6px 14px;
                font-size: 12px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s ease;
                display: flex;
                align-items: center;
                gap: 6px;
            }}
            .live-gps-btn:hover {{
                background: linear-gradient(135deg, #0369a1, #075985);
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(2, 132, 199, 0.4);
            }}
            .status-text {{
                color: #cbd5e1;
                font-size: 12px;
            }}
            .custom-pin-node:hover svg {{
                transform: scale(1.18) translateY(-4px);
            }}
            .pin-pulse-halo {{
                position: absolute;
                bottom: -5px;
                left: 50%;
                transform: translateX(-50%);
                width: 20px;
                height: 9px;
                border-radius: 50%;
                background: rgba(220, 38, 38, 0.5);
                animation: halo-anim 1.5s infinite ease-out;
            }}
            @keyframes halo-anim {{
                0% {{ transform: translateX(-50%) scale(0.6); opacity: 1; }}
                100% {{ transform: translateX(-50%) scale(2.3); opacity: 0; }}
            }}
            .user-radar-wave {{
                position: absolute;
                bottom: -6px;
                left: 50%;
                transform: translateX(-50%);
                width: 24px;
                height: 12px;
                border-radius: 50%;
                background: rgba(2, 132, 199, 0.45);
                animation: radar-anim 1.6s infinite ease-out;
            }}
            @keyframes radar-anim {{
                0% {{ transform: translateX(-50%) scale(0.6); opacity: 1; }}
                100% {{ transform: translateX(-50%) scale(2.4); opacity: 0; }}
            }}
            .leaflet-popup-content-wrapper {{
                background: #0f172a !important;
                color: #f8fafc !important;
                border-radius: 10px !important;
                border: 1px solid rgba(59, 130, 246, 0.3) !important;
                box-shadow: 0 10px 25px rgba(0,0,0,0.5) !important;
            }}
            .leaflet-popup-tip {{
                background: #0f172a !important;
            }}
            .leaflet-tooltip.pin-tooltip {{
                background: #0f172a;
                color: #f8fafc;
                border: 1px solid rgba(255, 255, 255, 0.18);
                font-size: 12px;
                font-weight: 600;
                border-radius: 6px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.4);
                padding: 4px 8px;
            }}
            .leaflet-tooltip.pin-tooltip::before {{
                border-top-color: #0f172a;
            }}
        </style>
    </head>
    <body>
        <div id="map-wrapper">
            <div id="map"></div>
        </div>

        <script>
            // 1. High-Detail Google Maps Roadmap
            var googleRoadmap = L.tileLayer('https://mt1.google.com/vt/lyrs=m&x={{x}}&y={{y}}&z={{z}}', {{
                maxZoom: 20,
                subdomains: ['mt0', 'mt1', 'mt2', 'mt3'],
                attribution: '&copy; Google Maps'
            }});

            // 2. Google Maps Satellite Hybrid
            var googleSatellite = L.tileLayer('https://mt1.google.com/vt/lyrs=y&x={{x}}&y={{y}}&z={{z}}', {{
                maxZoom: 20,
                subdomains: ['mt0', 'mt1', 'mt2', 'mt3'],
                attribution: '&copy; Google Satellite'
            }});

            // 3. OpenStreetMap Roadmap
            var osmRoadmap = L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                maxZoom: 19,
                attribution: '&copy; OpenStreetMap'
            }});

            var isFocused = {str(is_focused).lower()};
            var focusLat = {focus_lat};
            var focusLon = {focus_lon};
            var initialZoom = isFocused ? 17 : 13;

            var map = L.map('map', {{
                center: [focusLat, focusLon],
                zoom: initialZoom,
                layers: [googleRoadmap],
                scrollWheelZoom: true
            }});

            // Layer Switcher Control
            var baseMaps = {{
                "🗺️ Google Maps (Roads & Details)": googleRoadmap,
                "🛰️ Google Satellite (Hybrid)": googleSatellite,
                "🌍 OpenStreetMap": osmRoadmap
            }};
            L.control.layers(baseMaps, null, {{position: 'topright'}}).addTo(map);
            L.control.scale({{metric: true, imperial: false}}).addTo(map);

            // Teardrop Pin SVG Icon Generator for High-Visibility Complaint Markers
            function createPinIcon(severity, isSelected) {{
                var colors = {{
                    "Critical": "#dc2626", // Red (P1)
                    "High":     "#ea580c", // Orange (P2)
                    "Medium":   "#d97706", // Amber (P3)
                    "Low":      "#16a34a"  // Green (P4)
                }};
                var pinColor = colors[severity] || "#2563eb";
                var w = isSelected ? 36 : 28;
                var h = isSelected ? 50 : 40;
                var anchorX = w / 2;
                var anchorY = h;

                var svg = `
                    <div class="custom-pin-node" style="position:relative; width:${{w}}px; height:${{h}}px; cursor:pointer;">
                        <svg width="${{w}}" height="${{h}}" viewBox="0 0 30 42" fill="none" xmlns="http://www.w3.org/2000/svg"
                             style="filter: drop-shadow(0 4px 7px rgba(0,0,0,0.55)); transition: transform 0.2s ease;">
                            <path d="M15 0C6.71573 0 0 6.71573 0 15C0 26.25 15 42 15 42C15 42 30 26.25 30 15C30 6.71573 23.2843 0 15 0Z"
                                  fill="${{pinColor}}" stroke="#ffffff" stroke-width="2"/>
                            <circle cx="15" cy="15" r="7.5" fill="#ffffff"/>
                            <circle cx="15" cy="15" r="4.5" fill="${{pinColor}}"/>
                        </svg>
                        ${{isSelected ? '<div class="pin-pulse-halo"></div>' : ''}}
                    </div>
                `;

                return L.divIcon({{
                    className: 'leaflet-custom-pin',
                    html: svg,
                    iconSize: [w, h],
                    iconAnchor: [anchorX, anchorY],
                    popupAnchor: [0, -anchorY + 4]
                }});
            }}

            // User / Officer Blue Location Pin Icon Generator
            // 4. Render Complaints with Distinct Teardrop Map Pins
            var complaintsData = {complaints_json};
            var markers = [];
            var bounds = [];

            var colorMap = {{
                "Critical": "#dc2626",
                "High":     "#ea580c",
                "Medium":   "#d97706",
                "Low":      "#16a34a"
            }};

            complaintsData.forEach(function(c) {{
                var pinIcon = createPinIcon(c.severity, isFocused);
                var marker = L.marker([c.lat, c.lon], {{ icon: pinIcon }}).addTo(map);
                var color = colorMap[c.severity] || "#2563eb";

                var popupContent = `
                    <div style="font-size:13px; line-height:1.45; min-width:200px; padding:4px;">
                        <b style="color:${{color}}; font-size:14px;">📍 Ticket ${{c.id}}</b> (${{c.priority}})<br/>
                        <b>Problem:</b> ${{c.problem}}<br/>
                        <b>Severity:</b> ${{c.severity}}<br/>
                        <b>Location:</b> ${{c.location}}<br/>
                        <b>Status:</b> ${{c.status}}<br/>
                        <b>Reports:</b> ${{c.reports}} citizen report(s)
                    </div>
                `;
                marker.bindPopup(popupContent);
                marker.bindTooltip(`<b>${{c.id}}</b>: ${{c.problem}}`, {{
                    className: 'pin-tooltip',
                    direction: 'top',
                    offset: [0, isFocused ? -50 : -40]
                }});
                markers.push(marker);
                bounds.push([c.lat, c.lon]);

                if (isFocused) {{
                    marker.openPopup();
                }}
            }});

            if (!isFocused && bounds.length > 0) {{
                map.fitBounds(bounds, {{padding: [45, 45], maxZoom: 15}});
            }}

            // User / Officer Blue Location Pin Icon Generator
            function createUserPinIcon() {{
                var w = 34, h = 48;
                var svg = `
                    <div style="position:relative; width:${{w}}px; height:${{h}}px; cursor:pointer;">
                        <svg width="${{w}}" height="${{h}}" viewBox="0 0 30 42" fill="none" xmlns="http://www.w3.org/2000/svg"
                             style="filter: drop-shadow(0 4px 8px rgba(2,132,199,0.7));">
                            <path d="M15 0C6.71573 0 0 6.71573 0 15C0 26.25 15 42 15 42C15 42 30 26.25 30 15C30 6.71573 23.2843 0 15 0Z"
                                  fill="#0284c7" stroke="#ffffff" stroke-width="2.2"/>
                            <circle cx="15" cy="15" r="7.5" fill="#ffffff"/>
                            <circle cx="15" cy="15" r="4.5" fill="#0284c7"/>
                        </svg>
                        <div class="user-radar-wave"></div>
                    </div>
                `;
                return L.divIcon({{
                    className: 'leaflet-user-pin',
                    html: svg,
                    iconSize: [w, h],
                    iconAnchor: [w / 2, h],
                    popupAnchor: [0, -h + 4]
                }});
            }}

            var officerCoords = {officer_json};
            if (officerCoords && officerCoords.length === 2) {{
                var uLat = officerCoords[0];
                var uLon = officerCoords[1];
                var userMarker = L.marker([uLat, uLon], {{ icon: createUserPinIcon(), zIndexOffset: 1000 }}).addTo(map);
                userMarker.bindPopup(`
                    <div style="font-size:13px; line-height:1.45; min-width:180px; padding:3px;">
                        <b style="color:#0284c7; font-size:14px;">📍 Your Location (Officer)</b><br/>
                        <code>${{uLat.toFixed(5)}}° N, ${{uLon.toFixed(5)}}° E</code>
                    </div>
                `);
                userMarker.bindTooltip("<b>📍 Your Real Location</b>", {{
                    direction: 'top',
                    offset: [0, -48]
                }});

                if (isFocused && complaintsData.length > 0) {{
                    var targetLat = complaintsData[0].lat;
                    var targetLon = complaintsData[0].lon;
                    L.polyline([[uLat, uLon], [targetLat, targetLon]], {{
                        color: '#0284c7',
                        weight: 4,
                        dashArray: '8, 8',
                        opacity: 0.95
                    }}).addTo(map);
                    map.fitBounds([[uLat, uLon], [targetLat, targetLon]], {{padding: [55, 55], maxZoom: 16}});
                }} else {{
                    bounds.push([uLat, uLon]);
                    if (bounds.length === 1) {{
                        map.setView([uLat, uLon], 15);
                    }} else {{
                        map.fitBounds(bounds, {{padding: [50, 50], maxZoom: 15}});
                    }}
                }}
            }}

            </script>
    </body>
    </html>
    """

    components.html(html_code, height=540)

    # Map Legend
    if officer_loc:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown("🔴 **Critical (P1)**: Manholes / hazards")
        c2.markdown("🟠 **High (P2)**: Potholes / road floods")
        c3.markdown("🟡 **Medium (P3)**: Streetlights / leaks")
        c4.markdown("🟢 **Low (P4)**: Signs / minor debris")
        c5.markdown("🔵 **Live Pin**: Your GPS Location")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown("🔴 **Critical (P1)**: Manholes / hazards")
        c2.markdown("🟠 **High (P2)**: Potholes / road floods")
        c3.markdown("🟡 **Medium (P3)**: Streetlights / leaks")
        c4.markdown("🟢 **Low (P4)**: Signs / minor debris")


# ==========================================
# PAGE 1: CITIZEN INTAKE PORTAL
# ==========================================
def render_workflow_stepper(current_step: int):
    """Render a modern visual stepper bar for the 4-stage citizen intake workflow."""
    steps = [
        (1, "📷 Photo Evidence", "Upload, live camera, or tested preset"),
        (2, "📝 Description & Voice", "Multilingual text or live mic speech"),
        (3, "📍 Location Intelligence", "GPS detection & pre-submission review"),
        (4, "🎫 Municipal Ticket", "Multimodal AI analysis & work order"),
    ]
    
    cards_html = []
    for num, title, subtitle in steps:
        if current_step == num:
            cls = "step-card active"
            badge = f'<span class="step-num">{num}</span>'
        elif current_step > num:
            cls = "step-card completed"
            badge = '<span class="step-num">✓</span>'
        else:
            cls = "step-card pending"
            badge = f'<span class="step-num">{num}</span>'
        card = (
            f'<div class="{cls}">'
            f'{badge}'
            f'<div>'
            f'<div style="font-size:0.92rem; font-weight:700;">{title}</div>'
            f'<div style="font-size:0.75rem; opacity:0.85;">{subtitle}</div>'
            f'</div>'
            f'</div>'
        )
        cards_html.append(card)
        
    full_html = f'<div class="workflow-stepper">{"".join(cards_html)}</div>'
    st.markdown(full_html, unsafe_allow_html=True)


# ==========================================
# PAGE 1: CITIZEN INTAKE PORTAL
# ==========================================
def show_citizen_intake():
    st.markdown('<div id="workflow-top-anchor"></div>', unsafe_allow_html=True)
    st.title("🏙️ Citizen Complaint Intake Portal")
    st.caption("AI-Powered Urban Defect Registration & Live Multimodal Work Order Pipeline")

    # Workflow Stage Tracker (1: Photo, 2: Description, 3: Location, 4: Ticket)
    if "current_intake_step" not in st.session_state:
        st.session_state["current_intake_step"] = 1
    
    current_step = st.session_state["current_intake_step"]

    # Persistent Complaint Form State Initialization (immune to widget lifecycle deletion)
    if "complaint_description" not in st.session_state:
        st.session_state["complaint_description"] = ""
    if "complaint_landmark" not in st.session_state:
        st.session_state["complaint_landmark"] = ""
    if "complaint_lat" not in st.session_state:
        st.session_state["complaint_lat"] = 0.0
    if "complaint_lon" not in st.session_state:
        st.session_state["complaint_lon"] = 0.0
    if "complaint_photo" not in st.session_state:
        st.session_state["complaint_photo"] = None
    if "complaint_category" not in st.session_state:
        st.session_state["complaint_category"] = "🤖 Auto-Detect from Description (AI)"
    if "complaint_input_mode" not in st.session_state:
        st.session_state["complaint_input_mode"] = "Type text"

    # Visual Workflow Stepper Bar
    render_workflow_stepper(current_step)

    # Interactive Step Switcher Pills
    step_nav1, step_nav2, step_nav3, step_nav4 = st.columns(4)
    with step_nav1:
        if st.button("1️⃣ Step 1: Photo", key="btn_nav_step_1", use_container_width=True, type="primary" if current_step == 1 else "secondary"):
            st.session_state["current_intake_step"] = 1
            auto_scroll_to("workflow-top-anchor")
            st.rerun()
    with step_nav2:
        if st.button("2️⃣ Step 2: Description", key="btn_nav_step_2", use_container_width=True, type="primary" if current_step == 2 else "secondary"):
            st.session_state["current_intake_step"] = 2
            auto_scroll_to("workflow-top-anchor")
            st.rerun()
    with step_nav3:
        if st.button("3️⃣ Step 3: Location", key="btn_nav_step_3", use_container_width=True, type="primary" if current_step == 3 else "secondary"):
            st.session_state["current_intake_step"] = 3
            auto_scroll_to("workflow-top-anchor")
            st.rerun()
    with step_nav4:
        ticket_ready = "last_fusion_result" in st.session_state
        if st.button("4️⃣ Step 4: Ticket" if ticket_ready else "4️⃣ Step 4: Pending", key="btn_nav_step_4", use_container_width=True, disabled=not ticket_ready, type="primary" if current_step == 4 else "secondary"):
            st.session_state["current_intake_step"] = 4
            auto_scroll_to("workflow-step4-anchor")
            st.rerun()

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────
    # STEP 1: PHOTO OF THE PROBLEM
    # ─────────────────────────────────────────────────────────────
    if current_step == 1:
        st.markdown(clean_html("""
        <div class="intake-box">
            <h3 style="margin-top:0; color:#1d4ed8; display:flex; align-items:center; gap:8px;">
                <span>📷 Step 1: Photographic Evidence</span>
            </h3>
            <p style="color:#64748b; font-size:0.92rem; margin-bottom:12px;">
                Upload a photo, capture a live picture using your device camera, or select one of the tested demo complaints to test live duplicate clustering.
            </p>
        </div>
        """), unsafe_allow_html=True)

        # Quick Test Complaint Presets for Duplicate Demonstration
        st.markdown("##### 📋 Tested Complaint Presets (Quick Duplicate Detection Demo)")
        st.caption("Select any previously registered complaint to auto-fill its photo, description, and GPS coordinates to test AI duplicate clustering:")

        preset_names = list(DEMO_PRESETS.keys())
        preset_cols = st.columns(3)
        for i, name in enumerate(preset_names):
            col = preset_cols[i % 3]
            p_info = DEMO_PRESETS[name]
            btn_label = f"📋 {p_info['id']}: {p_info['title']}"
            if col.button(btn_label, key=f"demo_preset_btn_{i}", use_container_width=True):
                st.session_state["active_preset"] = name
                st.session_state["preset_manually_selected"] = True
                
                # Update persistent state (NEVER wiped by Streamlit)
                st.session_state["complaint_preset_id"] = p_info["id"]
                st.session_state["complaint_landmark"] = p_info["loc"]
                st.session_state["complaint_lat"] = float(p_info["lat"])
                st.session_state["complaint_lon"] = float(p_info["lon"])
                st.session_state["complaint_description"] = p_info.get("sample_text", "")
                st.session_state["complaint_input_mode"] = "Type text"
                st.session_state["location_badge"] = f"📋 Test Complaint Loaded: **{p_info['id']} — {p_info['title']}** (Photo, text & GPS auto-filled)"
                
                # Pre-sync widget states if widgets exist
                st.session_state["widget_complaint_text"] = p_info.get("sample_text", "")
                st.session_state["widget_loc_name"] = p_info["loc"]
                st.session_state["widget_lat"] = float(p_info["lat"])
                st.session_state["widget_lon"] = float(p_info["lon"])

                # Load preset photo
                photo_p = p_info.get("sample_photo")
                if photo_p and Path(photo_p).exists():
                    try:
                        loaded_img = Image.open(photo_p).convert("RGB")
                        st.session_state["complaint_photo"] = loaded_img
                        st.session_state["workflow_uploaded_image"] = loaded_img
                    except Exception:
                        pass
                st.rerun()

        if st.session_state.get("preset_manually_selected"):
            if st.button("🔄 Clear Selected Preset (Start Blank)", key="clear_preset_btn", use_container_width=True):
                st.session_state["active_preset"] = None
                st.session_state["preset_manually_selected"] = False
                st.session_state["complaint_preset_id"] = None
                st.session_state["complaint_landmark"] = ""
                st.session_state["complaint_lat"] = 0.0
                st.session_state["complaint_lon"] = 0.0
                st.session_state["complaint_description"] = ""
                st.session_state["complaint_photo"] = None
                st.session_state["workflow_uploaded_image"] = None
                st.session_state.pop("location_badge", None)
                st.session_state.pop("widget_complaint_text", None)
                st.session_state.pop("widget_loc_name", None)
                st.session_state.pop("widget_lat", None)
                st.session_state.pop("widget_lon", None)
                st.rerun()

        st.divider()

        # Photo Input Source (Upload or Camera)
        photo_source = st.radio("Photo input method:", ["Upload file", "Capture camera"], horizontal=True, key="photo_source_choice")
        uploaded_image = st.session_state.get("complaint_photo") or st.session_state.get("workflow_uploaded_image")
        photo_lat, photo_lon = None, None

        if photo_source == "Upload file":
            file_buffer = st.file_uploader("Upload issue photograph", type=["jpg", "jpeg", "png", "webp"], key="citizen_file_uploader")
            if file_buffer:
                raw_img = Image.open(BytesIO(file_buffer.getvalue()))
                photo_lat, photo_lon = extract_exif_gps(raw_img)
                uploaded_image = raw_img.convert("RGB")
                st.session_state["complaint_photo"] = uploaded_image
                st.session_state["workflow_uploaded_image"] = uploaded_image
        else:
            camera_buffer = st.camera_input("Capture live photo", key="citizen_cam_input")
            if camera_buffer:
                raw_img = Image.open(BytesIO(camera_buffer.getvalue()))
                photo_lat, photo_lon = extract_exif_gps(raw_img)
                uploaded_image = raw_img.convert("RGB")
                st.session_state["complaint_photo"] = uploaded_image
                st.session_state["workflow_uploaded_image"] = uploaded_image

        # Display preview of active photo
        if uploaded_image:
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            active_p_title = st.session_state.get("active_preset", "Incident Evidence")
            col_img1, col_img2 = st.columns([1.5, 1], gap="medium")
            with col_img1:
                st.image(uploaded_image, caption=f"📷 Photo Evidence ({active_p_title})", use_container_width=True)
            with col_img2:
                st.markdown(clean_html("""
                <div style="background:rgba(29, 78, 216, 0.08); border:1.5px solid rgba(37, 99, 235, 0.25);
                     border-radius:12px; padding:16px;">
                    <b style="color:#1d4ed8;">✅ Photo Evidence Attached</b>
                    <p style="font-size:0.85rem; color:#64748b; margin-top:6px; margin-bottom:12px;">
                        The YOLO-World municipal defect detector will scan this image for potholes, garbage heaps, open drains, and streetlights upon submission.
                    </p>
                </div>
                """), unsafe_allow_html=True)
                if st.button("🗑️ Remove / Replace Photo", key="btn_remove_photo", use_container_width=True):
                    st.session_state["complaint_photo"] = None
                    st.session_state["workflow_uploaded_image"] = None
                    st.rerun()

            if photo_lat is not None and photo_lon is not None:
                st.session_state["complaint_lat"] = round(float(photo_lat), 5)
                st.session_state["complaint_lon"] = round(float(photo_lon), 5)
                st.session_state["complaint_landmark"] = f"Photo Location ({photo_lat:.4f}, {photo_lon:.4f})"
                st.session_state["location_badge"] = f"📍 **GPS Extracted from Photo EXIF:** `{photo_lat:.5f}° N, {photo_lon:.5f}° E` (Auto-filled into Step 3!)"
                st.success(f"📍 **GPS Extracted from Photo EXIF Metadata:** `{photo_lat:.5f}° N, {photo_lon:.5f}° E` (Auto-filled in Step 3!)")
        else:
            st.info("ℹ️ No photo attached yet. You can attach a photo above or proceed directly with a text or voice complaint.")

        st.divider()

        # Step 1 Navigation Button to Step 2
        col_next1, col_next2 = st.columns([1, 1])
        with col_next2:
            if st.button("Continue to Step 2: Problem Description ➡️", type="primary", use_container_width=True, key="btn_to_step_2"):
                st.session_state["current_intake_step"] = 2
                auto_scroll_to("workflow-top-anchor")
                st.rerun()

    # ─────────────────────────────────────────────────────────────
    # STEP 2: LANGUAGE & DESCRIPTION / VOICE
    # ─────────────────────────────────────────────────────────────
    elif current_step == 2:
        st.markdown(clean_html("""
        <div class="intake-box">
            <h3 style="margin-top:0; color:#1d4ed8; display:flex; align-items:center; gap:8px;">
                <span>📝 Step 2: Problem Description & Multilingual Voice</span>
            </h3>
            <p style="color:#64748b; font-size:0.92rem; margin-bottom:12px;">
                Describe what happened in your preferred language. You can type freely or speak directly into your microphone.
            </p>
        </div>
        """), unsafe_allow_html=True)

        citizen_lang = st.radio(
            "Select Complaint Language / भाषा चुनें / భాషను ఎంచుకోండి:",
            LANGUAGE_CHOICES,
            index=0,
            horizontal=True,
            key="citizen_language_choice"
        )
        active_lang_code = normalize_lang_code(citizen_lang)
        active_speech_code = get_language_speech_code(active_lang_code)

        # Dynamic cross-lingual sentence translation on language switch
        prev_lang_code = st.session_state.get("prev_citizen_lang_code")
        if prev_lang_code is not None and prev_lang_code != active_lang_code:
            current_typed = st.session_state.get("complaint_description", "").strip()
            if current_typed:
                with st.spinner(f"Translating complaint to {citizen_lang}..."):
                    trans_typed, _ = translate_text(current_typed, source_lang=prev_lang_code, target_lang=active_lang_code)
                    if trans_typed and trans_typed.strip():
                        st.session_state["complaint_description"] = trans_typed
                        st.session_state["widget_complaint_text"] = trans_typed
                        st.session_state["widget_mic_text"] = trans_typed

        st.session_state["prev_citizen_lang_code"] = active_lang_code

        # Choose input mode
        input_modes = ["Type text", "🎤 Speak into Microphone (Live Voice-to-Text)", "📁 Audio File / Demo Voice Preset"]
        cur_mode = st.session_state.get("complaint_input_mode", "Type text")
        mode_idx = input_modes.index(cur_mode) if cur_mode in input_modes else 0
        input_type = st.radio(
            "Description input mode:",
            input_modes,
            index=mode_idx,
            horizontal=True,
            key="description_input_type",
        )
        st.session_state["complaint_input_mode"] = input_type

        final_text = ""
        if input_type == "Type text":
            if active_lang_code == "hi":
                sample_placeholder = "अपनी समस्या का विवरण यहाँ लिखें (उदा. 'सड़क पर बहुत सारा कचरा पड़ा हुआ है, बदबू आ रही है...')"
            elif active_lang_code == "te":
                sample_placeholder = "మీ సమస్యను ఇక్కడ రాయండి (ఉదా. 'రోడ్డుపై చాలా చెత్త పేరుకుపోయింది, దుర్వాసన వస్తోంది...')"
            else:
                sample_placeholder = "Describe what you see, e.g., 'Large dangerous pothole or overflowing waste on the road...'"

            # Sync persistent description into widget state if missing or updated
            if "widget_complaint_text" not in st.session_state:
                st.session_state["widget_complaint_text"] = st.session_state.get("complaint_description", "")

            final_text = st.text_area(
                f"Describe the civic problem ({citizen_lang}):",
                key="widget_complaint_text",
                height=110,
                placeholder=sample_placeholder,
                help=f"Type directly in {citizen_lang} or paste your description. Switching languages translates your text automatically into real sentences."
            )
            # Update persistent storage on every keystroke/rerun
            st.session_state["complaint_description"] = final_text

        elif input_type == "🎤 Speak into Microphone (Live Voice-to-Text)":
            if active_lang_code == "hi":
                st.markdown("🎙️ **लाइव माइक्रोफ़ोन (हिंदी में बोलें)**")
                st.caption("नीचे लाल रिकॉर्ड बटन दबाएं, हिंदी में अपनी शिकायत बोलें, और स्टॉप दबाएं। Google स्पीच इंजन आपकी आवाज़ को सीधे देवनागरी हिंदी में ट्रांसक्राइब करेगा!")
            elif active_lang_code == "te":
                st.markdown("🎙️ **లైవ్ మైక్రోఫోన్ (తెలుగులో మాట్లాడండి)**")
                st.caption("క్రింది రికార్డ్ బటన్ క్లిక్ చేసి తెలుగులో మాట్లాడండి. సిస్టమ్ నేరుగా తెలుగు లిపిలో రికార్డ్ చేసి టెక్స్ట్‌గా మారుస్తుంది!")
            else:
                st.markdown("🎙️ **Live Microphone (Speak in English)**")
                st.caption("Click the red record button below, speak in English, and click stop. The speech-to-text engine converts your speech into English text!")

            mic_audio = st.audio_input("Record your voice complaint:")
            
            if mic_audio is not None:
                audio_bytes = mic_audio.getvalue()
                audio_hash = hash(audio_bytes)
                if st.session_state.get("last_mic_audio_hash") != audio_hash:
                    with st.spinner(f"Transcribing your voice in {citizen_lang}..."):
                        transcribed_text, note = transcribe_audio(mic_audio, language_code=active_speech_code)
                        if transcribed_text:
                            st.session_state["complaint_description"] = transcribed_text
                            st.session_state["widget_mic_text"] = transcribed_text
                            st.session_state["last_mic_audio_hash"] = audio_hash
                            st.session_state["last_mic_lang"] = citizen_lang
                        else:
                            st.info(note)

            if "widget_mic_text" not in st.session_state:
                st.session_state["widget_mic_text"] = st.session_state.get("complaint_description", "")

            if st.session_state.get("complaint_description"):
                mic_lang_label = st.session_state.get("last_mic_lang", citizen_lang)
                st.success(f"**Current Voice Description ({mic_lang_label}):** {st.session_state['complaint_description']}")

            final_text = st.text_area(
                f"Transcribed Complaint Text ({citizen_lang}) — NLP will analyze this:",
                key="widget_mic_text",
                height=90,
                placeholder=f"Your spoken words in {citizen_lang} will appear here automatically. You can edit them before submitting...",
                help="You can review, edit, or add details to the transcribed text before submitting."
            )
            st.session_state["complaint_description"] = final_text

        else:
            st.info(f"📁 **Audio File or Sample Recording ({citizen_lang})**")
            available_samples = get_sample_voice_complaints(active_lang_code)
            voice_choice = st.selectbox(
                f"Select a sample citizen voice recording ({citizen_lang}) or upload an audio file:",
                ["Upload custom audio file (.wav/.mp3)"] + [item["title"] for item in available_samples]
            )
            if voice_choice == "Upload custom audio file (.wav/.mp3)":
                audio_file = st.file_uploader("Upload audio recording", type=["wav", "mp3", "m4a"])
                if audio_file:
                    st.audio(audio_file)
                    text, note = transcribe_audio(audio_file, language_code=active_speech_code)
                    st.caption(note)
                    if text:
                        st.session_state["complaint_description"] = text
                    final_text = st.text_area(f"Transcribed Complaint Text ({citizen_lang}):", value=st.session_state.get("complaint_description", ""), height=90)
                    st.session_state["complaint_description"] = final_text
            else:
                selected_voice = next((v for v in available_samples if v["title"] == voice_choice), available_samples[0])
                st.session_state["complaint_description"] = selected_voice["transcript"]
                st.success(f"**Loaded Sample Text ({selected_voice.get('language_name', citizen_lang)}):** '{selected_voice['transcript']}'")
                final_text = selected_voice["transcript"]

        # Optional explicit issue category selector (prioritized if selected)
        category_options = [
            "🤖 Auto-Detect from Description (AI)",
            "Pothole / Road Surface Damage (Roads)",
            "Garbage / Waste Accumulation (Sanitation)",
            "Open Drainage / Trench (Drainage)",
            "Drainage Blockage / Overflow (Sewage)",
            "Waterlogging / Flooding (Stormwater)",
            "Open / Damaged Manhole (Public Safety)",
            "Streetlight Outage / Electrical (Electrical)",
        ]
        cur_cat = st.session_state.get("complaint_category", category_options[0])
        cat_idx = category_options.index(cur_cat) if cur_cat in category_options else 0
        chosen_cat_option = st.selectbox(
            "🏷️ Issue Category (Optional — AI auto-detects from text if left default):",
            category_options,
            index=cat_idx,
            key="citizen_explicit_cat_choice",
            help="Leave as Auto-Detect to let CivicFusion NLP analyze your description, or explicitly select the civic problem."
        )
        st.session_state["complaint_category"] = chosen_cat_option

        st.divider()

        # Step 2 Navigation Buttons (Back & Continue)
        col_b1, col_b2 = st.columns([1, 1])
        with col_b1:
            if st.button("⬅️ Back to Step 1: Photo", use_container_width=True, key="btn_back_to_1"):
                st.session_state["current_intake_step"] = 1
                auto_scroll_to("workflow-top-anchor")
                st.rerun()
        with col_b2:
            if st.button("Continue to Step 3: Location Intelligence ➡️", type="primary", use_container_width=True, key="btn_to_step_3"):
                chk_text = st.session_state.get("complaint_description", "").strip()
                if not chk_text and "Auto-Detect" in chosen_cat_option:
                    st.warning("⚠️ Please provide a text/voice description or select an issue category before continuing.")
                else:
                    st.session_state["current_intake_step"] = 3
                    auto_scroll_to("workflow-top-anchor")
                    st.rerun()

    # ─────────────────────────────────────────────────────────────
    # STEP 3: LOCATION INTELLIGENCE & VERIFICATION
    # ─────────────────────────────────────────────────────────────
    elif current_step == 3:
        st.markdown(clean_html("""
        <div class="intake-box">
            <h3 style="margin-top:0; color:#1d4ed8; display:flex; align-items:center; gap:8px;">
                <span>📍 Step 3: Location Intelligence & Pre-Submission Review</span>
            </h3>
            <p style="color:#64748b; font-size:0.92rem; margin-bottom:12px;">
                Verify incident coordinates using device GPS or specify a landmark name. Review your complete complaint submission before dispatching to the municipality.
            </p>
        </div>
        """), unsafe_allow_html=True)

        st.markdown("##### 📍 Real-Time Location Auto-Detection")

        # High-Precision Device GPS (via custom component bridge)
        citizen_gps = gps_location_button(
            last_status=st.session_state.get("citizen_gps_status_msg", ""),
            key="citizen_gps_comp"
        )
        if citizen_gps and isinstance(citizen_gps, dict):
            if "latitude" in citizen_gps and "longitude" in citizen_gps:
                cg_lat = round(float(citizen_gps["latitude"]), 5)
                cg_lon = round(float(citizen_gps["longitude"]), 5)
                cg_acc = round(float(citizen_gps.get("accuracy", 0)))
                cg_ts = citizen_gps.get("timestamp")
                if st.session_state.get("last_citizen_gps_ts") != cg_ts:
                    st.session_state["last_citizen_gps_ts"] = cg_ts
                    st.session_state["complaint_lat"] = cg_lat
                    st.session_state["complaint_lon"] = cg_lon
                    st.session_state["widget_lat"] = cg_lat
                    st.session_state["widget_lon"] = cg_lon
                    st.session_state["citizen_gps_status_msg"] = f"✓ Got GPS: {cg_lat:.5f}N, {cg_lon:.5f}E (±{cg_acc}m). Filling inputs now..."
                    st.session_state["location_badge"] = f"🎯 Live Device GPS Captured: `{cg_lat:.5f}° N, {cg_lon:.5f}° E` (±{cg_acc}m accuracy)"
                    st.rerun()
            elif "error" in citizen_gps and citizen_gps.get("error"):
                st.error(f"❌ {citizen_gps['error']}")

        if "location_badge" in st.session_state:
            st.success(st.session_state["location_badge"])

        # Sync persistent landmark & coordinates into widgets if not already bound
        if "widget_loc_name" not in st.session_state:
            st.session_state["widget_loc_name"] = st.session_state.get("complaint_landmark", "")
        if "widget_lat" not in st.session_state:
            st.session_state["widget_lat"] = float(st.session_state.get("complaint_lat", 0.0))
        if "widget_lon" not in st.session_state:
            st.session_state["widget_lon"] = float(st.session_state.get("complaint_lon", 0.0))

        loc_name = st.text_input("Location / Landmark Name:", key="widget_loc_name", placeholder="e.g. Main Street, Sector 4")
        st.session_state["complaint_landmark"] = loc_name

        c_lat, c_lon = st.columns(2)
        latitude = c_lat.number_input("Latitude:", format="%.5f", key="widget_lat")
        longitude = c_lon.number_input("Longitude:", format="%.5f", key="widget_lon")
        st.session_state["complaint_lat"] = latitude
        st.session_state["complaint_lon"] = longitude

        st.caption("Coordinates represent the exact physical incident location on the municipal GIS map.")

        # Pre-Submission Review Summary Card
        cur_img = st.session_state.get("complaint_photo") or st.session_state.get("workflow_uploaded_image")
        cur_desc = st.session_state.get("complaint_description", "").strip()
        cur_cat = st.session_state.get("complaint_category", "🤖 Auto-Detect from Description (AI)")
        cur_lang = st.session_state.get("citizen_language_choice", "English")

        st.markdown(clean_html(f"""
        <div class="review-summary-card">
            <h4 style="margin-top:0; color:#059669; display:flex; align-items:center; gap:8px;">
                <span>📋 Pre-Submission Incident Summary</span>
            </h4>
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:12px; font-size:0.88rem; margin-top:8px;">
                <div><b>📷 Evidence:</b> {'Attached (Image loaded)' if cur_img is not None else 'None (Text-only report)'}</div>
                <div><b>🌐 Language:</b> {cur_lang}</div>
                <div><b>🏷️ Category:</b> {cur_cat.split(' (')[0]}</div>
                <div><b>📍 Coordinates:</b> {f'{latitude:.5f}° N, {longitude:.5f}° E' if (latitude != 0.0 or longitude != 0.0) else 'Pending / Unset'}</div>
            </div>
            <div style="margin-top:10px; font-size:0.88rem; color:var(--civic-text-body, #1e293b); border-top:1px dashed rgba(5, 150, 105, 0.3); padding-top:8px;">
                <b>📝 Description:</b> <em>"{cur_desc[:140] + ('...' if len(cur_desc) > 140 else '') if cur_desc else '(None provided)'}"</em>
            </div>
        </div>
        """), unsafe_allow_html=True)

        st.divider()

        # Step 3 Navigation Buttons (Back & Submit)
        col_sub1, col_sub2 = st.columns([1, 1])
        with col_sub1:
            if st.button("⬅️ Back to Step 2: Description", use_container_width=True, key="btn_back_to_2"):
                st.session_state["current_intake_step"] = 2
                auto_scroll_to("workflow-top-anchor")
                st.rerun()
        with col_sub2:
            submit_clicked = st.button("🚀 Process & Generate Municipal Ticket", type="primary", use_container_width=True, key="btn_submit_intake")

        if submit_clicked:
            final_text = cur_desc
            chosen_cat_option = cur_cat
            explicit_category_key = None
            if "Pothole" in chosen_cat_option:
                explicit_category_key = "road_damage"
            elif "Garbage" in chosen_cat_option:
                explicit_category_key = "waste"
            elif "Open Drainage" in chosen_cat_option:
                explicit_category_key = "open_drain"
            elif "Drainage Blockage" in chosen_cat_option:
                explicit_category_key = "drain_blockage"
            elif "Waterlogging" in chosen_cat_option:
                explicit_category_key = "waterlogging"
            elif "Manhole" in chosen_cat_option:
                explicit_category_key = "open_manhole"
            elif "Streetlight" in chosen_cat_option:
                explicit_category_key = "streetlight"

            if not final_text.strip():
                if explicit_category_key:
                    clean_cat_title = chosen_cat_option.split(" (")[0].replace("🏷️", "").strip()
                    final_text = f"Citizen reported {clean_cat_title} at {loc_name or 'this location'}."
                else:
                    st.warning("Please provide a text or voice description, or select an issue category from the dropdown.")
                    return

            active_lang_code = normalize_lang_code(cur_lang)
            with st.spinner("AI Processing: Running YOLO Computer Vision, NLP Entity Extraction, & Duplicate Matching..."):
                # 1. Computer Vision Detection
                detections = []
                vision_msg = None
                uploaded_image = cur_img
                if uploaded_image is not None:
                    detections, vision_msg = detect(uploaded_image)
                
                # 2. Multilingual NLP Extraction
                nlp_model = get_nlp_model()
                nlp_results = analyze_text_full(
                    nlp_model,
                    final_text,
                    user_lang=active_lang_code,
                    explicit_category=explicit_category_key,
                )

                # 3. Duplicate Detection (cross-lingual semantic matching)
                dup_check = check_for_duplicates(
                    new_text=final_text,
                    new_lat=latitude,
                    new_lon=longitude,
                    new_category=nlp_results["category"],
                    existing_reports=st.session_state["complaints"],
                )

                # 4. Multimodal Fusion (Citizen Description & NLP as ground truth authority)
                fusion_result = fuse_and_recommend(
                    text_category=nlp_results["category"],
                    nlp_confidence=nlp_results["confidence"],
                    urgency_cues=nlp_results["urgency_cues"],
                    detections=detections,
                    location_name=loc_name,
                    latitude=latitude,
                    longitude=longitude,
                    impact_text=nlp_results["impact"],
                    duplicate_info=dup_check,
                    secondary_categories=nlp_results.get("secondary_categories", []),
                    resolution_note=nlp_results.get("resolution_note", ""),
                )

                # Duplicate clustering or New ticket creation
                if dup_check["is_duplicate"]:
                    escalation = "P1" if fusion_result["priority"] == "P1" else ("P2" if fusion_result["priority"] == "P2" else None)
                    st.session_state["complaints"] = increment_cluster_count(dup_check["parent_id"], escalation)
                    ticket_id_created = dup_check["parent_id"]
                else:
                    new_id = f"CF-{st.session_state['ticket_counter']}"
                    st.session_state["ticket_counter"] += 1
                    initial_status = "Registered"
                    
                    photo_file_path = None
                    annotated_file_path = None
                    if uploaded_image is not None:
                        upload_dir = Path("data/uploads")
                        upload_dir.mkdir(parents=True, exist_ok=True)
                        target_img = upload_dir / f"{new_id}.jpg"
                        try:
                            uploaded_image.save(target_img, "JPEG")
                            photo_file_path = str(target_img)
                            if detections:
                                ann_img = upload_dir / f"{new_id}_annotated.jpg"
                                vision.save_annotated_image(uploaded_image, detections, ann_img)
                                annotated_file_path = str(ann_img)
                        except Exception:
                            photo_file_path = None

                    city_for_entry = "City Zone"

                    new_entry = {
                        "id": new_id,
                        "description": final_text,
                        "original_language": active_lang_code,
                        "language_name": cur_lang,
                        "translated_description_en": nlp_results.get("translated_text_en", final_text),
                        "category_raw": nlp_results["category"],
                        "category_display": fusion_result["category_display"],
                        "secondary_categories": nlp_results.get("secondary_categories", []),
                        "decision_reason": fusion_result.get("decision_reason", ""),
                        "visual_evidence_summary": fusion_result.get("visual_evidence_summary", ""),
                        "problem_title": fusion_result["problem_title"],
                        "department": fusion_result["department"],
                        "location_name": loc_name.strip() if loc_name and loc_name.strip() else (f"GPS ({latitude:.4f}, {longitude:.4f})" if (latitude != 0.0 or longitude != 0.0) else f"{city_for_entry} (Address Pending)"),
                        "city": city_for_entry,
                        "latitude": latitude,
                        "longitude": longitude,
                        "severity": fusion_result["severity"],
                        "priority": fusion_result["priority"],
                        "status": initial_status,
                        "citizen_reports_count": 1,
                        "ai_confidence": round(fusion_result["composite_confidence"], 2),
                        "needs_review": False,
                        "reported_date": date.today().isoformat(),
                        "photo_path": photo_file_path,
                        "annotated_photo_path": annotated_file_path,
                        "detections": [
                            {
                                "class_name": d["class"],
                                "category": d.get("category", d["class"]),
                                "confidence": d["confidence"],
                                "confidence_percent": d.get("confidence_percent", round(d["confidence"] * 100, 1)),
                                "bbox": d["box"],
                            }
                            for d in detections
                        ],
                        "model_version": ACTIVE_MODEL_VERSION,
                    }
                    st.session_state["complaints"] = insert_complaint(new_entry)
                    ticket_id_created = new_id

                    if fusion_result["agreement"] is False and uploaded_image is not None:
                        try:
                            log_cv_nlp_mismatch(
                                ticket_id=new_id,
                                citizen_description=final_text,
                                nlp_category=nlp_results["category"],
                                nlp_confidence=nlp_results["confidence"],
                                cv_categories=fusion_result["visual_categories"],
                                cv_confidence=fusion_result["best_cv_conf"],
                                final_category=fusion_result["category_raw"],
                                photo_path=photo_file_path,
                                model_version=ACTIVE_MODEL_VERSION,
                                decision_reason=fusion_result.get("decision_reason", ""),
                                original_language=active_lang_code,
                            )
                        except Exception:
                            pass

                # Store all generation outputs in session state for Step 4 presentation
                st.session_state["last_fusion_result"] = fusion_result
                st.session_state["last_nlp_results"] = nlp_results
                st.session_state["last_dup_check"] = dup_check
                st.session_state["last_detections"] = detections
                st.session_state["last_ticket_id"] = ticket_id_created
                st.session_state["last_submitted_text"] = final_text
                st.session_state["last_submitted_lang"] = cur_lang
                st.session_state["last_submitted_lang_code"] = active_lang_code
                st.session_state["current_intake_step"] = 4
                st.session_state["should_scroll_step4"] = True
                st.rerun()

    # ─────────────────────────────────────────────────────────────
    # STEP 4: MUNICIPAL TICKET & MULTIMODAL AI INTELLIGENCE
    # ─────────────────────────────────────────────────────────────
    elif current_step == 4:
        if st.session_state.get("should_scroll_step4"):
            auto_scroll_to("workflow-step4-anchor")
            st.session_state["should_scroll_step4"] = False

        st.markdown('<div id="workflow-step4-anchor"></div>', unsafe_allow_html=True)

        fusion_result = st.session_state.get("last_fusion_result")
        nlp_results = st.session_state.get("last_nlp_results")
        dup_check = st.session_state.get("last_dup_check", {"is_duplicate": False})
        detections = st.session_state.get("last_detections", [])
        uploaded_image = st.session_state.get("complaint_photo") or st.session_state.get("workflow_uploaded_image")
        final_text = st.session_state.get("last_submitted_text", "")
        citizen_lang = st.session_state.get("last_submitted_lang", "English")
        active_lang_code = st.session_state.get("last_submitted_lang_code", "en")
        ticket_id = st.session_state.get("last_ticket_id", "CF-XXXX")

        if not fusion_result or not nlp_results:
            st.warning("⚠️ No recent ticket generated in this session. Please submit a complaint in Steps 1-3.")
            if st.button("⬅️ Return to Step 1", type="primary"):
                st.session_state["current_intake_step"] = 1
                st.rerun()
            return

        # Duplicate Banner or New Ticket Success Banner
        if dup_check.get("is_duplicate"):
            st.markdown(clean_html(f"""
            <div class="duplicate-pill">
              <b>⚠️ DUPLICATE COMPLAINT DETECTED & CLUSTERED:</b><br/>
              This issue matches active Ticket <b>{dup_check['parent_id']}</b> located only <b>{dup_check['distance_meters']} meters away</b> with <b>{dup_check['text_similarity']:.0%} semantic similarity</b>.<br/>
              The municipality has clustered your report to prevent redundant work orders and <b>escalated priority to {fusion_result['priority']}</b>!
            </div>
            """), unsafe_allow_html=True)
        else:
            st.markdown(clean_html(f"""
            <div style="background:linear-gradient(135deg, rgba(5,150,105,0.14), rgba(16,185,129,0.08));
                 border:1.5px solid #059669; border-radius:12px; padding:16px 20px; margin-bottom:16px;">
                <h4 style="margin:0; color:#059669;">✅ Unique Civic Problem Registered & Dispatched!</h4>
                <p style="margin:4px 0 0 0; color:#0f172a; font-weight:600;">
                    Municipal Work Order <b>{ticket_id}</b> created and routed to <code>{fusion_result['department']}</code>.
                </p>
            </div>
            """), unsafe_allow_html=True)

        # Modality Alignment & Decision Card
        cv_seen = ", ".join(
            DISPLAY_NAMES.get(v, v.replace("_", " ").title())
            for v in fusion_result.get("visual_categories", [])
        ) or "None Detected"
        nlp_said = DISPLAY_NAMES.get(nlp_results["category"], nlp_results["category"].replace("_", " ").title())
        best_cv_pct = f"{fusion_result['best_cv_conf']:.0%}" if fusion_result.get('best_cv_conf', 0) > 0 else "N/A"
        decision_msg = fusion_result.get("decision_reason", "")
        agreement = fusion_result.get("agreement")
        status_color = "#10b981" if agreement is True else ("#f97316" if agreement is False else "#3b82f6")
        status_icon = "🟢" if agreement is True else ("🟠" if agreement is False else "🔵")

        st.markdown(clean_html(f"""
        <div style="background:var(--civic-card-bg, #ffffff); border: 1.5px solid {status_color};
                    border-radius: 12px; padding: 14px 18px; margin-bottom: 20px; box-shadow: var(--civic-card-shadow, 0 4px 18px rgba(0,0,0,0.05));">
          <div style="font-weight: 700; color: {status_color}; font-size: 1.0rem;">
            {status_icon} Multimodal Dispatch Decision: {fusion_result.get('category_display', 'Municipal Issue')}
          </div>
          <div style="font-size: 0.88rem; color: var(--civic-text-sub, #475569); margin-top: 4px;">
            <b>Decision Rule:</b> {decision_msg}
          </div>
          <div style="display: flex; gap: 20px; margin-top: 10px; font-size: 0.84rem; flex-wrap: wrap; color: var(--civic-text-body, #1e293b);">
            <span>📷 <b>CV Detected:</b> {cv_seen} ({best_cv_pct})</span>
            <span>🧠 <b>NLP Extracted:</b> {nlp_said} ({nlp_results['confidence']:.0%})</span>
            <span>⚖️ <b>Authority Signal:</b> {fusion_result.get('authority_modality', 'NLP & Citizen Description')}</span>
            <span>🏛️ <b>Target Department:</b> {fusion_result['department']}</span>
          </div>
        </div>
        """), unsafe_allow_html=True)

        res_col1, res_col2 = st.columns([1.2, 1], gap="large")

        with res_col1:
            # 1. Computer Vision Analysis
            st.markdown("#### 📷 1. YOLO Computer Vision Analysis")
            if uploaded_image is not None:
                if detections:
                    st.success(f"🎯 **YOLO Detection:** Found {len(detections)} defect bounding box(es).")
                    annotated_img = draw_detections(uploaded_image, detections)
                    st.image(annotated_img, caption="YOLO Detected Defect Localization", use_container_width=True)
                    cv_label = ", ".join(f"{d['class']} ({d['confidence']:.0%})" for d in detections)
                    st.markdown(clean_html(f"""
                    <div style="background:rgba(249,115,22,0.10);border:1px solid #f97316;
                         border-radius:8px;padding:10px 14px;margin-top:8px;">
                      <b style="color:#f97316;">📷 CV Verdict:</b> <span style="font-weight:600;">{cv_label}</span>
                      &nbsp;|&nbsp; Visual Confidence: <b>{fusion_result['best_cv_conf']:.1%}</b>
                    </div>"""), unsafe_allow_html=True)
                else:
                    st.info(f"ℹ️ No municipal hazard detected above the {DEFAULT_CONFIDENCE_THRESHOLD:.0%} confidence threshold in this photo.")
                    st.image(uploaded_image, caption="Uploaded Issue Photograph", use_container_width=True)
                    st.markdown(clean_html("""
                    <div style="background:rgba(100,116,139,0.10);border:1px solid #64748b;
                         border-radius:8px;padding:10px 14px;margin-top:8px;">
                      <b>📷 CV Verdict:</b> No visual detection above threshold — NLP serves as the primary dispatch signal.
                    </div>"""), unsafe_allow_html=True)
            else:
                st.info("No photo uploaded — classification and dispatch performed by NLP alone.")

            st.divider()

            # 2. NLP Multilingual Text Analysis
            st.markdown("#### 🧠 2. NLP Multilingual Text Analysis")
            nlp_display = DISPLAY_NAMES.get(nlp_results["category"], nlp_results["category"].replace("_", " ").title())
            sec_display = ", ".join(DISPLAY_NAMES.get(s, s.replace("_", " ").title()) for s in nlp_results.get("secondary_categories", [])) or "None"
            
            st.markdown(clean_html(f"""
            <div style="background:rgba(59,130,246,0.10);border:1px solid #3b82f6;
                 border-radius:8px;padding:10px 14px;margin-bottom:10px;">
              <b style="color:#3b82f6;">🧠 NLP Verdict:</b> <span style="font-weight:600;">{nlp_display}</span>
              &nbsp;|&nbsp; Model Confidence: <b>{nlp_results['confidence']:.1%}</b>
            </div>"""), unsafe_allow_html=True)
            escaped_final = escape(final_text)
            st.markdown(f"**Original Citizen Description:** *'{escaped_final}'*")
            st.markdown(f"**Detected Language:** `{active_lang_code}` ({citizen_lang})")
            st.markdown(f"**Primary Category:** **{nlp_display}**")
            st.markdown(f"**Secondary Condition:** `{sec_display}`")
            st.markdown(f"**Extracted Impact:** {nlp_results.get('impact', 'None noted')}")
            st.markdown(f"**Urgency Cues:** {', '.join(nlp_results.get('urgency_cues', [])) if nlp_results.get('urgency_cues') else 'None detected'}")
            st.markdown(f"**Department Assigned:** `{fusion_result['department']}`")

        with res_col2:
            st.markdown("#### 📋 Final Generated Municipal Ticket")
            st.markdown(f'<div class="ticket-box">{fusion_result["ascii_card"]}</div>', unsafe_allow_html=True)

            st.markdown("#### ⚖️ Multimodal Decision Transparency")
            cv_conf_str = f"{fusion_result['best_cv_conf']:.1%}" if fusion_result.get('best_cv_conf', 0) > 0 else "N/A (Text-Driven)"
            st.dataframe(pd.DataFrame([
                {"Parameter": "Primary Category", "Value": fusion_result["category_display"]},
                {"Parameter": "Secondary Condition", "Value": sec_display},
                {"Parameter": "Severity Assessment", "Value": fusion_result["severity"]},
                {"Parameter": "Assigned Priority", "Value": fusion_result["priority"]},
                {"Parameter": "Assigned Department", "Value": fusion_result["department"]},
                {"Parameter": "NLP Confidence", "Value": f"{nlp_results['confidence']:.1%}"},
                {"Parameter": "CV Confidence", "Value": cv_conf_str},
                {"Parameter": "Composite AI Confidence", "Value": f"{fusion_result['composite_confidence']:.1%}"},
            ]), hide_index=True, use_container_width=True)

            for r in fusion_result.get("reasons", []):
                st.markdown(f"- {r}")

        st.divider()

        # Action bar to reset workflow or navigate
        act_col1, act_col2 = st.columns(2)
        with act_col1:
            if st.button("📝 File Another Complaint (Reset Workflow)", type="primary", use_container_width=True, key="btn_reset_workflow"):
                st.session_state["current_intake_step"] = 1
                st.session_state["complaint_preset_id"] = None
                st.session_state["active_preset"] = None
                st.session_state["complaint_landmark"] = ""
                st.session_state["complaint_lat"] = 0.0
                st.session_state["complaint_lon"] = 0.0
                st.session_state["complaint_description"] = ""
                st.session_state["complaint_photo"] = None
                st.session_state["workflow_uploaded_image"] = None
                st.session_state.pop("location_badge", None)
                st.session_state.pop("last_fusion_result", None)
                st.session_state.pop("widget_complaint_text", None)
                st.session_state.pop("widget_loc_name", None)
                st.session_state.pop("widget_lat", None)
                st.session_state.pop("widget_lon", None)
                auto_scroll_to("workflow-top-anchor")
                st.rerun()



# ==========================================
# PAGE 2: MUNICIPAL OFFICIAL DASHBOARD & MAP
# ==========================================
def show_municipal_dashboard():
    st.markdown('<div id="municipal-top-anchor"></div>', unsafe_allow_html=True)
    st.title("🏛️ Municipal Official Command Center")
    st.caption("AI-Powered Urban Complaint Management, Dispatcher & GIS Geographic Triage")

    complaints = st.session_state["complaints"]
    df = pd.DataFrame(complaints)

    # Top Section Switcher Navigation
    SECTION_CHOICES = [
        "🌐 Full Command Overview (All Sections)",
        "⚡ Smart Municipal Dispatcher & Action Center",
        "🗺️ City Hazard GIS Map & Triage",
        "📋 Active Work Orders Queue",
        "📊 Sector & Priority Analytics",
    ]
    selected_section = st.radio(
        "Municipal Command Section View:",
        SECTION_CHOICES,
        index=0,
        horizontal=True,
        key="officer_section_view",
    )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Top KPI Metrics Cards (rendered in all views for instant situational awareness)
    total = len(df)
    critical = int((df["priority"] == "P1").sum()) if "priority" in df.columns else 0
    high = int((df["priority"] == "P2").sum()) if "priority" in df.columns else 0
    medium = int((df["priority"] == "P3").sum()) if "priority" in df.columns else 0
    low = int((df["priority"] == "P4").sum()) if "priority" in df.columns else 0

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Total Complaints", total)
    kpi2.metric("🚨 Critical (P1)", critical)
    kpi3.metric("🔴 High (P2)", high)
    kpi4.metric("🟡 Medium (P3)", medium)
    kpi5.metric("🟢 Low (P4)", low)

    # Quick Jump Buttons when in Full Command Overview
    if selected_section == "🌐 Full Command Overview (All Sections)":
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.markdown("##### ⚡ Quick Section Navigation (Click to Smoothly Auto-Scroll):")
        j1, j2, j3, j4 = st.columns(4)
        with j1:
            if st.button("⚡ Jump to Dispatcher", key="jump_btn_dispatcher", use_container_width=True):
                auto_scroll_to("section-dispatcher")
        with j2:
            if st.button("🗺️ Jump to GIS Map", key="jump_btn_gis_map", use_container_width=True):
                auto_scroll_to("section-gis-map")
        with j3:
            if st.button("📋 Jump to Work Orders", key="jump_btn_work_orders", use_container_width=True):
                auto_scroll_to("section-work-orders")
        with j4:
            if st.button("📊 Jump to Analytics", key="jump_btn_analytics", use_container_width=True):
                auto_scroll_to("section-analytics")

    st.divider()

    # ─────────────────────────────────────────────────────────────
    # 1. SMART MUNICIPAL DISPATCHER & ACTION CENTER
    # ─────────────────────────────────────────────────────────────
    show_dispatcher = (selected_section in [
        "🌐 Full Command Overview (All Sections)",
        "⚡ Smart Municipal Dispatcher & Action Center"
    ])

    if show_dispatcher:
        st.markdown('<div id="section-dispatcher"></div>', unsafe_allow_html=True)
        st.subheader("⚡ Smart Municipal Dispatcher & Action Center")
        st.caption("Inspect live complaint evidence, translate citizen descriptions, and dispatch real-time status updates directly to storage.")

        ticket_ids = [c["id"] for c in st.session_state["complaints"]]
        if not ticket_ids:
            st.info("ℹ️ No active complaints currently registered in the municipal database.")
        else:
            col_t1, col_t2 = st.columns([1.6, 1], gap="large")
            selected_tid = col_t1.selectbox("Select Ticket ID to Inspect or Update Status:", ticket_ids, key="dispatcher_ticket_select")
            
            selected_ticket = next((c for c in st.session_state["complaints"] if c["id"] == selected_tid), None)
            if selected_ticket:
                with col_t1:
                    render_complaint_card(selected_ticket, is_officer=True)
                    render_officer_translation_tool(selected_ticket)
                    render_complaint_media_and_detections(selected_ticket, is_officer=True)
                
                with col_t2:
                    st.markdown(clean_html("""
                    <div style="background:linear-gradient(135deg, rgba(29, 78, 216, 0.08) 0%, rgba(5, 150, 105, 0.08) 100%);
                         border:1.5px solid rgba(37, 99, 235, 0.25); border-radius:12px; padding:18px;">
                        <h4 style="margin:0 0 10px 0; color:#1d4ed8;">🛠️ Dispatch Actions</h4>
                    """), unsafe_allow_html=True)

                    current_status = selected_ticket.get("status", "Open")
                    status_options = ["Registered", "Open", "In Progress", "Assigned", "Resolved", "Pending Review"]
                    default_idx = status_options.index(current_status) if current_status in status_options else 0
                    new_status = st.selectbox("Update Status:", status_options, index=default_idx, key=f"status_select_{selected_tid}")
                    
                    if st.button("💾 Save Status Update", key=f"save_status_{selected_tid}", use_container_width=True, type="primary"):
                        st.session_state["complaints"] = update_ticket_status(
                            selected_tid,
                            new_status=new_status,
                        )
                        st.success(f"✅ Ticket {selected_tid} status updated to {new_status} in persistent storage!")
                        st.rerun()

                    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                    
                    if st.button("🗑️ Delete Complaint Permanently", key=f"delete_btn_{selected_tid}", use_container_width=True):
                        st.session_state["complaints"] = delete_complaint(selected_tid)
                        st.toast(f"Complaint {selected_tid} deleted successfully!", icon="🗑️")
                        st.rerun()

                    st.markdown("</div>", unsafe_allow_html=True)

        if selected_section == "🌐 Full Command Overview (All Sections)":
            st.divider()

    # ─────────────────────────────────────────────────────────────
    # 2. CITY HAZARD GIS MAP & GEOGRAPHIC TRIAGE
    # ─────────────────────────────────────────────────────────────
    show_gis = (selected_section in [
        "🌐 Full Command Overview (All Sections)",
        "🗺️ City Hazard GIS Map & Triage"
    ])

    if show_gis:
        st.markdown('<div id="section-gis-map"></div>', unsafe_allow_html=True)
        st.subheader("🗺️ City Hazard GIS Map & Geographic Triage")
        st.caption("Scroll over the map with your mouse wheel to zoom in and out. Select or search a ticket below to focus only on that complaint's exact location.")

        map_ctrl1, map_ctrl2 = st.columns([2, 1.2])
        with map_ctrl1:
            all_option_label = "🌐 All Reported Areas (City Overview)"
            ticket_options = [all_option_label] + [
                f"{c['id']} — {c.get('problem_title', c.get('category_display'))} ({c.get('location_name', 'Location')})"
                for c in complaints
            ]
            chosen_ticket_label = st.selectbox(
                "🎯 Search / Focus on Complaint Location:",
                ticket_options,
                index=0,
                key="map_ticket_selector"
            )

        with map_ctrl2:
            direct_search_id = st.text_input(
                "🔍 Quick Search by Ticket ID:",
                placeholder="e.g. CF-1052",
                key="direct_search_box"
            ).strip().upper()

        focus_ticket_id = None
        if direct_search_id:
            match = next((c for c in complaints if str(c["id"]).upper() == direct_search_id), None)
            if match:
                focus_ticket_id = match["id"]
            else:
                st.caption(f"ℹ️ No complaint with ID '{direct_search_id}' found.")

        if not focus_ticket_id and chosen_ticket_label != all_option_label:
            focus_ticket_id = chosen_ticket_label.split(" — ")[0].strip()

        # User / Officer Live Location & GPS Coordinates
        officer_loc = None
        with st.expander("📍 Officer Live Position & GPS Distance Calculation", expanded=False):
            st.markdown(
                "Click **📍 Detect & Auto-Fill My GPS Location** to capture your device's real physical coordinates "
                "and display your position marker on the map in relation to civic complaints."
            )

            gps_data = gps_location_button(
                last_status=st.session_state.get("officer_gps_status_msg", ""),
                key="officer_gps_comp"
            )

            if gps_data and isinstance(gps_data, dict):
                if "latitude" in gps_data and "longitude" in gps_data:
                    ret_lat = round(float(gps_data["latitude"]), 5)
                    ret_lon = round(float(gps_data["longitude"]), 5)
                    acc = round(float(gps_data.get("accuracy", 0)))
                    ts = gps_data.get("timestamp")
                    if st.session_state.get("last_officer_gps_ts") != ts:
                        st.session_state["last_officer_gps_ts"] = ts
                        st.session_state["user_lat_input"] = ret_lat
                        st.session_state["user_lon_input"] = ret_lon
                        st.session_state["officer_gps_status_msg"] = f"✓ Got GPS: {ret_lat:.5f}N, {ret_lon:.5f}E (±{acc}m). Filling inputs now..."
                        st.rerun()
                elif "error" in gps_data and gps_data.get("error"):
                    st.error(f"❌ {gps_data['error']}")

            col_off1, col_off2 = st.columns(2)
            if "user_lat_input" not in st.session_state:
                st.session_state["user_lat_input"] = 0.0
            if "user_lon_input" not in st.session_state:
                st.session_state["user_lon_input"] = 0.0

            off_lat = col_off1.number_input("User Latitude:", format="%.5f", key="user_lat_input")
            off_lon = col_off2.number_input("User Longitude:", format="%.5f", key="user_lon_input")

            officer_loc = (off_lat, off_lon) if (off_lat != 0.0 or off_lon != 0.0) else None

            if focus_ticket_id and officer_loc:
                fc = next((c for c in complaints if c["id"] == focus_ticket_id), None)
                if fc:
                    dist_to_c = haversine_distance_meters(off_lat, off_lon, float(fc["latitude"]), float(fc["longitude"]))
                    st.info(f"📍 **Distance from Your Coordinates to Incident ({focus_ticket_id}):** `{dist_to_c:.0f} meters` (~`{max(1, int(dist_to_c/250))} min(s)` transit)")

        # Render Map (Isolated ticket view if focus_ticket_id is active)
        render_map(complaints, focus_ticket_id=focus_ticket_id, officer_loc=officer_loc)

        if focus_ticket_id:
            sel_c = next((c for c in complaints if c["id"] == focus_ticket_id), None)
            if sel_c:
                st.markdown(f"#### 📋 Incident Dossier & Geographic Evidence: `{sel_c['id']}`")
                dossier_col1, dossier_col2 = st.columns([1.4, 1], gap="large")
                with dossier_col1:
                    render_complaint_card(sel_c, is_officer=True)
                    render_officer_translation_tool(sel_c)
                with dossier_col2:
                    render_complaint_media_and_detections(sel_c, is_officer=True)

        if selected_section == "🌐 Full Command Overview (All Sections)":
            st.divider()

    # ─────────────────────────────────────────────────────────────
    # 3. ACTIVE MUNICIPAL WORK ORDERS QUEUE
    # ─────────────────────────────────────────────────────────────
    show_work_orders = (selected_section in [
        "🌐 Full Command Overview (All Sections)",
        "📋 Active Work Orders Queue"
    ])

    if show_work_orders:
        st.markdown('<div id="section-work-orders"></div>', unsafe_allow_html=True)
        st.subheader("📋 Active Municipal Work Orders Queue")
        st.caption("Filter and review all open, assigned, and resolved municipal work orders across jurisdictions.")

        f_col1, f_col2, f_col3 = st.columns(3)
        all_depts = sorted(df["department"].unique().tolist()) if "department" in df.columns else []
        dept_filter = f_col1.selectbox("Filter by Department:", ["All Departments"] + all_depts, key="queue_dept_filter")
        prio_filter = f_col2.selectbox("Filter by Priority:", ["All Priorities", "P1", "P2", "P3", "P4"], key="queue_prio_filter")
        status_filter = f_col3.selectbox("Filter by Status:", ["All Statuses", "Registered", "Open", "In Progress", "Assigned", "Resolved", "Pending Review"], key="queue_status_filter")

        filtered_df = df.copy()
        if dept_filter != "All Departments" and "department" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["department"] == dept_filter]
        if prio_filter != "All Priorities" and "priority" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["priority"] == prio_filter]
        if status_filter != "All Statuses" and "status" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["status"] == status_filter]

        avail_cols = [c for c in ["id", "problem_title", "location_name", "department", "severity", "priority", "status", "citizen_reports_count", "reported_date"] if c in filtered_df.columns]
        display_table = filtered_df[avail_cols].rename(columns={
            "id": "Ticket ID",
            "problem_title": "Identified Problem",
            "location_name": "Location",
            "department": "Assigned Department",
            "severity": "Severity",
            "priority": "Priority",
            "status": "Status",
            "citizen_reports_count": "Citizen Reports",
            "reported_date": "Reported",
        })

        st.dataframe(display_table, use_container_width=True, hide_index=True)

        if selected_section == "🌐 Full Command Overview (All Sections)":
            st.divider()

    # ─────────────────────────────────────────────────────────────
    # 4. SECTOR & PRIORITY ANALYTICS
    # ─────────────────────────────────────────────────────────────
    show_analytics = (selected_section in [
        "🌐 Full Command Overview (All Sections)",
        "📊 Sector & Priority Analytics"
    ])

    if show_analytics:
        st.markdown('<div id="section-analytics"></div>', unsafe_allow_html=True)
        st.subheader("📊 Sector & Priority Municipal Analytics")
        st.caption("Distribution of urban defects across municipal sectors and operational emergency triage levels.")

        col_chart1, col_chart2 = st.columns(2, gap="large")
        with col_chart1:
            st.markdown("##### 📊 Complaints by Sector / Category")
            if "category_display" in df.columns and len(df) > 0:
                cat_df = df["category_display"].value_counts().reset_index()
                cat_df.columns = ["Category", "Count"]
                max_cat = max(4, int(cat_df["Count"].max()) + 1)
                
                bars_cat = alt.Chart(cat_df).mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, color="#1d4ed8").encode(
                    y=alt.Y("Category:N", sort="-x", axis=alt.Axis(title="", labelFontSize=12)),
                    x=alt.X("Count:Q", scale=alt.Scale(domain=[0, max_cat]), axis=alt.Axis(tickMinStep=1, title="Number of Reports")),
                    tooltip=["Category", "Count"]
                )
                text_cat = bars_cat.mark_text(align="left", baseline="middle", dx=6, fontSize=12, fontWeight="bold").encode(
                    text="Count:Q"
                )
                chart_cat = (bars_cat + text_cat).properties(height=280)
                st.altair_chart(chart_cat, use_container_width=True)
            else:
                st.info("No complaint data available for sector chart.")

        with col_chart2:
            st.markdown("##### 🎯 Priority Distribution")
            if "priority" in df.columns and len(df) > 0:
                p_order = ["P1", "P2", "P3", "P4"]
                p_labels = {"P1": "P1 (Critical)", "P2": "P2 (High)", "P3": "P3 (Medium)", "P4": "P4 (Low)"}
                p_data = []
                for p in p_order:
                    cnt = int((df["priority"] == p).sum())
                    p_data.append({"Priority": p, "Label": p_labels[p], "Count": cnt})
                p_df = pd.DataFrame(p_data)
                max_p = max(4, int(p_df["Count"].max()) + 1)
                
                bars_p = alt.Chart(p_df).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
                    x=alt.X("Priority:N", sort=p_order, axis=alt.Axis(labelAngle=0, title="Priority Level", labelFontSize=12)),
                    y=alt.Y("Count:Q", scale=alt.Scale(domain=[0, max_p]), axis=alt.Axis(tickMinStep=1, title="Number of Reports")),
                    color=alt.Color("Priority:N", scale=alt.Scale(domain=p_order, range=["#ef4444", "#f97316", "#eab308", "#059669"]), legend=None),
                    tooltip=["Label", "Count"]
                )
                text_p = bars_p.mark_text(align="center", baseline="bottom", dy=-5, fontSize=13, fontWeight="bold").encode(
                    text="Count:Q"
                )
                chart_p = (bars_p + text_p).properties(height=280)
                st.altair_chart(chart_p, use_container_width=True)
            else:
                st.info("No complaint data available for priority chart.")


# ==========================================
# PAGE 3: SYSTEM ARCHITECTURE & EXPLAINER
# ==========================================
def show_architecture():
    st.title("🧠 System Architecture & AI Pipeline")
    st.markdown("""
    ### Why this is a Multimodal AI System
    Instead of relying purely on a generic form or isolated model, **CivicFusion AI** fuses three distinct sensory modalities:
    
    1. **📷 Visual Modality (Computer Vision - YOLO-World v2)**: Analyzes photos for physical municipal hazards (potholes, garbage, manholes, broken streetlights).
    2. **📝/🎤 Linguistic Modality (NLP & Speech-to-Text - TF-IDF / BERT & Whisper)**: Extracts semantic meaning, specific locations, urgency cues, and citizen impact from text or voice.
    3. **📍 Spatial Modality (Geographic Intelligence & Duplicate Detection)**: Uses Haversine distance and semantic embeddings to cluster duplicate reports and assign responsible municipal departments.
    """)

    st.divider()

    st.markdown("""
    #### 🔄 End-to-End Processing Flow
    ```
                           CITIZEN
                              │
                   ┌──────────┼──────────┐
                   ↓          ↓          ↓
                📷 IMAGE    🎤 VOICE    📍 GPS
                   │          │
                   ↓          ↓
                 YOLO       Whisper
                   │          │
                   ↓          ↓
             VISUAL INFO    TEXT
                   │          │
                   │        NLP/BERT
                   │          │
                   └─────┬────┘
                         ↓
                  MULTIMODAL AI
                         ↓
            ┌────────────┼────────────┐
            ↓            ↓            ↓
        Category      Severity     Duplicate
        Detection     Detection    Detection
            │            │            │
            └────────────┼────────────┘
                         ↓
                   PRIORITY ENGINE
                         ↓
                  Complaint Created
                         ↓
                   MUNICIPAL DASHBOARD
    ```
    """)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### 10 Core Architectural Components:
        - **1️⃣ Computer Vision (YOLO):** Object detection predicting 10 civic issue classes.
        - **2️⃣ NLP Information Extraction:** Detects problem class, landmark mentions, and citizen impact.
        - **3️⃣ Voice / Speech-to-Text:** Converts spoken voice into clean text descriptions.
        - **4️⃣ Location Intelligence:** Latitude & Longitude mapped to municipal GIS.
        - **5️⃣ Multimodal Combine:** Validates whether visual cues match linguistic descriptions.
        """)
    with col2:
        st.markdown("""
        - **6️⃣ Severity Prediction:** Categorizes into Low, Medium, High, and Critical.
        - **7️⃣ Automatic Routing:** Automatically maps issues to responsible municipal departments.
        - **8️⃣ Duplicate Clustering:** Spatial (Haversine < 200m) + Semantic Cosine similarity.
        - **9️⃣ Priority Engine (P1-P4):** Escalates based on safety risks and duplicate volume.
        - **🔟 Structured Municipal Ticket:** Generates standardized work orders with confidence scores.
        """)

    st.divider()
    if MODEL_PATH.exists():
        st.success(f"✅ Active Vision Model: **`{MODEL_PATH.name}`** ({ACTIVE_MODEL_VERSION}) — Trained multi-class detector for Potholes, Damaged Roads, Open Drains/Trenches, Open Manholes, Garbage, and Streetlights.")
    else:
        st.info("Vision model export available via `export_model.py`.")

    st.divider()
    st.subheader("🖼️ Real-World Training Dataset & AI Defect Demonstration Gallery")
    st.markdown(
        "Demonstrate the real-world municipal dataset examples and their ground-truth annotated bounding boxes. "
        "These examples show how CivicFusion AI identifies macro-level infrastructure defects while eliminating false garbage detections on background specks."
    )

    demo_dir = Path("data/demonstration_examples")
    demo_options = [
        "Category A: Road Pothole Cavity (user_pothole_diagnostic.jpg)",
        "Category A: Roadside Cavity & Pothole under Hedge (user_pothole_hedge.jpg)",
        "Category C: Open Drainage & Roadside Excavation Trench (user_roadside_trench_diagnostic.jpg)",
        "Category D: Solid Waste Accumulation (garbage_sample_01.jpg)",
        "Category A: Broken Asphalt & Road Deterioration (road_damage_sample_01.jpg)",
        "Category B: Open Manhole Chamber Opening (manhole_sample_01.jpg)",
        "Category E: Streetlight Infrastructure Outage (streetlight_sample_01.jpg)",
        "🔬 YOLO Training Batch 0 Mosaic (train_batch0.jpg)",
        "🔬 YOLO Training Batch 1 Mosaic (train_batch1.jpg)",
    ]

    selected_demo = st.selectbox("Select training example to demonstrate:", demo_options, key="arch_demo_select")

    demo_data_map = {
        "Category A: Road Pothole Cavity (user_pothole_diagnostic.jpg)": {
            "img": demo_dir / "demo_pothole.jpg",
            "category": "Road Infrastructure (Category A)",
            "hazard": "Pothole (Cavity in asphalt surface)",
            "coords": "[0.53, 0.45, 0.38, 0.50] (Center X: 53%, Center Y: 45%, Width: 38%, Height: 50%)",
            "dept": "Roads & Bridges Department (PWD)",
            "split": "train (Included in Training Set)",
            "note": "Encloses the true road depression cavity rather than falsely classifying tiny pebbles or surface litter as garbage.",
        },
        "Category A: Roadside Cavity & Pothole under Hedge (user_pothole_hedge.jpg)": {
            "img": demo_dir / "demo_pothole_hedge_annotated.jpg",
            "category": "Road Infrastructure (Category A)",
            "hazard": "Pothole Cavity & Road Ground Subsidence",
            "coords": "[0.60, 0.55, 0.28, 0.37] (Center X: 60%, Center Y: 55%, Width: 28%, Height: 37%)",
            "dept": "Road Maintenance Department",
            "split": "train (Included in Training Set)",
            "note": "Ground truth bounding box precisely isolating the cavity next to sidewalk hedges, completely suppressing garbage speck misclassifications.",
        },
        "Category C: Open Drainage & Roadside Excavation Trench (user_roadside_trench_diagnostic.jpg)": {
            "img": demo_dir / "demo_trench.jpg",
            "category": "Drainage & Sewerage (Category C)",
            "hazard": "Open Drain / Excavated Roadside Trench",
            "coords": "[0.46, 0.60, 0.55, 0.70] (Center X: 46%, Center Y: 60%, Width: 55%, Height: 70%)",
            "dept": "Water Supply & Sewerage Board",
            "split": "train (Included in Training Set)",
            "note": "Encloses the hazardous open trench alongside the road corridor, preventing minor bottles/litter from masking the primary hazard.",
        },
        "Category D: Solid Waste Accumulation (garbage_sample_01.jpg)": {
            "img": demo_dir / "demo_garbage.jpg",
            "category": "Solid Waste Management (Category D)",
            "hazard": "Garbage Heap / Municipal Refuse Pile",
            "coords": "[0.48, 0.65, 0.52, 0.40] (Center X: 48%, Center Y: 65%, Width: 52%, Height: 40%)",
            "dept": "Sanitation & Solid Waste Management",
            "split": "train (Included in Training Set)",
            "note": "Classified as macro-level municipal refuse pile requiring bulk collection crew dispatch.",
        },
        "Category A: Broken Asphalt & Road Deterioration (road_damage_sample_01.jpg)": {
            "img": demo_dir / "demo_road_damage.jpg",
            "category": "Road Infrastructure (Category A)",
            "hazard": "Damaged Road / Aggregate Erosion & Surface Cracks",
            "coords": "[0.50, 0.65, 0.85, 0.50] (Center X: 50%, Center Y: 65%, Width: 85%, Height: 50%)",
            "dept": "Roads & Bridges Department (PWD)",
            "split": "train (Included in Training Set)",
            "note": "Identifies extensive structural degradation across the travel lane.",
        },
        "Category B: Open Manhole Chamber Opening (manhole_sample_01.jpg)": {
            "img": demo_dir / "demo_manhole.jpg",
            "category": "Manhole Hazard (Category B)",
            "hazard": "Open Manhole / Missing Chamber Cover",
            "coords": "[0.50, 0.60, 0.42, 0.42] (Center X: 50%, Center Y: 60%, Width: 42%, Height: 42%)",
            "dept": "Water Supply & Sewerage Board",
            "split": "train (Included in Training Set)",
            "note": "High-priority safety hazard identifying missing manhole cover over municipal utility chamber.",
        },
        "Category E: Streetlight Infrastructure Outage (streetlight_sample_01.jpg)": {
            "img": demo_dir / "demo_streetlight.jpg",
            "category": "Electrical Infrastructure (Category E)",
            "hazard": "Damaged / Non-Functional Streetlight Pole",
            "coords": "[0.52, 0.48, 0.38, 0.82] (Center X: 52%, Center Y: 48%, Width: 38%, Height: 82%)",
            "dept": "Electrical & Street Lighting Department",
            "split": "train (Included in Training Set)",
            "note": "Identifies damaged light post infrastructure requiring municipal electrical crew inspection.",
        },
        "🔬 YOLO Training Batch 0 Mosaic (train_batch0.jpg)": {
            "img": demo_dir / "train_batch0.jpg",
            "category": "YOLO Batch 0 Ground Truth Mosaic",
            "hazard": "Multi-Image Training Mosaic",
            "coords": "Generated during Ultralytics YOLO training run",
            "dept": "AI Model Training Pipeline",
            "split": "train_batch0",
            "note": "Actual YOLO data loader training mosaic displaying real-world complaint images with bounding boxes.",
        },
        "🔬 YOLO Training Batch 1 Mosaic (train_batch1.jpg)": {
            "img": demo_dir / "train_batch1.jpg",
            "category": "YOLO Batch 1 Ground Truth Mosaic",
            "hazard": "Multi-Image Training Mosaic",
            "coords": "Generated during Ultralytics YOLO training run",
            "dept": "AI Model Training Pipeline",
            "split": "train_batch1",
            "note": "Actual YOLO data loader training mosaic displaying real-world complaint images with bounding boxes.",
        },
    }

    if selected_demo in demo_data_map:
        d_info = demo_data_map[selected_demo]
        col_img, col_info = st.columns([1.3, 1], gap="medium")
        with col_img:
            if d_info["img"].exists():
                st.image(Image.open(d_info["img"]), caption=selected_demo, use_container_width=True)
            else:
                st.warning(f"Demonstration image {d_info['img'].name} not found.")
        with col_info:
            st.markdown(f"#### 🏷️ {d_info['hazard']}")
            st.markdown(f"- **Civic Domain:** `{d_info['category']}`")
            st.markdown(f"- **Dataset Split:** `{d_info['split']}`")
            st.markdown(f"- **Assigned Dept:** `{d_info['dept']}`")
            st.markdown(f"- **Ground Truth Box:** `{d_info['coords']}`")
            st.markdown(f"**Diagnostic Analysis:**\n{d_info['note']}")


# ==========================================
# PAGE 4: MUNICIPAL OFFICER LOGIN
# ==========================================
def show_login_page():
    st.markdown(clean_html("""
    <div style="display:flex;flex-direction:column;align-items:center;margin-top:20px;">
      <h1 style="margin-bottom:4px;">🔐 Municipal Officer Login</h1>
      <p style="color:rgba(125,125,125,0.8);margin-bottom:18px;">
        This area is restricted to authorized municipal officials only.
      </p>
    </div>
    """), unsafe_allow_html=True)

    col_pad, col_form, col_pad2 = st.columns([1, 1.5, 1])
    with col_form:
        st.markdown("#### Sign in to access the Command Dashboard")
        st.caption("Citizens do not need an account — use the **Submit Complaint** page instead.")
        st.divider()

        username = st.text_input("👤 Username", placeholder="Enter your officer username")
        password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")

        if st.button("🔓 Login", type="primary", use_container_width=True):
            if login(st.session_state, username, password):
                st.success(f"✅ Welcome, **{username.strip()}**! Redirecting to Command Dashboard…")
                st.session_state["nav"] = "🏛️ Municipal Command Dashboard"
                st.rerun()
            else:
                st.error("❌ Invalid username or password. Please try again.")

    st.info("🔒 **Access Control:** Citizens cannot see the internal dashboard, complaint analytics, or status update tools.")


# ==========================================
# PAGE 5: TRACK MY COMPLAINT (Citizen)
# ==========================================
def show_track_complaint():
    st.title("🔍 Track My Complaint")
    st.markdown(
        "Enter your **Ticket ID** (e.g., `CF-1052`) to check the current status of your complaint. "
        "Only the information related to your own ticket is shown here."
    )

    st.divider()

    col_search, col_pad = st.columns([1.2, 2])
    with col_search:
        ticket_input = st.text_input(
            "Enter Ticket ID:",
            placeholder="CF-1052",
            help="Your ticket ID was shown when you submitted the complaint."
        )
        search_btn = st.button("🔎 Search", type="primary", use_container_width=True)

    if search_btn or (ticket_input and ticket_input.strip()):
        if not ticket_input.strip():
            st.warning("Please enter a Ticket ID.")
            return

        ticket_id = ticket_input.strip().upper()
        complaints = st.session_state.get("complaints", [])
        found = next((c for c in complaints if str(c.get("id", "")).strip().upper() == ticket_id), None)

        if found is None:
            st.error(f"❌ No complaint found with Ticket ID **{ticket_id}**. Please check the ID and try again.")
            return

        # Render structured complaint card (unified citizen/officer view)
        render_complaint_card(found, is_officer=is_officer(st.session_state))

        # Render photographic evidence and YOLO AI vision detections
        render_complaint_media_and_detections(found, is_officer=is_officer(st.session_state))

        # What does this status mean?
        theme = get_status_theme(found.get("status", "Open"))
        if theme.get("explanation"):
            st.info(f"**What this means:** {theme['explanation']}")


# ==========================================
# SIDEBAR NAVIGATION  (Role-Based)
# ==========================================
with st.sidebar:
    st.markdown("# 🏙️ CivicFusion AI")
    st.caption("Multimodal Municipal Intelligence System")
    st.divider()

    # --- Role badge ---
    if is_officer(st.session_state):
        officer_name = st.session_state.get("logged_in_as", "officer").title()
        st.markdown(f'<div class="role-badge-officer">🏛️ Officer: {officer_name}</div>', unsafe_allow_html=True)
        # Logout button
        if st.button("🔓 Logout", use_container_width=True):
            logout(st.session_state)
            st.session_state["nav"] = "📱 Submit a Complaint"
            st.rerun()
        st.divider()
        # Officer navigation
        citizen_pages = ["📱 Submit a Complaint", "🔍 Track My Complaint"]
        officer_pages = ["🏛️ Municipal Command Dashboard", "🧠 System Architecture & AI Pipeline"]
        st.caption("🌐 Citizen Portal")
        nav = st.radio("", citizen_pages + officer_pages, key="nav_radio_officer",
                       label_visibility="collapsed")
    else:
        st.markdown('<div class="role-badge-citizen">👤 Citizen</div>', unsafe_allow_html=True)
        st.divider()
        # Citizen-only navigation
        nav = st.radio(
            "",
            ["📱 Submit a Complaint", "🔍 Track My Complaint", "🔐 Municipal Officer Login"],
            key="nav_radio_citizen",
            label_visibility="collapsed"
        )

    st.divider()
    st.caption("Active Categories: 10 Civic Domains")
    st.caption("CV: YOLO Municipal Vision · NLP: TF-IDF + Logistic · Geo: PyDeck")


# ==========================================
# PAGE ROUTER
# ==========================================
if nav == "📱 Submit a Complaint":
    show_citizen_intake()
elif nav == "🔍 Track My Complaint":
    show_track_complaint()
elif nav == "🔐 Municipal Officer Login":
    show_login_page()
elif nav == "🏛️ Municipal Command Dashboard":
    # Hard gate — double-check even if someone bypasses the sidebar
    if not is_officer(st.session_state):
        st.error("🚫 Access Denied. This page is restricted to authorized municipal officers.")
        st.info("Please use the **🔐 Municipal Officer Login** page in the sidebar.")
        st.stop()
    show_municipal_dashboard()
elif nav == "🧠 System Architecture & AI Pipeline":
    if not is_officer(st.session_state):
        st.error("🚫 Access Denied. This page is restricted to authorized municipal officers.")
        st.info("Please use the **🔐 Municipal Officer Login** page in the sidebar.")
        st.stop()
    show_architecture()
