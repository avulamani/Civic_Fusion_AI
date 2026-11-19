# CivicFusion AI: Development & Delivery Roadmap

## System Vision
An AI-powered municipal intelligence platform transforming citizen-reported images, speech/text descriptions, and geospatial locations into automatically categorized, prioritized, and actionable municipal complaints.

---

## The 10 Core Pillars & Implementation Status

| # | Pillar | Status | File / Technology |
|---|---|---|---|
| **1** | **Computer Vision ("What can I see?")** | ✅ Implemented | `vision.py` · YOLO-World v2 (`municipal_worldv2.onnx`), 10 civic classes |
| **2** | **NLP ("What is the citizen saying?")** | ✅ Implemented | `nlp.py` · TF-IDF + Logistic Regression, entity & impact extraction |
| **3** | **Voice / Speech-to-Text** | ✅ Implemented | `speech_utils.py` · Whisper / STT integration & preset audio samples |
| **4** | **Location & GIS Mapping** | ✅ Implemented | `dashboard.py` · PyDeck map with severity-colored markers (🔴 🟠 🟡 🟢) |
| **5** | **Multimodal Combine** | ✅ Implemented | `fusion.py` · Vision + NLP + Location verification & agreement check |
| **6** | **Severity Prediction** | ✅ Implemented | `fusion.py` · Critical, High, Medium, Low risk tiers |
| **7** | **Automatic Routing** | ✅ Implemented | `fusion.py` · Mapped to 10 specific municipal departments |
| **8** | **Duplicate Detection Engine** | ✅ Implemented | `duplicates.py` · Haversine spatial distance (< 250m) + Semantic similarity |
| **9** | **Priority Engine (P1-P4)** | ✅ Implemented | `fusion.py` · Risk escalation & duplicate frequency boosting |
| **10**| **Structured AI Ticket & Dashboard**| ✅ Implemented | `dashboard.py` · Standardized Work Order card + Municipal Command Center |

---

## File Architecture

```text
CivicFusion_AI/
├── dashboard.py           # Streamlit Dual-Portal (Citizen Intake + Official Dashboard)
├── nlp.py                 # Text classification, entity extraction, impact analysis
├── vision.py              # ONNX Runtime YOLO-World v2 visual detection
├── vision_config.py       # 10 classes and zero-shot descriptive prompts
├── duplicates.py          # Spatial (Haversine) and semantic (TF-IDF) duplicate engine
├── speech_utils.py        # Voice-to-text audio processing and preset voice clips
├── fusion.py              # Multimodal fusion, priority engine, ASCII ticket generator
├── export_model.py        # YOLO-World v2 prompt binding and ONNX export tool
├── data/
│   ├── labeled_complaints.csv           # 1,800 synthetic training complaints
│   └── generate_synthetic_complaints.py # Dataset generation script
├── models/
│   ├── municipal_worldv2.onnx           # Exported 10-class YOLO-World v2 ONNX model
│   └── municipal_worldv2.classes.txt    # Target class labels
├── requirements.txt       # Production dependencies
├── render.yaml            # Render deployment configuration
└── README.md              # Complete system guide
```

---

## Presentation & Academic Walkthrough

When presenting this project to evaluators, focus on what makes it unique:
1. **Not a generic form:** Citizens don't have to manually select departments or guess severity; AI automates it.
2. **Not just computer vision:** It is truly multimodal—combining camera images, spoken voice / typed text, and GPS coordinates.
3. **Solves real municipal duplicate bottlenecks:** The system clusters multiple citizens reporting the same pothole or open manhole, preventing 50 redundant work orders and boosting urgency instead.
4. **Transparent & explainable AI:** Every recommendation explains *why* it assigned a department, severity, and priority.
