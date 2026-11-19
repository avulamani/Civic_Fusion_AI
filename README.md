# CivicFusion AI: Multimodal Municipal Complaint Intelligence System

An enterprise-grade, multimodal civic intelligence platform combining **Computer Vision (fine-tuned YOLO)**, **Multilingual NLP (IndicBERT v2)**, and **Semantic Intelligence (LaBSE)** to automate municipal complaint triage, defect localization, duplicate clustering, and priority assignment across Indian cities.

---

## 🏛️ System Overview

When a citizen encounters a civic hazard (such as a hazardous road pothole, damaged streetlight, open sewer manhole, overflowing garbage, or uncovered drainage ditch), reporting and departmental dispatch are often impeded by manual reviews, language barriers, and duplicate tickets.

**CivicFusion AI automates the entire civic intake and resolution pipeline:**
1. 📷 **Citizen captures an infrastructure photo**
2. 📝 **Citizen submits a complaint in English, Hindi (Devanagari), or Telugu (Telugu script)** via text or 🎤 speech
3. 📍 **GPS coordinates and landmark metadata are captured**

The system executes genuine, offline-trained AI models on CPU:
- **Detects infrastructure defects** with genuine YOLO bounding boxes and confidence scores.
- **Understands multilingual complaints** natively in English, Hindi, and Telugu using IndicBERT.
- **Eliminates duplicate work orders** by combining LaBSE cross-lingual semantic similarity with geospatial Haversine distance (< 250m).
- **Enforces citizen ground-truth authority**: The citizen's statement and NLP interpretation govern complaint category and municipal department dispatch; Computer Vision provides supplementary visual defect localization without overriding citizen context.
- **Assigns explainable severity and priority** (**Critical, High, Medium, Low** / **P1 to P4**) based on physical hazards, commuter impact, and urgency cues.
- **Immediately auto-registers complaints** into active municipal work orders without blocking manual pre-verification gates.

---

## 🏗️ End-to-End System Architecture

```text
                               CITIZEN SUBMISSION
                                       │
                ┌──────────────────────┼──────────────────────┐
                ↓                      ↓                      ↓
         📷 INFRASTRUCTURE PHOTO  📝 COMPLAINT TEXT/VOICE   📍 GPS COORDINATES
                │                      │                      │
                ↓                      ↓                      ↓
      Dedicated YOLO11n ONNX     IndicBERT v2 (EN/HI/TE)  Haversine Distance
      (models/vision/yolo11n)    (models/indicbert/)      (< 250m Proximity)
                │                      │                      │
                │         ┌────────────┴────────────┐         │
                │         ↓                         ↓         │
                │   Category & Entity        LaBSE Semantic   │
                │   Extraction (Urgency)     Cross-Lingual    │
                │                               Embeddings    │
                │                                   │         │
                │                                   └────┬────┘
                │                                        ↓
                │                             DUPLICATE DETECTION ENGINE
                │                             (Safe Cross-Lingual Clustering)
                │                                        │
                └─────────────────┬──────────────────────┘
                                  ↓
                     MULTIMODAL FUSION & DECISION
                   (Citizen Description = Ground Truth)
                                  ↓
                ┌─────────────────┴─────────────────┐
                ↓                                   ↓
       EXPLAINABLE SEVERITY                 PRIORITY MATRIX
    (Critical / High / Med / Low)             (P1 to P4)
                │                                   │
                └─────────────────┬─────────────────┘
                                  ↓
                   AUTOMATED TICKET REGISTRATION
                   (No Blocking Pre-Verification)
                                  ↓
                ┌─────────────────┴─────────────────┐
                ↓                                   ↓
        CITIZEN PORTAL                      OFFICER DASHBOARD
   - Real-time YOLO bounding boxes     - GIS PyDeck 2D/3D Map
   - Native script preservation        - Multi-criteria triage & filters
   - Duplicate clustering notice       - Status lifecycle management
   - Instant ticket summary card       - On-demand dispatch translation
```

---

## 🧠 Core AI Capabilities

### 1. Dedicated Municipal Defect Detection (YOLO)
- **Primary Model**: Dedicated supervised fine-tuned **YOLO11 Nano (`models/vision/municipal_yolo11n.onnx`)**.
- **Model Footprint**: 10.14 MB (5x lighter than zero-shot YOLO-World v2 at 48.88 MB).
- **Inference Latency**: 76.3 ms/frame on CPU (3.3x faster than baseline 253.1 ms).
- **Trained Classes (6)**:
  1. `pothole` (Road Maintenance Department)
  2. `damaged_road` (Road Maintenance Department)
  3. `open_manhole` (Water Supply & Sewerage Board)
  4. `open_drain` (Stormwater Drainage & Sewerage Division)
  5. `garbage` (Sanitation & Waste Management Department)
  6. `streetlight_damage` (Electrical & Street Lighting Department)
- **Dataset**: `datasets/municipal_detection/` (37 genuine municipal photographs across train, val, and test splits with normalized YOLO bounding boxes; zero synthetic labels).
- **False-Positive Analysis**: Verified zero false-positives on clean asphalt road surfaces (max background confidence 0.0666, well below 0.15 threshold).
- **Error-Analysis System**: Real-time logging of prediction-versus-citizen mismatches to `data/error_analysis/cv_error_analysis.jsonl` for offline inspector audit.
- **Genuine Execution**: Eliminates all artificial hash caches (`_match_known_reference`). All bounding boxes and confidence scores originate purely from real ONNX Runtime execution.

### 2. IndicBERT Multilingual Classification & Information Extraction
- **Primary Model**: Fine-tuned **IndicBERT v2 (`ai4bharat/IndicBERTv2-MLM-only`)** sequence classifier with mean pooling and an MLP classification head.
- **Supported Languages**: English, Hindi (Devanagari script), and Telugu (Telugu script).
- **Classes (11)**: `road_damage`, `open_manhole`, `open_drain`, `drain_blockage`, `waste`, `streetlight_damage`, `water_leakage`, `waterlogging`, `damaged_sign`, `traffic_signal_damage`, `fallen_tree`.
- **Dataset**: `datasets/nlp/` (1,680 stratified genuine civic complaints: 560 English, 560 Hindi, 560 Telugu across train/val/test splits).
- **Baseline Comparison**: Benchmarked against a TF-IDF (1,2 n-gram) + Balanced Logistic Regression model trained on the exact same splits.
- **Native Script Preservation**: Citizen descriptions in Hindi and Telugu are stored and displayed in their authentic native script (`original_text`). Descriptions are never overwritten or mangled by forced translations.
- **Structured Information Extraction**:
  - Primary problem defect.
  - Secondary conditions (e.g., water accumulation inside structural road pothole).
  - Urgency markers (`Safety risk`, `Urgent obstruction`, `Severe damage`, `Night visibility risk`).
  - Commuter & pedestrian impact cues (e.g., vehicles swerving, skidding, pedestrians unable to cross).
  - Landmark mentions (e.g., "Near School Entrance", "Gol Chowk").
  - Resolution note explainability.

### 3. LaBSE Cross-Lingual Semantic Intelligence & Duplicate Detection
- **Primary Model**: **LaBSE (`sentence-transformers/LaBSE`)** mapping English, Hindi, and Telugu sentences into a shared 768-dimensional normalized embedding space.
- **Multi-Tier Semantic Engine (`semantic_engine.py`)**:
  - Tier 1: LaBSE cross-lingual semantic embedding dot product.
  - Tier 2: TF-IDF character/word n-gram cosine similarity fallback for memory-constrained environments.
- **Duplicate Decision Rules (`duplicates.py`)**:
  1. **Category Compatibility**: Complaints must match municipal category (`road_damage == road_damage`). Co-located defects of different categories (e.g., a broken streetlight 10m from an open drain) are kept strictly separate.
  2. **Haversine Distance**: Spatial distance must be $\le 250$ meters. Complaints with identical wording in different areas or cities (e.g., Phagwara vs. Jalandhar) are rejected by the spatial boundary filter.
  3. **Clustering Logic**: If distance $\le 50$ meters, complaints are grouped automatically; if between 50m and 250m, requires cross-lingual semantic similarity $\ge 0.58$.
- **Benchmark Evaluation**: Evaluated against 12 verified ground-truth pairs in `datasets/duplicate_pairs/duplicate_benchmark_dataset.json`:
  - **Accuracy**: 100.0% (12/12)
  - **Precision**: 1.0000 (0 false duplicates)
  - **Recall**: 1.0000 (0 missed duplicates)
  - **F1 Score**: 1.0000
- **Safe Ticket Clustering**: Increments parent ticket citizen report count, escalates priority to P2/P1, links child descriptions, and avoids creating redundant work orders.

---

## 📊 Measured Benchmark Summary

| Evaluation Dimension | Dedicated YOLO11n | Baseline YOLO-World v2 | IndicBERT v2 NLP | Baseline TF-IDF |
| :--- | :--- | :--- | :--- | :--- |
| **Model Size on Disk** | **10.14 MB** | 48.88 MB | 1061.5 MB | **0.28 MB** |
| **Inference Latency (CPU)** | **76.3 ms** | 253.1 ms | ~450 ms | **7.06 ms** |
| **Test Accuracy / mAP** | **mAP50: 0.112 (Rec 80%)** | Zero-shot (noisy) | **Test Acc: ~98-100%** | Test Acc: ~90% |
| **Clean Road False Positives** | **0 (PASS)** | 3+ background specks | N/A | N/A |
| **Cross-Lingual Support** | Visual (Domain Invariant) | English prompts only | **EN, HI, TE** | English / Hinglish |

### Duplicate Detection Benchmark (LaBSE)
- **Calibrated Similarity Threshold**: `0.58`
- **EN-HI Pothole Match Similarity**: `0.7275` (Distance 43.9m) $\rightarrow$ **Duplicate Detected (TP)**
- **EN-TE Open Manhole Similarity**: `0.6433` (Distance 48.4m) $\rightarrow$ **Duplicate Detected (TP)**
- **HI-TE Garbage Dump Similarity**: `0.8358` (Distance 36.2m) $\rightarrow$ **Duplicate Detected (TP)**
- **Co-located Different Categories (Same GPS)**: `0.5581` $\rightarrow$ **Separate Tickets (TN)**
- **Distant Identical Complaints (20 km apart)**: `0.8021` $\rightarrow$ **Distance Rejection (TN)**

---

## 📁 Repository Directory Structure

```text
CivicFusion_AI/
├── dashboard.py                   # Unified Citizen Portal & Officer Command Dashboard
├── vision.py                      # YOLO ONNX inference, annotation, & error logging
├── nlp.py                         # IndicBERT v2 classifier, entity extraction, & TF-IDF fallback
├── semantic_engine.py             # LaBSE cross-lingual embedding engine & benchmark evaluator
├── duplicates.py                  # Geospatial Haversine & semantic duplicate detection
├── fusion.py                      # Multimodal combine, ground-truth authority, & ticket builder
├── storage.py                     # Complaint persistence, JSON serialization, & mismatch audits
├── multilingual.py                # Language identification (EN, HI, TE) & on-demand translation
├── speech_utils.py                # Whisper & SpeechRecognition voice input utilities
├── train_vision.py                # YOLO fine-tuning pipeline & ONNX exporter
├── vision_evaluation.py           # Independent test-set vision evaluator & clean control test
├── train_indicbert.py             # IndicBERT fine-tuning pipeline & baseline comparison
├── benchmark_deployment.py        # CPU latency, disk footprint, and fallback test suite
├── test_multimodal_system.py      # End-to-end integration test runner (10/10 scenarios)
│
├── datasets/
│   ├── municipal_detection/       # 37 verified municipal images & YOLO annotations
│   │   ├── images/ (train, val, test)
│   │   ├── labels/ (train, val, test)
│   │   ├── data.yaml
│   │   └── metadata/ (sources.csv, class_mapping.yaml, annotation_metadata.json)
│   ├── nlp/                       # 1,680 stratified multilingual complaints (EN, HI, TE)
│   │   ├── train.csv, val.csv, test.csv
│   │   └── metadata.json
│   └── duplicate_pairs/           # Ground-truth duplicate benchmark dataset
│       └── duplicate_benchmark_dataset.json
│
├── models/
│   ├── vision/
│   │   ├── municipal_yolo11n.onnx # Trained dedicated municipal detector (10.14 MB)
│   │   ├── municipal_yolo11n_best.pt
│   │   ├── classes.txt
│   │   └── vision_evaluation_report.json
│   ├── indicbert/
│   │   ├── indicbert_classifier.pt # Fine-tuned IndicBERT classifier weights
│   │   ├── baseline_tfidf.joblib   # High-speed TF-IDF baseline model (297 KB)
│   │   └── classes.json
│   ├── semantic/
│   │   └── semantic_evaluation_report.json # 100% precision duplicate benchmark report
│   ├── deployment_benchmarks.json # CPU latency & footprint audit report
│   └── system_test_report.json    # 10/10 integration scenario verification report
│
└── data/
    ├── complaints.json            # Active municipal tickets & work orders
    ├── error_analysis/            # CV/NLP mismatch review logs
    └── uploads/                   # Uploaded citizen photos & annotated visuals
```

---

## 🚀 Running the Application Locally

### 1. Prerequisites & Environment Setup
Ensure Python 3.10+ is installed. Activate your virtual environment and install dependencies:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics transformers sentencepiece onnx onnxruntime scikit-learn pandas pillow streamlit pydeck
```

### 2. Launch the Streamlit Dashboard
```powershell
python -m streamlit run dashboard.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 3. Run the Evaluation & Test Suites
```powershell
# 1. Run Computer Vision Evaluation & Baseline Comparison
python vision_evaluation.py

# 2. Run LaBSE Semantic Duplicate Benchmark Evaluation
python semantic_engine.py

# 3. Run Resource & Latency Deployment Benchmarks
python benchmark_deployment.py

# 4. Run End-to-End System Integration Test (10 Scenarios)
python test_multimodal_system.py
```

---

## 🔒 Free-Tier CPU Optimization & Stability Safeguards

- **Zero GPU Requirement**: All inference pipelines operate natively on CPU with multi-threading limits (`intra_op_num_threads=2`).
- **Memory Footprint Protection**:
  - YOLO runs via lightweight ONNX Runtime with an in-memory footprint of ~35 MB.
  - IndicBERT operates with mean pooling and top-layer fine-tuning.
  - LaBSE embeddings use unit-norm cosine caching to prevent redundant tensor passes.
  - In extreme resource-constrained environments (e.g. Render free tier with 512MB RAM), the system seamlessly degrades to the ultra-fast TF-IDF baseline (0.28 MB, 7ms latency) and n-gram similarity without crashing.
- **Automated Work-Order Registration**: Citizens receive immediate work order numbers (`CF-XXXX`) upon submission. Unverified visual mismatches are logged asynchronously for offline review rather than blocking municipal intake.
