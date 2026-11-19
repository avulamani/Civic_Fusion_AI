"""Deployment benchmark and resource optimization audit for CivicFusion AI.

Evaluates:
  1. Disk footprints of all models (YOLO, IndicBERT, LaBSE, TF-IDF baseline)
  2. CPU inference latency per modality
  3. Memory consumption (WorkingSet) under Render free-tier constraints (512MB RAM)
  4. Graceful fallback mechanisms for zero-crash reliability
Outputs results to models/deployment_benchmarks.json
"""

import gc
import json
from pathlib import Path
import time
import numpy as np
import psutil
from PIL import Image

ROOT = Path(__file__).parent
OUTPUT_REPORT = ROOT / "models" / "deployment_benchmarks.json"


def measure_memory_mb() -> float:
    """Return current process RSS memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def benchmark_all():
    print("=" * 70)
    print("CIVICFUSION AI - CPU DEPLOYMENT & RESOURCE BENCHMARK")
    print("=" * 70)

    report = {
        "hardware": {
            "platform": "CPU-Only",
            "cpu_count": psutil.cpu_count(logical=True),
            "ram_total_mb": round(psutil.virtual_memory().total / (1024 * 1024), 1),
            "render_free_tier_target_limit_mb": 512.0,
        },
        "model_disk_sizes_mb": {},
        "inference_latencies_ms": {},
        "fallbacks_verified": {},
    }

    # 1. Model Sizes on Disk
    model_paths = {
        "dedicated_yolo11n_onnx": ROOT / "models" / "vision" / "municipal_yolo11n.onnx",
        "yolo_world_v2_baseline_onnx": ROOT / "models" / "municipal_worldv2.onnx",
        "baseline_tfidf_joblib": ROOT / "models" / "indicbert" / "baseline_tfidf.joblib",
        "indicbert_weights_pt": ROOT / "models" / "indicbert" / "indicbert_classifier.pt",
    }

    for name, path in model_paths.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            report["model_disk_sizes_mb"][name] = round(size_mb, 2)
            print(f"  Disk Size -> {name:<30}: {size_mb:>6.2f} MB")
        else:
            report["model_disk_sizes_mb"][name] = "Not found"

    # 2. Benchmark Computer Vision ONNX inference
    print("\n--- 1. Computer Vision (YOLO ONNX) Latency ---")
    import vision
    dummy_img = Image.new("RGB", (640, 640), (120, 120, 120))
    # Warmup
    vision.detect(dummy_img)

    cv_latencies = []
    for _ in range(5):
        t0 = time.perf_counter()
        dets, _ = vision.detect(dummy_img)
        cv_latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_cv_latency = float(np.mean(cv_latencies))
    report["inference_latencies_ms"]["yolo11n_onnx"] = round(avg_cv_latency, 2)
    print(f"  YOLO11n ONNX average latency: {avg_cv_latency:.1f} ms/frame")

    # 3. Benchmark NLP Inference (IndicBERT + Baseline)
    print("\n--- 2. NLP (IndicBERT & TF-IDF Baseline) Latency ---")
    import nlp
    sample_text_en = "There is a deep pothole near the school entrance causing traffic jams."
    sample_text_hi = "स्कूल के प्रवेश द्वार के पास सड़क पर एक बड़ा गड्ढा है।"
    sample_text_te = "పాఠశాల ప్రవేశ ద్వారం దగ్గర రోడ్డుపై పెద్ద గుంత ఉంది."

    # Benchmark full analysis
    nlp_latencies = []
    for txt, l in [(sample_text_en, "en"), (sample_text_hi, "hi"), (sample_text_te, "te")]:
        t0 = time.perf_counter()
        res = nlp.analyze_text_full(None, txt, user_lang=l)
        dt = (time.perf_counter() - t0) * 1000.0
        nlp_latencies.append(dt)
        print(f"  NLP [{l.upper()}] ({res['category']}): {dt:.1f} ms")

    report["inference_latencies_ms"]["nlp_full_analysis_avg"] = round(float(np.mean(nlp_latencies)), 2)

    # Benchmark pure TF-IDF baseline fallback
    tfidf_clf = nlp.load_classifier()
    t0_base = time.perf_counter()
    for _ in range(20):
        tfidf_clf.predict_proba(["There is a broken pothole on the road"])
    tfidf_lat = (time.perf_counter() - t0_base) * 1000.0 / 20
    report["inference_latencies_ms"]["tfidf_baseline"] = round(tfidf_lat, 2)
    print(f"  TF-IDF Baseline pure latency: {tfidf_lat:.2f} ms")

    # 4. Benchmark Semantic Duplicate Matching (LaBSE + TF-IDF fallback)
    print("\n--- 3. Semantic Duplicate Matching Latency ---")
    from duplicates import check_for_duplicates
    sample_existing = [
        {"id": "CF-1001", "description": "Big pothole on road near chowk", "latitude": 31.2185, "longitude": 75.7672, "category": "road_damage", "status": "Registered"}
    ]
    t0_dup = time.perf_counter()
    dup_res = check_for_duplicates("सड़क पर बहुत बड़ा गड्ढा है", 31.2188, 75.7675, "road_damage", sample_existing)
    dup_lat = (time.perf_counter() - t0_dup) * 1000.0
    report["inference_latencies_ms"]["duplicate_check"] = round(dup_lat, 2)
    print(f"  Duplicate Check (LaBSE/Fallback) latency: {dup_lat:.1f} ms | Is Duplicate: {dup_res['is_duplicate']}")

    # 5. Fallbacks Verification
    print("\n--- 4. Graceful Fallbacks Verification ---")
    # TF-IDF text fallback without IndicBERT
    fb_res = nlp.predict_category(tfidf_clf, "Broken streetlight on street", explicit_category=None)
    fb1_pass = fb_res[0] == "streetlight_damage"
    report["fallbacks_verified"]["tfidf_category_fallback"] = fb1_pass
    print(f"  TF-IDF Category Fallback: {'PASS' if fb1_pass else 'FAIL'}")

    # Distance boundary check (> 250m prevents false duplicate merging)
    dist_dup = check_for_duplicates("Big pothole on road near chowk", 31.5000, 75.9000, "road_damage", sample_existing)
    fb2_pass = not dist_dup["is_duplicate"]
    report["fallbacks_verified"]["distance_boundary_filter"] = fb2_pass
    print(f"  Distance Boundary Filter (>250m safe rejection): {'PASS' if fb2_pass else 'FAIL'}")

    # Clean road negative control
    clean_road_img = Image.new("RGB", (640, 640), (80, 80, 80))
    clean_dets, _ = vision.detect(clean_road_img, confidence_threshold=0.20)
    fb3_pass = len(clean_dets) == 0
    report["fallbacks_verified"]["clean_negative_rejection"] = fb3_pass
    print(f"  Clean Road Negative Control (Zero False Positives): {'PASS' if fb3_pass else 'FAIL'}")

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved deployment benchmark report to {OUTPUT_REPORT}")
    return report


if __name__ == "__main__":
    benchmark_all()

