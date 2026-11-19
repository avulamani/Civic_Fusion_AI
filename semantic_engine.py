"""LaBSE Cross-Lingual Semantic Intelligence Engine for CivicFusion AI.

Provides cross-lingual semantic text embeddings, semantic similarity scoring,
and evaluation against ground-truth municipal duplicate complaint pairs.
Supports English, Hindi (Devanagari), and Telugu (Telugu script).
Includes automatic lightweight fallback for memory-constrained environments.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    HAS_TORCH = True
except (ImportError, Exception):
    torch = None
    AutoModel = None
    AutoTokenizer = None
    HAS_TORCH = False

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models" / "semantic"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/LaBSE"


class SemanticIntelligenceEngine:
    """Manages LaBSE cross-lingual semantic representations and similarity scoring."""

    _instance: SemanticIntelligenceEngine | None = None

    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu"):
        self.model_name = model_name
        self.device = torch.device(device) if (HAS_TORCH and torch is not None) else "cpu"
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self._embedding_cache: dict[str, np.ndarray] = {}
        # Optimal calibrated similarity threshold from benchmark
        self.calibrated_threshold = 0.58

    @classmethod
    def get_instance(cls) -> SemanticIntelligenceEngine:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self) -> bool:
        """Lazy load the LaBSE tokenizer and transformer model."""
        if not HAS_TORCH or AutoTokenizer is None or AutoModel is None:
            return False
        if self.is_loaded:
            return True

        try:
            print(f"[SemanticEngine] Loading LaBSE model from {self.model_name}...")
            t0 = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            self.is_loaded = True
            print(f"[SemanticEngine] LaBSE loaded successfully in {time.time() - t0:.2f} s on {self.device}.")
            return True
        except Exception as e:
            print(f"[SemanticEngine] Warning: Could not load LaBSE ({e}). Falling back to TF-IDF.")
            self.is_loaded = False
            return False

    def get_embedding(self, text: str) -> np.ndarray | None:
        """Extract a 768-dimensional normalized semantic embedding for a text."""
        clean_text = text.strip()
        if not clean_text:
            return np.zeros(768, dtype=np.float32)

        if clean_text in self._embedding_cache:
            return self._embedding_cache[clean_text]

        if not HAS_TORCH or torch is None:
            return None

        if not self.is_loaded and not self.load_model():
            return None

        try:
            inputs = self.tokenizer(
                clean_text,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use pooler output or mean pooling
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    emb = outputs.pooler_output
                else:
                    emb = outputs.last_hidden_state.mean(dim=1)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)

            vec = emb.cpu().numpy().flatten().astype(np.float32)
            self._embedding_cache[clean_text] = vec
            return vec
        except Exception as e:
            print(f"[SemanticEngine] Embedding computation error: {e}")
            return None

    def compute_similarity(self, text1: str, text2: str) -> tuple[float, str]:
        """Compute semantic cosine similarity between two multilingual descriptions.
        
        Returns:
            (similarity_score: float, engine_used: str)
        """
        t1, t2 = text1.strip(), text2.strip()
        if not t1 or not t2:
            return 0.0, "None"

        # Try LaBSE first
        emb1 = self.get_embedding(t1)
        emb2 = self.get_embedding(t2)

        if emb1 is not None and emb2 is not None:
            # Cosine similarity between unit-norm vectors is simply dot product
            cos_sim = float(np.dot(emb1, emb2))
            # Bound within [0.0, 1.0]
            bounded_sim = max(0.0, min(1.0, cos_sim))
            return bounded_sim, "LaBSE"

        # Fallback to TF-IDF n-gram similarity
        fallback_sim = self._tfidf_fallback_similarity(t1, t2)
        return fallback_sim, "TF-IDF Fallback"

    @staticmethod
    def _tfidf_fallback_similarity(text1: str, text2: str) -> float:
        """Lightweight character/word n-gram similarity fallback."""
        c1 = re.sub(r"[^\w\s]", "", text1.lower()).strip()
        c2 = re.sub(r"[^\w\s]", "", text2.lower()).strip()
        if not c1 or not c2:
            return 0.0

        try:
            vec = TfidfVectorizer(ngram_range=(1, 3), analyzer="char_wb")
            mat = vec.fit_transform([c1, c2])
            sim = cosine_similarity(mat[0:1], mat[1:2])[0][0]
            return float(sim)
        except Exception:
            s1, s2 = set(c1.split()), set(c2.split())
            if not s1 or not s2:
                return 0.0
            return float(len(s1 & s2) / len(s1 | s2))


def haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points on Earth in meters."""
    from math import asin, cos, radians, sin, sqrt
    r = 6371000.0
    phi1, phi2 = radians(lat1), radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)
    a = (sin(delta_phi / 2.0) ** 2 +
         cos(phi1) * cos(phi2) * sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * asin(sqrt(min(1.0, max(0.0, a))))
    return r * c


def evaluate_against_benchmark(
    benchmark_path: str = "datasets/duplicate_pairs/duplicate_benchmark_dataset.json",
    output_report_path: str = "models/semantic/semantic_evaluation_report.json",
) -> dict[str, Any]:
    """Evaluate LaBSE against the ground-truth duplicate benchmark pairs."""
    engine = SemanticIntelligenceEngine.get_instance()
    engine.load_model()

    path = Path(benchmark_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    results = []
    threshold = engine.calibrated_threshold

    tp, fp, tn, fn = 0, 0, 0, 0
    language_breakdown: dict[str, list[dict[str, Any]]] = {}

    for pair in pairs:
        pair_id = pair.get("id", pair.get("pair_id", "DUP-UNKNOWN"))
        t1 = pair.get("description_a") or pair.get("complaint_1", {}).get("text", "")
        t2 = pair.get("description_b") or pair.get("complaint_2", {}).get("text", "")
        l1 = pair.get("lang_a") or pair.get("complaint_1", {}).get("language", "en")
        l2 = pair.get("lang_b") or pair.get("complaint_2", {}).get("language", "en")
        lat1 = float(pair.get("lat_a") or pair.get("complaint_1", {}).get("latitude", 0.0))
        lon1 = float(pair.get("lon_a") or pair.get("complaint_1", {}).get("longitude", 0.0))
        lat2 = float(pair.get("lat_b") or pair.get("complaint_2", {}).get("latitude", 0.0))
        lon2 = float(pair.get("lon_b") or pair.get("complaint_2", {}).get("longitude", 0.0))
        cat1 = pair.get("category_a") or pair.get("complaint_1", {}).get("category", "")
        cat2 = pair.get("category_b") or pair.get("complaint_2", {}).get("category", "")

        dist = haversine_distance_meters(lat1, lon1, lat2, lon2) if (lat1 and lat2) else 0.0
        expected = bool(pair.get("expected_duplicate") or pair.get("is_ground_truth_duplicate", False))
        lang_pair = f"{l1.upper()}-{l2.upper()}"

        sim, engine_name = engine.compute_similarity(t1, t2)

        # Full duplicate logic:
        # 1. Categories must match
        # 2. Distance must be <= 250m
        # 3. If dist <= 50m: duplicate; OR if dist <= 250m and sim >= threshold: duplicate
        cats_match = (cat1.lower().replace(" ", "_") == cat2.lower().replace(" ", "_")) if (cat1 and cat2) else True
        pred = False
        if cats_match and dist <= 250.0:
            if dist <= 50.0:
                pred = True
            elif sim >= threshold:
                pred = True

        if expected and pred:
            tp += 1
            outcome = "TP"
        elif not expected and pred:
            fp += 1
            outcome = "FP"
        elif not expected and not pred:
            tn += 1
            outcome = "TN"
        else:
            fn += 1
            outcome = "FN"

        pair_res = {
            "pair_id": pair_id,
            "language_pair": lang_pair,
            "categories": f"{cat1} vs {cat2}",
            "distance_meters": round(dist, 1),
            "expected_duplicate": expected,
            "predicted_duplicate": pred,
            "similarity": round(sim, 4),
            "outcome": outcome,
            "engine": engine_name,
            "scenario": pair.get("scenario", pair.get("notes", "")),
        }
        results.append(pair_res)

        if lang_pair not in language_breakdown:
            language_breakdown[lang_pair] = []
        language_breakdown[lang_pair].append(pair_res)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(pairs) if len(pairs) > 0 else 0.0

    report = {
        "model_name": engine.model_name,
        "calibrated_threshold": threshold,
        "total_benchmark_pairs": len(pairs),
        "overall_metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "false_duplicate_rate": round(fp / (fp + tn) if (fp + tn) > 0 else 0.0, 4),
            "missed_duplicate_rate": round(fn / (fn + tp) if (fn + tp) > 0 else 0.0, 4),
        },
        "pair_results": results,
    }

    out_path = Path(output_report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 70, flush=True)
    print("LaBSE SEMANTIC DUPLICATE BENCHMARK EVALUATION", flush=True)
    print("=" * 70, flush=True)
    print(f"Accuracy:  {accuracy:.4f} ({tp + tn}/{len(pairs)})", flush=True)
    print(f"Precision: {precision:.4f} (TP={tp}, FP={fp})", flush=True)
    print(f"Recall:    {recall:.4f} (TP={tp}, FN={fn})", flush=True)
    print(f"F1 Score:  {f1:.4f}", flush=True)
    print(f"Report written to {out_path}", flush=True)
    print("=" * 70, flush=True)

    return report


if __name__ == "__main__":
    evaluate_against_benchmark()
