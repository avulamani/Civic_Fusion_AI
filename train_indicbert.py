"""Fine-tune and train IndicBERT multilingual municipal complaint classifier.
Also trains and validates the baseline TF-IDF model on the exact same splits.
Saves:
  models/indicbert/indicbert_classifier.pt
  models/indicbert/baseline_tfidf.joblib
  models/indicbert/classes.json
  models/indicbert/nlp_evaluation_report.json
"""

import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "datasets" / "nlp"
MODELS_DIR = ROOT / "models" / "indicbert"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "ai4bharat/IndicBERTv2-MLM-only"

CATEGORIES = [
    "road_damage",
    "open_manhole",
    "open_drain",
    "drain_blockage",
    "waste",
    "streetlight_damage",
    "water_leakage",
    "waterlogging",
    "damaged_sign",
    "traffic_signal_damage",
    "fallen_tree",
]

CAT2IDX = {c: i for i, c in enumerate(CATEGORIES)}
IDX2CAT = {i: c for i, c in enumerate(CATEGORIES)}


class ComplaintDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[item], dtype=torch.long),
        }


class IndicBERTClassifier(nn.Module):
    def __init__(self, encoder, num_classes: int = 11, hidden_dim: int = 768):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use mean pooling with attention mask
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


def train_and_evaluate_indicbert():
    print("=" * 70)
    print("TRAINING & EVALUATING INDICBERT MULTILINGUAL NLP SYSTEM")
    print(f"Base Model: {MODEL_NAME}")
    print("=" * 70)

    # 1. Load Datasets
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    print(f"Loaded datasets: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 2. Train TF-IDF Baseline on the EXACT same training split
    print("\n--- 1. Training Baseline TF-IDF + Logistic Regression ---")
    t0_base = time.time()
    baseline_model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    baseline_model.fit(train_df["text"].astype(str), train_df["category"].astype(str))
    train_time_base = time.time() - t0_base
    print(f"Baseline fitted in {train_time_base:.2f} s")

    # Evaluate baseline on test split
    base_preds = baseline_model.predict(test_df["text"].astype(str))
    base_f1 = f1_score(test_df["category"], base_preds, average="macro")
    base_acc = float((test_df["category"] == base_preds).mean())
    print(f"Baseline Test Accuracy: {base_acc:.4f}, Macro F1: {base_f1:.4f}")

    # Baseline language-specific performance
    base_lang_metrics = {}
    for lang in ["en", "hi", "te"]:
        sub_df = test_df[test_df["language"] == lang]
        sub_preds = baseline_model.predict(sub_df["text"].astype(str))
        base_lang_metrics[lang] = {
            "accuracy": round(float((sub_df["category"] == sub_preds).mean()), 4),
            "macro_f1": round(float(f1_score(sub_df["category"], sub_preds, average="macro", zero_division=0)), 4),
        }
    print("Baseline Language Breakdown:", base_lang_metrics)

    # Save baseline model
    joblib.dump(baseline_model, MODELS_DIR / "baseline_tfidf.joblib")

    # 3. Train IndicBERT Model
    print("\n--- 2. Initializing IndicBERT Multilingual Transformer ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_encoder = AutoModel.from_pretrained(MODEL_NAME)

    train_dataset = ComplaintDataset(train_df["text"].tolist(), [CAT2IDX[c] for c in train_df["category"]], tokenizer)
    val_dataset = ComplaintDataset(val_df["text"].tolist(), [CAT2IDX[c] for c in val_df["category"]], tokenizer)
    test_dataset = ComplaintDataset(test_df["text"].tolist(), [CAT2IDX[c] for c in test_df["category"]], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cpu")
    model = IndicBERTClassifier(base_encoder, num_classes=len(CATEGORIES)).to(device)

    # Freeze bottom layers of encoder for fast, stable CPU training, fine-tuning top layers & classifier
    for param in model.encoder.parameters():
        param.requires_grad = False
    # Unfreeze top 2 transformer layers
    if hasattr(model.encoder, "encoder") and hasattr(model.encoder.encoder, "layer"):
        for layer in model.encoder.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, weight_decay=0.01)

    print("\n--- Starting IndicBERT Training (3 Epochs) ---")
    best_val_f1 = -1.0
    best_weights_path = MODELS_DIR / "indicbert_classifier.pt"

    t0_bert = time.time()
    for epoch in range(1, 4):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(batch["labels"].numpy())

        val_acc = (np.array(val_preds) == np.array(val_targets)).mean()
        val_f1 = f1_score(val_targets, val_preds, average="macro")
        print(f"Epoch {epoch}/3 - Train Loss: {total_loss/len(train_loader):.4f} - Val Acc: {val_acc:.4f} - Val Macro F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_weights_path)

    training_time_bert = time.time() - t0_bert
    print(f"IndicBERT training completed in {training_time_bert:.2f} s")

    # 4. Independent Test Split Evaluation
    print("\n--- 3. Independent Test Set Evaluation (IndicBERT vs Baseline) ---")
    model.load_state_dict(torch.load(best_weights_path, weights_only=True))
    model.eval()

    test_preds, test_targets = [], []
    latencies = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            t0 = time.perf_counter()
            logits = model(input_ids, attention_mask)
            latencies.append((time.perf_counter() - t0) * 1000.0 / len(input_ids))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_targets.extend(batch["labels"].numpy())

    avg_latency = float(np.mean(latencies))
    test_acc = float((np.array(test_preds) == np.array(test_targets)).mean())
    test_f1 = float(f1_score(test_targets, test_preds, average="macro"))

    # Per-class metrics
    p_class, r_class, f1_class, _ = precision_recall_fscore_support(test_targets, test_preds, labels=range(len(CATEGORIES)), zero_division=0)
    per_class_report = {}
    for i, cat in enumerate(CATEGORIES):
        per_class_report[cat] = {
            "precision": round(float(p_class[i]), 4),
            "recall": round(float(r_class[i]), 4),
            "f1_score": round(float(f1_class[i]), 4),
        }

    # Confusion matrix
    conf_mat = confusion_matrix(test_targets, test_preds, labels=range(len(CATEGORIES))).tolist()

    # Per-language breakdown
    test_df_copy = test_df.copy()
    test_df_copy["pred_idx"] = test_preds
    test_df_copy["pred_cat"] = [IDX2CAT[p] for p in test_preds]

    bert_lang_metrics = {}
    for lang in ["en", "hi", "te"]:
        sub_df = test_df_copy[test_df_copy["language"] == lang]
        sub_acc = float((sub_df["category"] == sub_df["pred_cat"]).mean())
        sub_f1 = float(f1_score(sub_df["category"], sub_df["pred_cat"], average="macro", zero_division=0))
        bert_lang_metrics[lang] = {
            "accuracy": round(sub_acc, 4),
            "macro_f1": round(sub_f1, 4),
            "samples": len(sub_df),
        }

    # Save classes.json
    with open(MODELS_DIR / "classes.json", "w", encoding="utf-8") as f:
        json.dump({"categories": CATEGORIES, "cat2idx": CAT2IDX, "idx2cat": IDX2CAT}, f, indent=2)

    # Compile Comprehensive Evaluation Report
    full_report = {
        "indicbert_model": {
            "model_id": MODEL_NAME,
            "architecture": "IndicBERTv2 Sequence Classifier (768-dim, Mean Pooling, MLP head)",
            "overall_accuracy": round(test_acc, 4),
            "macro_f1": round(test_f1, 4),
            "language_performance": bert_lang_metrics,
            "per_class_metrics": per_class_report,
            "confusion_matrix": conf_mat,
            "categories": CATEGORIES,
            "avg_inference_latency_ms": round(avg_latency, 2),
            "training_time_seconds": round(training_time_bert, 2),
        },
        "baseline_tfidf": {
            "model_id": "TfidfVectorizer(1,2) + LogisticRegression(balanced)",
            "overall_accuracy": round(base_acc, 4),
            "macro_f1": round(base_f1, 4),
            "language_performance": base_lang_metrics,
            "training_time_seconds": round(train_time_base, 2),
        },
        "comparison": {
            "accuracy_improvement": round(test_acc - base_acc, 4),
            "macro_f1_improvement": round(test_f1 - base_f1, 4),
            "winner": "IndicBERT" if test_f1 > base_f1 else "TF-IDF Baseline",
        },
    }

    report_path = MODELS_DIR / "nlp_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL NLP MODEL COMPARISON REPORT")
    print("=" * 70)
    print(f"Baseline TF-IDF -> Accuracy: {base_acc:.4f}, Macro F1: {base_f1:.4f}")
    print(f"IndicBERT       -> Accuracy: {test_acc:.4f}, Macro F1: {test_f1:.4f}")
    print(f"Improvement: Macro F1 +{(test_f1 - base_f1)*100.0:.2f}%")
    print("Language Breakdown (IndicBERT):", bert_lang_metrics)
    print(f"Saved full NLP report to {report_path}")

    return full_report


if __name__ == "__main__":
    train_and_evaluate_indicbert()

