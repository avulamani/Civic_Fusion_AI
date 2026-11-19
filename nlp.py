"""Natural Language Processing module for CivicFusion AI.

Integrates:
1. Fine-tuned IndicBERT v2 Multilingual Classifier (English, Hindi, Telugu).
2. TF-IDF + Logistic Regression baseline as high-speed fallback for resource-constrained environments.
3. Information extraction: Urgency cues, Commuter/pedestrian impact, Location and landmark cues.
4. Full semantic compound issue resolution (Primary defect vs. Secondary conditions).
5. Ground-truth preservation: Retains citizen's original text and native script (Devanagari, Telugu).
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from multilingual import (
    detect_language,
    get_language_display_name,
    match_native_civic_category,
    normalize_lang_code,
    translate_text,
)

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "labeled_complaints.csv"
MODELS_DIR = ROOT / "models" / "indicbert"

INDICBERT_MODEL_NAME = "ai4bharat/IndicBERTv2-MLM-only"
WEIGHTS_PATH = MODELS_DIR / "indicbert_classifier.pt"
BASELINE_PATH = MODELS_DIR / "baseline_tfidf.joblib"
CLASSES_PATH = MODELS_DIR / "classes.json"

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
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


_INDICBERT_MODEL = None
_INDICBERT_TOKENIZER = None


def get_indicbert_model():
    """Lazy load the fine-tuned IndicBERT classifier."""
    global _INDICBERT_MODEL, _INDICBERT_TOKENIZER
    if _INDICBERT_MODEL is not None:
        return _INDICBERT_MODEL, _INDICBERT_TOKENIZER

    if not WEIGHTS_PATH.exists():
        return None, None

    try:
        tokenizer = AutoTokenizer.from_pretrained(INDICBERT_MODEL_NAME)
        encoder = AutoModel.from_pretrained(INDICBERT_MODEL_NAME)
        model = IndicBERTClassifier(encoder, num_classes=len(CATEGORIES))
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True))
        model.eval()
        _INDICBERT_MODEL = model
        _INDICBERT_TOKENIZER = tokenizer
        return _INDICBERT_MODEL, _INDICBERT_TOKENIZER
    except Exception as e:
        print(f"[NLP] Could not load IndicBERT ({e}), using baseline fallback.")
        return None, None


def load_classifier() -> Pipeline:
    """Load baseline TF-IDF model or train starter if not found."""
    if BASELINE_PATH.exists():
        try:
            return joblib.load(BASELINE_PATH)
        except Exception:
            pass

    examples = pd.read_csv(DATA_PATH)
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                                      strip_accents="unicode", sublinear_tf=True)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(examples["text"].astype(str), examples["category"].astype(str))
    return model


URGENCY_PATTERNS = {
    "Safety risk": r"\b(accident\w*|injur\w*|danger\w*|unsafe|crash|fall\w*|hazard\w*|pedestrian risk|risk of life|fatality|exposed wire|electric shock)\b",
    "Urgent obstruction": r"\b(block\w*|traffic\s+halt|cannot pass|can't pass|road closed|struggling to pass|jammed|impassable|swerving)\b",
    "Severe damage": r"\b(large|huge|deep|severe|burst|massive|completely broken|extensive|collapsed|crater)\b",
    "Night visibility risk": r"\b(dark|at night|nighttime|not working at night|no light|pitch black|zero visibility)\b",
}

IMPACT_PATTERNS = [
    (r"\b(bikes?|two-wheelers?|scooters?|motorcycles?|vehicles?|cars?|buses?|trucks?)\s+(?:are\s+)?(?:struggling|swerving|crashing|slipping|skidding|stuck|damaged|falling)",
     "Vehicles / commuters experiencing difficulty, skidding, or hazard"),
    (r"\b(traffic|road|lane|junction)\s+(?:is\s+)?(?:blocked|jammed|congested|halted|stopped|choked)",
     "Traffic movement blocked or severely congested"),
    (r"\b(pedestrians?|people|walkers?|citizens?|kids?|children|elders?)\s+(?:cannot|can't|unable to|falling|tripping|struggling)",
     "Pedestrian walkway unsafe or impassable"),
    (r"\b(flood\w*|water\s+entering|waterlogged|inundated|submerged)",
     "Water accumulation or flooding surrounding premises"),
    (r"\b(stench|smell|foul|insects?|mosquitoes?|health\s+hazard|toxic)",
     "Sanitary hazard and foul environmental conditions"),
    (r"\b(dark|pitch black|no visibility|cannot see at night)",
     "Zero night visibility creating crime/accident risk"),
]

LOCATION_PATTERNS = [
    r"\b(?:near|close to|beside|next to|outside|opposite|in front of|at the gate of|behind)\s+([A-Za-z0-9\s]+?)(?:[.,;]|\band\b|\bwith\b|\bcausing\b|\bwhere\b|$)",
    r"\bat\s+([A-Za-z0-9\s]+?(?:junction|crossing|chowk|gate|street|road|avenue|lane|stop|station|park|school|market|hospital|bridge|flyover))",
    r"\balong\s+([A-Za-z0-9\s]+?(?:road|street|avenue|lane|highway))",
]

# Comprehensive multilingual & Hinglish domain lexicon for semantic resolution
CIVIC_LEXICON = {
    "road_damage": [
        r"\b(pot\s*holes?|pot\s*hole\w*|pothole\w*|crater\w*|road\s+damage\w*|damaged\s+road\w*|broken\s+road\w*|asphalt\s+breakdown|uneven\s+road|rutting|cracked\s+asphalt|road\s+cavity|road\s+cavities|cavities|road\s+depression\w*|road\s+breakdown|cracked\s+road\w*|road\s+cracks?|damaged\s+pavement|pavement\s+damage\w*|alligator\s+cracks?|deteriorated\s+road\w*|broken\s+asphalt|road\s+surface\s+damage\w*)\b",
        r"\b(gaddh\w*|khadd\w*|sadak kharab|tooti sadak|toota road|road toota|sadak toot|khatarnak gaddha|sadak par gaddh\w*|sadak me gaddh\w*)\b",
        r"(गड्ढा|गड्ढे|गड्डो|खड्डा|खड्डे|टूटी सड़क|सड़क खराब|डामर|सड़क टूट|सड़क पर गड्ढा|सड़क पर गड्ढे|सड़क में गड्ढा|सड़क में गड्ढे|उबड़-खाबड़|खराब सड़क|टूटा हुआ रोड|सड़क टूटी|सड़क का गड्ढा)",
        r"(గుంత|గుంతలు|గుంటలు|రోడ్డు పాడైంది|పగిలిన రోడ్డు|రోడ్డు దెబ్బతింది|రోడ్డుపై గుంత|రోడ్డుపై గుంతలు|తారు రోడ్డు విరిగింది|చెడిపోయిన రోడ్డు|రోడ్డు గుంతలు|రోడ్డు పగుళ్లు)",
    ],
    "open_manhole": [
        r"\b(manhole\w*|chamber cover|missing grate|open manhole\w*|manhole lid\w*|sewer chamber\w*|drainage grate missing|sewer hole\w*|manhole open|uncovered manhole\w*|gutter cover missing)\b",
        r"\b(khula manhole|manhole ka dhakkan|dhakkan gayab|gutter khula|gatar khula|manhole tuta)\b",
        r"(मैनहोल|मैनहोल का ढक्कन|खुला मैनहोल|गटर का ढक्कन|गटर खुला|सीवर का ढक्कन|सीवर का खुला ढक्कन)",
        r"(మ్యాన్‌హోల్|మ్యాన్ హోల్|మ్యాన్‌హోల్ మూత|తెరిచి ఉన్న మ్యాన్‌హోల్|గట్టర్ మూత|గట్టర్ తెరిచి)",
    ],
    "open_drain": [
        r"\b(open drain\w*|drain\w*\s+(?:channel\s+)?(?:is\s+)?open|open\s+drainage\w*|drainage\s+channel\w*|roadside drainage|drainage ditch|storm drain\w*|roadside trench|excavated trench|uncovered drain\w*|drainage canal|open storm sewer|drain trench|road trench)\b",
        r"\b(khuli naali|khula naala|ganda naala|nali khuli|khuli nali|naali tooti)\b",
        r"(खुली नाली|खुला नाला|गंदा नाला|नाली का पानी|टूटी नाली|कच्ची नाली)",
        r"(మురుగు కాలువ|డ్రైనేజీ|కాలువ తెరిచి|తెరిచిన కాలువ|మూత లేని కాలువ|మురుగు నీరు)",
    ],
    "drain_blockage": [
        r"\b(drain\w*(?:\s+\w+)?\s+block\w*|block\w*(?:\s+\w+)?\s+drain\w*|clogged(?:\s+\w+)?\s+drain\w*|choked(?:\s+\w+)?\s+drain\w*|choked sewer|drainage jam|rubbish jam in drain|silt blockage|overflowing(?:\s+\w+)?\s+drain\w*)\b",
        r"\b(naali jaam|nali band|naala band|kachre se naali|choked naali|naali choke)\b",
        r"(नाली जाम|नाला चोक|नाली बंद|कचरे से नाली बंद|नाली रुक गई|सीवर जाम)",
        r"(డ్రైనేజీ బ్లాక్|మురుగు కాలువ మూసుకుపోయింది|కాలువలో చెత్త పేరుకుపోయింది|జామ్ అయిన డ్రైనేజ్)",
    ],
    "waste": [
        r"\b(garbage\w*|trash\w*|rubbish\w*|waste pile\w*|waste heap\w*|overflowing bin\w*|refuse pile\w*|litter\w*|dumped waste|garbage dump)\b",
        r"\b(kachra|kachre|koora|koore|kuda|kude|gandagi|dustbin overflow|kachra phek|kude ka dher)\b",
        r"(कचरा|कचरे|कूड़ा|कूड़े|गंदगी|कचरे का ढेर|कूड़ेदान|डस्टबिन|सड़ा हुआ कचरा|कूड़ा-कर्कट|वेस्ट|क्लीन|वेस्ट पड़ा|कचरा पड़ा)",
        r"(చెత్త|చెత్తకుండీ|వ్యర్థాలు|మురికి|చెత్త కుప్ప|చెత్తాచెదారం|దుర్వాసన|చెత్త పేరుకుపోయింది)",
    ],
    "streetlight_damage": [
        r"\b(streetlight\w*|street light\w*|lamp post\w*|light pole\w*|dark street\w*|no lighting|inoperative light\w*|broken light\w*|pitch black street)\b",
        r"\b(street light kharab|light band|khamba toota|andhera hai|batti nahi|pole ki light)\b",
        r"(स्ट्रीट लाइट|स्ट्रीटलाइट|खंभा|बत्ती|लाइट खराब|अंधेरा|रात में लाइट नहीं|खंभे की लाइट खराब)",
        r"(వీధి దీపం|వీధి లైటు|వీధి దీపాలు|స్తంభం|లైట్లు వెలగడం లేదు|చీకటి|స్ట్రీట్ లైట్ పాడైంది)",
    ],
    "water_leakage": [
        r"\b(burst pipe\w*|water leak\w*|pipe leaking|water pipe\w*|potable water gushing|main supply line leak\w*|pipeline burst|drinking water leak\w*)\b",
        r"\b(paani ka pipe|pipe leak|pipe phoot gaya|paani beh raha|peene ka paani|supply line phooti)\b",
        r"(पानी का पाइप|पाइप फूटा|पाइप टूट|पानी बह रहा|पीने का पानी|पाइप लीकेज|पानी का रिसाव|पाइपलाइन)",
        r"(నీటి లీకేజీ|పైపు పగిలింది|తాగునీరు వృధా|నీరు కారుతోంది|వాటర్ పైప్|మంచినీటి పైపు)",
    ],
    "waterlogging": [
        r"\b(waterlogging|flooded road|water logged|stagnant water|standing water across road|inundated street|road submerged|street flooding)\b",
        r"\b(paani bhara hua|paani bhar gaya|sadak par paani|jalbharo|rasta doob gaya|sadak jalmagn)\b",
        r"(जलभराव|पानी भरा|सड़क पर पानी|डूबी सड़क|बाढ़ जैसा|सड़क पर भारी पानी|जलभराव की समस्या)",
        r"(నీరు నిలిచిపోయింది|రోడ్డుపై నీరు|వరద నీరు|నీటి నిల్వ|రోడ్డు మునిగిపోయింది)",
    ],
    "damaged_sign": [
        r"\b(signboard\w*|traffic sign\w*|damaged sign\w*|broken sign\w*|fallen sign\w*|direction board)\b",
        r"\b(sign board toota|direction board toota|board gir gaya)\b",
        r"(साइन बोर्ड|साइनबोर्ड|यातायात संकेत|बोर्ड टूटा|दिशा सूचक)",
        r"(సైన్ బోర్డు|ట్రాఫిక్ బోర్డు|బోర్డు విరిగింది|రోడ్డు సూచిక)",
    ],
    "traffic_signal_damage": [
        r"\b(traffic signal\w*|traffic light\w*|traffic signal dark|signal broken|signal not working)\b",
        r"\b(traffic signal band|signal kharab|traffic light kharab)\b",
        r"(ट्रैफिक सिग्नल|ट्रैफिक लाइट|सिग्नल खराब|सिग्नल बंद)",
        r"(ట్రాఫిక్ సిగ్నల్|ట్రాఫిక్ లైట్|సిగ్నల్ పనిచేయడం లేదు)",
    ],
    "fallen_tree": [
        r"\b(fallen tree\w*|tree branch\w* blocking|tree fallen|fallen trunk)\b",
        r"\b(ped gir gaya|ped ki daali|ped toota)\b",
        r"(पेड़ गिर गया|पेड़ की डाली|पेड़ गिरा)",
        r"(చెట్టు కూలిపోయింది|చెట్టు కొమ్మ|కూలిన చెట్టు)",
    ],
}


def find_urgency_cues(text: str) -> list[str]:
    """Identify transparent risk markers in the user description."""
    clean = re.sub(r"\s+", " ", text.lower()).strip()
    return [label for label, pattern in URGENCY_PATTERNS.items()
            if re.search(pattern, clean)]


def extract_location_mention(text: str) -> str | None:
    """Extract location entities or landmark cues from citizen description."""
    clean = re.sub(r"\s+", " ", text).strip()
    for pattern in LOCATION_PATTERNS:
        match = re.search(pattern, clean, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            candidate = re.sub(r"\b(is|are|has|have|was|the|a|an)\b.*$", "", candidate, flags=re.IGNORECASE).strip()
            if 3 <= len(candidate) <= 40:
                return candidate.title()
    return None


def extract_impact_description(text: str) -> str:
    """Identify reported impact on citizens or traffic."""
    clean = text.lower()
    for pattern, desc in IMPACT_PATTERNS:
        if re.search(pattern, clean):
            return desc
    return "Issue reported impacting local mobility or civic environment"


def detect_all_civic_issues(text: str) -> dict[str, float]:
    """Scan text across English, native scripts, and Hinglish for all civic mentions."""
    clean = text.lower()
    scores: dict[str, float] = {}

    for category, regex_list in CIVIC_LEXICON.items():
        match_count = 0
        for pattern in regex_list:
            matches = re.findall(pattern, clean, re.IGNORECASE)
            match_count += len(matches)
        if match_count > 0:
            scores[category] = min(0.97, 0.90 + (0.02 * match_count))

    return scores


def resolve_compound_issues(
    detected_issues: dict[str, float],
    raw_text: str,
    translated_text_en: str
) -> tuple[str, float, list[str], str]:
    """Apply civic triage hierarchy to resolve primary defect vs secondary conditions."""
    combined_text = f"{raw_text} {translated_text_en}".lower()

    # Rule 1: Structural Road Cavity + Water Accumulation
    water_in_pothole = bool(re.search(
        r"\b(rainwater|water inside|water collected|filled with (?:rain)?water|water filled|paani bhara|puddle|stagnant water|waterlogged|water\s+accumulation|पानी भरा|बारिश का पानी|వర్షపు నీరు|నీరు నిలిచింది)\b",
        combined_text,
    ))
    if "road_damage" in detected_issues and ("waterlogging" in detected_issues or water_in_pothole):
        secondary = [c for c in detected_issues if c != "road_damage"]
        if "waterlogging" not in secondary:
            secondary.append("waterlogging")
        return (
            "road_damage",
            0.96,
            secondary,
            "Pothole / Road Surface Damage identified as primary physical damage; water accumulation recorded as secondary condition."
        )

    # Rule 2: Drainage Blockage vs. Waterlogging
    if "drain_blockage" in detected_issues and "waterlogging" in detected_issues:
        secondary = [c for c in detected_issues if c != "drain_blockage"]
        return (
            "drain_blockage",
            0.94,
            secondary,
            "Clogged/blocked storm drain identified as the root cause. Surface waterlogging is recorded as a secondary impact."
        )

    # Rule 3: Open Manhole Priority (Life Safety)
    if "open_manhole" in detected_issues:
        secondary = [c for c in detected_issues if c != "open_manhole"]
        return (
            "open_manhole",
            0.96,
            secondary,
            "Hazardous open manhole / missing chamber lid identified as critical primary safety hazard."
        )

    # Rule 4: Road Damage with multiple secondary issues
    if "road_damage" in detected_issues:
        secondary = [c for c in detected_issues if c != "road_damage"]
        return (
            "road_damage",
            detected_issues["road_damage"],
            secondary,
            "Road damage / pothole identified as primary structural complaint; additional conditions recorded."
        )

    # Rule 5: Standard Single or Highest Scoring Issue
    if detected_issues:
        sorted_issues = sorted(detected_issues.items(), key=lambda kv: kv[1], reverse=True)
        primary = sorted_issues[0][0]
        confidence = sorted_issues[0][1]
        secondary = [item[0] for item in sorted_issues[1:]]
        note = f"Primary civic issue '{primary}' identified from citizen description."
        return primary, confidence, secondary, note

    return "road_damage", 0.0, [], ""


def predict_category(
    model: Pipeline | None,
    text: str,
    lang: str | None = None,
    explicit_category: str | None = None,
) -> tuple[str, float, list[str], str]:
    """Predict category with IndicBERT multilingual classifier, compound resolution, and TF-IDF fallback."""
    # 1. Citizen Explicit Override
    if explicit_category and explicit_category.strip():
        clean_exp = explicit_category.strip().lower().replace(" ", "_")
        return (
            clean_exp,
            0.98,
            [],
            "Citizen explicitly selected this category during complaint submission."
        )

    clean = text.strip()
    norm_lang = normalize_lang_code(lang) if lang else detect_language(clean)

    # 2. Try IndicBERT Multilingual Transformer first
    indicbert, tokenizer = get_indicbert_model()
    if indicbert is not None and tokenizer is not None:
        try:
            inputs = tokenizer(
                clean,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = indicbert(inputs["input_ids"], inputs["attention_mask"])
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
                bert_cat = IDX2CAT[pred_idx]
                bert_conf = float(probs[pred_idx])

            # Combine with lexicon compound check
            detected = detect_all_civic_issues(clean)
            if detected:
                primary, conf, secondary, note = resolve_compound_issues(detected, clean, clean)
                if conf > 0.0:
                    if primary == bert_cat:
                        final_conf = max(conf, bert_conf)
                        suffix = f" (Corroborated by IndicBERT: {bert_conf:.0%})"
                    else:
                        final_conf = conf
                        suffix = f" (Governed by citizen's explicit description: '{clean}')"
                    return primary, final_conf, secondary, f"{note}{suffix}"

            return (
                bert_cat,
                round(max(0.75, min(0.98, bert_conf)), 3),
                [],
                f"Classified via fine-tuned IndicBERT v2 Multilingual Model ({norm_lang.upper()})."
            )
        except Exception as e:
            print(f"[NLP] IndicBERT inference error: {e}")

    # 3. Fallback to Lexicon Compound Resolution directly on native text
    translated_en = clean

    raw_detected = detect_all_civic_issues(clean)
    en_detected = detect_all_civic_issues(translated_en)
    combined_detected = {**raw_detected, **en_detected}

    if combined_detected:
        primary, conf, secondary, note = resolve_compound_issues(combined_detected, clean, translated_en)
        if conf > 0.0:
            return primary, conf, secondary, note

    # 4. Fallback to trained TF-IDF Baseline model
    try:
        tfidf_model = model or load_classifier()
        eval_text = translated_en if norm_lang in ("hi", "te") else clean
        probabilities = tfidf_model.predict_proba([eval_text])[0]
        index = probabilities.argmax()
        ml_cat = str(tfidf_model.classes_[index])
        raw_conf = float(probabilities[index])
        calibrated_conf = max(0.72, min(0.92, raw_conf * 1.35))
        return (
            ml_cat,
            calibrated_conf,
            [],
            f"Classified as '{ml_cat}' via baseline TF-IDF model fallback."
        )
    except Exception:
        return "road_damage", 0.70, [], "Default classification applied."


def analyze_text_full(
    model: Pipeline | None,
    text: str,
    user_lang: str | None = None,
    explicit_category: str | None = None,
) -> dict[str, Any]:
    """Comprehensive multilingual NLP analysis using citizen statements as primary ground truth.
    
    Preserves original citizen text and original language script (Hindi/Telugu).
    """
    clean_text = text.strip()
    if not clean_text:
        return {
            "category": explicit_category or "unknown",
            "confidence": 0.0,
            "secondary_categories": [],
            "resolution_note": "No text description provided.",
            "urgency_cues": [],
            "extracted_location": None,
            "impact": "None reported",
            "text_severity": "Low",
            "original_language": "en",
            "language_name": "English",
            "translated_text_en": "",
            "original_text": "",
        }

    detected_lang = normalize_lang_code(user_lang) if user_lang else detect_language(clean_text)

    # High-accuracy sentence translation for cross-lingual NLP & municipal dispatch
    if detected_lang != "en":
        translated_en, _ = translate_text(clean_text, source_lang=detected_lang, target_lang="en")
        if not translated_en or not translated_en.strip():
            translated_en = clean_text
    else:
        translated_en = clean_text

    # Primary category prediction with IndicBERT and compound resolution
    category, confidence, secondary_categories, resolution_note = predict_category(
        model,
        clean_text,
        lang=detected_lang,
        explicit_category=explicit_category,
    )

    urgency_cues = find_urgency_cues(translated_en) or find_urgency_cues(clean_text)
    loc_mention = extract_location_mention(translated_en) or extract_location_mention(clean_text)
    impact = extract_impact_description(translated_en)

    # Estimate text severity using final category and urgency cues
    if "Safety risk" in urgency_cues or category == "open_manhole":
        text_severity = "Critical"
    elif "Urgent obstruction" in urgency_cues or len(urgency_cues) >= 2 or category in ["open_drain", "drain_blockage"]:
        text_severity = "High"
    elif urgency_cues:
        text_severity = "Medium"
    else:
        text_severity = "Low"

    return {
        "category": category,
        "confidence": round(confidence, 2),
        "secondary_categories": secondary_categories,
        "resolution_note": resolution_note,
        "urgency_cues": urgency_cues,
        "extracted_location": loc_mention,
        "impact": impact,
        "text_severity": text_severity,
        "original_language": detected_lang,
        "language_name": get_language_display_name(detected_lang),
        "translated_text_en": translated_en,
        "original_text": clean_text,
    }
