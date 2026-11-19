"""Multilingual Intelligence & Translation Module for CivicFusion AI.

Supports three official civic languages:
1. English (en / en-IN)
2. Hindi (hi / hi-IN / हिंदी)
3. Telugu (te / te-IN / తెలుగు)

Features:
- Unicode script-based language identification (Devanagari, Telugu, Latin).
- Live translation across all directional pairs (hi<->en, te<->en, hi<->te).
- In-memory LRU caching for sub-millisecond repeated translations.
- Native civic domain keyword lexicon for Hindi and Telugu.
- Robust offline fallbacks in case of network interruptions.
"""

from functools import lru_cache
import json
import re
from typing import Any
import urllib.parse
import urllib.request
import requests
import certifi
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


SUPPORTED_LANGUAGES = {
    "en": {
        "code": "en",
        "speech_code": "en-IN",
        "name": "English",
        "native_name": "English",
        "label": "English",
        "script": "Latin",
    },
    "hi": {
        "code": "hi",
        "speech_code": "hi-IN",
        "name": "Hindi",
        "native_name": "हिंदी (Hindi)",
        "label": "हिंदी (Hindi)",
        "script": "Devanagari",
    },
    "te": {
        "code": "te",
        "speech_code": "te-IN",
        "name": "Telugu",
        "native_name": "తెలుగు (Telugu)",
        "label": "తెలుగు (Telugu)",
        "script": "Telugu",
    },
}

LANGUAGE_CHOICES = ["English", "हिंदी (Hindi)", "తెలుగు (Telugu)"]

# Map various representations to ISO 639-1 code
LANG_CODE_MAP = {
    "english": "en",
    "en": "en",
    "en-in": "en",
    "en-us": "en",
    "hindi": "hi",
    "hi": "hi",
    "hi-in": "hi",
    "हिंदी": "hi",
    "हिंदी (hindi)": "hi",
    "telugu": "te",
    "te": "te",
    "te-in": "te",
    "తెలుగు": "te",
    "తెలుగు (telugu)": "te",
}

# Native civic keyword dictionaries for high-precision local classification
NATIVE_CIVIC_KEYWORDS = {
    "hi": {
        "road_damage": [
            r"गड्ढा", r"गड्ढे", r"गड्डो", r"खड्डा", r"टूटी सड़क", r"सड़क खराब",
            r"डामर", r"सड़क टूट", r"सड़क पर गड्ढा", r"उबड़-खाबड़", r"खराब सड़क"
        ],
        "waste": [
            r"कचरा", r"कचरे", r"कूड़ा", r"कूड़े", r"गंदगी", r"कचरे का ढेर",
            r"कूड़ेदान", r"डस्टबिन", r"सड़ा हुआ", r"कूड़ा-कर्कट"
        ],
        "open_drain": [
            r"खुली नाली", r"खुला नाला", r"गंदा नाला", r"नाली", r"नाले",
            r"सीवर", r"नाली का पानी", r"टूटी नाली"
        ],
        "open_manhole": [
            r"मैनहोल", r"मैनहोल का ढक्कन", r"खुला मैनहोल", r"गटर",
            r"गटर का ढक्कन", r"गटर खुला"
        ],
        "streetlight_damage": [
            r"स्ट्रीट लाइट", r"स्ट्रीटलाइट", r"खंभा", r"बत्ती", r"लाइट खराब",
            r"अंधेरा", r"रात में लाइट", r"खंभे की लाइट"
        ],
        "water_leakage": [
            r"पानी का पाइप", r"पाइप फूटा", r"पाइप टूट", r"पानी बह रहा",
            r"पीने का पानी", r"पाइप लीकेज", r"पानी का रिसाव"
        ],
        "waterlogging": [
            r"जलभराव", r"पानी भरा", r"सड़क पर पानी", r"डूबी सड़क", r"बाढ़ जैसा"
        ],
        "damaged_sign": [
            r"साइन बोर्ड", r"साइनबोर्ड", r"यातायात संकेत", r"बोर्ड टूटा", r"दिशा सूचक"
        ],
    },
    "te": {
        "road_damage": [
            r"గుంత", r"గుంతలు", r"గుంటలు", r"రోడ్డు పాడైంది", r"పగిలిన రోడ్డు",
            r"రోడ్డు దెబ్బతింది", r"రోడ్డుపై గుంత", r"తారు రోడ్డు విరిగింది"
        ],
        "waste": [
            r"చెత్త", r"చెత్తకుండీ", r"వ్యర్థాలు", r"మురికి", r"చెత్త కుప్ప",
            r"చెత్తాచెదారం", r"దుర్వాసన", r"చెత్త పేరుకుపోయింది"
        ],
        "open_drain": [
            r"మురుగు కాలువ", r"డ్రైనేజీ", r"కాలువ", r"తెరిచిన కాలువ",
            r"మూత లేని కాలువ", r"మురుగు నీరు", r"డ్రైనేజ్"
        ],
        "open_manhole": [
            r"మ్యాన్‌హోల్", r"మ్యాన్ హోల్", r"గుంత మూత", r"తెరిచి ఉన్న మ్యాన్‌హోల్",
            r"గట్టర్ మూత", r"గట్టర్ తెరిచి"
        ],
        "streetlight_damage": [
            r"వీధి దీపం", r"వీధి లైటు", r"వీధి దీపాలు", r"స్తంభం",
            r"లైట్లు వెలగడం లేదు", r"చీకటి", r"స్ట్రీట్ లైట్"
        ],
        "water_leakage": [
            r"నీటి లీకేజీ", r"పైపు పగిలింది", r"తాగునీరు వృధా", r"నీరు కారుతోంది",
            r"వాటర్ పైప్", r"మంచినీటి పైపు"
        ],
        "waterlogging": [
            r"నీరు నిలిచిపోయింది", r"రోడ్డుపై నీరు", r"వరద నీరు", r"నీటి నిల్వ"
        ],
        "damaged_sign": [
            r"సైన్ బోర్డు", r"ట్రాఫిక్ బోర్డు", r"బోర్డు విరిగింది", r"రోడ్డు సూచిక"
        ],
    },
}


def normalize_lang_code(lang: str) -> str:
    """Normalize any language string, name, or code to 'en', 'hi', or 'te'."""
    clean = str(lang).strip().lower()
    return LANG_CODE_MAP.get(clean, "en")


def detect_language(text: str) -> str:
    """Detect language based on Unicode character distribution.
    
    - Devanagari range (U+0900 to U+097F) -> 'hi'
    - Telugu range (U+0C00 to U+0C7F) -> 'te'
    - Latin / English / other -> 'en'
    """
    if not text or not text.strip():
        return "en"
    
    hi_count = sum(1 for c in text if "\u0900" <= c <= "\u097F")
    te_count = sum(1 for c in text if "\u0C00" <= c <= "\u0C7F")
    
    if hi_count > 0 and hi_count >= te_count:
        return "hi"
    if te_count > 0:
        return "te"
    return "en"


def get_language_display_name(lang_code: str) -> str:
    """Get full display name with native script for a language code."""
    norm = normalize_lang_code(lang_code)
    meta = SUPPORTED_LANGUAGES.get(norm, SUPPORTED_LANGUAGES["en"])
    return meta["native_name"]


def get_language_speech_code(lang_code: str) -> str:
    """Get Google SpeechRecognition locale code (e.g. 'hi-IN', 'te-IN', 'en-IN')."""
    norm = normalize_lang_code(lang_code)
    meta = SUPPORTED_LANGUAGES.get(norm, SUPPORTED_LANGUAGES["en"])
    return meta["speech_code"]


OFFLINE_CIVIC_DICTIONARY = {
    # Hindi -> English
    ("hi", "en"): [
        (r"गड्ढ[ाे]|सड़क.*गड्ढ", "potholes on the road"),
        (r"कचरा|कूड़ा", "garbage waste accumulation"),
        (r"नाली|गंदा पानी", "open drainage water"),
        (r"मैनहोल", "open manhole"),
        (r"स्ट्रीट\s*लाइट|बत्ती", "broken streetlight"),
    ],
    # Telugu -> English
    ("te", "en"): [
        (r"గుంత[లు]*|రోడ్డు.*గుంత", "potholes on the road"),
        (r"చెత్త|వ్యర్థాలు", "garbage waste accumulation"),
        (r"కాలువ|మురుగు", "open drainage canal"),
        (r"మ్యాన్‌హోల్", "open manhole"),
        (r"వీధి\s*దీపాలు|లైట్లు", "broken streetlights"),
    ],
    # English -> Hindi
    ("en", "hi"): [
        (r"streetlight", "स्ट्रीट लाइट"),
        (r"pothole", "गड्ढा"),
        (r"garbage", "कचरा"),
        (r"drainage|drain", "नाली"),
        (r"manhole", "मैनहोल"),
    ],
    # English -> Telugu
    ("en", "te"): [
        (r"streetlight", "వీధి దీపాలు లైట్లు"),
        (r"pothole", "రోడ్డుపై గుంతలు"),
        (r"garbage", "మున్సిపల్ చెత్త"),
        (r"drainage|drain", "మురుగు కాలువ"),
        (r"manhole", "మ్యాన్‌హోల్"),
    ],
}


@lru_cache(maxsize=2048)
def _cached_api_translate(text: str, sl: str, tl: str) -> tuple[str, str]:
    """Execute raw HTTP translation query with multi-engine fallback, SSL resilience, and LRU caching."""
    clean = text.strip()
    if not clean:
        return "", sl
    
    encoded = urllib.parse.quote(clean)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    }

    # Primary: Google Translate API with certifi and unverified fallback
    for client in ["gtx", "dict-chrome-ex"]:
        url = f"https://translate.googleapis.com/translate_a/single?client={client}&sl={sl}&tl={tl}&dt=t&q={encoded}"
        for verify_opt in [certifi.where(), False]:
            try:
                resp = requests.get(url, headers=headers, timeout=5, verify=verify_opt)
                if resp.status_code == 200:
                    payload = resp.json()
                    if payload and isinstance(payload, list) and len(payload) > 0 and isinstance(payload[0], list):
                        segments = [seg[0] for seg in payload[0] if seg and seg[0]]
                        result = "".join(segments).strip()
                        if result:
                            detected_src = sl
                            if len(payload) > 2 and isinstance(payload[2], str):
                                detected_src = payload[2]
                            return result, detected_src
            except Exception:
                continue

    # Secondary Fallback: MyMemory Translation API
    try:
        url_mm = f"https://api.mymemory.translated.net/get?q={encoded}&langpair={sl}|{tl}"
        for verify_opt in [certifi.where(), False]:
            try:
                resp_mm = requests.get(url_mm, headers=headers, timeout=4, verify=verify_opt)
                if resp_mm.status_code == 200:
                    data = resp_mm.json()
                    mm_trans = data.get("responseData", {}).get("translatedText", "").strip()
                    if mm_trans and "MYMEMORY WARNING" not in mm_trans:
                        return mm_trans, sl
            except Exception:
                continue
    except Exception:
        pass

    # Tertiary Fallback: Offline Domain Civic Dictionary
    dict_rules = OFFLINE_CIVIC_DICTIONARY.get((sl, tl), [])
    for pat, replacement in dict_rules:
        if re.search(pat, clean, re.IGNORECASE):
            return f"{clean} ({replacement})", sl

    return clean, sl


def translate_text(
    text: str,
    source_lang: str = "auto",
    target_lang: str = "en"
) -> tuple[str, str]:
    """Translate text between English, Hindi, and Telugu.
    
    Args:
        text: Input string.
        source_lang: 'en', 'hi', 'te', or 'auto'.
        target_lang: 'en', 'hi', or 'te'.
        
    Returns:
        (translated_text, effective_source_lang)
    """
    clean = text.strip()
    if not clean:
        return "", source_lang

    tl = normalize_lang_code(target_lang)
    
    # Resolve source language if auto
    if source_lang == "auto" or not source_lang:
        sl = detect_language(clean)
    else:
        sl = normalize_lang_code(source_lang)
        
    # If source and target are the same, no translation required
    if sl == tl:
        return clean, sl

    # Query API with caching
    try:
        translated, detected = _cached_api_translate(clean, sl, tl)
        return translated, (detected if detected in ("en", "hi", "te") else sl)
    except Exception:
        return clean, sl


def match_native_civic_category(text: str, lang: str | None = None) -> tuple[str | None, float]:
    """Check text against native Hindi or Telugu civic keyword patterns.
    
    Returns (category, confidence) if matched, else (None, 0.0).
    """
    if not text:
        return None, 0.0
        
    detected_lang = lang or detect_language(text)
    norm_lang = normalize_lang_code(detected_lang)
    
    lexicon = NATIVE_CIVIC_KEYWORDS.get(norm_lang)
    if not lexicon:
        return None, 0.0
        
    for category, patterns in lexicon.items():
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                # High confidence for explicit native municipal defect keywords
                return category, 0.95
                
    return None, 0.0

