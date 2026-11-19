"""Speech and voice processing module for CivicFusion AI.

Provides:
1. Multilingual microphone audio transcription (Speech-to-Text) using SpeechRecognition or Whisper.
2. Direct language support for English (en-IN), Hindi (hi-IN), and Telugu (te-IN).
3. Handling of recorded audio buffers (WAV from st.audio_input) or uploaded audio files.
4. Multilingual demo voice samples across English, Hindi, and Telugu.
"""

from io import BytesIO
from typing import Any


SAMPLE_VOICE_COMPLAINTS = [
    # English Samples
    {
        "id": "voice_en_1",
        "lang": "en",
        "language_name": "English",
        "city": "City Zone",
        "title": "🕳️ [EN] Large Pothole on GT Road (NH-44)",
        "transcript": "There is a very large dangerous pothole on the GT Road near the main flyover. Multiple two-wheelers are swerving to avoid accidents.",
        "location": "Main Highway Corridor",
        "lat": 31.2241,
        "lon": 75.7709,
    },
    {
        "id": "voice_en_2",
        "lang": "en",
        "language_name": "English",
        "city": "City Zone",
        "title": "🗑️ [EN] Garbage Pile at Sugar Mill Crossing",
        "transcript": "A huge pile of municipal garbage has accumulated on Hoshiarpur Road near the market crossing. The stench is unbearable and the footpath is blocked.",
        "location": "Market Crossing Road",
        "lat": 31.2294,
        "lon": 75.7788,
    },
    {
        "id": "voice_en_3",
        "lang": "en",
        "language_name": "English",
        "city": "City Zone",
        "title": "🚰 [EN] Water Pipe Leakage at BMC Chowk",
        "transcript": "Clean drinking water has been gushing from a burst municipal pipe near the central junction on the main road for two days.",
        "location": "Central Junction Road",
        "lat": 31.3195,
        "lon": 75.5828,
    },
    {
        "id": "voice_en_4",
        "lang": "en",
        "language_name": "English",
        "city": "City Zone",
        "title": "💡 [EN] Broken Streetlights at Model Town",
        "transcript": "The streetlight poles near the commercial market are broken and dark at night. The walkway is completely unlit and unsafe for pedestrians.",
        "location": "Commercial Market Area",
        "lat": 31.3090,
        "lon": 75.5802,
    },
    {
        "id": "voice_en_5",
        "lang": "en",
        "language_name": "English",
        "city": "City Zone",
        "title": "🚨 [EN] Open Drain Hazard at Gol Chowk",
        "transcript": "An open storm sewer drain cover is completely broken right at the railway station road crossing. It is an urgent hazard for public safety.",
        "location": "Station Road Crossing",
        "lat": 31.2185,
        "lon": 75.7672,
    },

    # Hindi Samples (हिंदी)
    {
        "id": "voice_hi_1",
        "lang": "hi",
        "language_name": "हिंदी (Hindi)",
        "city": "City Zone",
        "title": "🕳️ [HI] सड़क पर खतरनाक गड्ढा (Main Highway)",
        "transcript": "मुख्य फ्लाईओवर के पास सड़क पर एक बहुत खतरनाक और बड़ा गड्ढा है। दोपहिया वाहन दुर्घटना से बचने के लिए इधर-उधर मुड़ रहे हैं।",
        "location": "Main Highway Corridor",
        "lat": 31.2241,
        "lon": 75.7709,
    },
    {
        "id": "voice_hi_2",
        "lang": "hi",
        "language_name": "हिंदी (Hindi)",
        "city": "City Zone",
        "title": "🗑️ [HI] कचरे का बड़ा ढेर (Market Crossing)",
        "transcript": "मार्केट क्रॉसिंग के पास मुख्य सड़क पर कचरे का भारी ढेर जमा हो गया है। बदबू असहनीय है और पैदल चलने का रास्ता बंद है।",
        "location": "Market Crossing Road",
        "lat": 31.2294,
        "lon": 75.7788,
    },
    {
        "id": "voice_hi_3",
        "lang": "hi",
        "language_name": "हिंदी (Hindi)",
        "city": "City Zone",
        "title": "🚰 [HI] पानी की पाइपलाइन लीकेज (Central Junction)",
        "transcript": "सेंट्रल जंक्शन पर मुख्य सड़क पर पिछले दो दिनों से टूटी पाइप से साफ पीने का पानी लगातार बह रहा है और सड़क खराब हो रही है।",
        "location": "Central Junction Road",
        "lat": 31.3195,
        "lon": 75.5828,
    },
    {
        "id": "voice_hi_4",
        "lang": "hi",
        "language_name": "हिंदी (Hindi)",
        "city": "City Zone",
        "title": "💡 [HI] खराब स्ट्रीट लाइट और अंधेरा (Market Area)",
        "transcript": "कमर्शियल मार्केट के पास स्ट्रीट लाइट के खंभे खराब हैं। रात में पूरा अंधेरा रहता है जिससे राहगीरों के लिए खतरा बना रहता है।",
        "location": "Commercial Market Area",
        "lat": 31.3090,
        "lon": 75.5802,
    },
    {
        "id": "voice_hi_5",
        "lang": "hi",
        "language_name": "हिंदी (Hindi)",
        "city": "City Zone",
        "title": "🚨 [HI] खुला नाला व टूटा ढक्कन (Station Road)",
        "transcript": "स्टेशन रोड क्रॉसिंग पर खुली नाली का ढक्कन टूटा हुआ है। यह जनता और छोटे बच्चों के लिए बहुत गंभीर खतरा है।",
        "location": "Station Road Crossing",
        "lat": 31.2185,
        "lon": 75.7672,
    },

    # Telugu Samples (తెలుగు)
    {
        "id": "voice_te_1",
        "lang": "te",
        "language_name": "తెలుగు (Telugu)",
        "city": "City Zone",
        "title": "🕳️ [TE] రోడ్డుపై ప్రమాదకరమైన గుంత (Main Highway)",
        "transcript": "ప్రధాన ఫ్లైఓవర్ సమీపంలో చాలా ప్రమాదకరమైన పెద్ద గుంత ఉంది. ద్విచక్ర వాహనాలు ప్రమాదాలకు గురవుతున్నాయి.",
        "location": "Main Highway Corridor",
        "lat": 31.2241,
        "lon": 75.7709,
    },
    {
        "id": "voice_te_2",
        "lang": "te",
        "language_name": "తెలుగు (Telugu)",
        "city": "City Zone",
        "title": "🗑️ [TE] చెత్త కుప్ప మరియు దుర్వాసన (Market Crossing)",
        "transcript": "మార్కెట్ క్రాసింగ్ వద్ద ప్రధాన రహదారిపై మున్సిపల్ చెత్త కుప్పగా పేరుకుపోయింది. దుర్వాసన భరించలేకుండా ఉంది మరియు కాలినడక దారి మూసుకుపోయింది.",
        "location": "Market Crossing Road",
        "lat": 31.2294,
        "lon": 75.7788,
    },
    {
        "id": "voice_te_3",
        "lang": "te",
        "language_name": "తెలుగు (Telugu)",
        "city": "City Zone",
        "title": "🚰 [TE] మంచినీటి పైపు లీకేజీ (Central Junction)",
        "transcript": "సెంట్రల్ జంక్షన్ వద్ద ప్రధాన రహదారిపై పైపు పగిలి రెండు రోజులుగా తాగునీరు వృధాగా ప్రవహిస్తోంది, వెంటనే మరమ్మతు చేయండి.",
        "location": "Central Junction Road",
        "lat": 31.3195,
        "lon": 75.5828,
    },
    {
        "id": "voice_te_4",
        "lang": "te",
        "language_name": "తెలుగు (Telugu)",
        "city": "City Zone",
        "title": "💡 [TE] వీధి దీపాలు వెలగడం లేదు (Market Area)",
        "transcript": "కమర్షియల్ మార్కెట్ వద్ద వీధి లైట్లు పాడై రాత్రిపూట పూర్తిగా చీకటిగా ఉంటోంది. ప్రజలు నడవడానికి చాలా ఇబ్బంది పడుతున్నారు.",
        "location": "Commercial Market Area",
        "lat": 31.3090,
        "lon": 75.5802,
    },
    {
        "id": "voice_te_5",
        "lang": "te",
        "language_name": "తెలుగు (Telugu)",
        "city": "City Zone",
        "title": "🚨 [TE] తెరిచిన మురుగు కాలువ (Station Road)",
        "transcript": "స్టేషన్ రోడ్ క్రాసింగ్ వద్ద మురుగు కాలువ కాంక్రీట్ మూత విరిగిపోయి తెరిచి ఉంది. ఇది ప్రజల భద్రతకు తీవ్ర ముప్పు.",
        "location": "Station Road Crossing",
        "lat": 31.2185,
        "lon": 75.7672,
    },
]


def resolve_speech_locale(lang_code: str | None) -> tuple[str, str]:
    """Map language identifier to (Google Speech locale, Whisper language code)."""
    if not lang_code:
        return "en-IN", "en"
    clean = lang_code.strip().lower()
    if clean in ("hi", "hi-in", "hindi", "हिंदी"):
        return "hi-IN", "hi"
    if clean in ("te", "te-in", "telugu", "తెలుగు"):
        return "te-IN", "te"
    return "en-IN", "en"


def get_sample_voice_complaints(lang: str = "en") -> list[dict[str, Any]]:
    """Return voice complaint presets filtered or prioritized by language."""
    clean = lang.strip().lower()
    norm = "en"
    if clean in ("hi", "hi-in", "hindi", "हिंदी"):
        norm = "hi"
    elif clean in ("te", "te-in", "telugu", "తెలుగు"):
        norm = "te"
    
    # Priority samples matching the language first, followed by others
    matching = [item for item in SAMPLE_VOICE_COMPLAINTS if item.get("lang") == norm]
    other = [item for item in SAMPLE_VOICE_COMPLAINTS if item.get("lang") != norm]
    return matching + other


def transcribe_audio(audio_data: Any, language_code: str = "en-IN") -> tuple[str, str]:
    """Transcribe live microphone or uploaded audio into text.
    
    Accepts:
    - audio_data: st.audio_input file-like buffer (WAV), BytesIO, or file path.
    - language_code: Speech language ('hi-IN', 'te-IN', 'en-IN', or 'hi', 'te', 'en').
    
    Returns:
    - transcribed_text: str (Devanagari for Hindi, Telugu script for Telugu, English for English)
    - status_message: str
    """
    if audio_data is None:
        return "", "No audio provided."

    # If BytesIO or UploadedFile, reset position
    if hasattr(audio_data, "seek"):
        audio_data.seek(0)

    google_locale, whisper_lang = resolve_speech_locale(language_code)

    # 1. Google Speech Recognition (free, fast, built into SpeechRecognition package)
    try:
        import speech_recognition as sr  # type: ignore

        r = sr.Recognizer()
        
        # AudioFile accepts file path or file-like object with seek/read
        with sr.AudioFile(audio_data) as source:
            # Adjust for ambient noise and record
            audio_content = r.record(source)
            # Pass explicit locale: 'hi-IN' produces Devanagari, 'te-IN' produces Telugu script
            text = r.recognize_google(audio_content, language=google_locale)
            if text and text.strip():
                lang_display = "Hindi (हिंदी)" if "hi" in google_locale else ("Telugu (తెలుగు)" if "te" in google_locale else "English")
                return text.strip(), f"✅ Transcribed successfully in {lang_display} using Google Speech-to-Text!"
    except sr.UnknownValueError:
        return "", "⚠️ Speech could not be understood. Please speak clearly into the microphone."
    except sr.RequestError as req_err:
        return "", f"Speech API network error ({req_err}). Please check your internet connection."
    except Exception as exc:
        pass

    # 2. Try Whisper if installed
    try:
        import whisper  # type: ignore
        model = whisper.load_model("tiny")
        result = model.transcribe(audio_data, language=whisper_lang)
        txt = result.get("text", "").strip()
        if txt:
            return txt, f"✅ Transcribed successfully using OpenAI Whisper ({whisper_lang})!"
    except Exception:
        pass

    return (
        "",
        "Could not transcribe audio. Ensure your microphone is clear and internet connection is active."
    )
