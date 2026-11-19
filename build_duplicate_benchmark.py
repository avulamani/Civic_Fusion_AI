"""Generate verified ground-truth duplicate and non-duplicate benchmark pairs.
Used to calibrate and evaluate:
1. LaBSE multilingual semantic embeddings vs. TF-IDF cosine similarity
2. Haversine distance spatial filtering (< 250m)
3. Modality and category consistency
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "datasets" / "duplicate_pairs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK_PAIRS = [
    # 1. Cross-Lingual True Duplicates (Same location within 250m, same category, different language)
    {
        "id": "DUP-001",
        "description_a": "There is a large dangerous pothole near Gol Chowk. Two wheelers are swerving to avoid accidents.",
        "lang_a": "en",
        "category_a": "road_damage",
        "lat_a": 31.2185,
        "lon_a": 75.7672,
        "description_b": "गोल चौक के पास सड़क पर बहुत बड़ा गड्ढा है। गाड़ियां गिर रही हैं और दुर्घटना का खतरा है।",
        "lang_b": "hi",
        "category_b": "road_damage",
        "lat_b": 31.2188,
        "lon_b": 75.7675,
        "expected_duplicate": True,
        "scenario": "Cross-lingual EN-HI: Same pothole within 40 meters",
    },
    {
        "id": "DUP-002",
        "description_a": "A heavy sewer chamber cover is missing near the railway station road, leaving a deep open manhole.",
        "lang_a": "en",
        "category_a": "open_manhole",
        "lat_a": 31.2210,
        "lon_a": 75.7690,
        "description_b": "రైల్వే స్టేషన్ రోడ్డు వద్ద మ్యాన్‌హోల్ మూత లేకుండా తెరిచి ఉంది. పాదచారులకు తీవ్ర ప్రాణాపాయం ఉంది.",
        "lang_b": "te",
        "category_b": "open_manhole",
        "lat_b": 31.2214,
        "lon_b": 75.7692,
        "expected_duplicate": True,
        "scenario": "Cross-lingual EN-TE: Same open manhole within 50 meters",
    },
    {
        "id": "DUP-003",
        "description_a": "सब्जी मंडी के बाहर कचरे का विशाल ढेर लगा हुआ है। भारी दुर्गंध आ रही है और रास्ता बंद है।",
        "lang_a": "hi",
        "category_a": "waste",
        "lat_a": 31.2250,
        "lon_a": 75.7720,
        "description_b": "కూరగాయల మార్కెట్ బయట చెత్త కుప్పలు పేరుకుపోయి విపరీతమైన దుర్వాసన వస్తోంది.",
        "lang_b": "te",
        "category_b": "waste",
        "lat_b": 31.2252,
        "lon_b": 75.7723,
        "expected_duplicate": True,
        "scenario": "Cross-lingual HI-TE: Same garbage pile within 35 meters",
    },
    {
        "id": "DUP-004",
        "description_a": "The streetlights along Model Town market are broken and completely dark at night.",
        "lang_a": "en",
        "category_a": "streetlight_damage",
        "lat_a": 31.3090,
        "lon_a": 75.5802,
        "description_b": "मॉडल टाउन मार्केट में स्ट्रीट लाइटें खराब हैं और रात में पूरा अंधेरा रहता है।",
        "lang_b": "hi",
        "category_b": "streetlight_damage",
        "lat_b": 31.3093,
        "lon_b": 75.5806,
        "expected_duplicate": True,
        "scenario": "Cross-lingual EN-HI: Same streetlight outage within 50 meters",
    },
    {
        "id": "DUP-005",
        "description_a": "Municipal drinking water pipeline has burst at BMC Chowk and potable water is gushing out.",
        "lang_a": "en",
        "category_a": "water_leakage",
        "lat_a": 31.3195,
        "lon_a": 75.5828,
        "description_b": "BMC చౌక్ వద్ద మంచినీటి పైపు పగిలిపోయి నీరు రోడ్డుపై ఉధృతంగా ప్రవహిస్తోంది.",
        "lang_b": "te",
        "category_b": "water_leakage",
        "lat_b": 31.3196,
        "lon_b": 75.5830,
        "expected_duplicate": True,
        "scenario": "Cross-lingual EN-TE: Same pipe burst within 25 meters",
    },
    {
        "id": "DUP-006",
        "description_a": "There is a very deep crater on GT Road NH-44 near Phagwara flyover.",
        "lang_a": "en",
        "category_a": "road_damage",
        "lat_a": 31.2241,
        "lon_a": 75.7709,
        "description_b": "Deep pothole along GT road NH-44 right by the Phagwara flyover, damaging car tyres.",
        "lang_b": "en",
        "category_b": "road_damage",
        "lat_a": 31.2241,
        "lon_a": 75.7709,
        "lat_b": 31.2244,
        "lon_b": 75.7712,
        "expected_duplicate": True,
        "scenario": "Monolingual EN-EN: Same pothole within 45 meters",
    },

    # 2. Hard Non-Duplicates: Same Location / Nearby, but DIFFERENT Civic Problem
    {
        "id": "NON-001",
        "description_a": "Garbage is accumulating near the bus stand. Overflowing trash bin.",
        "lang_a": "en",
        "category_a": "waste",
        "lat_a": 31.2290,
        "lon_a": 75.7750,
        "description_b": "There is a broken streetlight pole near the bus stand. Pitch black at night.",
        "lang_b": "en",
        "category_b": "streetlight_damage",
        "lat_b": 31.2290,
        "lon_b": 75.7750,
        "expected_duplicate": False,
        "scenario": "Co-located Different Category: Garbage vs Streetlight at exact same GPS",
    },
    {
        "id": "NON-002",
        "description_a": "खुला मैनहोल रेलवे स्टेशन के पास अत्यंत खतरनाक है।",
        "lang_a": "hi",
        "category_a": "open_manhole",
        "lat_a": 31.2210,
        "lon_a": 75.7690,
        "description_b": "रेलवे स्टेशन के पास सड़क पर पानी की पाइपलाइन फट गई है और पानी बह रहा है।",
        "lang_b": "hi",
        "category_b": "water_leakage",
        "lat_b": 31.2212,
        "lon_b": 75.7691,
        "expected_duplicate": False,
        "scenario": "Co-located Different Category: Open Manhole vs Water Leakage in Hindi",
    },
    {
        "id": "NON-003",
        "description_a": "కూరగాయల మార్కెట్ వద్ద డ్రైనేజీ పూడికతో జామ్ అయి నీరు రోడ్డుపైకి వస్తోంది.",
        "lang_a": "te",
        "category_a": "drain_blockage",
        "lat_a": 31.2250,
        "lon_a": 75.7720,
        "description_b": "కూరగాయల మార్కెట్ వద్ద ట్రాఫిక్ సిగ్నల్ పనిచేయడం లేదు, తీవ్ర గందరగోళం నెలకొంది.",
        "lang_b": "te",
        "category_b": "traffic_signal_damage",
        "lat_b": 31.2251,
        "lon_b": 75.7721,
        "expected_duplicate": False,
        "scenario": "Co-located Different Category: Drain Blockage vs Traffic Signal in Telugu",
    },

    # 3. Spatial Non-Duplicates: SAME Category, Similar Description, but DISTANT Locations (> 250m)
    {
        "id": "NON-004",
        "description_a": "There is a deep road pothole near the school gate in Phagwara.",
        "lang_a": "en",
        "category_a": "road_damage",
        "lat_a": 31.2241,
        "lon_a": 75.7709,
        "description_b": "There is a deep road pothole near the school gate in Jalandhar.",
        "lang_b": "en",
        "category_b": "road_damage",
        "lat_b": 31.3195,
        "lon_b": 75.5828, # ~20 km away
        "expected_duplicate": False,
        "scenario": "Identical Wording at Distant Coordinates: Phagwara vs Jalandhar (~20km)",
    },
    {
        "id": "NON-005",
        "description_a": "सड़क पर गहरा गड्ढा है जिससे गाड़ियां गिर रही हैं।",
        "lang_a": "hi",
        "category_a": "road_damage",
        "lat_a": 31.2185,
        "lon_a": 75.7672, # Gol Chowk
        "description_b": "बाजार के पास सड़क पर गहरा गड्ढा है।",
        "lang_b": "hi",
        "category_b": "road_damage",
        "lat_b": 31.2294,
        "lon_b": 75.7788, # Sugar Mill (~1.6 km away)
        "expected_duplicate": False,
        "scenario": "Same Issue at Distant Spots: Gol Chowk vs Sugar Mill (1.6 km apart)",
    },
    {
        "id": "NON-006",
        "description_a": "Uncollected garbage heap along the road creating severe stench.",
        "lang_a": "en",
        "category_a": "waste",
        "lat_a": 31.2250,
        "lon_a": 75.7720,
        "description_b": "రోడ్డు పక్కన చెత్త కుప్పలు పేరుకుపోయి విపరీతమైన దుర్వాసన వస్తోంది.",
        "lang_b": "te",
        "category_b": "waste",
        "lat_b": 31.3090,
        "lon_b": 75.5802, # Model Town Jalandhar (~22 km away)
        "expected_duplicate": False,
        "scenario": "Semantically Similar in Different Cities: Phagwara vs Jalandhar (22 km)",
    },
]


def save_benchmark():
    out_file = OUTPUT_DIR / "duplicate_benchmark_dataset.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(BENCHMARK_PAIRS, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(BENCHMARK_PAIRS)} duplicate evaluation pairs to {out_file}")


if __name__ == "__main__":
    save_benchmark()

