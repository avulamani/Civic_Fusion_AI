"""Build balanced multilingual civic complaint dataset for English, Hindi, and Telugu.
Covers 11 municipal categories with genuine native domain expressions,
extracted landmarks, impact cues, and urgency markers.
Generates:
  datasets/nlp/multilingual_complaints.csv
  datasets/nlp/train.csv
  datasets/nlp/val.csv
  datasets/nlp/test.csv
  datasets/nlp/metadata.json
"""

import json
from pathlib import Path
import random
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "datasets" / "nlp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

# Real municipal landmarks across Punjab and Indian urban corridors
LANDMARKS_EN = [
    "near the school gate", "outside the main market", "at Gol Chowk crossing",
    "along GT Road NH-44", "near Model Town park", "opposite the hospital entrance",
    "near the railway station", "at Sugar Mill crossing", "along Rama Mandi highway",
    "in front of the bus stand", "near College Road", "beside the community center"
]

LANDMARKS_HI = [
    "स्कूल के मुख्य द्वार के पास", "सब्जी मंडी के बाहर", "गोल चौक चौराहे पर",
    "जीटी रोड एनएच-44 के पास", "मॉडल टाउन पार्क के पास", "अस्पताल के प्रवेश द्वार के सामने",
    "रेलवे स्टेशन के पास", "शुगर मिल क्रॉसिंग पर", "रामा मंडी मुख्य मार्ग पर",
    "बस स्टैंड के सामने", "कॉलेज रोड के पास", "सामुदायिक केंद्र के पास"
]

LANDMARKS_TE = [
    "పాఠశాల మెయిన్ గేట్ వద్ద", "కూరగాయల మార్కెట్ బయట", "గోల్ చౌక్ జంక్షన్ వద్ద",
    "జీటీ రోడ్ జాతీయ రహదారి వెంబడి", "మోడల్ టౌన్ పార్క్ దగ్గర", "ఆసుపత్రి ప్రవేశ ద్వారం ఎదురుగా",
    "రైల్వే స్టేషన్ సమీపంలో", "షుగర్ మిల్లు క్రాసింగ్ వద్ద", "రామా మండి ప్రధాన రహదారిపై",
    "బస్ స్టాండ్ ఎదురుగా", "కాలేజీ రోడ్డు దగ్గర", "కమ్యూనిటీ హాల్ పక్కన"
]

# Category-specific template patterns with genuine civic terminology
TEMPLATES = {
    "road_damage": {
        "en": [
            "There is a deep dangerous pothole {loc}. Two-wheelers are swerving and riders risk falling.",
            "Large crater and cracked asphalt {loc}. Vehicles are getting damaged and traffic is moving very slowly.",
            "The road surface has completely broken down {loc}. Huge potholes have formed making driving hazardous.",
            "Severe alligator cracks and road cavities {loc}. Commuters face great difficulty crossing the stretch.",
            "Deep road pothole {loc} causing accidents especially at night for bike riders.",
        ],
        "hi": [
            "{loc} सड़क पर एक बहुत बड़ा और गहरा गड्ढा है। दोपहिया वाहन गिर रहे हैं और दुर्घटना का खतरा है।",
            "{loc} सड़क पूरी तरह से टूट चुकी है और गहरे गड्ढे बन गए हैं। गाड़ियां हिचकोले खा रही हैं।",
            "{loc} डामर उखड़ गया है और सड़क पर खतरनाक खड्डे हैं। वाहनों को निकलने में भारी परेशानी हो रही है।",
            "{loc} सड़क पर बड़ा गड्ढा होने से गाड़ियां अनियंत्रित हो रही हैं। कृपया तुरंत सड़क की मरम्मत करें।",
            "{loc} उबड़-खाबड़ और टूटी सड़क के कारण आए दिन दुर्घटनाएं हो रही हैं।",
        ],
        "te": [
            "{loc} రోడ్డుపై చాలా పెద్ద ప్రమాదకరమైన గుంత ఉంది. ద్విచక్ర వాహనాలు జారిపడుతున్నాయి.",
            "{loc} రోడ్డు పూర్తిగా పాడైపోయి పెద్ద గుంతలు పడ్డాయి. వాహనాల రాకపోకలకు తీవ్ర అంతరాయం కలుగుతోంది.",
            "{loc} తారు రోడ్డు విరిగిపోయి రాళ్ళు తేలాయి. ప్రయాణికులు తీవ్ర ఇబ్బందులు పడుతున్నారు.",
            "{loc} రోడ్డుపై గుంతల వల్ల ప్రమాదాలు జరుగుతున్నాయి. వెంటనే రోడ్డు మరమ్మతులు చేయాలి.",
            "{loc} పగిలిన రోడ్డు ఉపరితలం వల్ల వాహనాలు నెమ్మదిగా కదులుతూ ట్రాఫిక్ జామ్ అవుతోంది.",
        ],
        "impact": "Vehicles / commuters experiencing difficulty, skidding, or hazard",
        "urgency": "Urgent obstruction",
    },
    "open_manhole": {
        "en": [
            "There is an open manhole without any cover {loc}. It is a fatal life safety hazard for pedestrians and cyclists.",
            "A heavy sewer chamber cover is missing {loc}, leaving a gaping deep hole on the roadway.",
            "The manhole lid is broken and collapsed inside {loc}. People might fall into the deep sewer.",
            "Extremely hazardous uncovered drainage manhole {loc}. Immediate replacement of chamber grate required.",
        ],
        "hi": [
            "{loc} गटर का ढक्कन गायब है और खुला मैनहोल राहगीरों के लिए जानलेवा साबित हो सकता है।",
            "{loc} खुला मैनहोल है जिसका ढक्कन टूटा हुआ है। रात में किसी के गिरने का गंभीर खतरा है।",
            "{loc} सड़क के बीचों-बीच मैनहोल खुला पड़ा है। तुरंत नया ढक्कन लगवाया जाए।",
            "{loc} सीवर का खुला गड्ढा बच्चों और बुजुर्गों के लिए अत्यंत खतरनाक है।",
        ],
        "te": [
            "{loc} మ్యాన్‌హోల్ మూత లేకుండా తెరిచి ఉంది. పాదచారులు లోపల పడిపోయే ప్రాణాపాయం ఉంది.",
            "{loc} రోడ్డుపై గట్టర్ మూత విరిగిపోయి ప్రమాదకరంగా ఉంది. తక్షణమే మూత ఏర్పాటు చేయాలి.",
            "{loc} తెరిచి ఉన్న మురుగునీటి మ్యాన్‌హోల్ వల్ల రాత్రి వేళల్లో తీవ్ర ప్రమాదాలు జరిగే అవకాశం ఉంది.",
            "{loc} సీవర్ చాంబర్ మూత ఎత్తుకెళ్ళడంతో పెద్ద గుంత ఏర్పడింది. తక్షణ భద్రతా చర్యలు అవసరం.",
        ],
        "impact": "Pedestrian walkway unsafe or impassable",
        "urgency": "Safety risk",
    },
    "open_drain": {
        "en": [
            "An open drainage canal along the roadside {loc} is uncovered and foul water is flowing dangerously.",
            "Excavated roadside trench and open storm drain {loc} left without safety barriers or barricades.",
            "Dangerous deep open drain {loc}. Pedestrians and stray animals are at risk of falling in.",
            "The concrete slabs covering the stormwater drain {loc} are broken and missing.",
        ],
        "hi": [
            "{loc} सड़क किनारे खुली नाली बह रही है जिससे बदबू आ रही है और गिरने का खतरा है।",
            "{loc} खुला नाला बिना किसी बैरिकेड के पड़ा है, कभी भी कोई गाड़ी नाले में गिर सकती है।",
            "{loc} जल निकासी की नाली पर स्लैब टूटे हुए हैं और गहरी खाई खुली हुई है।",
            "{loc} गहरी खुली नाली के कारण पैदल चलने वालों के लिए रास्ता बहुत खतरनाक हो गया है।",
        ],
        "te": [
            "{loc} రోడ్డు పక్కన ఉన్న మురుగు కాలువపై మూతలు లేవు, దుర్వాసనతో పాటు పడిపోయే ప్రమాదం ఉంది.",
            "{loc} కాలువ తవ్వి తెరిచి ఉంచారు, ఎటువంటి రక్షణ కంచెలు లేకపోవడంతో భయంగా ఉంది.",
            "{loc} పెద్ద మురుగునీటి డ్రైనేజ్ కాలువ తెరిచి ఉండటం వల్ల పాదచారులకు రక్షణ లేదు.",
            "{loc} రోడ్డు పక్క డ్రైనేజీ కాలువ స్లాబులు విరిగిపోయి ప్రమాదకరంగా మారాయి.",
        ],
        "impact": "Pedestrian walkway unsafe or impassable",
        "urgency": "Urgent obstruction",
    },
    "drain_blockage": {
        "en": [
            "The municipal storm drain {loc} is completely clogged with garbage and silt, causing dirty sewage to back up.",
            "Heavy rubbish blockage inside the roadside sewer {loc}. Water is not draining and overflowing onto the street.",
            "Clogged and choked drainage channel {loc}. Rotten waste has blocked water passage entirely.",
            "Drainage pipe is fully jammed with silt and plastic bottles {loc}. Dirty water is flooding the walkway.",
        ],
        "hi": [
            "{loc} कचरा और प्लास्टिक फंसने से नाली पूरी तरह जाम हो गई है और गंदा पानी सड़क पर आ रहा है।",
            "{loc} नाला चोक होने की वजह से सीवर का बदबूदार पानी दुकानों के आगे भर रहा है।",
            "{loc} नाली में गाद जमने से पानी की निकासी बंद है। तुरंत सफाई करवाई जाए।",
            "{loc} जाम नाले के कारण मच्छरों का प्रकोप बढ़ गया है और भारी दुर्गंध आ रही है।",
        ],
        "te": [
            "{loc} మురుగు కాలువలో చెత్త పేరుకుపోయి నీరు ప్రవహించకుండా జామ్ అయిపోయింది.",
            "{loc} డ్రైనేజీ పూడిక పేరుకుపోవడం వల్ల మురుగు నీరు రోడ్డుపైకి ఉప్పొంగుతోంది.",
            "{loc} ప్లాస్టిక్ వ్యర్థాలతో కాలువ మూసుకుపోయింది, నీటి పారుదల లేక దుర్వాసన వస్తోంది.",
            "{loc} బ్లాక్ అయిన డ్రైనేజ్ వల్ల దోమలు పెరిగి అనారోగ్య సమస్యలు తలెత్తుతున్నాయి.",
        ],
        "impact": "Sanitary hazard and foul environmental conditions",
        "urgency": "Urgent obstruction",
    },
    "waste": {
        "en": [
            "A huge illegal garbage dump and waste heap has accumulated {loc}. The stench is unbearable.",
            "Public dustbin is overflowing with solid waste and garbage has spilled across the street {loc}.",
            "Rotting municipal refuse and trash bags dumped on the pavement {loc}, attracting stray cattle and flies.",
            "Uncollected garbage pile {loc} creating a severe public health hazard and blocking the footpath.",
        ],
        "hi": [
            "{loc} कचरे का विशाल ढेर लग गया है और बदबू के कारण सांस लेना मुश्किल हो रहा है।",
            "{loc} कूड़ेदान पूरी तरह भर चुका है और चारों तरफ गंदगी और कूड़ा फैला हुआ है।",
            "{loc} सफाई कर्मचारी कई दिनों से कूड़ा नहीं उठा रहे हैं, सड़क पर कचरे का अंबार लगा है।",
            "{loc} सड़ा हुआ कचरा पड़ा होने से संक्रामक बीमारियों का खतरा बढ़ गया है।",
        ],
        "te": [
            "{loc} రోడ్డు పక్కన చెత్త కుప్పలు పేరుకుపోయి విపరీతమైన దుర్వాసన వస్తోంది.",
            "{loc} చెత్తకుండీ నిండిపోయి వ్యర్థాలు రోడ్డుపైకి చెల్లాచెదురుగా పడి ఉన్నాయి.",
            "{loc} చాలా రోజులుగా మునిసిపాలిటీ వారు చెత్తను ఎత్తకపోవడంతో ఈగలు, దోమలు ముసురుతున్నాయి.",
            "{loc} చెత్తాచెదారంతో కాలిబాట పూర్తిగా మూసుకుపోయింది, అపరిశుభ్ర వాతావరణం నెలకొంది.",
        ],
        "impact": "Sanitary hazard and foul environmental conditions",
        "urgency": "Urgent obstruction",
    },
    "streetlight_damage": {
        "en": [
            "The streetlights {loc} are completely non-functional and dark at night, causing high risk of accidents.",
            "Broken lamp post and fallen streetlight fixture {loc} leaving live electrical components exposed.",
            "Pitch black road {loc} because streetlight bulbs have fused and not been replaced for weeks.",
            "Damaged leaning light pole {loc} with dangling wires posing electrocution danger during rain.",
        ],
        "hi": [
            "{loc} स्ट्रीट लाइटें कई दिनों से बंद हैं और रात में पूरा इलाका अंधेरे में डूबा रहता है।",
            "{loc} बिजली का खंभा टूटा हुआ है और नंगी तारें लटक रही हैं जिससे करंट का भारी खतरा है।",
            "{loc} रात में लाइट न जलने से सड़क पर अंधेरा रहता है और अपराध एवं दुर्घटना का डर बना रहता है।",
            "{loc} स्ट्रीटलाइट का बल्ब फूटा हुआ है, तुरंत नई लाइट लगाई जाए।",
        ],
        "te": [
            "{loc} వీధి దీపాలు వెలగడం లేదు, రాత్రి వేళల్లో దారంతా చిమ్మచీకటిగా ఉంటోంది.",
            "{loc} వీధి స్తంభం విరిగిపోయి కరెంట్ తీగలు ప్రమాదకరంగా వేలాడుతున్నాయి.",
            "{loc} రాత్రి చీకటి వల్ల దొంగతనాలు మరియు ప్రమాదాలు జరిగే అవకాశం ఉంది, వెంటనే లైట్లు వేయించండి.",
            "{loc} స్ట్రీట్ లైట్ ఫిక్చర్ పాడైపోయింది, కొత్త లైటును అమర్చాలని కోరుతున్నాము.",
        ],
        "impact": "Zero night visibility creating crime/accident risk",
        "urgency": "Night visibility risk",
    },
    "water_leakage": {
        "en": [
            "A municipal drinking water supply pipeline has burst {loc}, and potable water is gushing onto the street.",
            "Continuous high-pressure water pipe leakage {loc}. Thousands of liters of clean drinking water are wasting.",
            "Water supply line cracked underground {loc}, eroding road foundation and wasting city water.",
            "Damaged water valve spraying water across the roadway {loc} for the past three days.",
        ],
        "hi": [
            "{loc} पीने के पानी की मुख्य पाइपलाइन फट गई है और सड़कों पर हजारों लीटर पानी बर्बाद हो रहा है।",
            "{loc} सरकारी पानी का पाइप लीक हो रहा है जिससे सड़क पर पानी का फव्वारा छूट रहा है।",
            "{loc} पेयजल आपूर्ति लाइन टूटने से घरों में पानी का दबाव कम हो गया है और सड़क धंस रही है।",
            "{loc} तीन दिनों से लगातार पानी बह रहा है, तुरंत जल विभाग पाइप की मरम्मत करे।",
        ],
        "te": [
            "{loc} మంచినీటి పైపు పగిలిపోయి వేల లీటర్ల తాగునీరు రోడ్డుపై వృధాగా పోతోంది.",
            "{loc} మున్సిపల్ వాటర్ పైప్ లైన్ లీకేజీ వల్ల నీరు ఉధృతంగా ప్రవహిస్తోంది.",
            "{loc} నీటి పైపు పగలడం వల్ల కాలనీలో నీటి సరఫరా నిలిచిపోయింది, రోడ్డు కోతకు గురవుతోంది.",
            "{loc} తాగునీటి పైపులైన్ నుండి నీరు కారుతోంది, వెంటనే మరమ్మతులు చేపట్టాలి.",
        ],
        "impact": "Water accumulation or flooding surrounding premises",
        "urgency": "Severe damage",
    },
    "waterlogging": {
        "en": [
            "Severe waterlogging across the entire road width {loc} after rainfall. Water depth is knee-high.",
            "Stagnant flooded water has submerged the street {loc}, entering shops and stopping small vehicles.",
            "Road is heavily inundated with rainwater {loc}, completely halting traffic and pedestrian movement.",
            "Persistent stagnant water ponding {loc} due to lack of drainage slope, creating vector-borne disease threat.",
        ],
        "hi": [
            "{loc} बारिश के बाद सड़क पर भारी जलभराव हो गया है और घुटनों तक पानी भर गया है।",
            "{loc} पानी जमा होने से सड़क पूरी तरह डूब गई है और गाड़ियां बंद पड़ रही हैं।",
            "{loc} जलभराव के कारण दुकानों और घरों के अंदर पानी घुस रहा है, जनजीवन अस्त-व्यस्त है।",
            "{loc} गंदा पानी कई दिनों से जमा है जिससे मलेरिया और डेंगू का खतरा पैदा हो गया है।",
        ],
        "te": [
            "{loc} భారీ వర్షం వల్ల రోడ్డుపై మోకాలి లోతు నీరు నిలిచిపోయి రాకపోకలు స్తంభించాయి.",
            "{loc} రోడ్డు జలమయమై వాహనాలు నీటిలో ఆగిపోతున్నాయి, రోడ్డు దాటడం కష్టంగా మారింది.",
            "{loc} నీరు నిల్వ ఉండటం వల్ల దుకాణాల్లోకి నీరు చేరుతోంది, తక్షణమే నీటిని తోడివేయాలి.",
            "{loc} వర్షపు నీరు రోడ్డుపై నిలిచిపోయి సరైన డ్రైనేజీ లేక తీవ్ర అసౌకర్యం కలుగుతోంది.",
        ],
        "impact": "Water accumulation or flooding surrounding premises",
        "urgency": "Urgent obstruction",
    },
    "damaged_sign": {
        "en": [
            "The speed limit and traffic direction signboard {loc} has fallen down and is lying on the footpath.",
            "Broken and bent municipal traffic sign {loc}. Reflective warning is unreadable, confusing drivers.",
            "Fallen overhead directional board {loc} obstructing pedestrian pathway.",
            "Signboard pole has corroded and collapsed onto the street {loc}.",
        ],
        "hi": [
            "{loc} दिशा सूचक और ट्रैफिक साइन बोर्ड टूटकर नीचे गिर गया है जिससे राहगीर रास्ता भटक रहे हैं।",
            "{loc} गति सीमा का बोर्ड मुड़ गया है और बिल्कुल पढ़ा नहीं जा रहा है।",
            "{loc} सड़क का साइनबोर्ड गिरकर रास्ते में अड़चन बन रहा है, इसे दोबारा लगवाया जाए।",
            "{loc} चेतावनी बोर्ड टूट जाने से चौराहे पर भ्रम की स्थिति बनी रहती है।",
        ],
        "te": [
            "{loc} ట్రాఫిక్ దిశా నిర్దేశక బోర్డు విరిగి కింద పడిపోయింది, వాహనదారులు గందరగోళానికి గురవుతున్నారు.",
            "{loc} వేగ పరిమితి సైన్ బోర్డు పాడైపోయి కనిపించకుండా ఉంది, కొత్తది అమర్చాలి.",
            "{loc} కూలిపోయిన బోర్డు రోడ్డుపై అడ్డంగా ఉంది, ప్రయాణికులకు ఆటంకం కలిగిస్తోంది.",
            "{loc} సూచిక బోర్డు తుప్పుపట్టి విరిగిపోయింది, వెంటనే సరిచేయాలని కోరుతున్నాం.",
        ],
        "impact": "Traffic movement blocked or severely congested",
        "urgency": "Urgent obstruction",
    },
    "traffic_signal_damage": {
        "en": [
            "The traffic signal lights {loc} are completely dead and dark, creating chaotic junction jams and collisions.",
            "Traffic light is stuck continuously on red {loc}, causing extreme gridlock and honking.",
            "Blinking damaged traffic control signal {loc} leading to multiple near-miss accidents.",
            "Traffic signal pole was hit by a truck and is dangerously tilted {loc}.",
        ],
        "hi": [
            "{loc} चौराहे का ट्रैफिक सिग्नल पूरी तरह बंद है जिससे भीषण जाम लग रहा है और दुर्घटनाएं हो रही हैं।",
            "{loc} ट्रैफिक लाइट खराब होने से सभी दिशाओं की गाड़ियां आपस में टकरा रही हैं।",
            "{loc} सिग्नल की बत्तियां काम नहीं कर रही हैं, ट्रैफिक पुलिस भी मौजूद नहीं है।",
            "{loc} ट्रैफिक लाइट का पोल झुक गया है, कभी भी सड़क पर गिर सकता है।",
        ],
        "te": [
            "{loc} ట్రాఫిక్ సిగ్నల్ పనిచేయడం లేదు, జంక్షన్ వద్ద విపరీతమైన ట్రాఫిక్ జామ్ ఏర్పడుతోంది.",
            "{loc} సిగ్నల్ లైట్లు ఆగిపోవడం వల్ల వాహనదారులు ఒకరినొకరు ఢీకొనే ప్రమాదం ఉంది.",
            "{loc} ట్రాఫిక్ లైట్ ఎప్పుడూ రెడ్ లోనే ఆగిపోయింది, వాహనాలు ముందుకు కదలడం లేదు.",
            "{loc} ట్రాఫిక్ సిగ్నల్ పోల్ దెబ్బతింది, వెంటనే బాగు చేయించాలి.",
        ],
        "impact": "Traffic movement blocked or severely congested",
        "urgency": "Safety risk",
    },
    "fallen_tree": {
        "en": [
            "A large tree branch has fallen across the road {loc}, blocking both traffic lanes completely.",
            "A huge uprooted tree has collapsed onto the street {loc}, snapping overhead cables and halting traffic.",
            "Fallen tree blocking the main thoroughfare {loc}. Emergency clearing crew needed immediately.",
            "Dangerous dead tree branch hanging precariously above traffic {loc}, ready to fall on passing cars.",
        ],
        "hi": [
            "{loc} एक भारी पेड़ गिरकर सड़क के बीच में आ गया है जिससे पूरा रास्ता बंद हो गया है।",
            "{loc} पेड़ की बड़ी डाल टूटकर सड़क पर गिरी है और बिजली के तारों को भी नुकसान पहुंचा है।",
            "{loc} सड़क पर गिरे पेड़ के कारण गाड़ियों का निकलना असंभव हो गया है, तुरंत पेड़ हटवाया जाए।",
            "{loc} सूखा पेड़ गिरने की कगार पर है और कभी भी किसी राहगीर पर गिर सकता है।",
        ],
        "te": [
            "{loc} భారీ వృక్షం కూలి రోడ్డుకు అడ్డంగా పడింది, ఇరువైపులా రాకపోకలు పూర్తిగా స్తంభించాయి.",
            "{loc} చెట్టు కొమ్మ విరిగి రోడ్డుపై పడి విద్యుత్ తీగలు తెగిపోయాయి, వెంటనే తొలగించాలి.",
            "{loc} రహదారిపై పడిన చెట్టు వల్ల అత్యవసర వాహనాలు కూడా వెళ్ళలేకపోతున్నాయి.",
            "{loc} ఎండిపోయిన చెట్టు కొమ్మ రోడ్డుపై వేలాడుతోంది, ప్రయాణికులకు తీవ్ర ముప్పు ఉంది.",
        ],
        "impact": "Traffic movement blocked or severely congested",
        "urgency": "Urgent obstruction",
    },
}


def generate_dataset():
    random.seed(42)
    records = []

    for cat, data in TEMPLATES.items():
        impact = data["impact"]
        urgency = data["urgency"]

        # 1. English
        for tpl in data["en"]:
            for loc in LANDMARKS_EN:
                text = tpl.format(loc=loc)
                records.append({
                    "text": text,
                    "category": cat,
                    "language": "en",
                    "location_mention": loc,
                    "impact": impact,
                    "urgency": urgency,
                    "is_compound": False,
                    "secondary_category": "",
                })

        # 2. Hindi (Devanagari)
        for tpl in data["hi"]:
            for loc in LANDMARKS_HI:
                text = tpl.format(loc=loc)
                records.append({
                    "text": text,
                    "category": cat,
                    "language": "hi",
                    "location_mention": loc,
                    "impact": impact,
                    "urgency": urgency,
                    "is_compound": False,
                    "secondary_category": "",
                })

        # 3. Telugu (Telugu Script)
        for tpl in data["te"]:
            for loc in LANDMARKS_TE:
                text = tpl.format(loc=loc)
                records.append({
                    "text": text,
                    "category": cat,
                    "language": "te",
                    "location_mention": loc,
                    "impact": impact,
                    "urgency": urgency,
                    "is_compound": False,
                    "secondary_category": "",
                })

    # Add compound multi-issue statements (e.g. pothole + waterlogging, pothole + garbage)
    compound_examples = [
        # English compound
        {"text": "There are deep potholes near the bus stand and garbage is overflowing beside the road.", "category": "road_damage", "language": "en", "location_mention": "near the bus stand", "impact": "Vehicles skidding and sanitation hazard", "urgency": "Urgent obstruction", "is_compound": True, "secondary_category": "waste"},
        {"text": "Dangerous potholes filled with rainwater outside the main market; two wheelers are falling.", "category": "road_damage", "language": "en", "location_mention": "outside the main market", "impact": "Vehicles / commuters experiencing difficulty", "urgency": "Urgent obstruction", "is_compound": True, "secondary_category": "waterlogging"},
        {"text": "The storm drain is blocked with trash causing road waterlogging near the hospital entrance.", "category": "drain_blockage", "language": "en", "location_mention": "near the hospital entrance", "impact": "Water accumulation and sanitary hazard", "urgency": "Urgent obstruction", "is_compound": True, "secondary_category": "waterlogging"},
        {"text": "Open manhole and broken streetlights near the railway station making the road pitch dark and deadly.", "category": "open_manhole", "language": "en", "location_mention": "near the railway station", "impact": "Life safety hazard and darkness", "urgency": "Safety risk", "is_compound": True, "secondary_category": "streetlight_damage"},
        
        # Hindi compound
        {"text": "सड़क पर कॉलेज रोड के पास गहरा गड्ढा है और बारिश का पानी भरा हुआ है। गाड़ियां फिसल रही हैं।", "category": "road_damage", "language": "hi", "location_mention": "कॉलेज रोड के पास", "impact": "दोपहिया वाहन फिसल रहे हैं", "urgency": "Urgent obstruction", "is_compound": True, "secondary_category": "waterlogging"},
        {"text": "गोल चौक के पास नाली कचरे से जाम है और गंदा पानी सड़क पर बह रहा है।", "category": "drain_blockage", "language": "hi", "location_mention": "गोल चौक के पास", "impact": "गंदगी और पानी का भराव", "urgency": "Urgent obstruction", "is_compound": True, "secondary_category": "waterlogging"},
        {"text": "सब्जी मंडी के बाहर बड़ा गड्ढा है और कूड़े का ढेर भी लगा हुआ है।", "category": "road_damage", "language": "hi", "location_mention": "सब्जी मंडी के बाहर", "impact": "सड़क अवरोध और कचरा", "urgency": "Urgent obstruction", "is_compound": True, "secondary_category": "waste"},
        {"text": "बस स्टैंड के पास मैनहोल खुला है और स्ट्रीट लाइट भी बंद पड़ी है।", "category": "open_manhole", "language": "hi", "location_mention": "बस स्टैंड के पास", "impact": "अंधेरे में गिरने का गंभीर खतरा", "urgency": "Safety risk", "is_compound": True, "secondary_category": "streetlight_damage"},

        # Telugu compound
        {"text": "పాఠశాల మెయిన్ గేట్ వద్ద రోడ్డుపై పెద్ద గుంత ఉంది మరియు వర్షపు నీరు నిలిచిపోయింది.", "category": "road_damage", "language": "te", "location_mention": "పాఠశాల మెయిన్ గేట్ వద్ద", "impact": "వాహనాలు నెమ్మదిగా కదలడం", "urgency": "Urgent obstruction", "is_compound": True, "secondary_category": "waterlogging"},
        {"text": "కూరగాయల మార్కెట్ వద్ద డ్రైనేజీ చెత్తతో జామ్ అయి నీరు రోడ్డుపైకి పొంగుతోంది.", "category": "drain_blockage", "language": "te", "location_mention": "కూరగాయల మార్కెట్ వద్ద", "impact": "మురుగునీటి నిల్వ", "urgency": "Urgent obstruction", "is_compound": True, "secondary_category": "waterlogging"},
        {"text": "కాలేజీ రోడ్డు దగ్గర రోడ్డుపై గుంతలు పడ్డాయి మరియు చెత్త పేరుకుపోయింది.", "category": "road_damage", "language": "te", "location_mention": "కాలేజీ రోడ్డు దగ్గర", "impact": "రవాణా ఇబ్బందులు", "urgency": "Urgent obstruction", "is_compound": True, "secondary_category": "waste"},
        {"text": "రైల్వే స్టేషన్ సమీపంలో మ్యాన్‌హోల్ తెరిచి ఉంది మరియు స్ట్రీట్ లైట్లు కూడా వెలగడం లేదు.", "category": "open_manhole", "language": "te", "location_mention": "రైల్వే స్టేషన్ సమీపంలో", "impact": "రాత్రి వేళల్లో ప్రమాదాలు", "urgency": "Safety risk", "is_compound": True, "secondary_category": "streetlight_damage"},
    ]
    records.extend(compound_examples * 5)

    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Stratified split: 80% train, 10% val, 10% test
    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df["category"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["category"])

    df.to_csv(OUTPUT_DIR / "multilingual_complaints.csv", index=False, encoding="utf-8")
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(OUTPUT_DIR / "val.csv", index=False, encoding="utf-8")
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False, encoding="utf-8")

    meta = {
        "total_records": len(df),
        "train_records": len(train_df),
        "val_records": len(val_df),
        "test_records": len(test_df),
        "languages": {
            "en": int((df["language"] == "en").sum()),
            "hi": int((df["language"] == "hi").sum()),
            "te": int((df["language"] == "te").sum()),
        },
        "categories": {cat: int((df["category"] == cat).sum()) for cat in CATEGORIES},
        "compound_records": int(df["is_compound"].sum()),
    }
    with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Generated multilingual dataset with {len(df)} records in {OUTPUT_DIR}")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print("Language distribution:", meta["languages"])
    print("Category distribution:", meta["categories"])


if __name__ == "__main__":
    generate_dataset()

