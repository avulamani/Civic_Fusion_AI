from __future__ import annotations

import csv
import itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "labeled_complaints.csv"

# All phrases below are generated for an academic prototype; they are not reports
# from actual residents or municipal authorities.
ISSUES = {
    "road_damage": {
        "labels": ["pothole", "road surface", "pavement", "asphalt", "road"],
        "places": ["near the bus stop", "outside the market", "by the school gate", "at the junction", "along our street", "beside the park"],
        "conditions": ["has a deep hole", "is cracked and uneven", "has broken apart", "is sinking after the rain", "has several dangerous potholes", "is difficult for vehicles to cross"],
    },
    "streetlight_damage": {
        "labels": ["streetlight", "street lamp", "lamp post", "road light", "light pole"],
        "places": ["near the bus stop", "outside the park", "by the school gate", "at the crossing", "along our street", "beside the community hall"],
        "conditions": ["is not turning on after dark", "keeps flickering and going out", "has a broken lamp", "is leaning over the footpath", "has exposed wires", "leaves this stretch completely dark"],
    },
    "waste": {
        "labels": ["garbage pile", "rubbish", "waste bags", "public bin", "trash"],
        "places": ["near the bus stop", "outside the market", "by the school gate", "beside the drain", "along our street", "near the park entrance"],
        "conditions": ["has been left uncollected for days", "is overflowing onto the footpath", "is blocking the walkway", "is scattered across the road", "is attracting stray animals", "has piled up beside the public bin"],
    },
    "open_manhole": {
        "labels": ["manhole", "drain cover", "sewer opening", "manhole lid", "storm drain grate"],
        "places": ["near the bus stop", "outside the market", "by the school gate", "at the crossing", "along our street", "beside the footpath"],
        "conditions": ["is uncovered and dangerous", "has a missing cover", "is broken and could cause someone to fall", "has shifted out of place", "is open in the middle of the road", "is leaving a deep hole exposed"],
    },
    "water_leakage": {
        "labels": ["water pipe", "municipal water line", "roadside tap", "supply pipe", "water main"],
        "places": ["near the bus stop", "outside the market", "by the school gate", "at the junction", "along our street", "beside the park"],
        "conditions": ["is leaking continuously", "has burst and water is flowing onto the road", "is wasting clean water", "is flooding the footpath", "has a damaged joint", "is gushing water into the street"],
    },
    "damaged_sign": {
        "labels": ["street name sign", "direction board", "road sign", "warning sign", "traffic sign"],
        "places": ["near the bus stop", "outside the market", "by the school gate", "at the junction", "along our street", "beside the park"],
        "conditions": ["is bent and unreadable", "has fallen onto the footpath", "is missing from its pole", "is damaged and hard to see", "is leaning over the road", "has a broken sign panel"],
    },
    "drain_blockage": {
        "labels": ["storm drain", "roadside drain", "sewer drain", "street drain", "drain channel"],
        "places": ["near the bus stop", "outside the market", "by the school gate", "at the crossing", "along our street", "beside the park"],
        "conditions": ["is blocked with silt and rubbish", "is overflowing onto the road", "has stagnant water and a bad smell", "is clogged after the rain", "is not carrying water away", "is full of plastic waste"],
    },
    "traffic_signal_damage": {
        "labels": ["traffic signal", "signal light", "pedestrian crossing signal", "junction signal", "traffic light"],
        "places": ["near the bus stop", "outside the market", "by the school gate", "at the main junction", "near the hospital", "beside the crossing"],
        "conditions": ["is not working", "keeps showing the wrong light", "has a dark signal lamp", "is damaged after a collision", "is stuck on red", "has a broken pedestrian signal"],
    },
    "fallen_tree": {
        "labels": ["tree", "large branch", "fallen tree", "broken branch", "roadside tree"],
        "places": ["near the bus stop", "outside the market", "by the school gate", "at the junction", "along our street", "beside the park"],
        "conditions": ["has fallen across the footpath", "is blocking one lane", "has broken branches hanging over the road", "is obstructing the sidewalk", "fell during the storm", "is blocking access for pedestrians"],
    },
    "waterlogging": {
        "labels": ["rainwater", "standing water", "flood water", "water on the road", "street flooding"],
        "places": ["near the bus stop", "outside the market", "by the school gate", "at the junction", "along our street", "beside the park"],
        "conditions": ["is covering the road after rain", "has flooded the footpath", "is making the lane difficult to pass", "has collected because the drain is blocked", "is deep enough to hide the road surface", "is pooling around nearby homes"],
    },
}

FORMS = [
    "There is a {label} {place}; it {condition}.",
    "Please inspect the {label} {place}. It {condition}.",
    "Residents report that the {label} {place} {condition}.",
    "A municipal issue: the {label} {place} {condition}.",
    "We need help with the {label} {place} because it {condition}.",
    "The {label} {place} {condition}, creating a problem for people nearby.",
]

rows = []
for category, values in ISSUES.items():
    combos = list(itertools.product(values["labels"], values["places"], values["conditions"]))
    for index, (label, place, condition) in enumerate(combos):
        template = FORMS[index % len(FORMS)]
        rows.append((template.format(label=label, place=place, condition=condition), category))

with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["text", "category"])
    writer.writerows(rows)

print(f"Wrote {len(rows)} synthetic examples across {len(ISSUES)} categories to {OUTPUT}")
