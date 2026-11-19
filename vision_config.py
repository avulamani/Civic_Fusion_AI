"""Prompt vocabulary for the CivicFusion YOLO-World v2 detector."""


# Keep these machine labels aligned with NLP categories and fusion.py.
CV_CLASSES = [
    "road_damage",
    "streetlight_damage",
    "waste",
    "open_manhole",
    "water_leakage",
    "damaged_sign",
    "drain_blockage",
    "traffic_signal_damage",
    "fallen_tree",
    "waterlogging",
]

# Descriptive prompts help the open-vocabulary detector target visible evidence.
# These are zero-shot prompts, not a substitute for municipal labeled training data.
CV_PROMPTS = [
    "a pothole, crack, or damaged section of road pavement",
    "a damaged streetlight or broken lamp post",
    "a pile of garbage or overflowing public trash bin",
    "an open manhole or uncovered drain opening in a road or footpath",
    "water visibly leaking or spraying from a pipe beside a street",
    "a broken, fallen, or unreadable public road sign",
    "a street drain blocked by rubbish or debris",
    "a damaged or broken traffic signal at a road intersection",
    "a fallen tree or large branch blocking a road or footpath",
    "a flooded or waterlogged urban road",
]
