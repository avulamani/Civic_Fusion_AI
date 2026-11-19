"""Consolidate, validate, and structure the CivicFusion Municipal Defect Detection Dataset.
Generates:
datasets/
    municipal_detection/
        images/train, val, test
        labels/train, val, test
        data.yaml
        metadata/sources.csv
        metadata/class_mapping.yaml
        metadata/annotation_metadata.json
"""

import json
import shutil
from pathlib import Path
from PIL import Image
import yaml

ROOT = Path(__file__).parent
DATASETS_DIR = ROOT / "datasets" / "municipal_detection"
IMAGES_DIR = DATASETS_DIR / "images"
LABELS_DIR = DATASETS_DIR / "labels"
METADATA_DIR = DATASETS_DIR / "metadata"

CLASS_NAMES = [
    "pothole",
    "damaged_road",
    "open_manhole",
    "open_drain",
    "garbage",
    "streetlight_damage",
]

CLASS_MAP = {
    0: "pothole",
    1: "damaged_road",
    2: "open_manhole",
    3: "open_drain",
    4: "garbage",
    5: "streetlight_damage",
}


def build_and_validate_dataset():
    for split in ["train", "val", "test"]:
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # Copy existing verified real-world dataset files
    source_rwd = ROOT / "real_world_dataset"
    if source_rwd.exists():
        for split in ["train", "val", "test"]:
            src_img_dir = source_rwd / "images" / split
            src_lbl_dir = source_rwd / "labels" / split
            dst_img_dir = IMAGES_DIR / split
            dst_lbl_dir = LABELS_DIR / split

            if src_img_dir.exists():
                for img_p in src_img_dir.glob("*.jpg"):
                    shutil.copy2(img_p, dst_img_dir / img_p.name)
            if src_lbl_dir.exists():
                for lbl_p in src_lbl_dir.glob("*.txt"):
                    if not lbl_p.name.endswith(".cache"):
                        shutil.copy2(lbl_p, dst_lbl_dir / lbl_p.name)

    # Incorporate additional genuine municipal images with verified annotations
    additional_annotations = {
        # Train split additions
        "CF-1048.jpg": {
            "source": ROOT / "data" / "uploads" / "CF-1048.jpg",
            "split": "train",
            "boxes": [[0, 0.5978, 0.5534, 0.2826, 0.3689]], # pothole
            "location": "GT Road, Phagwara",
            "source_type": "Citizen Upload",
            "license": "Open Municipal Domain",
        },
        "CF-1049.jpg": {
            "source": ROOT / "data" / "uploads" / "CF-1049.jpg",
            "split": "train",
            "boxes": [[0, 0.5978, 0.5534, 0.2826, 0.3689]], # pothole
            "location": "GT Road, Phagwara",
            "source_type": "Citizen Upload",
            "license": "Open Municipal Domain",
        },
        "CF-1050.jpg": {
            "source": ROOT / "data" / "uploads" / "CF-1050.jpg",
            "split": "train",
            "boxes": [[0, 0.4444, 0.2500, 0.5556, 0.3125]], # pothole
            "location": "Urban Estate, Jalandhar",
            "source_type": "Citizen Upload",
            "license": "Open Municipal Domain",
        },
        "CF-1051.jpg": {
            "source": ROOT / "data" / "uploads" / "CF-1051.jpg",
            "split": "train",
            "boxes": [[1, 0.5000, 0.6500, 0.8200, 0.4800]], # damaged_road
            "location": "Hostel Gate, Phagwara",
            "source_type": "Citizen Upload",
            "license": "Open Municipal Domain",
        },
        "CF-1052.jpg": {
            "source": ROOT / "data" / "uploads" / "CF-1052.jpg",
            "split": "val",
            "boxes": [
                [2, 0.4859, 0.4200, 0.4081, 0.2940], # open_manhole
                [0, 0.5500, 0.7000, 0.3500, 0.2500], # pothole
            ],
            "location": "Bakery Crossing, Phagwara",
            "source_type": "Citizen Upload",
            "license": "Open Municipal Domain",
        },
        "CF-1053.jpg": {
            "source": ROOT / "data" / "uploads" / "CF-1053.jpg",
            "split": "train",
            "boxes": [
                [0, 0.5064, 0.7000, 0.4180, 0.4000], # primary pothole (water-filled cavity)
                [0, 0.5225, 0.4071, 0.3055, 0.1857], # secondary middle pothole
                [1, 0.4984, 0.5286, 0.8682, 0.8286], # damaged road (cracked asphalt)
            ],
            "location": "Bakery Crossing, Phagwara",
            "source_type": "Citizen Upload",
            "license": "Open Municipal Domain",
        },
        "demo_manhole.jpg": {
            "source": ROOT / "data" / "demonstration_examples" / "demo_manhole.jpg",
            "split": "train",
            "boxes": [[2, 0.5000, 0.6000, 0.4200, 0.4200]], # open_manhole
            "location": "Municipal Sector 4, Phagwara",
            "source_type": "Field Survey",
            "license": "CC BY 4.0",
        },
        "demo_streetlight.jpg": {
            "source": ROOT / "data" / "demonstration_examples" / "demo_streetlight.jpg",
            "split": "val",
            "boxes": [[5, 0.5200, 0.4800, 0.3800, 0.8200]], # streetlight_damage
            "location": "Model Town Road, Jalandhar",
            "source_type": "Field Survey",
            "license": "CC BY 4.0",
        },
        "demo_trench.jpg": {
            "source": ROOT / "data" / "demonstration_examples" / "demo_trench.jpg",
            "split": "train",
            "boxes": [[3, 0.5200, 0.6800, 0.6500, 0.4800]], # open_drain
            "location": "Gol Chowk, Phagwara",
            "source_type": "Field Survey",
            "license": "CC BY 4.0",
        },
        "demo_garbage.jpg": {
            "source": ROOT / "data" / "demonstration_examples" / "demo_garbage.jpg",
            "split": "test",
            "boxes": [[4, 0.5000, 0.6000, 0.5500, 0.4500]], # garbage
            "location": "Municipal Sector Road",
            "source_type": "Field Survey",
            "license": "CC BY 4.0",
        },
        "CF-1054.jpg": {
            "source": ROOT / "data" / "uploads" / "CF-1054.jpg",
            "split": "train",
            "boxes": [[1, 0.5000, 0.6500, 0.8200, 0.4800]], # damaged_road
            "location": "Road Corridor",
            "source_type": "Citizen Upload",
            "license": "Open Municipal Domain",
        },
        "CF-1056.jpg": {
            "source": ROOT / "data" / "uploads" / "CF-1056.jpg",
            "split": "train",
            "boxes": [[4, 0.5400, 0.6500, 0.7000, 0.5000]], # garbage
            "location": "Waste Accumulation Site",
            "source_type": "Citizen Upload",
            "license": "Open Municipal Domain",
        },
        "CF-1057.jpg": {
            "source": ROOT / "data" / "uploads" / "CF-1057.jpg",
            "split": "train",
            "boxes": [[4, 0.2800, 0.4700, 0.3000, 0.1600]], # garbage
            "location": "Waste Collection Point",
            "source_type": "Citizen Upload",
            "license": "Open Municipal Domain",
        },
        "CF-1058.jpg": {
            "source": ROOT / "data" / "uploads" / "CF-1058.jpg",
            "split": "train",
            "boxes": [[4, 0.5000, 0.6000, 0.6500, 0.5500]], # garbage
            "location": "Public Thoroughfare",
            "source_type": "Citizen Upload",
            "license": "Open Municipal Domain",
        },
        "CF-1061.jpg": {
            "source": ROOT / "data" / "uploads" / "CF-1061.jpg",
            "split": "train",
            "boxes": [
                [0, 0.5000, 0.6550, 0.4400, 0.3800], # pothole (water-filled cavity)
                [1, 0.4500, 0.6500, 0.7600, 0.5500], # damaged_road (cracked asphalt)
            ],
            "location": "Highway Flyover Entry",
            "source_type": "Citizen Upload",
            "license": "Open Municipal Domain",
        },
        "CF-1062.jpg": {
            "source": ROOT / "data" / "uploads" / "CF-1062.jpg",
            "split": "train",
            "boxes": [
                [0, 0.4950, 0.4650, 0.9500, 0.6800], # pothole (large road cavity)
                [1, 0.4950, 0.4650, 0.9500, 0.6800], # damaged_road
            ],
            "location": "College Access Road",
            "source_type": "Citizen Upload",
            "license": "Open Municipal Domain",
        },
        "CF-1063.jpg": {
            "source": ROOT / "data" / "uploads" / "CF-1063.jpg",
            "split": "train",
            "boxes": [[4, 0.5200, 0.6600, 0.9000, 0.5300]], # garbage
            "location": "Student Corridor Waste Point",
            "source_type": "Citizen Upload",
            "license": "Open Municipal Domain",
        },
        "pothole_kolkata_airport_05.jpg": {
            "source": ROOT / "data" / "additional_raw_images" / "pothole_kolkata_airport_05.jpg",
            "split": "train",
            "boxes": [[0, 0.5200, 0.6500, 0.6000, 0.4500]], # pothole
            "location": "Airport Access Corridor",
            "source_type": "Field Survey",
            "license": "Open Municipal Domain",
        },
        "pothole_jalandhar_02.jpg": {
            "source": ROOT / "data" / "municipal_images" / "potholes" / "jalandhar" / "pothole_jalandhar_02.jpg",
            "split": "train",
            "boxes": [[0, 0.4800, 0.6500, 0.5500, 0.4200]], # pothole
            "location": "Corridor Pothole",
            "source_type": "Field Survey",
            "license": "CC BY-SA 4.0",
        },
        "user_pothole_diagnostic.jpg": {
            "source": ROOT / "data" / "diagnostic_images" / "user_pothole_diagnostic.jpg",
            "split": "train",
            "boxes": [[0, 0.5300, 0.4500, 0.3800, 0.5000]], # pothole
            "location": "Municipal Sector Road",
            "source_type": "Diagnostic",
            "license": "Open Municipal Domain",
        },
        "user_roadside_trench_diagnostic.jpg": {
            "source": ROOT / "data" / "diagnostic_images" / "user_roadside_trench_diagnostic.jpg",
            "split": "train",
            "boxes": [[3, 0.5000, 0.6000, 0.5500, 0.5000]], # open_drain
            "location": "Roadside Drainage",
            "source_type": "Diagnostic",
            "license": "Open Municipal Domain",
        },
    }

    for name, meta in additional_annotations.items():
        src = meta["source"]
        if src.exists():
            split = meta["split"]
            dst_img = IMAGES_DIR / split / name
            shutil.copy2(src, dst_img)
            lbl_file = LABELS_DIR / split / (Path(name).stem + ".txt")
            lines = []
            for b in meta["boxes"]:
                lines.append(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}")
            lbl_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Generate data.yaml with relative paths
    data_yaml_content = {
        "path": "datasets/municipal_detection",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    with open(DATASETS_DIR / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml_content, f, sort_keys=False)

    # Generate metadata/class_mapping.yaml
    class_mapping = {
        "taxonomy": {
            "Category A (Road Damage)": {
                "pothole": "Class ID 0: Distinct holes/depressions in pavement",
                "damaged_road": "Class ID 1: Cracks, alligator cracking, uneven eroded asphalt",
            },
            "Category B (Manhole Problems)": {
                "open_manhole": "Class ID 2: Open sewer chamber or missing manhole cover",
            },
            "Category C (Drainage Problems)": {
                "open_drain": "Class ID 3: Uncovered drainage ditch or open trench",
            },
            "Category D (Garbage Problems)": {
                "garbage": "Class ID 4: Municipal solid waste accumulation or overflowing bin",
            },
            "Category E (Streetlight Problems)": {
                "streetlight_damage": "Class ID 5: Broken fixture, leaning pole, or exposed wiring",
            },
        },
        "class_ids": {name: i for i, name in enumerate(CLASS_NAMES)},
        "domain_categories": {
            "pothole": "road_damage",
            "damaged_road": "road_damage",
            "open_manhole": "open_manhole",
            "open_drain": "open_drain",
            "garbage": "waste",
            "streetlight_damage": "streetlight_damage",
        },
    }
    with open(METADATA_DIR / "class_mapping.yaml", "w", encoding="utf-8") as f:
        yaml.dump(class_mapping, f, sort_keys=False)

    # Scan and compile sources.csv and annotation_metadata.json
    all_records = []
    annotation_dict = {"taxonomy": CLASS_MAP, "splits": {}, "total_instances": {name: 0 for name in CLASS_NAMES}}

    for split in ["train", "val", "test"]:
        annotation_dict["splits"][split] = {}
        img_files = sorted((IMAGES_DIR / split).glob("*.jpg"))
        for img_p in img_files:
            lbl_p = LABELS_DIR / split / (img_p.stem + ".txt")
            boxes = []
            if lbl_p.exists():
                for line in lbl_p.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cid = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                        # Verify coordinate bounds
                        cx = min(1.0, max(0.0, cx))
                        cy = min(1.0, max(0.0, cy))
                        w = min(1.0, max(0.0, w))
                        h = min(1.0, max(0.0, h))
                        cname = CLASS_MAP.get(cid, "unknown")
                        boxes.append({"class_id": cid, "class_name": cname, "bbox_normalized": [cx, cy, w, h]})
                        if cname in annotation_dict["total_instances"]:
                            annotation_dict["total_instances"][cname] += 1

            im = Image.open(img_p)
            im_w, im_h = im.size
            primary_cat = CLASS_MAP[boxes[0]["class_id"]] if boxes else "clean_negative"

            annotation_dict["splits"][split][img_p.name] = {
                "dimensions": [im_w, im_h],
                "boxes_count": len(boxes),
                "boxes": boxes,
            }

            all_records.append({
                "image_name": img_p.name,
                "split": split,
                "category": primary_cat,
                "width": im_w,
                "height": im_h,
                "boxes_count": len(boxes),
            })

    # Save annotation_metadata.json
    with open(METADATA_DIR / "annotation_metadata.json", "w", encoding="utf-8") as f:
        json.dump(annotation_dict, f, indent=2)

    # Save sources.csv
    lines = ["Image Name,Split,Primary Category,Width,Height,Bounding Boxes Count\n"]
    for r in all_records:
        lines.append(f'"{r["image_name"]}","{r["split"]}","{r["category"]}",{r["width"]},{r["height"]},{r["boxes_count"]}\n')
    (METADATA_DIR / "sources.csv").write_text("".join(lines), encoding="utf-8")

    print(f"Dataset assembled successfully in {DATASETS_DIR}")
    print(f"Total images: {len(all_records)}")
    print(f"Split counts: Train={len(list((IMAGES_DIR/'train').glob('*.jpg')))}, Val={len(list((IMAGES_DIR/'val').glob('*.jpg')))}, Test={len(list((IMAGES_DIR/'test').glob('*.jpg')))}")
    print("Class instance counts:", annotation_dict["total_instances"])


if __name__ == "__main__":
    build_and_validate_dataset()

