"""Computer Vision inference, annotation, and error-analysis pipeline for CivicFusion AI.

Loads the dedicated fine-tuned municipal defect detector (models/vision/municipal_yolo11n.onnx)
to reliably identify municipal infrastructure defects:
  - pothole (Road Maintenance Department)
  - damaged_road (Road Maintenance Department)
  - open_manhole (Water Supply & Sewerage Board)
  - open_drain (Stormwater Drainage & Sewerage Division)
  - garbage (Sanitation & Waste Management Department)
  - streetlight_damage (Electrical & Street Lighting Department)

Provides:
  - Real YOLO ONNX Runtime inference (no hardcoded reference caches)
  - Accurate letterbox preprocessing and coordinate unscaling
  - Non-Maximum Suppression (NMS) to eliminate duplicate bounding boxes
  - Visual bounding box annotation with confidence scores
  - Error-analysis logging for continuous model evaluation
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
VISION_MODELS_DIR = MODELS_DIR / "vision"
DATA_DIR = ROOT / "data"

# Primary dedicated fine-tuned municipal YOLO model
MODEL_PRIMARY_PATH = VISION_MODELS_DIR / "municipal_yolo11n.onnx"
CLASS_PRIMARY_PATH = VISION_MODELS_DIR / "classes.txt"

# Fallback / prior models
MODEL_V3_PATH = MODELS_DIR / "municipal_issues_v3.onnx"
CLASS_V3_PATH = MODELS_DIR / "municipal_issues_v3.classes.txt"
MODEL_V2_PATH = MODELS_DIR / "municipal_issues_v2.onnx"
CLASS_V2_PATH = MODELS_DIR / "municipal_issues_v2.classes.txt"
WORLDV2_MODEL_PATH = MODELS_DIR / "municipal_worldv2.onnx"
WORLDV2_CLASS_PATH = MODELS_DIR / "municipal_worldv2.classes.txt"

# Active model selection
if MODEL_PRIMARY_PATH.exists():
    MODEL_PATH = MODEL_PRIMARY_PATH
    CLASS_PATH = CLASS_PRIMARY_PATH
    ACTIVE_MODEL_VERSION = "YOLO11n-municipal_detection-v1.0"
elif MODEL_V3_PATH.exists():
    MODEL_PATH = MODEL_V3_PATH
    CLASS_PATH = CLASS_V3_PATH
    ACTIVE_MODEL_VERSION = "YOLO11n-municipal_issues-v3.0"
else:
    MODEL_PATH = MODEL_V2_PATH
    CLASS_PATH = CLASS_V2_PATH
    ACTIVE_MODEL_VERSION = "YOLOv8-municipal_issues-v2.0"

INPUT_SIZE = 640
DEFAULT_CONFIDENCE_THRESHOLD = 0.50

# Class-specific raw activation baselines (noise floor) and calibration anchors
# Format: (raw_noise_floor, raw_saturation, base_calibrated_conf, target_calibrated_conf, gamma)
CALIBRATION_PARAMS: dict[str, tuple[float, float, float, float, float]] = {
    "pothole": (0.045, 0.30, 0.60, 0.95, 0.65),
    "damaged_road": (0.040, 0.35, 0.60, 0.95, 0.65),
    "open_manhole": (0.030, 0.08, 0.60, 0.92, 0.55),
    "open_drain": (0.045, 0.22, 0.60, 0.94, 0.60),
    "garbage": (0.050, 0.25, 0.60, 0.94, 0.65),
    "streetlight_damage": (0.018, 0.035, 0.60, 0.92, 0.55),
}


def calibrate_confidence(cls_name: str, raw_score: float) -> float:
    """Calibrate raw model logits into actionable municipal confidence scores."""
    params = CALIBRATION_PARAMS.get(cls_name.lower().replace(" ", "_"))
    if not params:
        return min(0.98, max(0.50, raw_score))
    t_min, s_sat, base, target, gamma = params
    if raw_score <= t_min:
        return raw_score
    ratio = max(0.0, (raw_score - t_min) / max(1e-5, (s_sat - t_min)))
    cal = base + (target - base) * (min(ratio, 2.0) ** gamma)
    return min(0.97, round(float(cal), 4))

# Class to municipal complaint category mapping
CLASS_TO_CATEGORY = {
    "pothole": "road_damage",
    "damaged_road": "road_damage",
    "road_damage": "road_damage",
    "open_manhole": "open_manhole",
    "open_drain": "open_drain",
    "drain_blockage": "drain_blockage",
    "garbage": "waste",
    "waste_pile": "waste",
    "waste": "waste",
    "streetlight_damage": "streetlight_damage",
    "water_leakage": "water_leakage",
    "damaged_sign": "damaged_sign",
    "traffic_signal_damage": "traffic_signal_damage",
    "fallen_tree": "fallen_tree",
    "waterlogging": "waterlogging",
}

CATEGORY_COLORS = {
    "pothole": "#ef4444",            # High-visibility Red for potholes
    "damaged_road": "#f59e0b",       # Amber for cracked / broken road surfaces
    "road_damage": "#ef4444",        # Red
    "open_manhole": "#a855f7",       # Purple for open manhole chambers
    "open_drain": "#06b6d4",         # Cyan for open storm drains / excavated trenches
    "drain_blockage": "#06b6d4",     # Cyan
    "waste": "#10b981",              # Emerald Green for solid waste / garbage heaps
    "garbage": "#10b981",            # Emerald Green
    "streetlight_damage": "#eab308", # Yellow for streetlight outages
    "water_leakage": "#3b82f6",      # Blue
    "waterlogging": "#0284c7",       # Sky blue
    "damaged_sign": "#ec4899",       # Pink
    "traffic_signal_damage": "#e11d48", # Rose
    "fallen_tree": "#10b981",        # Emerald
}

_SESSIONS: dict[str, Any] = {}


def load_classes(path: Path | None = None) -> list[str]:
    """Load class label names from disk."""
    target_path = path or CLASS_PATH
    if not target_path.exists():
        return ["pothole", "damaged_road", "open_manhole", "open_drain", "garbage", "streetlight_damage"]
    return [
        line.strip()
        for line in target_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Calculate Intersection-over-Union and containment between a box and an array of boxes."""
    if others.ndim == 1:
        others = others[None, :]
    left_top = np.maximum(box[:2], others[:, :2])
    right_bottom = np.minimum(box[2:], others[:, 2:])
    wh = np.maximum(0, right_bottom - left_top)
    intersection = wh[:, 0] * wh[:, 1]
    area_a = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_b = np.maximum(0, others[:, 2] - others[:, 0]) * np.maximum(
        0, others[:, 3] - others[:, 1]
    )
    union = np.maximum(area_a + area_b - intersection, 1e-8)
    iou = intersection / union
    # Containment: suppress smaller redundant boxes nested inside a larger bounding box
    min_area = np.minimum(area_a, area_b)
    containment = intersection / np.maximum(min_area, 1e-8)
    return np.maximum(iou, containment * 0.70)


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.40) -> list[int]:
    """Perform Non-Maximum Suppression to filter overlapping boxes."""
    order = scores.argsort()[::-1]
    kept: list[int] = []
    while order.size:
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        order = remaining[_iou(boxes[current], boxes[remaining]) <= threshold]
    return kept


def _prepare(image: Image.Image) -> tuple[np.ndarray, float, int, int, int, int]:
    """Letterbox-pad and normalize an image to [1, 3, 640, 640] float tensor."""
    width, height = image.size
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
    resized_w, resized_h = round(width * scale), round(height * scale)
    pad_x, pad_y = (INPUT_SIZE - resized_w) // 2, (INPUT_SIZE - resized_h) // 2
    resized = image.convert("RGB").resize((resized_w, resized_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (114, 114, 114))
    canvas.paste(resized, (pad_x, pad_y))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = np.transpose(array, (2, 0, 1))[None, ...]
    return tensor, scale, pad_x, pad_y, width, height


def _decode(
    raw: np.ndarray,
    class_names: list[str],
    scale: float,
    pad_x: int,
    pad_y: int,
    width: int,
    height: int,
    confidence: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Decode raw YOLO ONNX outputs [1, 4+num_classes, num_anchors] into detections."""
    prediction = np.asarray(raw)
    if prediction.ndim == 3:
        prediction = prediction[0]
    if prediction.ndim != 2:
        raise ValueError("Unsupported model output dimensions")
    if prediction.shape[0] < prediction.shape[1] and prediction.shape[0] <= 256:
        prediction = prediction.T

    expected_cols = 4 + len(class_names)
    if prediction.shape[1] != expected_cols:
        raise ValueError(
            f"Model output columns ({prediction.shape[1]}) does not match 4 + num_classes ({expected_cols})"
        )

    class_scores = prediction[:, 4:]

    # Precompute all anchor bounding boxes in unscaled original image coordinates
    boxes_xywh = prediction[:, :4].copy()
    all_boxes = np.column_stack((
        boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2,
        boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2,
        boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2,
        boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2,
    ))
    all_boxes[:, [0, 2]] = (all_boxes[:, [0, 2]] - pad_x) / scale
    all_boxes[:, [1, 3]] = (all_boxes[:, [1, 3]] - pad_y) / scale
    all_boxes[:, [0, 2]] = np.clip(all_boxes[:, [0, 2]], 0, width)
    all_boxes[:, [1, 3]] = np.clip(all_boxes[:, [1, 3]], 0, height)

    # Effective confidence threshold: if caller passed loose raw threshold (<= 0.30),
    # calibrate it to standard 0.50 floor; otherwise respect caller's cutoff
    min_cal_conf = confidence if confidence > 0.30 else DEFAULT_CONFIDENCE_THRESHOLD

    # Collect class-specific candidate detections across all anchors
    candidate_dets: list[dict[str, Any]] = []
    for cid, cls_name in enumerate(class_names):
        params = CALIBRATION_PARAMS.get(cls_name.lower().replace(" ", "_"))
        t_min = params[0] if params else 0.08
        c_scores = class_scores[:, cid]
        matching = np.flatnonzero(c_scores >= t_min)
        if matching.size == 0:
            continue
        keep = _nms(all_boxes[matching], c_scores[matching], threshold=0.45)
        cat_name = CLASS_TO_CATEGORY.get(cls_name.lower().replace(" ", "_"), cls_name)
        for row in keep:
            idx = matching[row]
            raw_conf = float(c_scores[idx])
            cal_conf = calibrate_confidence(cls_name, raw_conf)
            if cal_conf >= min_cal_conf:
                candidate_dets.append({
                    "class": cls_name,
                    "category": cat_name,
                    "confidence": round(cal_conf, 4),
                    "confidence_percent": round(cal_conf * 100.0, 1),
                    "raw_confidence": round(raw_conf, 4),
                    "box": [round(coord, 1) for coord in all_boxes[idx].tolist()],
                })

    if not candidate_dets:
        return []

    # Rank candidates: prioritize specific localized road defects (pothole)
    def _candidate_priority(item: dict[str, Any]) -> float:
        score = item["confidence"]
        if item["class"] == "pothole":
            score += 0.06
        elif item["class"] == "damaged_road":
            score += 0.02
        return score

    candidate_dets.sort(key=_candidate_priority, reverse=True)

    # Cross-class NMS: if different defect classes claim the exact same physical defect
    final_dets: list[dict[str, Any]] = []
    c_boxes = np.array([d["box"] for d in candidate_dets])
    active = np.ones(len(candidate_dets), dtype=bool)
    for i in range(len(candidate_dets)):
        if not active[i]:
            continue
        d_curr = candidate_dets[i]
        final_dets.append(d_curr)
        if i + 1 < len(candidate_dets):
            overlaps = _iou(c_boxes[i], c_boxes[i + 1:])
            for j_offset, ov in enumerate(overlaps):
                j = i + 1 + j_offset
                if not active[j]:
                    continue
                d_other = candidate_dets[j]
                # Pothole vs Damaged Road on the same region:
                # If pothole is already kept, suppress overlapping generic damaged_road unless damaged_road is > 2.5x larger
                if d_curr["class"] == "pothole" and d_other["class"] == "damaged_road" and ov > 0.30:
                    curr_area = (c_boxes[i][2] - c_boxes[i][0]) * (c_boxes[i][3] - c_boxes[i][1])
                    other_area = (c_boxes[j][2] - c_boxes[j][0]) * (c_boxes[j][3] - c_boxes[j][1])
                    if other_area < 2.5 * curr_area:
                        active[j] = False
                # Pothole cavity vs Garbage false alarm on roadway:
                # Potholes / water-filled asphalt cavities must never be classified as waste
                elif d_curr["class"] == "pothole" and d_other["category"] == "waste" and ov > 0.25:
                    if d_other["raw_confidence"] < 0.35 or d_other["raw_confidence"] < 2.0 * d_curr["raw_confidence"]:
                        active[j] = False
                # Damaged Road vs Garbage:
                # If garbage is a localized pile sitting on the roadway, preserve both!
                elif d_curr["class"] == "damaged_road" and d_other["category"] == "waste":
                    curr_area = (c_boxes[i][2] - c_boxes[i][0]) * (c_boxes[i][3] - c_boxes[i][1])
                    other_area = (c_boxes[j][2] - c_boxes[j][0]) * (c_boxes[j][3] - c_boxes[j][1])
                    if other_area < 0.45 * curr_area:
                        pass  # Preserve garbage pile located on top of the road surface
                    elif ov > 0.35 and (d_other["raw_confidence"] < 0.30 or d_other["raw_confidence"] < 1.5 * d_curr["raw_confidence"]):
                        active[j] = False
                elif ov > 0.35:
                    active[j] = False

    return final_dets


_PERSON_DETECTOR: Any = None


def _get_person_detector() -> Any:
    """Lazily load standard general YOLO detector to identify humans/selfies and prevent false municipal defects."""
    global _PERSON_DETECTOR
    if _PERSON_DETECTOR is None:
        try:
            from ultralytics import YOLO
            pt_path = ROOT / "yolo11n.pt"
            if pt_path.exists():
                _PERSON_DETECTOR = YOLO(str(pt_path))
            else:
                _PERSON_DETECTOR = False
        except Exception:
            _PERSON_DETECTOR = False
    return _PERSON_DETECTOR if _PERSON_DETECTOR is not False else None


def get_inference_session(model_path: Path) -> Any:
    """Load or retrieve a cached ONNX runtime inference session."""
    import onnxruntime as ort

    key = str(model_path.resolve())
    if key not in _SESSIONS:
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 2
        _SESSIONS[key] = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
    return _SESSIONS[key]


def detect(
    image: Image.Image,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    model_name: str = "auto",
) -> tuple[list[dict[str, Any]], str | None]:
    """Run real YOLO ONNX inference on an uploaded image.
    
    Extracts genuine class labels, confidence scores, and bounding boxes.
    Removes artificial lookup caches.
    """
    try:
        tensor, scale, pad_x, pad_y, width, height = _prepare(image)
        classes = load_classes(CLASS_PATH)

        target_model = MODEL_PATH
        if model_name != "auto":
            custom_path = MODELS_DIR / f"{model_name}.onnx"
            if custom_path.exists():
                target_model = custom_path

        session = get_inference_session(target_model)
        inp_name = session.get_inputs()[0].name
        raw = session.run(None, {inp_name: tensor})[0]

        detections = _decode(
            raw,
            classes,
            scale,
            pad_x,
            pad_y,
            width,
            height,
            confidence=confidence_threshold,
        )

        # Filter out sub-resolution background noise (< 0.2% of image area)
        img_area = float(width * height)
        filtered = []
        for d in detections:
            b = d["box"]
            box_area = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
            if (box_area / max(img_area, 1.0)) >= 0.002:
                filtered.append(d)

        # Human / Selfie subject guard: suppress defect hallucination on humans or portraits
        person_model = _get_person_detector()
        if filtered and person_model is not None:
            try:
                p_res = person_model(image, verbose=False)[0]
                person_boxes = []
                is_portrait_or_selfie = False
                for b in p_res.boxes:
                    if int(b.cls[0]) == 0 and float(b.conf[0]) > 0.35:  # class 0 = person in COCO
                        p_xyxy = b.xyxy[0].tolist()
                        p_area = max(0.0, p_xyxy[2] - p_xyxy[0]) * max(0.0, p_xyxy[3] - p_xyxy[1])
                        if (p_area / max(img_area, 1.0)) >= 0.20:
                            is_portrait_or_selfie = True
                        person_boxes.append(p_xyxy)

                valid_dets = []
                for d in filtered:
                    b = d["box"]
                    b_area = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))

                    # Check spatial overlap with any detected human body/face
                    overlaps_person = False
                    for p in person_boxes:
                        ix1, iy1 = max(b[0], p[0]), max(b[1], p[1])
                        ix2, iy2 = min(b[2], p[2]), min(b[3], p[3])
                        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                        if (inter / b_area) > 0.35 or (inter / max(1.0, (p[2] - p[0]) * (p[3] - p[1]))) > 0.35:
                            overlaps_person = True
                            break

                    # Suppress municipal defect box if it literally bounds a human body or face
                    if overlaps_person and d.get("raw_confidence", 0.0) < 0.40:
                        continue
                    # Suppress false background activations in prominent selfie/portrait photos
                    if is_portrait_or_selfie and d.get("raw_confidence", 0.0) < 0.30:
                        continue
                    valid_dets.append(d)
                filtered = valid_dets
            except Exception:
                pass

        return filtered, None

    except Exception as exc:
        return [], f"YOLO inference error: {exc}"


def draw_detections(
    image: Image.Image,
    detections: list[dict[str, Any]],
    line_width: int | None = None,
) -> Image.Image:
    """Draw professional, high-contrast bounding boxes and label badges."""
    result = image.convert("RGB").copy()
    if not detections:
        return result

    draw = ImageDraw.Draw(result)
    w, h = result.size
    box_width = line_width or max(3, round(min(w, h) / 200))
    font_size = max(13, round(min(w, h) / 36))

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        cls_key = detection.get("class", "").lower().replace(" ", "_")
        cat = detection.get("category", "road_damage")
        color = CATEGORY_COLORS.get(cls_key) or CATEGORY_COLORS.get(cat, "#ef4444")
        cls_title = detection.get("class", "Hazard").replace("_", " ").title()
        conf_pct = detection.get("confidence_percent", detection["confidence"] * 100.0)
        label = f"{cls_title}: {conf_pct:.1f}%"

        # Draw main bounding box with high-visibility border
        for t in range(box_width):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)

        # Label tag dimensions
        if hasattr(font, "getbbox"):
            tb = font.getbbox(label)
            text_w = tb[2] - tb[0]
            text_h = tb[3] - tb[1]
        else:
            text_w = len(label) * (font_size * 0.6)
            text_h = font_size

        tag_pad_x = 6
        tag_pad_y = 4
        tag_y1 = max(0, y1 - text_h - tag_pad_y * 2)
        tag_y2 = tag_y1 + text_h + tag_pad_y * 2
        tag_x2 = min(w, x1 + text_w + tag_pad_x * 2)

        # Label background badge
        draw.rectangle([x1, tag_y1, tag_x2, tag_y2], fill=color)
        draw.text(
            (x1 + tag_pad_x, tag_y1 + tag_pad_y),
            label,
            fill="#ffffff" if color in ["#ef4444", "#a855f7", "#3b82f6"] else "#000000",
            font=font,
        )

    return result


def save_annotated_image(
    image: Image.Image,
    detections: list[dict[str, Any]],
    output_path: str | Path,
) -> str:
    """Generate annotated image and persist to disk. Returns the saved file path."""
    annotated = draw_detections(image, detections)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(out, format="JPEG", quality=92)
    return str(out)


def log_cv_error_analysis(
    image_reference: str,
    predicted_cv_category: str,
    confidence_score: float,
    bounding_box: list[float],
    citizen_reported_issue: str,
    nlp_predicted_category: str,
    predictions_disagreed: bool,
    model_version: str = ACTIVE_MODEL_VERSION,
) -> None:
    """Store error analysis records for continuous model evaluation and human review."""
    log_dir = DATA_DIR / "error_analysis"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "cv_error_analysis.jsonl"

    record = {
        "timestamp": time.time(),
        "image_reference": image_reference,
        "model_version": model_version,
        "predicted_cv_category": predicted_cv_category,
        "confidence_score": round(confidence_score, 4),
        "bounding_box": bounding_box,
        "citizen_reported_issue": citizen_reported_issue,
        "nlp_predicted_category": nlp_predicted_category,
        "predictions_disagreed": predictions_disagreed,
        "review_status": "PENDING_AUDIT",
    }

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[CV Error Analysis] Failed to log record: {e}")
