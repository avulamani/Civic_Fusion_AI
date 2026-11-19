"""Prepare a prompt-configured YOLO-World v2 ONNX model for Render.

Run in the project virtual environment after installing requirements-export.txt:
    python export_model.py

Ultralytics downloads yolov8s-worldv2.pt on first run. The generated ONNX model
has a fixed 10-prompt vocabulary and is run by ONNX Runtime in the dashboard.
"""

from pathlib import Path
import shutil

from ultralytics import YOLO

from vision_config import CV_CLASSES, CV_PROMPTS


ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

SOURCE_MODEL = "yolov8s-worldv2.pt"
ONNX_PATH = MODEL_DIR / "municipal_worldv2.onnx"
CLASS_PATH = MODEL_DIR / "municipal_worldv2.classes.txt"


def main() -> None:
    model = YOLO(SOURCE_MODEL)
    model.set_classes(CV_PROMPTS)

    exported_file = Path(model.export(
        format="onnx",
        imgsz=640,
        dynamic=False,
        simplify=False,
    ))
    if exported_file.resolve() != ONNX_PATH.resolve():
        shutil.copy2(exported_file, ONNX_PATH)

    CLASS_PATH.write_text("\n".join(CV_CLASSES) + "\n", encoding="utf-8")
    print(f"Exported prompt-configured YOLO-World v2 model to {ONNX_PATH}")
    print(f"Saved its {len(CV_CLASSES)} class labels to {CLASS_PATH}")


if __name__ == "__main__":
    main()
