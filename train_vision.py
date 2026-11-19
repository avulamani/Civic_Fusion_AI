"""Train and fine-tune dedicated YOLO11 municipal defect detection model.
Exports trained weights to ONNX for CPU-optimized deployment.
"""

from pathlib import Path
import shutil
from ultralytics import YOLO

ROOT = Path(__file__).parent
DATA_YAML = ROOT / "datasets" / "municipal_detection" / "data.yaml"
OUTPUT_DIR = ROOT / "models" / "vision"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Prefer fine-tuning from existing municipal weights if present, otherwise base yolo11n
EXISTING_BEST = OUTPUT_DIR / "municipal_yolo11n_best.pt"
BASE_WEIGHTS = EXISTING_BEST if EXISTING_BEST.exists() else (ROOT / "yolo11n.pt")


def train_municipal_detector(epochs: int = 10, batch: int = 8):
    print("=" * 70)
    print("TRAINING CIVICFUSION DEDICATED YOLO11 MUNICIPAL DEFECT DETECTOR")
    print("=" * 70)
    print(f"Data config: {DATA_YAML}")
    print(f"Base weights: {BASE_WEIGHTS}")

    model = YOLO(str(BASE_WEIGHTS))

    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        batch=batch,
        imgsz=640,
        device="cpu",
        project=str(ROOT / "runs" / "municipal_train_yolo11"),
        name="dedicated_detector",
        exist_ok=True,
        verbose=True,
        seed=42,
        workers=0,
        lr0=0.001,
        lrf=0.01,
        cls=1.5,
        box=7.5,
        augment=True,
        mosaic=0.1,
        close_mosaic=3,
    )

    print("\nTraining completed successfully!")
    run_dir = ROOT / "runs" / "municipal_train_yolo11" / "dedicated_detector"
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = run_dir / "weights" / "last.pt"

    dest_pt = OUTPUT_DIR / "municipal_yolo11n_best.pt"
    shutil.copy2(best_pt, dest_pt)
    print(f"Saved best weights to {dest_pt}")

    # Export to ONNX for fast, lightweight CPU inference on Render
    print("\nExporting dedicated municipal detector to ONNX...")
    trained_model = YOLO(str(dest_pt))
    exported_path = Path(
        trained_model.export(
            format="onnx",
            imgsz=640,
            dynamic=False,
            simplify=False,
        )
    )
    dest_onnx = OUTPUT_DIR / "municipal_yolo11n.onnx"
    shutil.copy2(exported_path, dest_onnx)
    print(f"Exported ONNX model to {dest_onnx}")

    # Write class list
    classes_txt = OUTPUT_DIR / "classes.txt"
    class_names = [
        "pothole",
        "damaged_road",
        "open_manhole",
        "open_drain",
        "garbage",
        "streetlight_damage",
    ]
    classes_txt.write_text("\n".join(class_names) + "\n", encoding="utf-8")
    print(f"Saved classes to {classes_txt}")

    return dest_pt, dest_onnx


if __name__ == "__main__":
    train_municipal_detector(epochs=10, batch=8)

