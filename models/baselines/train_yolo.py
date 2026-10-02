import argparse
from pathlib import Path
from ultralytics import YOLO, RTDETR

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_YAML = PROJECT_ROOT / "configs" / "data.yaml"

def train_model(model_name: str = "yolov8s.pt", epochs: int = 100, batch: int = 16, imgsz: int = 640):
    """
    Huấn luyện mô hình YOLO / RT-DETR trên Master Dataset.
    Hỗ trợ:
    - yolov5s.pt
    - yolov8s.pt
    - yolo11s.pt
    - rtdetr-l.pt
    """
    print(f"[*] Bắt đầu huấn luyện mô hình: {model_name}")
    print(f"[*] File cấu hình dataset: {DATA_YAML}")

    if "rtdetr" in model_name.lower():
        model = RTDETR(model_name)
    else:
        model = YOLO(model_name)

    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(PROJECT_ROOT / "runs" / "train"),
        name=model_name.replace(".pt", ""),
        seed=42,
        save=True,
        plots=True
    )

    print(f"[+] Hoàn tất huấn luyện {model_name}!")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện Baseline YOLO / RT-DETR")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Tên model checkpoint ban đầu (yolov5s.pt, yolov8s.pt, yolo11s.pt, rtdetr-l.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Số epoch huấn luyện")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đầu vào")
    args = parser.parse_args()

    train_model(args.model, args.epochs, args.batch, args.imgsz)
