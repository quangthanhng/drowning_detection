import os
from pathlib import Path
from PIL import Image

def verify_dataset_structure(dataset_dir: Path):
    """
    Kiểm tra cấu trúc và tính hợp lệ của nhãn YOLO trong thư mục dataset.
    """
    print(f"\n[*] Đang kiểm tra thư mục: {dataset_dir}")
    if not dataset_dir.exists():
        print(f"[-] Thư mục {dataset_dir} chưa tồn tại.")
        return

    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    if not images_dir.exists():
        print(f"[-] Không tìm thấy thư mục images tại {images_dir}")
        return

    image_files = list(images_dir.glob("*.*"))
    valid_images = [f for f in image_files if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    print(f"[+] Tìm thấy {len(valid_images)} file ảnh hợp lệ.")

    if not labels_dir.exists():
        print(f"[-] Cảnh báo: Không tìm thấy thư mục labels tại {labels_dir}")
        return

    missing_labels = 0
    corrupt_boxes = 0
    valid_samples = 0

    for img_path in valid_images:
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            missing_labels += 1
            continue

        try:
            with open(lbl_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        corrupt_boxes += 1
                        continue
                    cls_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    # Kiểm tra tọa độ chuẩn hóa YOLO (0.0 -> 1.0)
                    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                        corrupt_boxes += 1
            valid_samples += 1
        except Exception:
            corrupt_boxes += 1

    print(f"[+] Số mẫu hợp lệ hoàn toàn: {valid_samples}")
    if missing_labels > 0:
        print(f"[!] Cảnh báo: {missing_labels} ảnh thiếu file nhãn .txt tương ứng.")
    if corrupt_boxes > 0:
        print(f"[!] Cảnh báo: {corrupt_boxes} bounding box có định dạng không hợp lệ.")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"
    for d in raw_dir.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            verify_dataset_structure(d)
