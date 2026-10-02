import os
import shutil
import random
from pathlib import Path

# Cố định random seed để đảm bảo tính tái lập trong nghiên cứu khoa học
random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "dataset_master"

# Ánh xạ nhãn từ các nguồn khác nhau về chuẩn 2 lớp:
# 0: swimming, 1: drowning
LABEL_MAPPING = {
    # Figshare: 0: swimming, 1: struggling, 2: drowning
    "figshare": {0: 0, 1: 1, 2: 1},
    # Springer / Roboflow: 0: swimming, 1: drowning
    "springer": {0: 0, 1: 1},
    # Drone / Default:
    "default": {0: 0, 1: 1}
}

def create_target_directories():
    for split in ["train", "val", "test"]:
        (PROCESSED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (PROCESSED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

def collect_valid_samples():
    samples = []
    print("[1/4] Đang quét các nguồn dữ liệu raw trong:", RAW_DIR)

    for source_dir in RAW_DIR.iterdir():
        if not source_dir.is_dir() or source_dir.name.startswith("."):
            continue

        images_dir = source_dir / "images"
        labels_dir = source_dir / "labels"

        # Nếu dataset chia sẵn train/val/test trong raw, quét đệ quy
        if not images_dir.exists():
            img_candidates = list(source_dir.glob("**/images/*.*")) + list(source_dir.glob("**/*.*"))
        else:
            img_candidates = list(images_dir.glob("*.*"))

        source_name = source_dir.name
        source_count = 0

        for img_file in img_candidates:
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            
            # Tìm file label tương ứng
            lbl_file = img_file.parent.parent / "labels" / f"{img_file.stem}.txt"
            if not lbl_file.exists():
                lbl_file = img_file.with_suffix(".txt")
            
            if lbl_file.exists():
                samples.append((img_file, lbl_file, source_name))
                source_count += 1

        print(f"  -> Nguồn '{source_name}': tìm thấy {source_count} cặp ảnh-nhãn.")

    return samples

def main():
    create_target_directories()
    samples = collect_valid_samples()

    if len(samples) == 0:
        print("\n[!] CHÚ Ý: Chưa tìm thấy dữ liệu trong 'data/raw/'.")
        print("    Vui lòng tải dataset theo hướng dẫn trong 'scripts/01_download_datasets.sh'")
        print("    sau đó chạy lại script này.")
        return

    print(f"\n[2/4] Tổng số mẫu thu thập được: {len(samples)} ảnh.")
    
    # Xáo trộn dữ liệu ngẫu nhiên với seed=42
    random.shuffle(samples)

    # Chia theo tỷ lệ 70% Train, 15% Val, 15% Test
    total = len(samples)
    train_end = int(0.70 * total)
    val_end = int(0.85 * total)

    train_set = samples[:train_end]
    val_set = samples[train_end:val_end]
    test_set = samples[val_end:]

    splits = [("train", train_set), ("val", val_set), ("test", test_set)]

    print(f"[3/4] Phân bổ tập dữ liệu:")
    print(f"  - Train Set : {len(train_set)} ảnh (70%)")
    print(f"  - Val Set   : {len(val_set)} ảnh (15%)")
    print(f"  - Test Set  : {len(test_set)} ảnh (15% - Dùng chung để đánh giá 6 Baseline)")

    print("\n[4/4] Đang sao chép file và chuẩn hóa nhãn về 2 lớp (0: swimming, 1: drowning)...")
    for split_name, split_samples in splits:
        for idx, (img_path, lbl_path, source) in enumerate(split_samples):
            new_filename = f"{source}_{split_name}_{idx:05d}"
            target_img = PROCESSED_DIR / "images" / split_name / f"{new_filename}{img_path.suffix}"
            target_lbl = PROCESSED_DIR / "labels" / split_name / f"{new_filename}.txt"

            shutil.copy2(img_path, target_img)

            # Chuẩn hóa ID nhãn
            source_key = "figshare" if "figshare" in source.lower() else ("springer" if "springer" in source.lower() else "default")
            mapping = LABEL_MAPPING.get(source_key, {0: 0, 1: 1})

            with open(lbl_path, "r", encoding="utf-8") as f_in, open(target_lbl, "w", encoding="utf-8") as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            orig_cls = int(parts[0])
                            new_cls = mapping.get(orig_cls, 1)
                            f_out.write(f"{new_cls} {' '.join(parts[1:5])}\n")
                        except ValueError:
                            continue

    print("\n[+] HOÀN TẤT! Master Dataset đã được tạo thành công tại:")
    print(f"    {PROCESSED_DIR}")

if __name__ == "__main__":
    main()
