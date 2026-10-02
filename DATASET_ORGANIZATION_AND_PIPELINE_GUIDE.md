# HƯỚNG DẪN TỔ CHỨC DỮ LIỆU, CẤU TRÚC THƯ MỤC & PIPELINE THỰC NGHIỆM DROWNING DETECTION

> **Mục tiêu tài liệu**: Quy định chi tiết cấu trúc thư mục, quy trình thu thập dữ liệu từ các nguồn học thuật uy tín (2022–2026), nguyên tắc gộp dữ liệu thành **Master Dataset** dùng chung để so sánh công bằng 6 mô hình Baseline, và cơ chế tách bạch tập kiểm thử **Video Testbed** để đánh giá bộ lọc thời gian (Temporal Filtering).

---

## 1. CẤU TRÚC THƯ MỤC TOÀN DIỆN CỦA DỰ ÁN

Toàn bộ dự án tại thư mục gốc `/Users/nguyenquangthanh/Downloads/drowning-detection` được chuẩn hóa theo cấu trúc sau:

```
drowning-detection/
│
├── configs/                                 # Chứa các file cấu hình huấn luyện & dữ liệu
│   ├── data.yaml                            # Cấu hình Master Dataset (dùng chung cho 6 Baselines)
│   ├── hyperparameters.yaml                 # Tham số train (epochs=100, lr=0.01, batch=16, imgsz=640)
│   └── temporal_config.yaml                 # Cấu hình bộ lọc thời gian (threshold N, window size, tolerance)
│
├── data/                                    # QUẢN LÝ DỮ LIỆU CÁC GIAI ĐOẠN
│   ├── raw/                                 # [NƠI CHỨA DỮ LIỆU GỐC TẢI VỀ - KHÔNG CHỈNH SỬA]
│   │   ├── figshare_underwater_2025/        # Dataset từ Figshare (DOI: 10.6084/m9.figshare.29497235.v2)
│   │   ├── swimming_yolo_springer_2025/     # Dataset từ bài báo Springer SIVP 2025
│   │   ├── drone_pool_wang_2023/            # Dataset góc nhìn trên cao UAV
│   │   └── roboflow_academic_subset/        # Dữ liệu mở bổ sung đã lọc
│   │
│   ├── processed/                           # [NƠI GỘP CHUNG - MASTER IMAGE DATASET]
│   │   └── dataset_master/                  # Dùng chung để Train & Test 6 Model Baseline
│   │       ├── images/
│   │       │   ├── train/                   # 70% tổng số ảnh (~4,500 - 5,000 ảnh)
│   │       │   ├── val/                     # 15% tổng số ảnh (~1,000 ảnh)
│   │       │   └── test/                    # 15% tổng số ảnh (~1,000 ảnh - CHẤM ĐIỂM BẢNG 1)
│   │       └── labels/
│   │           ├── train/                   # File .txt tương ứng (YOLO format: class x y w h)
│   │           ├── val/
│   │           └── test/
│   │
│   └── video_testbed/                       # [NƠI KIỂM THỬ VIDEO CHUỖI THỜI GIAN - GIAI ĐOẠN 2]
│       ├── videos/                          # Thư mục chứa 15 - 20 video liên tục (25 - 30 fps)
│       │   ├── true_drowning/               # 8 - 10 video có tình huống nguy hiểm thật/diễn tập cứu hộ
│       │   │   ├── drown_clip_01.mp4
│       │   │   └── ...
│       │   └── fake_triggers/               # 15 - 20 video chứa hành vi gây nhiễu (vẫy tay, lặn, té nước)
│       │       ├── fake_splash_01.mp4
│       │       ├── fake_waving_02.mp4
│       │       └── ...
│       └── ground_truth_events.csv          # File nhãn thời gian chuẩn để đo FAR & Detection Delay
│
├── models/                                  # Chứa mã nguồn kiến trúc mô hình & trọng số
│   ├── baselines/                           # Mã nguồn huấn luyện 6 mô hình Baseline
│   │   ├── train_faster_rcnn.py             # Baseline 1: Faster R-CNN (ResNet-50 FPN)
│   │   ├── train_ssd.py                     # Baseline 2: SSD-MobileNetV2
│   │   ├── train_yolov5.py                  # Baseline 3: YOLOv5s
│   │   ├── train_yolov8.py                  # Baseline 4: YOLOv8s
│   │   ├── train_yolov11.py                 # Baseline 5: YOLOv11s
│   │   └── train_rtdetr.py                  # Baseline 6: RT-DETR-R18
│   └── weights/                             # Nơi lưu checkpoint tốt nhất
│       ├── best_faster_rcnn.pth
│       ├── best_yolov5s.pt
│       ├── best_yolov8s.pt
│       ├── best_yolov11s.pt
│       └── best_rtdetr.pt
│
├── src/                                     # MÃ NGUỒN CỐT LÕI (TEMPORAL FILTERING PIPELINE)
│   ├── detector.py                          # Lớp Wrapper tải model detector tốt nhất
│   ├── tracker.py                           # Tích hợp ByteTrack gán ID ổn định cho từng người bơi
│   ├── temporal_filter.py                   # Thuật toán Cửa sổ trượt Debounce (Cốt lõi đóng góp đề tài)
│   └── alert_system.py                      # Bộ điều phối phát còi & vẽ bounding box cảnh báo
│
├── scripts/                                 # Các kịch bản tự động hóa chuẩn bị dữ liệu
│   ├── 01_download_datasets.sh              # Tải tự động các nguồn dữ liệu học thuật
│   ├── 02_clean_and_deduplicate.py          # Lọc ảnh trùng lặp bằng pHash & kiểm tra bounding box
│   ├── 03_merge_and_split_master.py         # Gộp các nguồn và chia Train/Val/Test 70/15/15
│   └── 04_evaluate_temporal_testbed.py      # Chạy kiểm thử trên 20 video test và tính FAR, Delay
│
├── experiments/                             # Nơi lưu kết quả thực nghiệm để đưa vào báo cáo
│   ├── table1_baseline_frame_results.csv    # Kết quả so sánh 6 model trên tập test ảnh tĩnh
│   ├── table2_temporal_filter_results.csv   # Kết quả đo FAR, Miss Rate, Delay trên Video Testbed
│   └── plots/                               # Đồ thị PR-Curve, Trade-off Curve giữa FAR và Delay
│
└── demo/                                    # Ứng dụng Demo trực quan bảo vệ đề tài
    └── app.py                               # Giao diện Demo Streamlit/OpenCV vẽ tiến trình nghi vấn
```

---

## 2. NGUỒN DATASET CHO TỪNG GIAI ĐOẠN & CÁCH THỨC THU THẬP

Dưới đây là chi tiết từng nguồn dữ liệu, đường dẫn định danh học thuật và hướng dẫn lưu trữ vào hệ thống thư mục:

```
                                  [ NGUỒN THU THẬP DỮ LIỆU DỰ ÁN ]
                                                 │
         ┌───────────────────────────────────────┴───────────────────────────────────────┐
         ▼                                                                               ▼
[ DỮ LIỆU ẢNH TĨNH: GỘP MASTER DATASET ]                                 [ DỮ LIỆU CHUỖI VIDEO: TESTBED ]
  • Figshare Academic (DOI: 10.6084/...)                                    • Video thực tế camera hồ bơi
  • Springer SIVP 2025 (Swimming-YOLO)                                      • Video diễn tập cứu hộ bể bơi
  • UAV Drone Pool Dataset (Wang et al.)                                    • Video tình huống giả định (vẫy tay, lặn)
  ──► Gộp & Chia 70% Train - 15% Val - 15% Test                             ──► Gán nhãn CSV: ground_truth_events.csv
```

### 2.1. Nguồn Dữ liệu 1: Underwater Drowning Detection Dataset (Figshare 2025)
* **Nguồn học thuật**: Figshare Academic Repository (Công bố 2025).
* **Định danh DOI**: `10.6084/m9.figshare.29497235.v2`
* **Quy mô**: **5,613 ảnh** phân giải chuẩn 640×640, định dạng sẵn nhãn YOLO `.txt`.
* **Phân lớp gốc**: 3 lớp gồm `0: swimming` (1,871 ảnh), `1: struggling` (1,871 ảnh), `2: drowning` (1,871 ảnh).
* **Thư mục lưu trữ**: `data/raw/figshare_underwater_2025/`
* **Cách sử dụng**: Đây là tập dữ liệu nền móng có chất lượng gán nhãn thủ công cao nhất và có định danh học thuật DOI citable.

### 2.2. Nguồn Dữ liệu 2: Swimming-YOLO Pool Dataset (Springer SIVP 2025)
* **Nguồn học thuật**: Xuất bản trên tạp chí **Signal, Image and Video Processing** (*Springer Nature*, 2025 - Jiang et al., DOI: `10.1007/s11760-024-03712-x`).
* **Quy mô**: **4,800 ảnh** chụp từ camera bờ hồ bơi bao quát (tilt/overhead angle).
* **Đặc điểm**: Bổ sung các góc chụp thực tế từ trên bờ nhìn xuống, có người bơi bình thường, người chìm, phao bơi và bóng nước.
* **Thư mục lưu trữ**: `data/raw/swimming_yolo_springer_2025/`

### 2.3. Nguồn Dữ liệu 3: UAV Drone Pool Dataset (Wang-Kaikai et al., 2022-2023)
* **Nguồn học thuật**: Nghiên cứu phát hiện đuối nước bằng thiết bị bay không người lái (UAV) của nhóm Kaikai Wang (ResearchGate/GitHub).
* **Quy mô**: **8,572 ảnh** góc nhìn từ trên cao thẳng đứng (Top-down 90 độ).
* **Thư mục lưu trữ**: `data/raw/drone_pool_wang_2023/`
* **Cách sử dụng**: Trích xuất ngẫu nhiên khoảng 1,500 - 2,000 ảnh đặc trưng để bổ sung góc nhìn thẳng đứng, chống điểm mù.

### 2.4. Nguồn Dữ liệu 4: Custom Temporal Video Testbed (Kiểm thử chuỗi thời gian)
* **Nguồn thu thập**: Thu thập từ các video camera giám sát hồ bơi thực tế, các đoạn video diễn tập cứu hộ của nhân viên cứu trợ, và các tình huống mô phỏng bơi lội an toàn nhưng dễ nhầm lẫn.
* **Quy mô**: **15 - 20 video** liên tục (thời lượng từ 30 giây đến 2 phút/video, 25 - 30 FPS).
* **Thư mục lưu trữ**: `data/video_testbed/videos/`
* **Tách bạch hoàn toàn**: **Tập này TUYỆT ĐỐI KHÔNG cắt ảnh đưa vào tập Train** để đảm bảo tính khách quan 100% khi đo đạc độ trễ và báo động giả.

---

## 3. NGUYÊN TẮC GỘP DỮ LIỆU (MASTER DATASET) & CHUẨN HÓA NHÃN

Để đảm bảo việc so sánh 6 mô hình Baseline (Faster R-CNN, SSD, YOLOv5, YOLOv8, YOLOv11, RT-DETR) là **hoàn toàn công bằng và chuẩn mực khoa học**, toàn bộ các nguồn ảnh tĩnh ở mục 2.1, 2.2, 2.3 sẽ được đưa qua quy trình **Data Fusion** để tạo thành **MỘT BỘ MASTER DATASET DUY NHẤT**.

```
[ Figshare (5,613 ảnh) ]  ──┐
[ Springer (4,800 ảnh) ]  ──┼──► [ Lọc trùng pHash ] ──► [ Ánh xạ nhãn 2 Class ] ──► [ MASTER DATASET ]
[ Drone UAV (2,000 ảnh) ] ──┘                                                          ├── Train (70%)
                                                                                       ├── Val   (15%)
                                                                                       └── Test  (15%)
```

### 3.1. Bảng ánh xạ chuẩn hóa nhãn (Label Mapping Matrix)
Các nguồn dữ liệu khác nhau có số lượng lớp khác nhau. Khi gộp vào `dataset_master/`, toàn bộ được quy chuẩn về **2 lớp duy nhất**:

| Lớp chuẩn hóa chung | Class ID | Nhãn gốc từ Figshare | Nhãn gốc từ Springer / Drone / Roboflow | Ý nghĩa thực tế trong hệ thống |
| :--- | :---: | :--- | :--- | :--- |
| **`swimming`** | **`0`** | `swimming` (Bơi lội an toàn) | `swimming`, `safe_swimmer`, `treading_water` | Trạng thái an toàn: Không kích hoạt chuông báo. |
| **`drowning`** | **`1`** | `struggling` (chới với) + `drowning` (chìm) | `drowning`, `distress`, `submerged` | Trạng thái nguy hiểm: Kích hoạt đếm thời gian nghi vấn. |

> **Giải thích học thuật**: Trong bài toán giám sát an toàn bể bơi thực tế, cả hành vi chới với hoảng loạn (*struggling*) và chìm bất động (*drowning*) đều là trạng thái khẩn cấp cần nhân viên cứu hộ phản ứng ngay lập tức. Việc gộp lớp `struggling` vào lớp `drowning` giúp mô hình nhận diện nguy hiểm từ giai đoạn sớm nhất.

### 3.2. Quy trình làm sạch & Khử trùng lặp (Deduplication)
1. **Lọc trùng lặp bằng Perceptual Hashing (pHash)**:
   * Do một số bộ dữ liệu được trích xuất từ video, các frame liên tiếp (cách nhau 1/30s) có thể giống hệt nhau.
   * Nếu 1 frame rơi vào tập Train và 1 frame giống hệt rơi vào tập Test, kết quả sẽ bị **Rò rỉ dữ liệu (Data Leakage)** làm điểm mAP ảo.
   * Chạy script `02_clean_and_deduplicate.py` tính khoảng cách Hamming pHash: Nếu 2 ảnh có độ tương đồng $> 92\%$, chỉ giữ lại 1 ảnh.
2. **Kiểm tra Bounding Box hợp lệ**:
   * Loại bỏ các bounding box có tọa độ âm, tọa độ vượt quá kích thước ảnh ($> 1.0$), hoặc diện tích $w \times h \le 0$.

### 3.3. Phân chia tập dữ liệu cố định (Train / Val / Test Split)
Sau khi lọc sạch, tập Master Dataset đạt quy mô khoảng **7,000 ảnh sạch**:
* **`dataset_master/images/train/`**: 70% (~4,900 ảnh) — Dùng để huấn luyện cả 6 baseline.
* **`dataset_master/images/val/`**: 15% (~1,050 ảnh) — Dùng để tune hyperparameter và lưu `best.pt`.
* **`dataset_master/images/test/`**: 15% (~1,050 ảnh) — **DÙNG CHUNG ĐỂ ĐÁNH GIÁ VÀ XUẤT RA BẢNG 1**.

### 3.4. File cấu hình chuẩn `configs/data.yaml`
Cả 6 mô hình Baseline đều trỏ về file cấu hình duy nhất này:

```yaml
# configs/data.yaml - Master Dataset Drowning Detection
path: /Users/nguyenquangthanh/Downloads/drowning-detection/data/processed/dataset_master
train: images/train
val: images/val
test: images/test

# Số lớp phân loại
nc: 2

# Tên các lớp
names:
  0: swimming
  1: drowning
```

---

## 4. CƠ CHẾ KIỂM THỬ: TÁCH BẠCH RÕ RÀNG GIỮA ẢNH TĨNH VÀ VIDEO

Nghiên cứu của bạn có 2 bài toán kiểm thử riêng biệt phục vụ 2 đóng góp khác nhau:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        HAI HỆ THỐNG KIỂM THỬ TÁCH BẠCH                                 │
├────────────────────────────────────────────┬───────────────────────────────────────────┤
│ HỆ THỐNG 1: KIỂM THỬ ẢNH TĨNH              │ HỆ THỐNG 2: KIỂM THỬ CHUỖI VIDEO          │
│ (Frame-level Benchmark)                    │ (Temporal Video Testbed)                  │
├────────────────────────────────────────────┼───────────────────────────────────────────┤
│ • Thư mục: `dataset_master/images/test/`   │ • Thư mục: `data/video_testbed/videos/`   │
│ • Đối tượng: Cả 6 mô hình Baseline         │ • Đối tượng: Detector tốt nhất + Temporal │
│ • Dữ liệu: 1,050 ảnh tĩnh độc lập          │ • Dữ liệu: 20 video liên tục (25-30 fps)  │
│ • Mục đích: Đo mAP@0.5, P, R, FPS          │ • Mục đích: Đo False Alarm Rate & Delay   │
│ • Kết quả xuất ra: BẢNG 1 BÁO CÁO          │ • Kết quả xuất ra: BẢNG 2 BÁO CÁO         │
└────────────────────────────────────────────┴───────────────────────────────────────────┘
```

### 4.1. Kiểm thử Cấp độ 1: Đánh giá 6 Model Baseline trên Ảnh tĩnh (`test/`)
* **Mục tiêu**: So sánh năng lực trích xuất đặc trưng không gian (*spatial features*) giữa các kiến trúc khác nhau (Two-stage vs One-stage vs Transformer).
* **Quy tắc thực nghiệm**:
  * Chạy file `val` hoặc `test` với cùng một tập ảnh `dataset_master/images/test/`.
  * Ghi nhận: Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95, GFLOPs, Số tham số (Params), và Tốc độ suy luận (FPS trên GPU T4).
  * Điền số liệu vào **Bảng 1 của Báo cáo**.

### 4.2. Kiểm thử Cấp độ 2: Đánh giá Temporal Filtering trên Video Testbed
* **Mục tiêu**: Chứng minh đóng góp cốt lõi của đề tài — **Cắt giảm triệt để báo động giả (False Alarms) bằng suy luận chuỗi thời gian**.
* **Cấu trúc File Annotation `ground_truth_events.csv`**:
  Mỗi sự kiện trong 20 video kiểm thử được định nghĩa chính xác theo mốc thời gian:

```csv
video_id,video_name,fps,total_frames,start_sec,end_sec,event_type,description
1,true_drown_01.mp4,30,900,12.5,24.0,true_drowning,Bé chới với rồi chìm ở góc hồ
2,true_drown_02.mp4,25,750,08.0,18.2,true_drowning,Bé trượt chân chìm dần dưới phao
3,fake_splash_01.mp4,30,600,04.0,06.5,fake_trigger,Bé đập tay té nước đùa giỡn
4,fake_diving_02.mp4,30,900,15.2,18.0,fake_trigger,Bé ngụp lặn nín thở dưới đáy hồ
5,fake_waving_03.mp4,30,600,10.0,12.8,fake_trigger,Bé đứng nước vẫy tay chào phụ huynh
```

* **Các chỉ số đo đạc trên Video Testbed**:
  1. **Số lần Báo động giả (False Alarms)**: Số lần hệ thống phát chuông cảnh báo trong các đoạn video `fake_trigger` hoặc ngoài khoảng `[start_sec, end_sec]` của sự kiện đuối nước thật.
  2. **False Alarm Rate (FAR)**: $\text{FAR} = \frac{\text{Tổng số lần báo động giả}}{\text{Tổng số phút video kiểm thử}}$.
  3. **Độ trễ phát hiện (Detection Delay $\tau_d$)**: $\tau_d = T_{\text{kích hoạt chuông}} - T_{\text{start\_sec}}$ (Đo bằng giây, yêu cầu $\tau_d \le 2.0 - 2.5s$).
  4. **Tỷ lệ Bỏ sót (Miss Rate)**: Tỷ lệ các sự kiện `true_drowning` mà hệ thống không phát chuông cảnh báo (Yêu cầu nghiêm ngặt $= 0\%$).

---

## 5. MÃ NGUỒN SCRIPT MẪU TỰ ĐỘNG HÓA PIPELINE

Dưới đây là các script Python hoàn chỉnh, sẵn sàng thực thi để bạn xây dựng pipeline ngay lập tức:

### 5.1. Script Gộp dữ liệu & Chia tập 70/15/15 (`scripts/03_merge_and_split_master.py`)

```python
import os
import shutil
import random
from pathlib import Path
from PIL import Image
import imagehash

# Cố định seed ngẫu nhiên để tái lập kết quả khoa học
random.seed(42)

PROJECT_ROOT = Path("/Users/nguyenquangthanh/Downloads/drowning-detection")
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "dataset_master"

# Ánh xạ nhãn về 2 lớp: 0 (swimming), 1 (drowning)
LABEL_MAPPING = {
    "figshare": {0: 0, 1: 1, 2: 1},  # 0: swim -> 0, 1: struggling -> 1, 2: drown -> 1
    "springer": {0: 0, 1: 1},         # 0: swim, 1: drown
}

def create_dirs():
    for split in ["train", "val", "test"]:
        (PROCESSED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (PROCESSED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

def is_duplicate(image_path, seen_hashes, threshold=5):
    """Khử trùng lặp ảnh bằng Perceptual Hashing (pHash)"""
    try:
        with Image.open(image_path) as img:
            h = imagehash.phash(img)
            for seen_h in seen_hashes:
                if h - seen_h <= threshold:
                    return True
            seen_hashes.append(h)
            return False
    except Exception:
        return True

def process_and_merge():
    create_dirs()
    all_samples = []
    seen_hashes = []

    print("[1/3] Đang quét và kiểm tra trùng lặp từ các nguồn dữ liệu raw...")
    # Quét qua các thư mục raw đã tải về
    for source_name in os.listdir(RAW_DIR):
        source_path = RAW_DIR / source_name
        if not source_path.is_dir():
            continue
        
        img_dir = source_path / "images"
        lbl_dir = source_path / "labels"
        if not img_dir.exists():
            continue

        for img_file in img_dir.glob("*.*"):
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            lbl_file = lbl_dir / f"{img_file.stem}.txt"
            if not lbl_file.exists():
                continue

            # Kiểm tra trùng lặp
            if not is_duplicate(img_file, seen_hashes):
                all_samples.append((img_file, lbl_file, source_name))

    print(f"Tổng số ảnh sạch không trùng lặp: {len(all_samples)}")

    # Xáo trộn ngẫu nhiên
    random.shuffle(all_samples)

    # Chia tỷ lệ 70% Train, 15% Val, 15% Test
    total = len(all_samples)
    train_end = int(0.70 * total)
    val_end = int(0.85 * total)

    train_data = all_samples[:train_end]
    val_data = all_samples[train_end:val_end]
    test_data = all_samples[val_end:]

    splits = [("train", train_data), ("val", val_data), ("test", test_data)]

    print("[2/3] Đang sao chép ảnh và chuẩn hóa nhãn về 2 class...")
    for split_name, split_samples in splits:
        for idx, (img_path, lbl_path, source) in enumerate(split_samples):
            new_stem = f"{source}_{split_name}_{idx:05d}"
            target_img = PROCESSED_DIR / "images" / split_name / f"{new_stem}{img_path.suffix}"
            target_lbl = PROCESSED_DIR / "labels" / split_name / f"{new_stem}.txt"

            shutil.copy(img_path, target_img)

            # Chuẩn hóa nhãn
            with open(lbl_path, "r") as f_in, open(target_lbl, "w") as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        orig_cls = int(parts[0])
                        # Ánh xạ về 0 (swim) hoặc 1 (drown)
                        mapping = LABEL_MAPPING.get("figshare" if "figshare" in source else "springer", {0:0, 1:1})
                        new_cls = mapping.get(orig_cls, 1)
                        f_out.write(f"{new_cls} {' '.join(parts[1:5])}\n")

    print("[3/3] Hoàn tất! Master Dataset đã sẵn sàng tại data/processed/dataset_master")

if __name__ == "__main__":
    process_and_merge()
```

---

### 5.2. Module Temporal Filtering Cốt lõi (`src/temporal_filter.py`)

```python
from collections import deque
import numpy as np

class TemporalDrowningFilter:
    """
    Module lọc chuỗi thời gian Debounce Cửa sổ trượt
    Chỉ kích hoạt cảnh báo khi phát hiện trạng thái Drowning liên tục >= N giây.
    """
    def __init__(self, fps=30, duration_threshold_sec=2.0, tolerance_ratio=0.85):
        self.fps = fps
        self.duration_threshold_sec = duration_threshold_sec
        self.window_size = int(fps * duration_threshold_sec)
        self.tolerance_ratio = tolerance_ratio
        
        # Lưu trữ bộ đệm lịch sử cho từng swimmer ID: {track_id: deque(maxlen=window_size)}
        self.track_buffers = {}
        # Ghi nhận trạng thái đã kích hoạt chuông cảnh báo chưa: {track_id: bool}
        self.alert_states = {}

    def update(self, track_id, is_drowning_frame):
        """
        track_id: ID duy nhất của người bơi do ByteTrack gán
        is_drowning_frame: bool (True nếu Detector dự đoán là drowning với conf >= 0.5)
        """
        if track_id not in self.track_buffers:
            self.track_buffers[track_id] = deque(maxlen=self.window_size)
            self.alert_states[track_id] = False

        buffer = self.track_buffers[track_id]
        buffer.append(1 if is_drowning_frame else 0)

        # Chưa đủ số frame trong cửa sổ trượt -> Chưa kích hoạt
        if len(buffer) < self.window_size:
            progress = len(buffer) / self.window_size if is_drowning_frame else 0.0
            return False, progress

        # Tính tỷ lệ frame drowning trong cửa sổ trượt
        drowning_ratio = sum(buffer) / self.window_size

        if drowning_ratio >= self.tolerance_ratio:
            self.alert_states[track_id] = True
            return True, 1.0  # KÍCH HOẠT BÁO ĐỘNG CHÍNH THỨC
        else:
            self.alert_states[track_id] = False
            return False, drowning_ratio
```

---

## 6. CHECKLIST THỰC HIỆN ĐỀ TÀI DÀNH CHO BẠN

Để đảm bảo tiến độ bảo vệ đề tài tại UIT, hãy thực hiện theo đúng 5 bước sau:

1. **Bước 1 (Tuần 1)**: Chạy lệnh tạo toàn bộ cây thư mục dự án theo đúng Section 1.
2. **Bước 2 (Tuần 2)**: Tải dữ liệu từ Figshare DOI và Springer, đặt vào `data/raw/`. Chạy script `03_merge_and_split_master.py` để tạo ra thư mục `data/processed/dataset_master/` có sẵn 3 tập `train`, `val`, `test`.
3. **Bước 3 (Tuần 3 - 5)**: Huấn luyện 6 mô hình Baseline (Faster R-CNN, SSD, YOLOv5s, YOLOv8s, YOLOv11s, RT-DETR) trên tập `train/`. Chấm điểm cả 6 mô hình trên tập `test/` để xuất ra **Bảng 1 (So sánh Baseline)**.
4. **Bước 4 (Tuần 6 - 7)**: Chuẩn bị 15 - 20 video kiểm thử đưa vào `data/video_testbed/videos/` và điền mốc thời gian vào file `ground_truth_events.csv`.
5. **Bước 5 (Tuần 8 - 11)**: Lắp module `temporal_filter.py` vào mô hình detector tốt nhất từ Bước 3. Chạy qua 20 video test ở cả 2 chế độ (không lọc vs có lọc với $N=1s, 2s, 3s$) để đo FAR và độ trễ, xuất ra **Bảng 2 (Chứng minh hiệu quả Temporal Filtering)**.
