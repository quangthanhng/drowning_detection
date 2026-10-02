#!/bin/bash
# ==============================================================================
# scripts/01_download_datasets.sh
# HƯỚNG DẪN CHI TIẾT & LỆNH TẢI DATASET HỌC THUẬT (2022 - 2026)
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${PROJECT_ROOT}/data/raw"

echo "=========================================================================="
echo "          HƯỚNG DẪN CHI TIẾT TẢI DATASET NGUỒN 2 VÀ NGUỒN 3              "
echo "=========================================================================="

# ------------------------------------------------------------------------------
# [NGUỒN 2]: SWIMMING-YOLO POOL DATASET (SPRINGER SIVP 2025)
# ------------------------------------------------------------------------------
SPRINGER_DIR="${RAW_DIR}/swimming_yolo_springer_2025"
mkdir -p "${SPRINGER_DIR}/images" "${SPRINGER_DIR}/labels"

echo ""
echo "--------------------------------------------------------------------------"
echo ">>> [NGUỒN 2]: SWIMMING-YOLO POOL DATASET (Springer SIVP 2025 - Jiang et al.)"
echo "--------------------------------------------------------------------------"
echo "Bản chất học thuật: Bài báo Springer SIVP 2025 sử dụng tập dữ liệu thu thập"
echo "từ camera bờ hồ bơi bao quát (tilt/overhead) và các kho mở Roboflow Universe."
echo ""
echo "CÁCH 1: Tải trực tiếp qua Trình duyệt Web (Khuyến nghị, dễ thực hiện nhất):"
echo "  1. Mở trình duyệt và truy cập vào 1 trong 2 kho dữ liệu chuẩn của bài báo:"
echo "     Link A (DrowningDetectionTracking - 9.5k ảnh):"
echo "     https://universe.roboflow.com/drowning-detection/drowningdetectiontracking"
echo "     hoặc Link B (Swimming and Drowning Detection - 7.3k ảnh):"
echo "     https://universe.roboflow.com/university-g3h71/swimming-and-drowning-detection"
echo ""
echo "  2. Tại trang web Roboflow, nhấn nút màu tím: [Download Dataset] ở góc phải."
echo "  3. Tại mục 'Export Format', chọn: [YOLOv8] hoặc [YOLOv5 PyTorch]."
echo "  4. Tích chọn: [download zip to computer] -> Nhấn [Continue]."
echo "  5. Sau khi tải về file .zip, giải nén và chuyển các file vào thư mục:"
echo "     - Chép toàn bộ ảnh vào:  ${SPRINGER_DIR}/images/"
echo "     - Chép toàn bộ nhãn vào: ${SPRINGER_DIR}/labels/"
echo ""
echo "CÁCH 2: Tải tự động bằng Python Roboflow (Nếu bạn có API Key miễn phí):"
echo "  Chạy đoạn code sau trong Python:"
echo "  --------------------------------------------------"
echo "  from roboflow import Roboflow"
echo "  rf = Roboflow(api_key='YOUR_ROBOFLOW_API_KEY')"
echo "  project = rf.workspace('drowning-detection').project('drowningdetectiontracking')"
echo "  version = project.version(1)"
echo "  dataset = version.download('yolov8', location='${SPRINGER_DIR}')"
echo "  --------------------------------------------------"

# ------------------------------------------------------------------------------
# [NGUỒN 3]: WANG-KAIKAI DRONE POOL DATASET (GITHUB / RESEARCHGATE 2022-2023)
# ------------------------------------------------------------------------------
DRONE_DIR="${RAW_DIR}/drone_pool_wang_2023"
mkdir -p "${DRONE_DIR}/images" "${DRONE_DIR}/labels"

echo ""
echo "--------------------------------------------------------------------------"
echo ">>> [NGUỒN 3]: WANG-KAIKAI DRONE POOL DATASET (GitHub / ResearchGate)"
echo "--------------------------------------------------------------------------"
echo "Bản chất học thuật: Dataset góc nhìn trên cao (Top-down UAV) trong hồ bơi"
echo "của nhóm nghiên cứu Kaikai Wang et al. Kho mã nguồn: Wang-Kaikai/drowning-detection-dataset."
echo "Trong repository có sẵn thư mục 'self-made dataset' chứa 'images' và 'labels'."
echo ""
echo "CÁCH 1: Tải tự động bằng 1 dòng lệnh Curl (Nhanh nhất - Không cần cài Git):"
echo "  Bạn có thể chạy ngay lệnh sau trong Terminal để tự động tải và giải nén:"
echo ""
cat << 'EOF'
  curl -L "https://github.com/Wang-Kaikai/drowning-detection-dataset/archive/refs/heads/main.zip" -o "/tmp/wang_dataset.zip"
  unzip -q "/tmp/wang_dataset.zip" -d "/tmp/wang_unzipped"
  cp -r "/tmp/wang_unzipped/drowning-detection-dataset-main/self-made dataset/images/"* "${RAW_DIR}/drone_pool_wang_2023/images/"
  cp -r "/tmp/wang_unzipped/drowning-detection-dataset-main/self-made dataset/labels/"* "${RAW_DIR}/drone_pool_wang_2023/labels/"
  rm -rf "/tmp/wang_dataset.zip" "/tmp/wang_unzipped"
  echo "[+] Đã tải và trích xuất thành công Nguồn 3 vào ${RAW_DIR}/drone_pool_wang_2023/!"
EOF
echo ""
echo "CÁCH 2: Tải thủ công qua Trình duyệt Web:"
echo "  1. Truy cập trực tiếp: https://github.com/Wang-Kaikai/drowning-detection-dataset"
echo "  2. Bấm nút màu xanh [<> Code] -> Chọn [Download ZIP]."
echo "  3. Mở file .zip tải về, truy cập vào thư mục: 'self-made dataset/'"
echo "  4. Sao chép:"
echo "     - Thư mục 'images' bỏ vào: ${DRONE_DIR}/images/"
echo "     - Thư mục 'labels' bỏ vào: ${DRONE_DIR}/labels/"
echo ""
echo "=========================================================================="
echo "HỎI NHANH: Bạn có muốn tự động chạy tải NGUỒN 3 (Wang-Kaikai UAV) ngay bây giờ không? (y/n)"
read -r -p "Lựa chọn của bạn: " choice || true
if [[ "$choice" =~ ^[Yy]$ ]]; then
    echo "[*] Đang tải Nguồn 3 từ GitHub..."
    curl -L "https://github.com/Wang-Kaikai/drowning-detection-dataset/archive/refs/heads/main.zip" -o "/tmp/wang_dataset.zip"
    echo "[*] Đang giải nén..."
    unzip -q "/tmp/wang_dataset.zip" -d "/tmp/wang_unzipped"
    echo "[*] Đang sao chép vào ${DRONE_DIR}..."
    cp -r "/tmp/wang_unzipped/drowning-detection-dataset-main/self-made dataset/images/"* "${DRONE_DIR}/images/"
    cp -r "/tmp/wang_unzipped/drowning-detection-dataset-main/self-made dataset/labels/"* "${DRONE_DIR}/labels/"
    rm -rf "/tmp/wang_dataset.zip" "/tmp/wang_unzipped"
    echo "[+] Tải và bố trí Nguồn 3 THÀNH CÔNG!"
    echo "    Tổng số ảnh: $(ls "${DRONE_DIR}/images" | wc -l)"
    echo "    Tổng số file nhãn: $(ls "${DRONE_DIR}/labels" | wc -l)"
else
    echo "[i] Bỏ qua tải tự động. Bạn có thể tự thực hiện bất cứ lúc nào."
fi

echo ""
echo "=========================================================================="
echo "SAU KHI ĐẶT ĐỦ DỮ LIỆU VÀO CẢ 3 NGUỒN TRONG data/raw/, CHẠY LỆNH:"
echo "python3 scripts/03_merge_and_split_master.py"
echo "=========================================================================="
