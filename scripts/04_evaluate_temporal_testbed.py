import os
import cv2
import pandas as pd
from pathlib import Path
import sys

# Thêm thư mục gốc vào PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.temporal_filter import TemporalDrowningFilter
from src.tracker import SimpleSwimmerTracker
from src.metrics import calculate_temporal_metrics

def load_ground_truth(csv_path: Path):
    if not csv_path.exists():
        print(f"[-] Không tìm thấy file ground truth: {csv_path}")
        return []
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")

def evaluate_video(
    video_path: Path,
    detector_model,
    temporal_filter: TemporalDrowningFilter,
    tracker: SimpleSwimmerTracker,
    conf_threshold: float = 0.5
):
    """
    Chạy suy luận trên 1 video, áp dụng tracker và temporal filter,
    ghi nhận các thời điểm phát chuông cảnh báo.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[-] Không thể mở video: {video_path}")
        return [], 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    temporal_filter.fps = int(fps)
    temporal_filter.reset_all()

    frame_idx = 0
    predicted_alerts = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_sec = frame_idx / fps

        # Giả lập hoặc gọi mô hình Detector thực tế
        # detector_model nhận frame -> trả về detections [{'box': [x1, y1, x2, y2], 'class_id': int, 'conf': float}]
        detections = []
        if detector_model is not None:
            results = detector_model(frame, verbose=False)[0]
            for box in results.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                if conf >= conf_threshold:
                    coords = box.xyxy[0].cpu().numpy().tolist()
                    detections.append({
                        "box": coords,
                        "class_id": cls_id,
                        "conf": conf
                    })

        # Cập nhật ID theo dõi cho người bơi
        tracked_swimmers = tracker.update(detections)

        # Đưa qua Temporal Filter cho từng người bơi
        for swimmer in tracked_swimmers:
            t_id = swimmer["track_id"]
            # Class 1: Drowning
            is_drown = (swimmer["class_id"] == 1)
            is_alert, progress = temporal_filter.update(t_id, is_drown, frame_idx)

            if is_alert:
                predicted_alerts.append({
                    "video": video_path.name,
                    "frame_idx": frame_idx,
                    "alert_sec": round(current_sec, 2),
                    "track_id": t_id
                })

        frame_idx += 1

    total_duration_minutes = (frame_idx / fps) / 60.0
    cap.release()
    return predicted_alerts, total_duration_minutes

def main():
    gt_csv = PROJECT_ROOT / "data" / "video_testbed" / "ground_truth_events.csv"
    video_dir = PROJECT_ROOT / "data" / "video_testbed" / "videos"
    output_csv = PROJECT_ROOT / "experiments" / "table2_temporal_filter_results.csv"

    print("==========================================================================")
    print(" [ĐÁNH GIÁ THỰC NGHIỆM VIDEO TESTBED & ĐÓNG GÓP TEMPORAL FILTERING] ")
    print("==========================================================================")

    gt_events = load_ground_truth(gt_csv)
    print(f"[+] Đã tải {len(gt_events)} sự kiện ground-truth từ file CSV.")

    # Khảo sát các ngưỡng thời gian N
    thresholds = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0]
    print(f"[+] Các ngưỡng thời gian khảo sát N (giây): {thresholds}")
    print("[*] Kết quả sau khi chạy thực nghiệm sẽ được xuất ra:", output_csv)

if __name__ == "__main__":
    main()
