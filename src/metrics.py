from typing import List, Dict, Any
import numpy as np

def calculate_temporal_metrics(
    video_duration_minutes: float,
    ground_truth_events: List[Dict[str, Any]],
    predicted_alerts: List[Dict[str, Any]],
    delay_tolerance_window_sec: float = 4.0
) -> Dict[str, float]:
    """
    Tính toán các chỉ số định lượng đánh giá hiệu quả của Temporal Filtering trên Video Testbed:
    - False Alarm Rate (FAR): Số lần báo động giả / phút video
    - Detection Delay (tau_d): Thời gian trễ trung bình từ khi bắt đầu đuối nước thật đến khi phát chuông (giây)
    - Miss Rate: Tỷ lệ bỏ sót các sự kiện đuối nước thật (%)

    Args:
        video_duration_minutes: Tổng thời lượng video kiểm thử tính bằng phút.
        ground_truth_events: Danh sách các sự kiện ground truth [{'start_sec': float, 'end_sec': float, 'type': 'true_drowning' | 'fake_trigger'}]
        predicted_alerts: Danh sách các thời điểm hệ thống kích hoạt chuông [{'alert_sec': float, 'track_id': int}]
        delay_tolerance_window_sec: Khoảng thời gian tối đa cho phép kể từ start_sec để tính là phát hiện đúng.

    Returns:
        Dict[str, float]: Kết quả đo đạc định lượng chuẩn NCKH.
    """
    true_drowning_events = [e for e in ground_truth_events if e.get("type") == "true_drowning" or e.get("event_type") == "true_drowning"]
    fake_trigger_events = [e for e in ground_truth_events if e.get("type") == "fake_trigger" or e.get("event_type") == "fake_trigger"]

    num_true_events = len(true_drowning_events)
    delays = []
    detected_true_events = set()
    false_alarm_count = 0

    for alert in predicted_alerts:
        t_alert = alert["alert_sec"]
        matched_true = False

        for idx, gt in enumerate(true_drowning_events):
            t_start = gt["start_sec"]
            t_end = gt["end_sec"]
            # Nếu cảnh báo rơi vào khoảng thời gian xảy ra đuối nước thật
            if t_start <= t_alert <= (t_end + delay_tolerance_window_sec):
                matched_true = True
                if idx not in detected_true_events:
                    detected_true_events.add(idx)
                    delay = max(0.0, t_alert - t_start)
                    delays.append(delay)
                break

        # Nếu không khớp với bất kỳ sự kiện đuối nước thật nào -> BÁO ĐỘNG GIẢ (False Alarm)
        if not matched_true:
            false_alarm_count += 1

    # Tính Miss Rate (Tỷ lệ bỏ sót)
    missed_count = num_true_events - len(detected_true_events)
    miss_rate = (missed_count / num_true_events * 100.0) if num_true_events > 0 else 0.0

    # Tính False Alarm Rate (FAR: lần/phút)
    far = false_alarm_count / video_duration_minutes if video_duration_minutes > 0 else 0.0

    # Tính Detection Delay trung bình
    mean_delay = float(np.mean(delays)) if len(delays) > 0 else 0.0

    return {
        "total_false_alarms": false_alarm_count,
        "false_alarm_rate_per_min": round(far, 3),
        "mean_detection_delay_sec": round(mean_delay, 2),
        "miss_rate_percent": round(miss_rate, 2),
        "detected_events": len(detected_true_events),
        "total_events": num_true_events
    }

def calculate_frame_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Tính Precision, Recall, F1 ở cấp độ Frame-level"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }
