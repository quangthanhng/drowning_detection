from collections import deque
from typing import Dict, Tuple

class TemporalDrowningFilter:
    """
    Module lọc chuỗi thời gian (Sliding Window Debounce) cho bài toán Drowning Detection.
    
    Nguyên lý:
    - Nhận tín hiệu nhị phân (0: Safe/Swimming, 1: Drowning) từ Detector tại từng frame theo từng Track ID.
    - Duy trì bộ đệm cửa sổ trượt kích thước W = FPS * duration_threshold_sec.
    - Chỉ kích hoạt còi cảnh báo khi tỷ lệ xuất hiện trạng thái 'Drowning' trong cửa sổ >= tolerance_ratio.
    - Triệt tiêu các tín hiệu nhiễu thoáng qua (vẫy tay, lặn, té nước trong 1-2 frames).
    """

    def __init__(self, fps: int = 30, duration_threshold_sec: float = 2.0, tolerance_ratio: float = 0.85):
        self.fps = fps
        self.duration_threshold_sec = duration_threshold_sec
        self.window_size = max(1, int(fps * duration_threshold_sec))
        self.tolerance_ratio = tolerance_ratio

        # Bộ đệm trạng thái cho từng người bơi: {track_id: deque(maxlen=window_size)}
        self.track_buffers: Dict[int, deque] = {}
        # Trạng thái cảnh báo hiện tại: {track_id: bool}
        self.alert_states: Dict[int, bool] = {}
        # Thời điểm lần đầu phát hiện nghi vấn: {track_id: int (frame_index)}
        self.first_suspicious_frame: Dict[int, int] = {}

    def set_duration_threshold(self, duration_sec: float):
        """Thay đổi ngưỡng thời gian xác thực N (dùng cho Ablation Study)"""
        self.duration_threshold_sec = duration_sec
        self.window_size = max(1, int(self.fps * duration_sec))
        # Reset lại maxlen của các deque
        for track_id, old_buf in self.track_buffers.items():
            self.track_buffers[track_id] = deque(list(old_buf), maxlen=self.window_size)

    def update(self, track_id: int, is_drowning_frame: bool, current_frame_idx: int) -> Tuple[bool, float]:
        """
        Cập nhật trạng thái của đối tượng tại khung hình hiện tại.

        Args:
            track_id (int): ID duy nhất của đối tượng bơi do tracker cung cấp.
            is_drowning_frame (bool): True nếu frame-level detector dự đoán là 'drowning'.
            current_frame_idx (int): Chỉ số frame hiện tại trong video.

        Returns:
            Tuple[bool, float]:
                - is_alert_triggered (bool): True nếu đã đủ điều kiện xác thực đuối nước thật.
                - suspicion_progress (float): Tiến trình nghi vấn [0.0 -> 1.0] để vẽ thanh progress bar HUD.
        """
        if track_id not in self.track_buffers:
            self.track_buffers[track_id] = deque(maxlen=self.window_size)
            self.alert_states[track_id] = False
            self.first_suspicious_frame[track_id] = -1

        buffer = self.track_buffers[track_id]
        buffer.append(1 if is_drowning_frame else 0)

        # Ghi nhận frame đầu tiên phát hiện nghi vấn
        if is_drowning_frame and self.first_suspicious_frame[track_id] == -1:
            self.first_suspicious_frame[track_id] = current_frame_idx
        elif not is_drowning_frame and sum(buffer) == 0:
            self.first_suspicious_frame[track_id] = -1

        # Nếu cửa sổ trượt chưa được lấp đầy
        if len(buffer) < self.window_size:
            progress = len(buffer) / self.window_size if is_drowning_frame else 0.0
            return False, min(1.0, progress)

        # Tính tỷ lệ trạng thái drowning trong cửa sổ trượt
        drowning_ratio = sum(buffer) / self.window_size

        if drowning_ratio >= self.tolerance_ratio:
            self.alert_states[track_id] = True
            return True, 1.0
        else:
            self.alert_states[track_id] = False
            return False, drowning_ratio

    def reset_track(self, track_id: int):
        """Xóa theo dõi đối tượng khi người bơi rời khỏi khung hình"""
        self.track_buffers.pop(track_id, None)
        self.alert_states.pop(track_id, None)
        self.first_suspicious_frame.pop(track_id, None)

    def reset_all(self):
        """Reset toàn bộ bộ đệm khi bắt đầu video mới"""
        self.track_buffers.clear()
        self.alert_states.clear()
        self.first_suspicious_frame.clear()
