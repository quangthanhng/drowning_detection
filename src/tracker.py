import numpy as np
from typing import List, Dict, Any

class SimpleSwimmerTracker:
    """
    Tracker cơ bản dựa trên IoU (Intersection over Union) và khoảng cách tâm
    để gán ID ổn định cho từng người bơi trong hồ bơi qua các khung hình liên tiếp.
    """

    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 15):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.next_track_id = 1
        # Lưu các track hiện tại: {track_id: {'box': [x1, y1, x2, y2], 'lost': int, 'class_id': int, 'conf': float}}
        self.tracks: Dict[int, Dict[str, Any]] = {}

    @staticmethod
    def compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Cập nhật danh sách track dựa trên detections mới tại khung hình.

        Args:
            detections: Danh sách dict [{'box': [x1, y1, x2, y2], 'class_id': int, 'conf': float}]

        Returns:
            List[Dict[str, Any]]: Detections đã được gán 'track_id'
        """
        matched_tracks = set()
        matched_dets = set()
        tracked_results = []

        if len(self.tracks) > 0 and len(detections) > 0:
            track_ids = list(self.tracks.keys())
            iou_matrix = np.zeros((len(track_ids), len(detections)))

            for i, tid in enumerate(track_ids):
                for j, det in enumerate(detections):
                    iou_matrix[i, j] = self.compute_iou(self.tracks[tid]["box"], det["box"])

            # Khớp tham lam (Greedy Matching)
            while True:
                max_val = np.max(iou_matrix)
                if max_val < self.iou_threshold:
                    break
                i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                tid = track_ids[i]
                
                # Cập nhật track
                self.tracks[tid]["box"] = detections[j]["box"]
                self.tracks[tid]["class_id"] = detections[j]["class_id"]
                self.tracks[tid]["conf"] = detections[j]["conf"]
                self.tracks[tid]["lost"] = 0

                res = dict(detections[j])
                res["track_id"] = tid
                tracked_results.append(res)

                matched_tracks.add(tid)
                matched_dets.add(j)

                iou_matrix[i, :] = -1
                iou_matrix[:, j] = -1

        # Tạo track mới cho các detections chưa khớp
        for j, det in enumerate(detections):
            if j not in matched_dets:
                tid = self.next_track_id
                self.next_track_id += 1
                self.tracks[tid] = {
                    "box": det["box"],
                    "class_id": det["class_id"],
                    "conf": det["conf"],
                    "lost": 0
                }
                res = dict(det)
                res["track_id"] = tid
                tracked_results.append(res)

        # Cập nhật các track bị mất dấu
        lost_track_ids = []
        for tid, tinfo in self.tracks.items():
            if tid not in matched_tracks and tid not in [r["track_id"] for r in tracked_results]:
                tinfo["lost"] += 1
                if tinfo["lost"] > self.max_lost_frames:
                    lost_track_ids.append(tid)

        # Xóa các track mất quá lâu
        for tid in lost_track_ids:
            del self.tracks[tid]

        return tracked_results
