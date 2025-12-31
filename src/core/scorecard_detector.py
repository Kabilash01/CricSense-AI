import cv2
import numpy as np


class ScorecardDetector:
    """
    Phase-1: Detect scorecard presence using ROI + edge density.
    No OCR, no ML, FPS-safe.
    """

    def __init__(self, edge_threshold=0.015, cooldown_frames=30):
        self.edge_threshold = edge_threshold
        self.cooldown_frames = cooldown_frames
        self.last_seen_frame = -999

    def detect(self, frame, frame_id):
        h, w = frame.shape[:2]

        # Define ROIs
        rois = {
            "bottom_left": frame[int(h * 0.80):int(h * 0.98),
                                 int(w * 0.02):int(w * 0.45)],
            "bottom_right": frame[int(h * 0.80):int(h * 0.98),
                                  int(w * 0.55):int(w * 0.98)],
        }

        for name, roi in rois.items():
            if roi.size == 0:
                continue

            if self._has_scorecard_edges(roi):
                if frame_id - self.last_seen_frame > self.cooldown_frames:
                    self.last_seen_frame = frame_id
                    return True, name, roi

        return False, None, None

    def _has_scorecard_edges(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        edge_ratio = np.count_nonzero(edges) / edges.size
        return edge_ratio > self.edge_threshold
