# ByteTrack implementation
# src/vision/tracking/bytetrack_wrapper.py
import numpy as np
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.track import STrack

class ByteTrackWrapper:
    def __init__(self,
                 track_thresh=0.5,
                 match_thresh=0.8,
                 track_buffer=30,
                 frame_rate=30):

        self.tracker = BYTETracker(
            {
                "track_thresh": track_thresh,
                "match_thresh": match_thresh,
                "track_buffer": track_buffer,
                "mot20": False
            },
            frame_rate
        )
        self.frame_id = 0

    def update(self, detections, img):
        """
        detections: list of [x1, y1, x2, y2, score, class_name]
        img: original frame (not used, but required by ByteTrack)
        returns list of tracks:
            [{"track_id":int, "bbox":[x1,y1,x2,y2], "score":float, "label":str}]
        """

        self.frame_id += 1

        if len(detections) == 0:
            dets = np.empty((0, 5))
        else:
            # YOLO output → ByteTrack format
            dets = np.array([
                [x1, y1, x2, y2, score]
                for (x1, y1, x2, y2, score, label) in detections
            ], dtype=np.float32)

        online_targets = self.tracker.update(dets, img.shape, img.shape)

        tracks_out = []
        for t in online_targets:
            if not t.is_activated:
                continue

            tlwh = t.tlwh
            x1 = int(tlwh[0])
            y1 = int(tlwh[1])
            x2 = int(tlwh[0] + tlwh[2])
            y2 = int(tlwh[1] + tlwh[3])
            tracks_out.append({
                "track_id": int(t.track_id),
                "bbox": [x1, y1, x2, y2],
                "score": float(t.score),
                "label": "player"  # you can refine later
            })

        return tracks_out
