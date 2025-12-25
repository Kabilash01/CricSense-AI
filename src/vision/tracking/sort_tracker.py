# SORT — Simple Online Real-Time Tracking (clean portable version)

import numpy as np
from collections import deque

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / (
        (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) - wh
        + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    )
    return o

class Track:
    _id = 0
    def __init__(self, bbox):
        self.id = Track._id
        Track._id += 1
        self.bbox = bbox
        self.hits = 1
        self.no_updates = 0

    def update(self, bbox):
        self.bbox = bbox
        self.hits += 1
        self.no_updates = 0

class SortTracker:
    def __init__(self, iou_threshold=0.3, max_age=5):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = []

    def update(self, detections):
        updated = []

        if len(self.tracks) == 0:
            for d in detections:
                self.tracks.append(Track(d[:4]))

        for d in detections:
            best_iou = 0
            best_track = None

            for t in self.tracks:
                i = iou(d[:4], t.bbox)
                if i > best_iou:
                    best_iou = i
                    best_track = t

            if best_iou > self.iou_threshold:
                best_track.update(d[:4])
            else:
                self.tracks.append(Track(d[:4]))

        # Aging
        alive = []
        for t in self.tracks:
            t.no_updates += 1
            if t.no_updates <= self.max_age:
                alive.append(t)
        self.tracks = alive

        output = []
        for t in self.tracks:
            output.append({
                "track_id": t.id,
                "bbox": t.bbox,
                "label": "player"
            })
        return output
