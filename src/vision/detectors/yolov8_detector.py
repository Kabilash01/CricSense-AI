from ultralytics import YOLO
import torch
from pathlib import Path

class YoloV8Tracker:
    def __init__(self, model_path, device="cuda:0"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)

        if self.device != "cpu":
            self.model.to(self.device)

        # 🔑 Absolute path to ByteTrack config
        self.tracker_config = (
            Path(__file__)
            .resolve()
            .parents[3] / "configs" / "bytetrack.yaml"
        )

        if not self.tracker_config.exists():
            raise FileNotFoundError(
                f"ByteTrack config not found: {self.tracker_config}"
            )

    def track(self, frame, conf=0.3):
        results = self.model.track(
            frame,
            conf=conf,
            tracker=str(self.tracker_config),
            persist=True,
            device=self.device,
            verbose=False
        )[0]

        detections = []

        if results.boxes is None or results.boxes.id is None:
            return detections

        for box, track_id in zip(results.boxes, results.boxes.id):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            score = float(box.conf[0])
            name = self.model.names[cls_id]

            detections.append({
                "track_id": int(track_id),
                "box": [x1, y1, x2, y2],
                "score": score,
                "class_id": cls_id,
                "name": name
            })

        return detections
