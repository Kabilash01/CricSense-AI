from ultralytics import YOLO
import torch


class YoloV8Tracker:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)

    def track(self, frame, conf=0.35):
        results = self.model.track(
            source=frame,
            conf=conf,
            iou=0.5,
            persist=True,                 # 🔥 MUST
            tracker="bytetrack.yaml",     # 🔥 MUST
            device=self.device,
            half=True,                    # 🔥 FP16
            verbose=False
        )

        detections = []

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                if box.id is None:
                    continue
                detections.append({
                    "track_id": int(box.id.item()),
                    "box": box.xyxy.cpu().numpy()[0],
                    "score": float(box.conf.item()),
                    "name": results[0].names[int(box.cls.item())]
                })

        return detections
