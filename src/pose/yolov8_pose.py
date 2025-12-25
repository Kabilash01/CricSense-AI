from ultralytics import YOLO
import cv2

class YOLOv8Pose:
    def __init__(self, model="yolov8n-pose.pt", device="cuda"):
        self.model = YOLO(model)
        self.model.to(device)
        self.device = device

    def estimate(self, crop):
        if crop is None or crop.size == 0:
            return None
        
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = self.model(rgb, verbose=False, device=self.device)

        if len(results) == 0 or results[0].keypoints is None:
            return None

        kpts = results[0].keypoints[0]  # first detection
        final = []
        for x, y, conf in kpts:
            final.append((int(x), int(y), float(conf)))
        return final
