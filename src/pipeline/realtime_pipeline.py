# src/pipeline/realtime_pipeline.py
import sys
from pathlib import Path
import time
import cv2
import os
import torch


# add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest.video_reader import VideoReader
from ui.overlay import draw_boxes
from vision.detectors.yolov8_detector import YoloV8Detector
from core.event_timeline import EventTimeline

def draw_labeled_boxes(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = map(int, d['box'])
        label = f"{d['name']}:{d['score']:.2f}"
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        # put label
        cv2.putText(frame, label, (x1, max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    return frame

def main():
    video_path = Path("C:\\cricket-ai\\data\\samples\\test.mp4").resolve()
    if not video_path.exists():
        print("ERROR: sample.mp4 missing at:", video_path)
        return

    # Choose device: 'cpu' or 'cuda:0' as available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"  # None = let ultralytics auto-select; set 'cpu' or 'cuda:0' if you prefer
    model_path = "yolov8n.pt"  # small, fast model. Replace with your fine-tuned weights later.
    detector = YoloV8Detector(model_path=model_path, device=device)

    reader = VideoReader(str(video_path))
    frame_count = 0
    t0 = time.time()

    for frame in reader:
        frame_count += 1
        t_start = time.time()

        # run detection (adjust conf for speed/precision)
        detections = detector.predict(frame, conf=0.35)

        # draw boxes
        frame = draw_labeled_boxes(frame, detections)

        # compute FPS
        elapsed = time.time() - t_start
        total_elapsed = time.time() - t0
        fps = frame_count / total_elapsed if total_elapsed>0 else 0.0

        # overlay simple stats
        cv2.putText(frame, f"Frame: {frame_count} FPS: {fps:.1f}", (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)

        cv2.imshow("Cricket AI Pipeline - Detections", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Total frames processed:", frame_count)
    reader.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
