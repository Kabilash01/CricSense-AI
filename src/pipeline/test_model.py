import sys
from pathlib import Path
import cv2
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest.video_reader import VideoReader
from vision.detectors.yolov8_detector import YoloV8Tracker
from core.player_registry import PlayerRegistry

def center_of_box(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def infer_role_from_class(name):
    if name in [
        "Batsman",
        "Bowler",
        "Wicket_Keeper",
        "Umpire",
        "Player_Generic"
    ]:
        return name
    return None


def main():
    video_path = r"C:\\cricket-ai\\data\\samples\\test3.mp4"
    model_path = r"C:\\cricket-ai\\yolov8m_production4\\weights\\best.pt"

    tracker = YoloV8Tracker(model_path)
    registry = PlayerRegistry()

    reader = VideoReader(video_path)
    frame_id = 0

    for frame in reader:
        frame_id += 1
        detections = tracker.track(frame, conf=0.3)

        for det in detections:
            track_id = det["track_id"]
            box = det["box"]
            role_vote = infer_role_from_class(det["name"])
            center = center_of_box(box)

            registry.update(track_id, role_vote, center, frame_id)

            role = registry.get_role(track_id)

            # Draw
            x1, y1, x2, y2 = map(int, box)
            label = f"ID {track_id} | {role}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("YOLOv8 + ByteTrack + Player Registry", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    reader.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
