import sys
import time
import cv2
from pathlib import Path

# --------------------------------------------------
# OpenCV safety (Windows)
# --------------------------------------------------
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

# --------------------------------------------------
# Add src/ to PYTHONPATH
# --------------------------------------------------
SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

# --------------------------------------------------
# Imports
# --------------------------------------------------
from vision.detectors.yolov8_detector import YoloV8Tracker
from ingest.video_reader import VideoReader
from core.event_timeline import EventTimeline

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def infer_role(cls_name: str) -> str:
    if cls_name in ["Batsman", "Bowler", "Wicket_Keeper", "Player_Generic"]:
        return cls_name
    return "Ignore"

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():

    video_path = r"C:\cricket-ai\data\samples\test3.mp4"
    model_path = r"C:\cricket player train\cricket-ai\yolov8m_production4\weights\best.pt"

    # ------------------------------
    # Core components
    # ------------------------------
    tracker = YoloV8Tracker(model_path)
    reader = VideoReader(video_path, resize=(1280, 720), skip=0)

    timeline = EventTimeline()

    frame_id = 0
    t0 = time.time()

    # --------------------------------------------------
    # Frame loop
    # --------------------------------------------------
    for frame in reader:
        frame_id += 1

        # ------------------------------
        # YOLO + ByteTrack inference
        # ------------------------------
        detections = tracker.track(frame, conf=0.35)

        # ------------------------------
        # DRAW BOUNDING BOXES (IMMEDIATE)
        # ------------------------------
        for det in detections:
            role = infer_role(det["name"])
            if role == "Ignore":
                continue

            x1, y1, x2, y2 = map(int, det["box"])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{role} | ID {det['track_id']}",
                (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Optional logging
            timeline.log({
                "frame": frame_id,
                "sid": det["track_id"],
                "role": role,
                "event": "PLAYER_DETECTED",
                "meta": {
                    "conf": round(det["score"], 3)
                }
            })

        # ------------------------------
        # FPS overlay
        # ------------------------------
        fps = frame_id / (time.time() - t0 + 1e-6)
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.imshow("DEBUG – YOLO ONLY (OCR OFF)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------
    reader.release()
    cv2.destroyAllWindows()

    # --------------------------------------------------
    # Export JSON (optional)
    # --------------------------------------------------
    out_path = Path("events_timeline_yolo_only.json")
    timeline.export(out_path)
    print(f"[OK] YOLO-only timeline saved to {out_path}")

# --------------------------------------------------
if __name__ == "__main__":
    main()
