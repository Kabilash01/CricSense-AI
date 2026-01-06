import sys
import time
from pathlib import Path
import cv2

# --------------------------------------------------
# OpenCV safety
# --------------------------------------------------
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

# --------------------------------------------------
# Add src/ to PYTHONPATH
# --------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.detectors.yolov8_detector import YoloV8Tracker
from core.player_registry import PlayerRegistry
from core.event_timeline import EventTimeline
from core.run_detector import RunDetector
from ingest.video_reader import VideoReader


# --------------------------------------------------
# Helper
# --------------------------------------------------
def box_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():

    video_path = r"C:\cricket-ai\data\samples\test3.mp4"
    model_path = r"C:\cricket player train\cricket-ai\yolov8m_production4\weights\best.pt"

    tracker = YoloV8Tracker(model_path)
    registry = PlayerRegistry()
    timeline = EventTimeline()
    run_detector = RunDetector()

    reader = VideoReader(
        video_path,
        resize=(1280, 720),
        skip=0   # ❗ no frame skipping
    )

    print("[INFO] Pipeline started")

    frame_id = 0
    t0 = time.time()
    frame_count = 0

    last_shot_frame = -999

    for frame in reader:
        frame_id += 1
        frame_count += 1

        detections = tracker.track(frame, conf=0.35)

        for d in detections:
            if "box" not in d or "track_id" not in d:
                continue

            center = box_center(d["box"])
            role = d["name"]
            raw_id = d["track_id"]

            sid = registry.resolve_id(
                raw_id=raw_id,
                center=center,
                role=role,
                frame_id=frame_id
            )

            registry.update(sid, center)

            # -------------------------------
            # RUN DETECTION (after shot)
            # -------------------------------
            if role == "Batsman":
                if run_detector.update(
                    sid=sid,
                    frame_id=frame_id,
                    center=center,
                    last_shot_frame=last_shot_frame
                ):
                    timeline.log({
                        "frame": frame_id,
                        "sid": sid,
                        "role": "Batsman",
                        "event": "RUN_START",
                        "meta": {}
                    })
                    print(f"🏃 RUN_START @ frame {frame_id} (SID {sid})")

            # -------------------------------
            # Visualization
            # -------------------------------
            x1, y1, x2, y2 = map(int, d["box"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"SID {sid} | {role}",
                (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # -------------------------------
        # FPS overlay
        # -------------------------------
        fps = frame_count / (time.time() - t0 + 1e-6)
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow("Cricket AI – Stable Pipeline", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    reader.release()
    cv2.destroyAllWindows()

    timeline.save("events_timeline.json")
    print("[INFO] Timeline saved → events_timeline.json")


if __name__ == "__main__":
    main()
