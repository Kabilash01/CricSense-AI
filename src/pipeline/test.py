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
# Path setup
# --------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --------------------------------------------------
# Imports
# --------------------------------------------------
from vision.detectors.yolov8_detector import YoloV8Tracker
from ingest.video_reader import VideoReader

from core.player_registry import PlayerRegistry
from core.event_timeline import EventTimeline
from core.identity_binder import IdentityBinder
from core.name_stabilizer import NameStabilizer
from core.score_engine import ScoreEngine
from core.run_detector import RunDetector

from ocr.scoreboard_semantic_reader import ScoreboardSemanticReader
from ocr.async_scoreboard_ocr import AsyncScoreboardOCR

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def center_of_box(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def infer_role(name):
    if name in ["Batsman", "Bowler", "Wicket_Keeper", "Player_Generic"]:
        return name
    return "Ignore"

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():

    print("[INFO] Pipeline started (RUN → STRIKER → SCORE ENGINE)")

    video_path = r"C:\cricket-ai\data\samples\test3.mp4"
    model_path = r"C:\cricket player train\cricket-ai\yolov8m_production4\weights\best.pt"

    # ---------------- Core components ----------------
    tracker = YoloV8Tracker(model_path)
    registry = PlayerRegistry()
    timeline = EventTimeline()
    binder = IdentityBinder()
    name_stabilizer = NameStabilizer()

    score_engine = ScoreEngine()
    run_detector = RunDetector()

    scoreboard_reader = ScoreboardSemanticReader(gpu=False)
    async_ocr = AsyncScoreboardOCR(
        scoreboard_reader,
        interval_frames=90
    )

    reader = VideoReader(
        video_path,
        resize=(1280, 720),
        skip=0
    )

    frame_id = 0
    t0 = time.time()

    current_striker_sid = None
    last_shot_frame = -999

    # --------------------------------------------------
    # Frame loop
    # --------------------------------------------------
    for frame in reader:
        frame_id += 1

        detections = tracker.track(frame, conf=0.3)

        # ---------------- PLAYER TRACKING ----------------
        for det in detections:
            role = infer_role(det["name"])
            if role == "Ignore":
                continue

            raw_id = det["track_id"]
            box = det["box"]
            center = center_of_box(box)

            sid = registry.resolve_id(
                raw_id=raw_id,
                center=center,
                role=role,
                frame_id=frame_id
            )
            registry.update(sid, center)

            # Draw box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"SID {sid} | {role}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            # ---------------- RUN DETECTION (STRIKER ONLY) ----------------
            if sid == current_striker_sid:
                if run_detector.update(sid, frame_id, center, last_shot_frame):
                    timeline.log({
                        "frame": frame_id,
                        "sid": sid,
                        "role": "Batsman",
                        "event": "RUN_START",
                        "meta": {}
                    })

                    score_engine.ball_start(sid)

        # ---------------- ASYNC SCOREBOARD OCR ----------------
        h, w = frame.shape[:2]
        score_roi = frame[int(0.85*h):int(0.95*h),
                          int(0.05*w):int(0.95*w)]

        async_ocr.try_submit(frame_id, score_roi)
        parsed = async_ocr.get_latest()

        if parsed:
            name_stabilizer.update("striker", parsed.get("striker"))
            name_stabilizer.update("non_striker", parsed.get("non_striker"))
            name_stabilizer.update("bowler", parsed.get("bowler"))

            active_players = registry.get_active_players()
            identity = binder.bind(parsed, active_players)

            # Inject stabilized names
            for r in ["striker", "non_striker", "bowler"]:
                if r in identity and identity[r] is not None:
                    identity[r]["name"] = name_stabilizer.get(r)

            # Lock striker SID
            if "striker" in identity and identity["striker"]:
                current_striker_sid = identity["striker"]["sid"]

            timeline.log({
                "frame": frame_id,
                "sid": -1,
                "role": "Match",
                "event": "IDENTITY_BINDING",
                "meta": identity
            })

        # ---------------- BALL END (SIMPLIFIED TRIGGER) ----------------
        # (You already have BALL_END logic elsewhere — this is safe placeholder)
        if frame_id % 150 == 0:
            summary = score_engine.ball_end()
            if summary:
                timeline.log({
                    "frame": frame_id,
                    "sid": -1,
                    "role": "Match",
                    "event": "BALL_END",
                    "meta": summary
                })
                run_detector.reset()

        # ---------------- FPS ----------------
        fps = frame_id / (time.time() - t0 + 1e-6)
        cv2.putText(
            frame,
            f"FPS {fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.imshow("Cricket AI", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ---------------- CLEANUP ----------------
    reader.release()
    cv2.destroyAllWindows()
    timeline.save("events.json")

    print("[INFO] Pipeline finished")

# --------------------------------------------------
if __name__ == "__main__":
    main()
