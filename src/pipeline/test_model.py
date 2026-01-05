import sys
from pathlib import Path
import cv2
import time
import math
from collections import deque

# --------------------------------------------------
# OpenCV safety
# --------------------------------------------------
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

# --------------------------------------------------
# Path setup
# --------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.detectors.yolov8_detector import YoloV8Tracker
from ingest.video_reader import VideoReader
from core.player_registry import PlayerRegistry
from core.event_timeline import EventTimeline
from core.score_engine import ScoreEngine
from core.scorecard_detector import ScorecardDetector
from core.scorecard_roi import ScorecardROILocker
from core.score_delta_detector import ScoreDeltaDetector
from core.run_detector import RunDetector

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
VIDEO_PATH = r"C:\cricket-ai\data\samples\test3.mp4"
MODEL_PATH = r"C:\cricket player train\cricket-ai\yolov8m_production4\weights\best.pt"
BALL_END_TIMEOUT = 30

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def center_of_box(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def infer_role_from_class(name):
    if name in ["Batsman", "Bowler", "Wicket_Keeper", "Player_Generic"]:
        return name
    return "Ignore"

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():

    tracker = YoloV8Tracker(MODEL_PATH)
    registry = PlayerRegistry()
    timeline = EventTimeline()
    score_engine = ScoreEngine()

    scorecard_detector = ScorecardDetector()
    roi_locker = ScorecardROILocker()
    score_delta = ScoreDeltaDetector()   # EasyOCR
    run_detector = RunDetector()

    reader = VideoReader(VIDEO_PATH, resize=(1280, 720), skip=1)

    frame_id = 0
    t0 = time.time()

    batsman_motion = {}
    last_shot_frame = {}
    last_run_start = {}

    # BALL STATE
    ball_active = False
    ball_id = 0
    ball_closed = set()

    cv2.namedWindow("Cricket AI – Score Engine", cv2.WINDOW_NORMAL)

    # --------------------------------------------------
    # Frame loop
    # --------------------------------------------------
    for frame in reader:
        frame_id += 1

        # ---------- Scorecard visibility ----------
        visible, location, _ = scorecard_detector.detect(frame, frame_id)
        if visible:
            timeline.log(
                frame_id=frame_id,
                sid=-1,
                role="Scorecard",
                jersey=None,
                event="SCORECARD_VISIBLE",
                meta={"location": location}
            )

        detections = tracker.track(frame, conf=0.3)

        for det in detections:

            role = infer_role_from_class(det["name"])
            if role == "Ignore":
                continue

            x1, y1, x2, y2 = map(int, det["box"])
            if (x2 - x1) < 60 or (y2 - y1) < 100:
                continue

            center = center_of_box(det["box"])

            sid = registry.resolve_id(
                raw_id=det["track_id"],
                center=center,
                role=role,
                frame_id=frame_id
            )
            if sid is None:
                continue

            registry.update(sid, center)

            # ---------- DRAW BOX ----------
            color = (0, 255, 0) if role == "Batsman" else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"SID {sid} | {role}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            # ---------- PLAYER EVENTS ----------
            if sid not in registry.seen_ids:
                registry.seen_ids.add(sid)
                timeline.log(
                    frame_id=frame_id,
                    sid=sid,
                    role=role,
                    jersey=None,
                    event="PLAYER_APPEAR",
                    meta={}
                )

            if sid not in registry.role_confirmed:
                registry.role_confirmed.add(sid)
                timeline.log(
                    frame_id=frame_id,
                    sid=sid,
                    role=role,
                    jersey=None,
                    event="ROLE_CONFIRMED",
                    meta={}
                )

            # ---------- BATSMAN LOGIC ----------
            if role == "Batsman":

                batsman_motion.setdefault(sid, deque(maxlen=6))
                batsman_motion[sid].append(center)

                if len(batsman_motion[sid]) >= 6:
                    dx = batsman_motion[sid][-1][0] - batsman_motion[sid][0][0]
                    dy = batsman_motion[sid][-1][1] - batsman_motion[sid][0][1]
                    movement = math.hypot(dx, dy)

                    # BALL RELEASE
                    if movement > 18 and not ball_active:
                        ball_active = True
                        ball_id += 1
                        score_engine.on_ball_start()

                        timeline.log(
                            frame_id=frame_id,
                            sid=sid,
                            role="Bowler",
                            jersey=None,
                            event="BALL_RELEASE",
                            meta={"ball_id": ball_id}
                        )

                    # SHOT ATTEMPT
                    if movement > 30:
                        last_shot_frame[sid] = frame_id
                        timeline.log(
                            frame_id=frame_id,
                            sid=sid,
                            role=role,
                            jersey=None,
                            event="SHOT_ATTEMPT",
                            meta={"movement": round(movement, 2)}
                        )

                # RUN START
                if ball_active and run_detector.update(
                    sid, frame_id, center, last_shot_frame.get(sid, -999)
                ):
                    last_run_start[sid] = frame_id
                    timeline.log(
                        frame_id=frame_id,
                        sid=sid,
                        role=role,
                        jersey=None,
                        event="RUN_START",
                        meta={"ball_id": ball_id}
                    )

        # ---------- BALL END ----------
        if ball_active and ball_id not in ball_closed:
            last_event = max(
                list(last_run_start.values()) + list(last_shot_frame.values()),
                default=-999
            )

            if frame_id - last_event > BALL_END_TIMEOUT:
                ball_active = False
                ball_closed.add(ball_id)

                score_crop = roi_locker.crop(frame)
                delta = score_delta.detect_delta(score_crop)

                if delta:
                    score_engine.apply_scoreboard_delta(
                        runs=delta["runs_delta"],
                        wickets=delta["wicket_delta"]
                    )
                    timeline.log(
                        frame_id=frame_id,
                        sid=-1,
                        role="Scoreboard",
                        jersey=None,
                        event="SCORE_UPDATE",
                        meta=delta
                    )

                summary = score_engine.on_ball_end() or {}
                timeline.log(
                    frame_id=frame_id,
                    sid=-1,
                    role="Match",
                    jersey=None,
                    event="BALL_END",
                    meta={"ball_id": ball_id, **summary}
                )

        # ---------- FPS ----------
        fps = frame_id / (time.time() - t0 + 1e-6)
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.imshow("Cricket AI – Score Engine", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    reader.release()
    cv2.destroyAllWindows()
    timeline.export_json("event_timeline.json")

    print("Event summary:", timeline.summary())


if __name__ == "__main__":
    main()
