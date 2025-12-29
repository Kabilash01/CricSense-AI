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
from core.scorecard_detector import ScorecardDetector
from core.run_detector import RunDetector
from core.run_count_detector import RunCountDetector
from core.score_engine import ScoreEngine


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

    scorecard_detector = ScorecardDetector()
    run_detector = RunDetector()
    run_count_detector = RunCountDetector()
    score_engine = ScoreEngine()

    reader = VideoReader(VIDEO_PATH, resize=(1280, 720), skip=1)

    frame_id = 0
    t0 = time.time()

    batsman_motion = {}

    # -------- BALL STATE --------
    ball_active = False
    ball_id = 0
    ball_closed = set()
    run_started_this_ball = False

    last_ball_release = -999
    last_shot_frame = {}
    last_run_start = {}

    cv2.namedWindow("Cricket AI – Score Engine", cv2.WINDOW_NORMAL)

    # --------------------------------------------------
    # Frame loop
    # --------------------------------------------------
    for frame in reader:
        frame_id += 1
        frame_h = frame.shape[0]

        # ---------------- SCORECARD (backup) ----------------
        visible, location, _ = scorecard_detector.detect(frame, frame_id)
        if visible:
            timeline.log(
                frame_id=frame_id,
                sid=-1,
                role="Scorecard",
                jersey=None,
                event="SCORECARD_VISIBLE",
                meta={"location": location},
            )

        detections = tracker.track(frame, conf=0.3)

        for det in detections:

            role = infer_role_from_class(det["name"])
            if role == "Ignore":
                continue

            box = det["box"]
            x1, y1, x2, y2 = map(int, box)

            if (x2 - x1) < 60 or (y2 - y1) < 100:
                continue

            center = center_of_box(box)

            sid = registry.resolve_id(
                raw_id=det["track_id"],
                center=center,
                role=role,
                frame_id=frame_id,
            )
            if sid is None:
                continue

            registry.update(sid, center)

            # ---------------- PLAYER EVENTS ----------------
            if sid not in registry.seen_ids:
                registry.seen_ids.add(sid)
                timeline.log(
                    frame_id=frame_id,
                    sid=sid,
                    role=role,
                    jersey=None,
                    event="PLAYER_APPEAR",
                    meta={},
                )

            if sid not in registry.role_confirmed:
                registry.role_confirmed.add(sid)
                timeline.log(
                    frame_id=frame_id,
                    sid=sid,
                    role=role,
                    jersey=None,
                    event="ROLE_CONFIRMED",
                    meta={},
                )

            # --------------------------------------------------
            # BATSMAN LOGIC
            # --------------------------------------------------
            if role == "Batsman":

                batsman_motion.setdefault(sid, deque(maxlen=6))
                batsman_motion[sid].append(center)

                if len(batsman_motion[sid]) >= 6:
                    dx = batsman_motion[sid][-1][0] - batsman_motion[sid][0][0]
                    dy = batsman_motion[sid][-1][1] - batsman_motion[sid][0][1]
                    movement = math.hypot(dx, dy)

                    if movement > 18:
                        timeline.log(
                            frame_id=frame_id,
                            sid=sid,
                            role=role,
                            jersey=None,
                            event="BATSMAN_REACTION",
                            meta={"movement": round(movement, 2)},
                        )

                        if frame_id - last_ball_release > 40 and not ball_active:
                            ball_active = True
                            ball_id += 1
                            run_started_this_ball = False
                            score_engine.on_ball_start()

                            timeline.log(
                                frame_id=frame_id,
                                sid=sid,
                                role="Bowler",
                                jersey=None,
                                event="BALL_RELEASE",
                                meta={"ball_id": ball_id},
                            )

                            last_ball_release = frame_id

                    last_shot = last_shot_frame.get(sid, -999)
                    if movement > 30 and frame_id - last_shot > 20:
                        last_shot_frame[sid] = frame_id
                        timeline.log(
                            frame_id=frame_id,
                            sid=sid,
                            role=role,
                            jersey=None,
                            event="SHOT_ATTEMPT",
                            meta={"movement": round(movement, 2)},
                        )

                # ---------------- RUN_START ----------------
                if (
                    ball_active
                    and not run_started_this_ball
                    and run_detector.update(
                        sid, frame_id, center, last_shot_frame.get(sid, -999)
                    )
                ):
                    run_started_this_ball = True
                    last_run_start[sid] = frame_id

                    timeline.log(
                        frame_id=frame_id,
                        sid=sid,
                        role=role,
                        jersey=None,
                        event="RUN_START",
                        meta={"ball_id": ball_id},
                    )

                # ---------------- RUN_COUNT ----------------
                if (
                    ball_active
                    and run_started_this_ball
                    and run_count_detector.update(
                        sid,
                        frame_id,
                        center,
                        frame_h,
                        last_run_start.get(sid, -999),
                    )
                ):
                    score_engine.on_run(1)
                    timeline.log(
                        frame_id=frame_id,
                        sid=sid,
                        role=role,
                        jersey=None,
                        event="RUN_COUNT",
                        meta={"runs": 1, "ball_id": ball_id},
                    )

            # ---------------- DRAW ----------------
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"SID {sid} | {role}",
                (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # --------------------------------------------------
        # BALL_END (ONCE PER BALL)
        # --------------------------------------------------
        if ball_active and ball_id not in ball_closed:

            if run_started_this_ball:
                last_run = max(last_run_start.values(), default=-999)
                if frame_id - last_run > 25:
                    summary = score_engine.on_ball_end()
                    ball_active = False
                    ball_closed.add(ball_id)

                    timeline.log(
                        frame_id=frame_id,
                        sid=-1,
                        role="Match",
                        jersey=None,
                        event="BALL_END",
                        meta={"ball_id": ball_id, **summary},
                    )

            else:
                last_shot = max(last_shot_frame.values(), default=-999)
                if frame_id - last_shot > BALL_END_TIMEOUT:
                    summary = score_engine.on_ball_end()
                    ball_active = False
                    ball_closed.add(ball_id)

                    timeline.log(
                        frame_id=frame_id,
                        sid=-1,
                        role="Match",
                        jersey=None,
                        event="BALL_END",
                        meta={"ball_id": ball_id, **summary},
                    )

        # ---------------- FPS ----------------
        fps = frame_id / (time.time() - t0 + 1e-6)
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )

        cv2.imshow("Cricket AI – Score Engine", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    reader.release()
    cv2.destroyAllWindows()
    timeline.export_json("events_timeline.json")

    print("Event summary:", timeline.summary())


if __name__ == "__main__":
    main()
