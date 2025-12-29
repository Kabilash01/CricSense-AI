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


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
VIDEO_PATH = r"C:\cricket-ai\data\samples\test3.mp4"
MODEL_PATH = r"C:\cricket player train\cricket-ai\yolov8m_production4\weights\best.pt"


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

    # ---------------- Core systems ----------------
    tracker = YoloV8Tracker(MODEL_PATH)
    registry = PlayerRegistry()
    timeline = EventTimeline()

    scorecard_detector = ScorecardDetector()   # backup only
    run_detector = RunDetector()               # RUN_START
    run_count_detector = RunCountDetector()    # RUN_COUNT

    reader = VideoReader(
        VIDEO_PATH,
        resize=(1280, 720),
        skip=1          # NO frame skipping
    )

    # ---------------- Runtime state ----------------
    frame_id = 0
    t0 = time.time()

    batsman_motion = {}        # sid -> deque
    last_ball_release = -999
    last_shot_frame = {}       # sid -> frame_id
    last_run_start = {}        # sid -> frame_id

    cv2.namedWindow("Cricket AI – Run Engine", cv2.WINDOW_NORMAL)

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
                meta={"location": location}
            )

        # ---------------- PLAYER TRACKING ----------------
        detections = tracker.track(frame, conf=0.3)

        for det in detections:

            role = infer_role_from_class(det["name"])
            if role == "Ignore":
                continue

            box = det["box"]
            x1, y1, x2, y2 = map(int, box)

            # Reject tiny / noisy boxes
            if (x2 - x1) < 60 or (y2 - y1) < 100:
                continue

            center = center_of_box(box)

            # ---------------- SID RESOLUTION ----------------
            sid = registry.resolve_id(
                raw_id=det["track_id"],
                center=center,
                role=role,
                frame_id=frame_id
            )

            if sid is None:
                continue

            registry.update(sid, center)

            # ---------------- PLAYER_APPEAR ----------------
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

            # ---------------- ROLE_CONFIRMED ----------------
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

            # --------------------------------------------------
            # BATSMAN LOGIC
            # --------------------------------------------------
            if role == "Batsman":

                if sid not in batsman_motion:
                    batsman_motion[sid] = deque(maxlen=6)

                batsman_motion[sid].append(center)

                if len(batsman_motion[sid]) >= 6:
                    dx = batsman_motion[sid][-1][0] - batsman_motion[sid][0][0]
                    dy = batsman_motion[sid][-1][1] - batsman_motion[sid][0][1]
                    movement = math.hypot(dx, dy)

                    # -------- BATSMAN_REACTION --------
                    if movement > 18:
                        timeline.log(
                            frame_id=frame_id,
                            sid=sid,
                            role=role,
                            jersey=None,
                            event="BATSMAN_REACTION",
                            meta={"movement": round(movement, 2)}
                        )

                        # virtual BALL_RELEASE
                        if frame_id - last_ball_release > 40:
                            last_ball_release = frame_id
                            timeline.log(
                                frame_id=frame_id,
                                sid=sid,
                                role="Bowler",
                                jersey=None,
                                event="BALL_RELEASE",
                                meta={"source": "batsman_reaction"}
                            )

                    # -------- SHOT_ATTEMPT --------
                    last_shot = last_shot_frame.get(sid, -999)
                    if movement > 30 and frame_id - last_shot > 20:
                        last_shot_frame[sid] = frame_id
                        timeline.log(
                            frame_id=frame_id,
                            sid=sid,
                            role=role,
                            jersey=None,
                            event="SHOT_ATTEMPT",
                            meta={"movement": round(movement, 2)}
                        )

                # ---------------- RUN_START ----------------
                shot_frame = last_shot_frame.get(sid, -999)
                if run_detector.update(
                    sid=sid,
                    frame_id=frame_id,
                    center=center,
                    last_shot_frame=shot_frame
                ):
                    timeline.log(
                        frame_id=frame_id,
                        sid=sid,
                        role=role,
                        jersey=None,
                        event="RUN_START",
                        meta={}
                    )
                    last_run_start[sid] = frame_id
                    print(f"🏃 RUN_START @ frame {frame_id} (SID {sid})")

                # ---------------- RUN_COUNT ----------------
                run_start_frame = last_run_start.get(sid, -999)
                if run_count_detector.update(
                    sid=sid,
                    frame_id=frame_id,
                    center=center,
                    frame_h=frame_h,
                    run_started_frame=run_start_frame
                ):
                    timeline.log(
                        frame_id=frame_id,
                        sid=sid,
                        role=role,
                        jersey=None,
                        event="RUN_COUNT",
                        meta={"runs": 1}
                    )
                    print(f"🏏 RUN_COUNT @ frame {frame_id} (SID {sid})")

            # ---------------- Visualization ----------------
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

        # ---------------- FPS ----------------
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

        cv2.imshow("Cricket AI – Run Engine", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------
    reader.release()
    cv2.destroyAllWindows()

    timeline.export_json("event_timeline.json")
    print("Event summary:", timeline.summary())


if __name__ == "__main__":
    main()
