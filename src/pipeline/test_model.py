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
from core.player_registry import PlayerRegistry
from ingest.video_reader import VideoReader
from core.event_timeline import EventTimeline


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
VIDEO_PATH = r"C:\cricket-ai\data\samples\test3.mp4"
MODEL_PATH = r"C:\cricket player train\cricket-ai\yolov8m_production4\weights\best.pt"

REACTION_HISTORY = 6
REACTION_THRESHOLD = 18
SHOT_THRESHOLD = 30
EVENT_COOLDOWN = 40


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

    reader = VideoReader(
        VIDEO_PATH,
        resize=(1280, 720),
        skip=2
    )

    frame_id = 0
    t0 = time.time()

    batsman_motion = {}
    last_ball_release = -999

    cv2.namedWindow("Cricket AI – Full Event Timeline", cv2.WINDOW_NORMAL)

    # --------------------------------------------------
    # Frame loop
    # --------------------------------------------------
    for frame in reader:
        frame_id += 1

        detections = tracker.track(frame, conf=0.3)

        for det in detections:
            role = infer_role_from_class(det["name"])
            if role == "Ignore":
                continue

            raw_id = det["track_id"]
            box = det["box"]
            x1, y1, x2, y2 = map(int, box)

            if (x2 - x1) < 60 or (y2 - y1) < 100:
                continue

            center = center_of_box(box)

            sid = registry.resolve_id(
                raw_id=raw_id,
                center=center,
                role=role,
                frame_id=frame_id
            )
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

            # ---------------- BATSMAN EVENTS ----------------
            if role == "Batsman":
                if sid not in batsman_motion:
                    batsman_motion[sid] = deque(maxlen=REACTION_HISTORY)

                batsman_motion[sid].append(center)

                if len(batsman_motion[sid]) >= REACTION_HISTORY:
                    dx = batsman_motion[sid][-1][0] - batsman_motion[sid][0][0]
                    dy = batsman_motion[sid][-1][1] - batsman_motion[sid][0][1]
                    movement = math.hypot(dx, dy)

                    # BATSMAN_REACTION
                    if movement > REACTION_THRESHOLD:
                        timeline.log(
                            frame_id=frame_id,
                            sid=sid,
                            role=role,
                            jersey=None,
                            event="BATSMAN_REACTION",
                            meta={"movement": round(movement, 2)}
                        )

                        # BALL_RELEASE (virtual)
                        if frame_id - last_ball_release > EVENT_COOLDOWN:
                            last_ball_release = frame_id
                            timeline.log(
                                frame_id=frame_id,
                                sid=sid,
                                role="Bowler",
                                jersey=None,
                                event="BALL_RELEASE",
                                meta={"defined_as": "batsman_reaction"}
                            )
                            print(f"🎯 BALL_RELEASE @ frame {frame_id}")

                    # SHOT_ATTEMPT
                    if movement > SHOT_THRESHOLD and frame_id - last_ball_release < 15:
                        timeline.log(
                            frame_id=frame_id,
                            sid=sid,
                            role=role,
                            jersey=None,
                            event="SHOT_ATTEMPT",
                            meta={"movement": round(movement, 2)}
                        )
                        print(f"🏏 SHOT_ATTEMPT @ frame {frame_id}")

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

        cv2.imshow("Cricket AI – Full Event Timeline", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    reader.release()
    cv2.destroyAllWindows()

    timeline.export_json("event_timeline.json")
    print("Event summary:", timeline.summary())


if __name__ == "__main__":
    main()
