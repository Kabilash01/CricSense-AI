import sys
from pathlib import Path
import cv2
import math

# --------------------------------------------------
# Add src/ to PYTHONPATH
# --------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.detectors.yolov8_detector import YoloV8Tracker
from core.player_registry import PlayerRegistry
from ingest.video_reader import VideoReader

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def center_of_box(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def infer_role_from_class(name):
    if name in [
        "Batsman",
        "Bowler",
        "Wicket_Keeper",
        "Umpire",
        "Player_Generic",
        "Ball"
    ]:
        return name
    return "Unknown"


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# --------------------------------------------------
# Pitch / zone logic
# --------------------------------------------------
def define_pitch_zones(w, h):
    pitch_top = int(h * 0.35)
    pitch_bottom = int(h * 0.65)

    batting_crease = (
        int(w * 0.45),
        pitch_bottom - 20,
        int(w * 0.55),
        pitch_bottom + 20
    )

    bowling_crease = (
        int(w * 0.45),
        pitch_top - 20,
        int(w * 0.55),
        pitch_top + 20
    )

    return batting_crease, bowling_crease


def inside_zone(point, zone):
    x, y = point
    x1, y1, x2, y2 = zone
    return x1 <= x <= x2 and y1 <= y <= y2


def refine_role(role, center, batting_crease, bowling_crease):
    if inside_zone(center, batting_crease):
        return "Batsman"
    if inside_zone(center, bowling_crease):
        return "Bowler"
    return role


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    video_path = r"C:\cricket-ai\data\samples\test3.mp4"
    model_path = r"C:\cricket player train\cricket-ai\yolov8m_production4\weights\best.pt"

    tracker = YoloV8Tracker(model_path)
    registry = PlayerRegistry(max_idle_frames=60)

    reader = VideoReader(video_path)
    frame_id = 0

    batting_crease = bowling_crease = None
    shot_counter = {}

    SHOT_DISTANCE_THRESHOLD = 60
    SHOT_FRAMES = 3

    for frame in reader:
        frame_id += 1
        h, w = frame.shape[:2]

        if frame_id == 1:
            batting_crease, bowling_crease = define_pitch_zones(w, h)

        detections = tracker.track(frame, conf=0.3)
        ball_center = None

        for det in detections:
            raw_id = det["track_id"]
            box = det["box"]
            role = infer_role_from_class(det["name"])
            center = center_of_box(box)

            # Zone-based role refinement
            role = refine_role(role, center, batting_crease, bowling_crease)

            # Resolve to stable ID
            stable_id = registry.resolve_id(
                raw_id=raw_id,
                center=center,
                role=role,
                frame_id=frame_id
            )

            registry.update(stable_id, center)

            if role == "Ball":
                ball_center = center

            # Draw ONLY clean bounding box + label
            x1, y1, x2, y2 = map(int, box)
            label = f"SID {stable_id} | {role}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # --------------------------------------------------
        # BALL ↔ BATSMAN INTERACTION (SHOT DETECTION)
        # --------------------------------------------------
        if ball_center is not None:
            for sid, player in registry.players.items():
                if player["final_role"] == "Batsman":
                    batsman_center = player["last_center"]

                    if distance(ball_center, batsman_center) < SHOT_DISTANCE_THRESHOLD:
                        shot_counter[sid] = shot_counter.get(sid, 0) + 1
                        if shot_counter[sid] >= SHOT_FRAMES:
                            cv2.putText(
                                frame,
                                "SHOT DETECTED",
                                (50, 60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0,
                                (0, 0, 255),
                                3
                            )
                    else:
                        shot_counter[sid] = 0

        cv2.imshow("Cricket AI – Clean Output", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    reader.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
