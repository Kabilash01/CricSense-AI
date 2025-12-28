import sys
from pathlib import Path
import cv2
import time

# --------------------------------------------------
# OpenCV safety (Windows)
# --------------------------------------------------
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

# --------------------------------------------------
# Add src/ to PYTHONPATH
# --------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.detectors.yolov8_detector import YoloV8Tracker
from core.player_registry import PlayerRegistry
from ingest.video_reader import VideoReader
from core.event_timeline import EventTimeline

from jersey.smolvlm_reader import SmolVLMJerseyReader
from jersey.jersey_vote_tracker import JerseyVoteTracker


# --------------------------------------------------
# CONFIG (FPS + STABILITY TUNED)
# --------------------------------------------------
VIDEO_PATH = r"C:\cricket-ai\data\samples\test3.mp4"
MODEL_PATH = r"C:\cricket player train\cricket-ai\yolov8m_production4\weights\best.pt"

JERSEY_SAMPLE_INTERVAL = 6
MAX_JERSEY_CROPS_PER_PLAYER = 12
MAX_PLAYERS = 10


# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def center_of_box(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def infer_role_from_class(name):
    if name in ["Batsman", "Bowler", "Player_Generic", "Wicket_Keeper"]:
        return name
    return "Ignore"


# --------------------------------------------------
# Jersey helpers
# --------------------------------------------------
def is_jersey_visible(player_crop):
    h, w = player_crop.shape[:2]
    return h >= 80 and w >= 40


def extract_jersey_regions(player_crop):
    h, w = player_crop.shape[:2]
    return [
        player_crop[int(h * 0.30):int(h * 0.55),
                    int(w * 0.25):int(w * 0.75)],
        player_crop[int(h * 0.20):int(h * 0.50),
                    int(w * 0.15):int(w * 0.85)]
    ]


def is_good_region(crop):
    if crop is None:
        return False
    h, w = crop.shape[:2]
    return h >= 30 and w >= 40


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():

    tracker = YoloV8Tracker(MODEL_PATH)
    registry = PlayerRegistry()

    jersey_reader = SmolVLMJerseyReader()
    jersey_tracker = JerseyVoteTracker()

    timeline = EventTimeline()

    reader = VideoReader(
        VIDEO_PATH,
        resize=(1280, 720),
        skip=2
    )

    frame_id = 0
    frame_count = 0
    t0 = time.time()

    # --------------------------------------------------
    # Frame loop
    # --------------------------------------------------
    for frame in reader:
        frame_id += 1
        frame_count += 1

        jersey_enabled = len(jersey_tracker.locked) < MAX_PLAYERS
        jersey_processed_this_frame = False

        detections = tracker.track(frame, conf=0.3)

        for det in detections:
            raw_id = det["track_id"]
            box = det["box"]
            role = infer_role_from_class(det["name"])

            if role == "Ignore":
                continue

            x1, y1, x2, y2 = map(int, box)

            # Early reject tiny players
            if (y2 - y1) < 120 or (x2 - x1) < 60:
                continue

            player_crop = frame[y1:y2, x1:x2]
            if player_crop.size == 0:
                continue

            center = center_of_box(box)

            stable_id = registry.resolve_id(
                raw_id=raw_id,
                center=center,
                role=role,
                frame_id=frame_id
            )

            registry.update(stable_id, center)

            # ---------------- EVENT: PLAYER_APPEAR ----------------
            if stable_id not in registry.seen_ids:
                registry.seen_ids.add(stable_id)
                timeline.log(
                    frame_id=frame_id,
                    sid=stable_id,
                    role=role,
                    event="PLAYER_APPEAR"
                )

            # ---------------- EVENT: ROLE_CONFIRMED ----------------
            if stable_id not in registry.role_confirmed:
                registry.role_confirmed.add(stable_id)
                timeline.log(
                    frame_id=frame_id,
                    sid=stable_id,
                    role=role,
                    event="ROLE_CONFIRMED"
                )

            # ---------------- Jersey logic ----------------
            if (
                jersey_enabled
                and not jersey_processed_this_frame
                and frame_id % JERSEY_SAMPLE_INTERVAL == 0
                and is_jersey_visible(player_crop)
                and len(registry.players[stable_id]["jersey_crops"])
                    < MAX_JERSEY_CROPS_PER_PLAYER
            ):
                for region in extract_jersey_regions(player_crop):
                    if not is_good_region(region):
                        continue

                    number, _ = jersey_reader.read_jersey_number(region)
                    jersey_tracker.add_vote(stable_id, number)
                    registry.add_jersey_crop(stable_id, region)
                    jersey_processed_this_frame = True

                    # ---------------- EVENT: JERSEY_LOCKED ----------------
                    if (
                        jersey_tracker.is_locked(stable_id)
                        and stable_id not in registry.jersey_logged
                    ):
                        locked = jersey_tracker.get_locked(stable_id)
                        registry.jersey_logged.add(stable_id)

                        timeline.log(
                            frame_id=frame_id,
                            sid=stable_id,
                            role=role,
                            jersey=locked["jersey"],
                            event="JERSEY_LOCKED",
                            meta={"confidence": locked["confidence"]}
                        )

                        print(
                            f"🔒 Jersey LOCKED → SID {stable_id}: "
                            f"#{locked['jersey']} (confidence {locked['confidence']})"
                        )
                    break

            # ---------------- Visualization ----------------
            label = f"SID {stable_id} | {role}"
            if jersey_tracker.is_locked(stable_id):
                label += f" | #{jersey_tracker.get_locked(stable_id)['jersey']}"

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

        # ---------------- FPS display ----------------
        fps = frame_count / (time.time() - t0 + 1e-6)
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.imshow("Cricket AI – Event Timeline Enabled", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        del frame

    # --------------------------------------------------
    # Cleanup + Export
    # --------------------------------------------------
    reader.release()
    cv2.destroyAllWindows()

    timeline.export_json("event_timeline.json")
    print("Event summary:", timeline.summary())


if __name__ == "__main__":
    main()
