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
# Jersey logic
# --------------------------------------------------
def is_jersey_visible(player_crop):
    h, w = player_crop.shape[:2]
    return h >= 80 and w >= 40


def extract_jersey_regions(player_crop):
    h, w = player_crop.shape[:2]
    regions = []

    # Front torso
    regions.append(
        player_crop[int(h * 0.30):int(h * 0.55),
                    int(w * 0.25):int(w * 0.75)]
    )

    # Back torso
    regions.append(
        player_crop[int(h * 0.20):int(h * 0.50),
                    int(w * 0.15):int(w * 0.85)]
    )

    return regions


def is_good_region(crop):
    if crop is None:
        return False
    h, w = crop.shape[:2]
    return h >= 30 and w >= 40


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():

    # ------------------------------
    # Paths
    # ------------------------------
    video_path = r"C:\cricket-ai\data\samples\test3.mp4"
    model_path = r"C:\cricket player train\cricket-ai\yolov8m_production4\weights\best.pt"

    # ------------------------------
    # Core components
    # ------------------------------
    tracker = YoloV8Tracker(model_path)
    registry = PlayerRegistry()

    reader = VideoReader(
        video_path,
        resize=(1280, 720),  # 🔥 memory-safe
        skip=2               # 🔥 reduces pressure
    )

    # ------------------------------
    # Output directory
    # ------------------------------
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    JERSEY_OUT_DIR = PROJECT_ROOT / "jersey_crops"
    JERSEY_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Jersey crops will be saved to:", JERSEY_OUT_DIR)

    # ------------------------------
    # Runtime stats
    # ------------------------------
    frame_id = 0
    frame_count = 0
    t0 = time.time()

    JERSEY_SAMPLE_INTERVAL = 3

    # --------------------------------------------------
    # Frame loop
    # --------------------------------------------------
    for frame in reader:
        frame_id += 1
        frame_count += 1

        detections = tracker.track(frame, conf=0.3)

        for det in detections:
            raw_id = det["track_id"]
            box = det["box"]
            role = infer_role_from_class(det["name"])

            if role == "Ignore":
                continue

            x1, y1, x2, y2 = map(int, box)
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

            # ------------------------------
            # Jersey sampling
            # ------------------------------
            if (
                frame_id % JERSEY_SAMPLE_INTERVAL == 0
                and is_jersey_visible(player_crop)
            ):
                jersey_regions = extract_jersey_regions(player_crop)

                sid_dir = JERSEY_OUT_DIR / f"SID_{stable_id}"
                sid_dir.mkdir(parents=True, exist_ok=True)

                for idx, region in enumerate(jersey_regions):
                    if region is None or region.size == 0:
                        continue

                    debug_path = sid_dir / f"DEBUG_f{frame_id}_r{idx}.jpg"
                    cv2.imwrite(str(debug_path), region)

                    if is_good_region(region):
                        registry.add_jersey_crop(stable_id, region)
                        good_path = sid_dir / f"GOOD_f{frame_id}_r{idx}.jpg"
                        cv2.imwrite(str(good_path), region)

            # ------------------------------
            # Visualization
            # ------------------------------
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

        # ------------------------------
        # FPS display
        # ------------------------------
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

        cv2.imshow("Cricket AI – Jersey Crop Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        del frame  # 🔥 explicit release

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------
    reader.release()
    cv2.destroyAllWindows()

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------
    print("\nJERSEY CROP SUMMARY")
    for sid, player in registry.players.items():
        print(f"SID {sid}: {len(player['jersey_crops'])} jersey regions collected")


if __name__ == "__main__":
    main()
