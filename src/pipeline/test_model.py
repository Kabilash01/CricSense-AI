import sys
import cv2
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO
from jersey.smolvlm_reader import SmolVLMJerseyReader
from jersey.jersey_vote_tracker import JerseyVoteTracker


# ---------------- CONFIG ----------------
VIDEO_PATH = r"C:\cricket-ai\data\samples\test3.mp4"
MODEL_PATH = r"C:\cricket-ai\yolov8m_production4\weights\best.pt"
DEVICE = 0
FRAME_SKIP = 30  # 🔥 increased for FPS


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Cannot open video")
        return

    model = YOLO(MODEL_PATH)

    jersey_reader = SmolVLMJerseyReader()
    jersey_tracker = JerseyVoteTracker()

    frame_count = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        jersey_processed_this_frame = False

        results = model.track(
            frame,
            persist=True,
            conf=0.4,
            iou=0.5,
            device=DEVICE,
            verbose=False
        )[0]

        if results.boxes is not None:
            for box in results.boxes:
                if box.id is None:
                    continue

                sid = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw player box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"SID {sid}", (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Already locked
                if jersey_tracker.is_locked(sid):
                    jersey = jersey_tracker.get_locked(sid)["jersey"]
                    cv2.putText(frame, f"#{jersey}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    continue

                # 🔥 performance guards
                if jersey_processed_this_frame:
                    continue
                if frame_count % FRAME_SKIP != 0:
                    continue

                # Jersey crop
                h = y2 - y1
                w = x2 - x1

                cy1 = int(y1 + 0.25 * h)
                cy2 = int(y1 + 0.65 * h)
                cx1 = int(x1 + 0.2 * w)
                cx2 = int(x2 - 0.2 * w)

                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue

                number, _ = jersey_reader.read_jersey_number(crop)
                jersey_tracker.add_vote(sid, number)
                jersey_processed_this_frame = True

                if jersey_tracker.is_locked(sid):
                    locked = jersey_tracker.get_locked(sid)
                    print(f"🔒 LOCKED → SID {sid}: #{locked['jersey']} "
                          f"(confidence {locked['confidence']})")

        # FPS
        fps = frame_count / (time.time() - t0 + 1e-6)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Cricket AI", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
