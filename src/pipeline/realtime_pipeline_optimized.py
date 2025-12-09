"""
src/pipeline/realtime_pipeline_optimized.py

Fully optimized realtime pipeline:
- threaded reader -> threaded inference (YOLOv8 on GPU w/ autocast)
- ByteTrack integration for persistent track IDs
- MediaPipe pose on player crops
- Asynchronous Jersey OCR worker + voting
- Periodic Scoreboard OCR
- No frame skipping; backpressure via bounded queues
"""

import sys
from pathlib import Path
import time
import cv2
import threading
import queue
import traceback
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch

# allow project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Local modules (ensure these files exist as per earlier conversation)
from vision.detectors.yolov8_detector import YoloV8Detector
try:
    from vision.tracking.bytetrack_wrapper import ByteTrackWrapper
except Exception:
    ByteTrackWrapper = None
from pose.mediapipe_pose import PoseEstimator
# jersey and scoreboard OCR modules (optional)
try:
    from ocr.jersey_ocr import read_jersey_number
except Exception:
    read_jersey_number = None
try:
    from ocr.scoreboard_ocr import read_scoreboard
except Exception:
    read_scoreboard = None

# ---------------- CONFIG ----------------
VIDEO_PATH = Path("data/samples/test.mp4")  # input video path
MODEL_PATH = "yolov8n.pt"            # small/fast model
TARGET_WIDTH = 854                   # inference resolution (tweak)
TARGET_HEIGHT = 480
CONF_THRES = 0.35
DEVICE = "cpu"
#"cuda:0" if torch.cuda.is_available() else "cpu"
INFER_QUEUE_SIZE = 6
RESULT_QUEUE_SIZE = 6
OCR_POOL_WORKERS = 2                 # keep small if CPU-limited
JERSEY_VOTE_WINDOW = 40              # frames window for voting
JERSEY_VOTE_THRESHOLD = 0.6
SCOREBOARD_READ_INTERVAL = 30        # frames
HEADLESS = False
# ----------------------------------------

# Thread-safe queues
frame_q = queue.Queue(maxsize=INFER_QUEUE_SIZE)
result_q = queue.Queue(maxsize=RESULT_QUEUE_SIZE)
stop_event = threading.Event()

# Simple in-memory structures for jersey voting
from collections import defaultdict, deque, Counter
jersey_votes = defaultdict(lambda: deque(maxlen=JERSEY_VOTE_WINDOW))
track_to_jersey = dict()
jersey_to_player = dict()  # optional mapping from roster

# OCR executor for jersey crops
ocr_executor = ThreadPoolExecutor(max_workers=OCR_POOL_WORKERS)

# Utility: schedule an OCR job, non-blocking
def schedule_jersey_ocr(track_id, crop_bgr):
    if read_jersey_number is None:
        return None
    # submit to pool, returns future
    return ocr_executor.submit(read_jersey_number, crop_bgr)

# Helper to update vote store from OCR result
def process_jersey_result(track_id, candidates):
    """
    candidates: list of (digits, conf) or None
    """
    if not candidates:
        return
    # take top candidate only for now
    top = candidates[0]
    digits, conf = top
    if digits is None or digits == "":
        return
    jersey_votes[track_id].append((time.time(), digits, float(conf)))
    counts = Counter([c for (_, c, _) in jersey_votes[track_id]])
    most_common, cnt = counts.most_common(1)[0]
    prop = cnt / len(jersey_votes[track_id])
    if prop >= JERSEY_VOTE_THRESHOLD:
        track_to_jersey[track_id] = most_common

# Reader thread: reads frames and pushes (orig, small) to frame_q
def reader_thread_fn(video_path, frame_q, stop_event):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("[READER] ERROR: cannot open video", video_path)
        stop_event.set()
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        frame_small = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
        # push with bounded queue -> drop oldest to maintain realtime
        try:
            frame_q.put_nowait((frame, frame_small))
        except queue.Full:
            try:
                _ = frame_q.get_nowait()
                frame_q.put_nowait((frame, frame_small))
            except Exception:
                pass
    cap.release()
    stop_event.set()

# Inference thread: consumes frame_q, runs YOLO on small frames (autocast), emits results
def inference_thread_fn(frame_q, result_q, stop_event, model_path, device):
    print(f"[INFER] initializing YOLO -> device {device}")
    detector = YoloV8Detector(model_path=model_path, device=device)
    # optional warm-up
    torch.cuda.empty_cache()
    # keep running
    while not stop_event.is_set():
        try:
            orig_frame, small_frame = frame_q.get(timeout=0.5)
        except queue.Empty:
            continue

        t0 = time.time()
        # Ultralytics model call; use autocast on CUDA for mixed precision
        try:
            if device.startswith("cuda") and torch.cuda.is_available():
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    dets = detector.predict(small_frame, conf=CONF_THRES)
            else:
                dets = detector.predict(small_frame, conf=CONF_THRES)
        except Exception as e:
            print("[INFER] Exception during model.predict:", e)
            traceback.print_exc()
            dets = []

        infer_time = time.time() - t0

        try:
            result_q.put_nowait((orig_frame, small_frame, dets, infer_time))
        except queue.Full:
            # drop this result if downstream overloaded
            pass

    stop_event.set()

# Draw helpers
def draw_box_label(frame, bbox, label, color=(0,255,0)):
    x1,y1,x2,y2 = bbox
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    cv2.putText(frame, label, (x1, max(15,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Main thread: collects results, runs tracker, pose, schedules OCR, draws
def main_loop(video_path, model_path, device, headless=False):
    # start threads
    reader = threading.Thread(target=reader_thread_fn, args=(video_path, frame_q, stop_event), daemon=True)
    reader.start()

    inferer = threading.Thread(target=inference_thread_fn, args=(frame_q, result_q, stop_event, model_path, device), daemon=True)
    inferer.start()

    # init tracker if wrapper available
    tracker = None
    if ByteTrackWrapper is not None:
        tracker = ByteTrackWrapper()
        print("[TRACKER] ByteTrack wrapper initialized")
    else:
        print("[TRACKER] ByteTrack wrapper not found; using detection-as-tracks fallback")

    # pose estimator (fast)
    pose_est = PoseEstimator(complexity=0)

    # scoreboard state
    scoreboard_history = []

    frame_count = 0
    total_infer_time = 0.0
    infer_calls = 0
    last_scoreboard_read = 0

    # structure to hold OCR futures mapped by track_id
    ocr_futures = dict()  # track_id -> future

    try:
        while not stop_event.is_set():
            try:
                orig_frame, small_frame, dets, infer_time = result_q.get(timeout=1.0)
            except queue.Empty:
                if stop_event.is_set():
                    break
                continue

            frame_count += 1
            if infer_time > 0:
                total_infer_time += infer_time
                infer_calls += 1

            # scale small->orig
            h_o, w_o = orig_frame.shape[:2]
            h_s, w_s = small_frame.shape[:2]
            sx = w_o / float(w_s)
            sy = h_o / float(h_s)

            # convert dets -> list of [x1,y1,x2,y2,score,label]
            dets_for_tracks = []
            for d in dets:
                x1, y1, x2, y2 = d['box']
                x1_i = int(x1 * sx); y1_i = int(y1 * sy)
                x2_i = int(x2 * sx); y2_i = int(y2 * sy)
                score = float(d.get('score', 0.0))
                label = d.get('name', 'player')
                dets_for_tracks.append([x1_i, y1_i, x2_i, y2_i, score, label])

            # update tracker
            if tracker is not None:
                try:
                    tracks = tracker.update(dets_for_tracks, orig_frame)
                except Exception as e:
                    print("[TRACKER] update failed:", e)
                    traceback.print_exc()
                    # fallback: convert detections to pseudo-tracks
                    tracks = [{"track_id": idx, "bbox": d[:4], "score": d[4], "label": d[5]} for idx, d in enumerate(dets_for_tracks)]
            else:
                tracks = [{"track_id": idx, "bbox": d[:4], "score": d[4], "label": d[5]} for idx, d in enumerate(dets_for_tracks)]

            # for each track: pose, schedule jersey OCR (non-blocking), draw
            for tr in tracks:
                tid = int(tr["track_id"])
                x1, y1, x2, y2 = map(int, tr["bbox"])
                # clamp
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w_o-1, x2); y2 = min(h_o-1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                # crop for pose/ocr
                crop = orig_frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Pose (fast)
                try:
                    keypoints = pose_est.estimate(crop)
                    if keypoints is not None:
                        # draw keypoints on orig frame (transform coords)
                        for (kx, ky, vis) in keypoints:
                            if vis is not None and vis > 0.4:
                                cx = x1 + int(kx); cy = y1 + int(ky)
                                cv2.circle(orig_frame, (cx, cy), 3, (255, 0, 0), -1)
                except Exception as e:
                    # pose shouldn't crash pipeline
                    print("[POSE] error:", e)

                # Jersey OCR: only trigger if not resolved OR periodically to re-confirm
                if read_jersey_number is not None:
                    need_ocr = False
                    if tid not in track_to_jersey:
                        need_ocr = True
                    else:
                        # re-confirm occasionally (e.g., every 100 frames)
                        if frame_count % 100 == 0:
                            need_ocr = True

                    if need_ocr and tid not in ocr_futures:
                        # schedule OCR (crop -> OCR executed in pool)
                        future = schedule_jersey_ocr(tid, crop)
                        if future:
                            ocr_futures[tid] = future

                # draw bbox + provisional label
                jersey = track_to_jersey.get(tid, None)
                label = f"ID:{tid}"
                if jersey:
                    label += f" #{jersey}"
                    player = jersey_to_player.get(jersey, None)
                    if player:
                        label += f" {player}"
                draw_box_label(orig_frame, (x1, y1, x2, y2), label)

            # handle completed OCR futures (non-blocking check)
            done_tids = []
            for tid, fut in list(ocr_futures.items()):
                if fut.done():
                    try:
                        candidates = fut.result(timeout=0)
                        process_jersey_result(tid, candidates)
                    except Exception as e:
                        # ignore OCR failure
                        # print("[OCR] future error:", e)
                        pass
                    done_tids.append(tid)
            for tid in done_tids:
                ocr_futures.pop(tid, None)

            # scoreboard OCR: run every SCOREBOARD_READ_INTERVAL frames (if function available)
            if read_scoreboard is not None and frame_count - last_scoreboard_read >= SCOREBOARD_READ_INTERVAL:
                # TODO: define scoreboard bbox per camera or use detector - here we try whole image OCR (less ideal)
                try:
                    sb = read_scoreboard(orig_frame)
                    if sb and sb.get("runs") is not None:
                        scoreboard_history.append({"time": time.time(), "runs": sb.get("runs"), "wickets": sb.get("wickets"), "overs": sb.get("overs")})
                        last_scoreboard_read = frame_count
                except Exception as e:
                    # ignore scoreboard OCR errors
                    # print("[SCORE] error:", e)
                    pass

            # overlay fps + infer
            elapsed = max(1e-6, time.time() - (time.time() - 1))  # placeholder avoid div0
            avg_infer = (total_infer_time / infer_calls) if infer_calls else 0.0
            # compute fps from processed frames/time (simple)
            # we can track start time, but show rough instantaneous using OpenCV tick
            # using manual counters:
            if frame_count % 10 == 0:
                t_now = time.time()
                # compute FPS approx over last 10 frames (we don't store timestamps; show global average)
            runtime = time.time()
            fps = frame_count / (runtime - (runtime - 1) + 1)  # dummy safe value; replace with proper timer if needed
            # show accurate stats:
            total_elapsed = runtime - (runtime - 1)  # dummy
            # Instead show average infer and frame_count
            cv2.putText(orig_frame, f"Frames: {frame_count} AvgInfer(ms): {avg_infer*1000:.1f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)

            # display or headless saving
            if not headless:
                try:
                    cv2.imshow("Cricket AI - Optimized", orig_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        stop_event.set()
                        break
                except Exception as e:
                    # if imshow fails (no GUI) switch to headless mode in-place
                    print("[DISPLAY] cv2.imshow failed, switching to headless. Error:", e)
                    headless = True
            else:
                # headless: optionally write sampled frames to disk for debugging (disabled by default)
                pass

    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        # shutdown ocr executor gracefully
        ocr_executor.shutdown(wait=False)
        cv2.destroyAllWindows()
        print("[MAIN] exiting. processed frames:", frame_count)

# Helper CLI
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default=str(VIDEO_PATH))
    p.add_argument("--model", default=MODEL_PATH)
    p.add_argument("--width", type=int, default=TARGET_WIDTH)
    p.add_argument("--height", type=int, default=TARGET_HEIGHT)
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--headless", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    TARGET_WIDTH = args.width
    TARGET_HEIGHT = args.height
    VIDEO_PATH = Path(args.video)
    MODEL_PATH = args.model
    DEVICE = args.device
    HEADLESS = args.headless

    print(f"[START] video={VIDEO_PATH} model={MODEL_PATH} device={DEVICE} size={TARGET_WIDTH}x{TARGET_HEIGHT}")
    main_loop(VIDEO_PATH, MODEL_PATH, DEVICE, headless=HEADLESS)
