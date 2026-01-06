import threading
import time

class AsyncScoreboardOCR:
    def __init__(self, reader, interval_frames=60):
        self.reader = reader
        self.interval_frames = interval_frames
        self.last_result = None
        self.last_frame = -999
        self.lock = threading.Lock()
        self.running = False

    def try_submit(self, frame_id, scoreboard_roi):
        if self.running:
            return
        if frame_id - self.last_frame < self.interval_frames:
            return

        self.running = True
        self.last_frame = frame_id

        thread = threading.Thread(
            target=self._run_ocr,
            args=(scoreboard_roi.copy(),),
            daemon=True
        )
        thread.start()

    def _run_ocr(self, roi):
        try:
            result = self.reader.parse(roi)
            with self.lock:
                self.last_result = result
        finally:
            self.running = False

    def get_latest(self):
        with self.lock:
            return self.last_result
