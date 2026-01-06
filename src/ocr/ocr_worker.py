import threading
import queue
import time
import easyocr

class OCRWorker:
    def __init__(self, use_gpu=False):
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)
        self.queue = queue.Queue(maxsize=1)
        self.latest_result = None
        self.running = True

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, image):
        if not self.queue.full():
            self.queue.put(image)

    def _run(self):
        while self.running:
            try:
                img = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                result = self.reader.readtext(img, detail=0)
                self.latest_result = result
            except Exception:
                pass

    def get_latest(self):
        return self.latest_result

    def stop(self):
        self.running = False
