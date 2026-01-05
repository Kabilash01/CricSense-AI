import cv2
import numpy as np
import re
import easyocr


class ScoreDeltaDetector:
    def __init__(self):
        # Initialize EasyOCR ONCE
        self.reader = easyocr.Reader(
            ['en'],
            gpu=False,          # 🔒 keep FPS stable
            verbose=False
        )

        self.prev_crop = None
        self.prev_score = None  # (runs, wickets)

    # --------------------------------------------------
    # Preprocess for OCR
    # --------------------------------------------------
    def _preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        return th

    # --------------------------------------------------
    # OCR score: "12-3"
    # --------------------------------------------------
    def _read_score(self, crop):
        proc = self._preprocess(crop)

        results = self.reader.readtext(
            proc,
            detail=0,            # text only
            allowlist='0123456789-'
        )

        if not results:
            return None

        text = " ".join(results)
        match = re.search(r"(\d+)\s*-\s*(\d+)", text)

        if not match:
            return None

        return int(match.group(1)), int(match.group(2))

    # --------------------------------------------------
    # Detect score delta
    # --------------------------------------------------
    def detect_delta(self, crop):
        if self.prev_crop is None:
            self.prev_crop = crop
            self.prev_score = self._read_score(crop)
            return None

        # Fast pixel-change check (skip OCR if unchanged)
        diff = cv2.absdiff(crop, self.prev_crop)
        change_score = np.mean(diff)

        if change_score < 2.0:
            return None

        curr_score = self._read_score(crop)

        if not curr_score or not self.prev_score:
            return None

        runs_delta = curr_score[0] - self.prev_score[0]
        wicket_delta = curr_score[1] - self.prev_score[1]

        self.prev_crop = crop
        self.prev_score = curr_score

        return {
            "runs_delta": runs_delta,
            "wicket_delta": wicket_delta,
            "current_score": curr_score
        }
