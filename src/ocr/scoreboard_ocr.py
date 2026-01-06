import cv2
import easyocr
import numpy as np


class ScoreboardReader:
    """
    EasyOCR-based scoreboard reader for single-line broadcast overlays.
    """

    def __init__(self, use_gpu=False):
        self.reader = easyocr.Reader(
            ['en'],
            gpu=use_gpu,
            verbose=False
        )

    # -------------------------------------------------
    # Scoreboard ROI crop (single-line broadcast)
    # -------------------------------------------------
    def crop(self, frame):
        """
        Crop the scoreboard region from the frame.
        Assumes bottom-left scoreboard.
        """
        if frame is None:
            return None

        h, w = frame.shape[:2]

        x1 = int(0.01 * w)
        x2 = int(0.65 * w)

        y2 = int(0.98 * h)
        y1 = int(0.55 * h)   # tall enough for text

        roi = frame[y1:y2, x1:x2]
        return roi if roi.size > 0 else None

    # -------------------------------------------------
    # OCR helpers
    # -------------------------------------------------
    def _read(self, img, allowlist=None):
        if img is None or img.size == 0:
            return ""

        result = self.reader.readtext(
            img,
            detail=0,
            allowlist=allowlist,
            paragraph=False
        )
        return " ".join(result).strip()

    # -------------------------------------------------
    # Main read function
    # -------------------------------------------------
    def read(self, scoreboard_roi):
        """
        Read raw text fields from the scoreboard ROI.
        """
        h, w = scoreboard_roi.shape[:2]

        # Horizontal segmentation (single-line layout)
        team_roi        = scoreboard_roi[:, int(0.00*w):int(0.16*w)]
        over_roi        = scoreboard_roi[:, int(0.16*w):int(0.32*w)]
        striker_roi     = scoreboard_roi[:, int(0.32*w):int(0.56*w)]
        non_striker_roi = scoreboard_roi[:, int(0.56*w):int(0.76*w)]
        bowler_roi      = scoreboard_roi[:, int(0.76*w):int(1.00*w)]

        return {
            "team_score_raw": self._read(
                team_roi,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789- "
            ),
            "overs_raw": self._read(
                over_roi,
                allowlist="0123456789./"
            ),
            "striker_raw": self._read(
                striker_roi,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789() "
            ),
            "non_striker_raw": self._read(
                non_striker_roi,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789() "
            ),
            "bowler_raw": self._read(
                bowler_roi,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-./() "
            ),
        }
