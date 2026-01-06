import easyocr

class ScoreboardSemanticReader:
    """
    Layout-locked semantic OCR for cricket broadcast scoreboard.
    Extracts striker, non-striker, bowler ONLY.
    """

    def __init__(self, gpu=False):
        self.reader = easyocr.Reader(['en'], gpu=gpu, verbose=False)

    def _ocr(self, img, allowlist=None):
        if img is None or img.size == 0:
            return None
        text = self.reader.readtext(
            img,
            detail=0,
            allowlist=allowlist
        )
        return " ".join(text).strip() if text else None

    def parse(self, scoreboard_roi):
        """
        scoreboard_roi: full horizontal scoreboard strip
        """
        h, w = scoreboard_roi.shape[:2]

        striker_roi     = scoreboard_roi[:, int(0.48*w):int(0.66*w)]
        non_striker_roi = scoreboard_roi[:, int(0.66*w):int(0.82*w)]
        bowler_roi      = scoreboard_roi[:, int(0.88*w):int(1.00*w)]

        return {
            "striker": self._ocr(
                striker_roi,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789() "
            ),
            "non_striker": self._ocr(
                non_striker_roi,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789() "
            ),
            "bowler": self._ocr(
                bowler_roi,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-(). "
            )
        }
