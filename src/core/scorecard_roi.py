import cv2

class ScorecardROILocker:
    """
    Locks the scoreboard ROI once and reuses it forever.
    """

    def __init__(self):
        # These coordinates are based on your provided image
        # You can fine-tune once, then never touch again
        self.roi = {
            "x1": 40,
            "y1": 610,
            "x2": 360,
            "y2": 700
        }

    def crop(self, frame):
        x1, y1, x2, y2 = self.roi.values()
        return frame[y1:y2, x1:x2].copy()
