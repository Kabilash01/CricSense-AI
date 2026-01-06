from collections import deque
import math


class RunDetector:
    """
    Detects RUN_START based on sustained batsman translation
    after a SHOT_ATTEMPT.
    """

    def __init__(
        self,
        history=10,          # frames to observe
        min_distance=28,     # pixels
        min_frames_after_shot=3,
        cooldown=40          # frames between run starts
    ):
        self.history = history
        self.min_distance = min_distance
        self.min_frames_after_shot = min_frames_after_shot
        self.cooldown = cooldown

        self.motion = {}          # sid -> deque[(x,y)]
        self.last_run_frame = {}  # sid -> frame_id

    def update(self, sid, frame_id, center, last_shot_frame):
        """
        Returns True if RUN_START detected for this sid.
        """

        # Must be after a shot
        if frame_id - last_shot_frame < self.min_frames_after_shot:
            return False

        # Init motion history
        if sid not in self.motion:
            self.motion[sid] = deque(maxlen=self.history)

        self.motion[sid].append(center)

        if len(self.motion[sid]) < self.history:
            return False

        # Cooldown check
        last_run = self.last_run_frame.get(sid, -999)
        if frame_id - last_run < self.cooldown:
            return False

        # Net displacement
        x0, y0 = self.motion[sid][0]
        x1, y1 = self.motion[sid][-1]
        distance = math.hypot(x1 - x0, y1 - y0)

        if distance >= self.min_distance:
            self.last_run_frame[sid] = frame_id
            return True

        return False

    def reset(self):
        """
        Reset motion history at BALL_END.
        """
        self.motion.clear()
