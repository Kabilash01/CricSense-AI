from collections import deque


class BallReleaseDetector:
    """
    Detect BALL_RELEASE via bowler vertical speed drop.
    No ML, no ball tracking, robust across camera angles.
    """

    def __init__(self, history=8, cooldown_frames=40):
        self.history = history
        self.cooldown_frames = cooldown_frames

        # SID -> deque[(frame_id, center_y)]
        self.center_history = {}

        # SID -> last release frame
        self.last_release = {}

    def update(self, sid, frame_id, center_y):
        if sid not in self.center_history:
            self.center_history[sid] = deque(maxlen=self.history)

        self.center_history[sid].append((frame_id, center_y))

        if len(self.center_history[sid]) < self.history:
            return False

        last = self.last_release.get(sid, -9999)
        if frame_id - last < self.cooldown_frames:
            return False

        return self._check_release(sid, frame_id)

    def _check_release(self, sid, frame_id):
        points = self.center_history[sid]

        ys = [p[1] for p in points]
        velocities = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]

        # Key cricket pattern:
        # Fast downward motion followed by sudden slowdown
        min_v = min(velocities)
        max_v = max(velocities)

        if min_v > 20 and max_v < 5:
            self.last_release[sid] = frame_id
            return True

        return False
