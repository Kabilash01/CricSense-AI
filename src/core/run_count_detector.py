class RunCountDetector:
    """
    Detects RUN_COUNT using crease crossing geometry.
    Pure spatial logic, no ML, no FPS impact.
    """

    def __init__(
        self,
        top_crease_ratio=0.22,
        bottom_crease_ratio=0.78
    ):
        self.top_crease_ratio = top_crease_ratio
        self.bottom_crease_ratio = bottom_crease_ratio

        # sid -> state info
        self.state = {}
        # sid -> frame_id where run already counted
        self.run_locked = {}

    def _which_side(self, y, frame_h):
        """
        Returns 'top' or 'bottom' based on batsman position.
        """
        if y < frame_h * 0.5:
            return "top"
        return "bottom"

    def update(
        self,
        sid,
        frame_id,
        center,
        frame_h,
        run_started_frame
    ):
        """
        Returns True if RUN_COUNT detected.
        """

        # RUN_START must exist
        if frame_id < run_started_frame:
            return False

        # Init state
        if sid not in self.state:
            side = self._which_side(center[1], frame_h)
            self.state[sid] = {
                "start_side": side,
                "counted": False
            }

        # Already counted this run
        if self.state[sid]["counted"]:
            return False

        # Lock to avoid duplicates
        last_count = self.run_locked.get(sid, -999)
        if frame_id - last_count < 30:
            return False

        y = center[1]

        # Define creases
        top_crease_y = int(frame_h * self.top_crease_ratio)
        bottom_crease_y = int(frame_h * self.bottom_crease_ratio)

        start_side = self.state[sid]["start_side"]

        crossed = False

        if start_side == "bottom" and y < top_crease_y:
            crossed = True

        if start_side == "top" and y > bottom_crease_y:
            crossed = True

        if crossed:
            self.state[sid]["counted"] = True
            self.run_locked[sid] = frame_id
            return True

        return False

    def reset_for_new_delivery(self):
        """
        Call after BALL_END (future).
        """
        self.state.clear()
