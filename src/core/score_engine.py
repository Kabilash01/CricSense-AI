class ScoreEngine:
    """
    Deterministic cricket score engine.
    No ML. Event driven.
    """

    def __init__(self):
        self.total_runs = 0
        self.total_balls = 0
        self.overs = 0
        self.ball_in_over = 0

        self.current_ball_runs = 0
        self.ball_active = False
        self.current_striker_sid = None

    # ------------------------
    # BALL LIFECYCLE
    # ------------------------
    def ball_start(self, striker_sid):
        self.ball_active = True
        self.current_ball_runs = 0
        self.current_striker_sid = striker_sid

    def add_run(self, runs=1):
        if not self.ball_active:
            return
        self.current_ball_runs += runs

    def ball_end(self):
        if not self.ball_active:
            return None

        self.total_runs += self.current_ball_runs
        self.total_balls += 1

        self.ball_in_over += 1
        if self.ball_in_over == 6:
            self.ball_in_over = 0
            self.overs += 1

        summary = {
            "runs_this_ball": self.current_ball_runs,
            "total_runs": self.total_runs,
            "overs": f"{self.overs}.{self.ball_in_over}",
            "striker_sid": self.current_striker_sid
        }

        self.ball_active = False
        self.current_ball_runs = 0
        self.current_striker_sid = None

        return summary
