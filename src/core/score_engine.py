class ScoreEngine:
    """
    Event-driven cricket score engine.
    Consumes RUN_COUNT and BALL_END events.
    """

    def __init__(self):
        self.total_runs = 0
        self.total_balls = 0
        self.current_over = 0
        self.ball_in_over = 0

        self.current_ball_runs = 0
        self.ball_active = False

    def on_ball_start(self):
        self.current_ball_runs = 0
        self.ball_active = True

    def on_run(self, runs=1):
        if self.ball_active:
            self.current_ball_runs += runs

    def on_ball_end(self):
        if not self.ball_active:
            return None

        self.total_runs += self.current_ball_runs
        self.total_balls += 1

        self.ball_in_over += 1
        if self.ball_in_over == 6:
            self.ball_in_over = 0
            self.current_over += 1

        summary = {
            "runs_this_ball": self.current_ball_runs,
            "total_runs": self.total_runs,
            "overs": f"{self.current_over}.{self.ball_in_over}"
        }

        self.ball_active = False
        self.current_ball_runs = 0

        return summary
