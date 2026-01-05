class ScoreEngine:
    """
    Scoreboard-authoritative cricket score engine.
    """

    def __init__(self):
        # 🔒 Always initialize state
        self.total_runs = 0
        self.wickets = 0
        self.balls = 0

    # --------------------------------------------------
    # Ball lifecycle
    # --------------------------------------------------
    def on_ball_start(self):
        pass

    def apply_scoreboard_delta(self, runs: int, wickets: int):
        """
        Apply OCR-derived scoreboard delta.
        """
        if runs is not None:
            self.total_runs += max(0, runs)

        if wickets is not None:
            self.wickets += max(0, wickets)

    def on_ball_end(self):
        """
        Close the ball and return match summary.
        ALWAYS returns a dict.
        """
        self.balls += 1
        overs = f"{self.balls // 6}.{self.balls % 6}"

        return {
            "total_runs": self.total_runs,
            "wickets": self.wickets,
            "overs": overs
        }
