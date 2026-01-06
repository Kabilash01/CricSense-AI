class IdentityBinder:
    def __init__(self):
        self.mapping = {
            "striker": None,
            "non_striker": None,
            "bowler": None
        }

    def bind(self, scoreboard_state, players):
        """
        scoreboard_state:
            {
              striker: "R Sharma",
              non_striker: "Gill",
              bowler: "Starc"
            }

        players:
            list of dicts:
            {
              sid, role, center, speed
            }
        """

        batsmen = [p for p in players if p["role"] == "Batsman"]
        bowlers = [p for p in players if p["role"] == "Bowler"]

        # --- Bowler binding ---
        if bowlers:
            self.mapping["bowler"] = {
                "name": scoreboard_state.get("bowler"),
                "sid": bowlers[0]["sid"]
            }

        # --- Batsman binding ---
        if len(batsmen) >= 2:
            batsmen_sorted = sorted(
                batsmen,
                key=lambda b: (-b["speed"], b["center"][1])
            )

            self.mapping["striker"] = {
                "name": scoreboard_state.get("striker"),
                "sid": batsmen_sorted[0]["sid"]
            }

            self.mapping["non_striker"] = {
                "name": scoreboard_state.get("non_striker"),
                "sid": batsmen_sorted[1]["sid"]
            }

        return self.mapping
