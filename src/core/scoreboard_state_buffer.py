from collections import deque, Counter

class ScoreboardStateBuffer:
    def __init__(self, size=15):
        self.buffer = deque(maxlen=size)

    def add(self, state):
        self.buffer.append(state)

    def _vote(self, key):
        values = [s.get(key) for s in self.buffer if s.get(key)]
        if not values:
            return None
        return Counter(values).most_common(1)[0][0]

    def get_stable_state(self):
        return {
            "team_score": self._vote("team_score"),
            "overs": self._vote("overs"),
            "striker": self._vote("striker"),
            "non_striker": self._vote("non_striker"),
            "bowler": self._vote("bowler"),
        }
