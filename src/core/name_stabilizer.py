from collections import defaultdict, Counter

class NameStabilizer:
    def __init__(
        self,
        fast_window=5,
        strict_window=15,
        strict_ratio=0.6
    ):
        self.fast_window = fast_window
        self.strict_window = strict_window
        self.strict_ratio = strict_ratio

        self.buffers = {
            "striker": [],
            "non_striker": [],
            "bowler": []
        }

        self.locked = {
            "striker": None,
            "non_striker": None,
            "bowler": None
        }

    def update(self, role, name):
        if not name or name in ["-", ""]:
            return

        buf = self.buffers[role]
        buf.append(name)

        if len(buf) > self.strict_window:
            buf.pop(0)

        # Try strict lock first
        if self.locked[role] is None:
            counts = Counter(buf)
            top, freq = counts.most_common(1)[0]

            if freq / len(buf) >= self.strict_ratio:
                self.locked[role] = top

    def get(self, role):
        # STRICT wins always
        if self.locked[role]:
            return self.locked[role]

        buf = self.buffers[role]
        if len(buf) < self.fast_window:
            return None

        counts = Counter(buf[-self.fast_window:])
        return counts.most_common(1)[0][0]
