from collections import Counter


class JerseyVoteTracker:
    """
    Cricket-aware jersey number temporal validator + locker.
    """

    def __init__(self, min_votes=3, min_confidence=0.6):
        self.min_votes = min_votes
        self.min_confidence = min_confidence
        self.votes = {}
        self.locked = {}

    def is_locked(self, sid):
        return sid in self.locked

    def get_locked(self, sid):
        return self.locked.get(sid)

    def add_vote(self, sid, jersey):
        if self.is_locked(sid):
            return

        jersey = self._validate(jersey, sid)
        if jersey is None:
            return

        if sid not in self.votes:
            self.votes[sid] = Counter()

        self.votes[sid][jersey] += 1
        self._try_lock(sid)

    # ---------------- VALIDATION ----------------

    def _validate(self, jersey, sid):
        if jersey is None or not jersey.isdigit():
            return None

        val = int(jersey)

        # Reject impossible
        if val == 0:
            return None

        # Strong signal
        if 10 <= val <= 99:
            return jersey

        # Single digit → very strict
        if 1 <= val <= 9:
            votes = self.votes.get(sid, Counter())
            count = votes.get(jersey, 0)
            total = sum(votes.values()) + 1
            confidence = count / total if total else 0

            if count >= 5 and confidence >= 0.7:
                return jersey

        return None

    # ---------------- LOCKING ----------------

    def _try_lock(self, sid):
        counter = self.votes.get(sid)
        if not counter:
            return

        total = sum(counter.values())
        jersey, count = counter.most_common(1)[0]
        confidence = count / total

        if count >= self.min_votes and confidence >= self.min_confidence:
            self.locked[sid] = {
                "jersey": jersey,
                "votes": dict(counter),
                "confidence": round(confidence, 2)
            }
