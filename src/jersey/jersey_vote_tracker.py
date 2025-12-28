from collections import Counter


class JerseyVoteTracker:
    """
    Cricket-aware jersey number temporal validator + locker.
    """

    def __init__(self, min_votes=3, min_confidence=0.6):
        self.min_votes = min_votes
        self.min_confidence = min_confidence

        self.votes = {}      # sid -> Counter
        self.locked = {}     # sid -> locked info

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------

    def is_locked(self, sid):
        return sid in self.locked

    def get_locked(self, sid):
        return self.locked.get(sid)

    def add_vote(self, sid, jersey_number):
        """
        Adds a jersey prediction AFTER validation.
        """

        if self.is_locked(sid):
            return

        jersey_number = self._validate(jersey_number, sid)

        if jersey_number is None:
            return

        if sid not in self.votes:
            self.votes[sid] = Counter()

        self.votes[sid][jersey_number] += 1
        self._try_lock(sid)

    # --------------------------------------------------
    # VALIDATION LOGIC (CRITICAL)
    # --------------------------------------------------

    def _validate(self, jersey, sid):
        """
        Returns validated jersey number or None.
        """

        if jersey is None:
            return None

        if not jersey.isdigit():
            return None

        jersey_int = int(jersey)

        # ❌ Rule 1: Reject impossible numbers
        if jersey_int == 0:
            return None

        # ✅ Rule 2: Accept strong two-digit jerseys
        if 10 <= jersey_int <= 99:
            return jersey

        # ⚠️ Rule 3: Single-digit jerseys (rare)
        if 1 <= jersey_int <= 9:
            votes = self.votes.get(sid, Counter())
            count = votes.get(jersey, 0)
            total = sum(votes.values()) + 1

            confidence = count / total if total > 0 else 0

            # Allow only after strong evidence
            if count >= 5 and confidence >= 0.7:
                return jersey
            else:
                return None

        return None

    # --------------------------------------------------
    # LOCKING LOGIC
    # --------------------------------------------------

    def _try_lock(self, sid):
        counter = self.votes.get(sid)
        if not counter:
            return

        total_votes = sum(counter.values())
        jersey, count = counter.most_common(1)[0]

        confidence = count / total_votes

        if count >= self.min_votes and confidence >= self.min_confidence:
            self.locked[sid] = {
                "jersey": jersey,
                "votes": dict(counter),
                "confidence": round(confidence, 2)
            }
