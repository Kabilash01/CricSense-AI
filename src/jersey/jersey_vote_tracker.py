from collections import Counter

class JerseyVoteTracker:
    """
    Tracks jersey number predictions per Stable ID (SID),
    applies temporal voting, and locks jersey numbers.
    """

    def __init__(self, min_votes=3, min_confidence=0.6):
        self.min_votes = min_votes
        self.min_confidence = min_confidence

        # sid -> Counter({jersey_number: count})
        self.votes = {}

        # sid -> locked jersey number
        self.locked = {}

    def is_locked(self, sid):
        return sid in self.locked

    def get_locked(self, sid):
        return self.locked.get(sid)

    def add_vote(self, sid, jersey_number):
        """
        Add a jersey prediction for a SID.
        jersey_number can be str or None.
        """

        if self.is_locked(sid):
            return  # already locked, ignore further votes

        if jersey_number is None:
            return  # ignore empty predictions

        if sid not in self.votes:
            self.votes[sid] = Counter()

        self.votes[sid][jersey_number] += 1

        self._try_lock(sid)

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

    def summary(self):
        return self.locked
