import json
from collections import Counter
import time


class EventTimeline:
    def __init__(self):
        self.events = []
        self.counter = Counter()

    def log(
        self,
        *,
        frame_id,
        sid,
        role,
        jersey,
        event,
        meta
    ):
        record = {
            "ts": time.time(),
            "frame": frame_id,
            "sid": sid,
            "role": role,
            "jersey": jersey,
            "event": event,
            "meta": meta or {}
        }

        self.events.append(record)
        self.counter[event] += 1

    def export_json(self, path="event_timeline.json"):
        with open(path, "w") as f:
            json.dump(self.events, f, indent=2)

    def summary(self):
        return self.counter
