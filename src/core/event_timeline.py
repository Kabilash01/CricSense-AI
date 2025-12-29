import time
import json
from collections import Counter


class EventTimeline:
    """
    Lightweight event logger for real-time pipelines.
    - O(1) append
    - No CV / ML
    - JSON export
    """

    def __init__(self):
        self.events = []
        self._seen = set()  # used for one-time events per SID (optional)

    def log(self, *, frame_id, sid, event, role=None, jersey=None, meta=None):
        """
        Append a single event.
        meta: optional dict for extra info (kept small)
        """
        self.events.append({
            "ts": round(time.time(), 3),
            "frame": int(frame_id),
            "sid": int(sid),
            "role": role,
            "jersey": jersey,
            "event": event,
            "meta": meta or {}
        })

    def log_once(self, *, key, frame_id, sid, event, role=None, jersey=None, meta=None):
        """
        Log an event only once per unique key (e.g., per SID).
        """
        if key in self._seen:
            return
        self._seen.add(key)
        self.log(
            frame_id=frame_id,
            sid=sid,
            event=event,
            role=role,
            jersey=jersey,
            meta=meta
        )

    def export_json(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.events, f, indent=2)

    def summary(self):
        return Counter(e["event"] for e in self.events)
