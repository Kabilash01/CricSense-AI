import time


class EventTimeline:
    def __init__(self):
        self.events = []

    def log(self, event: dict):
        """
        Log a structured event dictionary.
        """
        event["ts"] = time.time()
        self.events.append(event)

    def export(self, path):
        import json
        with open(path, "w") as f:
            json.dump(self.events, f, indent=2)
