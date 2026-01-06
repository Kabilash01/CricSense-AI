import math

class PlayerRegistry:
    def __init__(self, max_idle_frames=60, max_trail_len=40):
        self.players = {}
        self.raw_to_stable = {}
        self.next_stable_id = 1

        self.max_idle_frames = max_idle_frames
        self.max_trail_len = max_trail_len

        self.seen_ids = set()
        self.role_confirmed = set()
        self.jersey_logged = set()

    # -------------------------------
    # Utility
    # -------------------------------
    def _distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    # -------------------------------
    # Stable ID resolution
    # -------------------------------
    def resolve_id(self, raw_id, center, role, frame_id):
        # Existing mapping
        if raw_id in self.raw_to_stable:
            sid = self.raw_to_stable[raw_id]
            self.players[sid]["last_center"] = center
            self.players[sid]["last_seen"] = frame_id
            return sid

        # Spatial match
        for sid, p in self.players.items():
            if p["role"] != role:
                continue
            if self._distance(center, p["last_center"]) < 60:
                if frame_id - p["last_seen"] <= self.max_idle_frames:
                    self.raw_to_stable[raw_id] = sid
                    p["last_center"] = center
                    p["last_seen"] = frame_id
                    return sid

        # New stable ID
        sid = self.next_stable_id
        self.next_stable_id += 1

        self.raw_to_stable[raw_id] = sid
        self.players[sid] = {
            "role": role,
            "last_center": center,
            "last_seen": frame_id,
            "trajectory": [],
            "jersey_crops": []
        }
        return sid

    # -------------------------------
    # Update motion
    # -------------------------------
    def update(self, stable_id, center):
        traj = self.players[stable_id]["trajectory"]
        traj.append(center)
        if len(traj) > self.max_trail_len:
            traj.pop(0)

        self.players[stable_id]["last_center"] = center

    # -------------------------------
    # Jersey crops
    # -------------------------------
    def add_jersey_crop(self, stable_id, crop):
        self.players[stable_id]["jersey_crops"].append(crop)

    def get_jersey_crops(self, stable_id):
        return self.players.get(stable_id, {}).get("jersey_crops", [])

    # -------------------------------
    # Active player snapshot (FIXED)
    # -------------------------------
    def get_active_players(self):
        """
        Returns lightweight player info for
        identity binding and analytics.
        """
        snapshot = []

        for sid, p in self.players.items():
            traj = p["trajectory"]

            # Compute speed (pixel/frame)
            speed = 0.0
            if len(traj) >= 2:
                speed = self._distance(traj[-1], traj[-2])

            snapshot.append({
                "sid": sid,
                "role": p["role"],
                "center": p["last_center"],
                "speed": speed
            })

        return snapshot
