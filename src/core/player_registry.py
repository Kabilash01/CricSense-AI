import math
import cv2

class PlayerRegistry:
    def __init__(self, max_idle_frames=60, max_trail_len=40):
        self.players = {}
        self.raw_to_stable = {}
        self.next_stable_id = 1
        self.max_idle_frames = max_idle_frames
        self.max_trail_len = max_trail_len

    def _distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    # -------------------------------
    # STABLE ID RESOLUTION
    # -------------------------------
    def resolve_id(self, raw_id, center, role, frame_id):
        if raw_id in self.raw_to_stable:
            sid = self.raw_to_stable[raw_id]
            self.players[sid]["last_center"] = center
            self.players[sid]["last_seen"] = frame_id
            return sid

        for sid, player in self.players.items():
            if player["final_role"] != role:
                continue
            if self._distance(center, player["last_center"]) < 60:
                if frame_id - player["last_seen"] <= self.max_idle_frames:
                    self.raw_to_stable[raw_id] = sid
                    player["last_center"] = center
                    player["last_seen"] = frame_id
                    return sid

        sid = self.next_stable_id
        self.next_stable_id += 1

        self.raw_to_stable[raw_id] = sid
        self.players[sid] = {
            "final_role": role,
            "last_center": center,
            "last_seen": frame_id,
            "trajectory": [],
            "jersey_crops": []   # 🔥 NEW
        }
        return sid

    def update(self, stable_id, center):
        traj = self.players[stable_id]["trajectory"]
        traj.append(center)
        if len(traj) > self.max_trail_len:
            traj.pop(0)

    # -------------------------------
    # JERSEY CROP STORAGE
    # -------------------------------
    def add_jersey_crop(self, stable_id, crop):
        self.players[stable_id]["jersey_crops"].append(crop)

    def get_jersey_crops(self, stable_id):
        return self.players.get(stable_id, {}).get("jersey_crops", [])
