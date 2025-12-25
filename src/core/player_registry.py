import math

class PlayerRegistry:
    """
    Maintains STABLE player identities on top of ByteTrack IDs
    and stores trajectory + role information.
    """

    def __init__(self, max_idle_frames=60, max_trail_len=40):
        # stable_id -> player data
        self.players = {}

        # raw ByteTrack ID -> stable_id
        self.raw_to_stable = {}

        self.next_stable_id = 1
        self.max_idle_frames = max_idle_frames
        self.max_trail_len = max_trail_len

    # --------------------------------------------------
    # Utility
    # --------------------------------------------------
    def _distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    # --------------------------------------------------
    # ID STABILIZATION LOGIC (CORE)
    # --------------------------------------------------
    def resolve_id(self, raw_id, center, role, frame_id):
        """
        Resolve a ByteTrack raw ID into a stable ID.
        """

        # Case 1: raw ID already mapped
        if raw_id in self.raw_to_stable:
            sid = self.raw_to_stable[raw_id]
            self.players[sid]["last_center"] = center
            self.players[sid]["last_seen"] = frame_id
            return sid

        # Case 2: merge with an existing stable ID
        for sid, player in self.players.items():
            # Role must match
            if player["final_role"] != role:
                continue

            # Spatial proximity check
            if self._distance(center, player["last_center"]) < 60:
                # Temporal continuity check
                if frame_id - player["last_seen"] <= self.max_idle_frames:
                    self.raw_to_stable[raw_id] = sid
                    player["last_center"] = center
                    player["last_seen"] = frame_id
                    return sid

        # Case 3: create new stable ID
        sid = self.next_stable_id
        self.next_stable_id += 1

        self.raw_to_stable[raw_id] = sid
        self.players[sid] = {
            "final_role": role,
            "last_center": center,
            "last_seen": frame_id,
            "trajectory": []
        }

        return sid

    # --------------------------------------------------
    # UPDATE TRAJECTORY
    # --------------------------------------------------
    def update(self, stable_id, center):
        traj = self.players[stable_id]["trajectory"]
        traj.append(center)

        # keep trajectory bounded
        if len(traj) > self.max_trail_len:
            traj.pop(0)

    # --------------------------------------------------
    # ACCESSORS
    # --------------------------------------------------
    def get_trajectory(self, stable_id):
        return self.players.get(stable_id, {}).get("trajectory", [])

    def get_player(self, stable_id):
        return self.players.get(stable_id, None)
