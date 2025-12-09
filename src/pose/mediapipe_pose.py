import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

class PoseEstimator:
    def __init__(self, complexity=0):
        """
        complexity 0 = fastest
        complexity 1 = balanced
        """
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=complexity,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def estimate(self, crop_bgr):
        """
        crop_bgr: BGR player crop
        returns: list of (x, y, visibility)
        """
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)

        if not res.pose_landmarks:
            return None

        h, w = crop_bgr.shape[:2]
        keypoints = []
        for lm in res.pose_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            keypoints.append((x, y, lm.visibility))
        return keypoints

    def draw(self, crop, keypoints):
        if keypoints is None:
            return crop
        for (x,y,v) in keypoints:
            if v > 0.5:
                cv2.circle(crop, (x, y), 3, (255, 0, 0), -1)
        return crop
