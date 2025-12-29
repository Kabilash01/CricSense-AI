import cv2


class VideoReader:
    """
    Optimized, memory-safe video reader for real-time CV pipelines.

    Features:
    - Forced resize (prevents OpenCV OOM on Windows)
    - Frame skipping
    - Explicit resource cleanup
    - Iterator-safe
    """

    def __init__(
        self,
        video_path,
        resize=(1280, 720),   # 🔥 CRITICAL: reduce memory
        skip=1,               # process every Nth frame
    ):
        self.video_path = video_path
        self.resize = resize
        self.skip = max(1, skip)

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        # Disable OpenCV threading (Windows stability)
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)

        self.frame_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.release()
                raise StopIteration

            self.frame_idx += 1

            # -------------------------------
            # Frame skipping
            # -------------------------------
            if self.frame_idx % self.skip != 0:
                continue

            # -------------------------------
            # Force resize (MOST IMPORTANT)
            # -------------------------------
            if self.resize is not None:
                frame = cv2.resize(frame, self.resize, interpolation=cv2.INTER_LINEAR)

            return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
