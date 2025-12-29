import cv2

class VideoReader:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)

    def __iter__(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame

    def release(self):
        self.cap.release()