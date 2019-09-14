import threading
import time

import cv2


class ThreadedVideoCapture:
    def __init__(self, src):
        self.src = src
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.frame_count = 0
        self.frame_lock = threading.Lock()
        self.frame_update = threading.Event()
        self._thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.cap.release()

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self):
        if self.started:
            raise ValueError("Video Capture Thread has already been initialized")
        self.started = True
        self._thread = threading.Thread(target=self.update, args=())
        self._thread.start()

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.frame_lock:
                self.frame_update.set()
                self.grabbed = grabbed
                self.frame = frame
                self.frame_count += 1

    def read(self):
        with self.frame_lock:
            frame = self.frame
            grabbed = self.grabbed
        return grabbed, frame

    def read_block(self, last_frame):
        if self.frame_count == last_frame:
            self.frame_update.clear()
            self.frame_update.wait()
        else:
            print("New frame appeared!")
        return self.frame_count, self.read()

    def stop(self):
        self.started = False
        self._thread.join()
