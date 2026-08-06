import cv2
import threading
from queue import Queue, Empty
from typing import Optional
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger("Camera")

class ThreadedCamera:
    def __init__(self, device_id: int = 0, width: int = 1280, height: int = 720):
        self.cap = cv2.VideoCapture(device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.q: Queue[np.ndarray] = Queue(maxsize=1)
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)

    def start(self) -> "ThreadedCamera":
        if not self.cap.isOpened():
            raise RuntimeError("Camera is not opened")
        self.thread.start()
        logger.info("Camera started")
        return self

    def _update(self) -> None:
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Frame not received")
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except Empty:
                    pass
            self.q.put(frame)

    def read(self) -> Optional[np.ndarray]:
        try:
            return self.q.get_nowait()
        except Empty:
            return None

    def stop(self) -> None:
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()
        logger.info("Camera stopped")

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()