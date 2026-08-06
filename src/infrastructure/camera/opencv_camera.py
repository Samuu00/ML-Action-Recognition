import cv2
from typing import Optional
import numpy as np


class OpenCVCamera:
    """Adapter per la gestione dell'I/O dello stream video via OpenCV."""

    def __init__(self, device_id: int = 0, width: int = 1280, height: int = 720):
        self.device_id = device_id
        self.cap = cv2.VideoCapture(self.device_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"Impossibile aprire la fotocamera con device_id={device_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> Optional[np.ndarray]:
        """Legge un singolo frame dallo stream."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self) -> None:
        """Rilascia le risorse della camera."""
        if self.cap.isOpened():
            self.cap.release()