import cv2
import logging
from typing import Optional

logger = logging.getLogger("OpenCVCamera")

class OpenCVCamera:
    def __init__(self, device_id: int = 0, width: int = 1280, height: int = 720):
        self.device_id = device_id
        self.width = width
        self.height = height

        self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.device_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"Impossibile aprire la webcam (device_id: {device_id})")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        logger.info(f"Webcam {device_id} inizializzata con successo.")

    def read(self) -> Optional[cv2.Mat]:
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Impossibile leggere il frame dalla webcam.")
            return None
        return frame

    def release(self) -> None:
        if self.cap and self.cap.isOpened():
            self.cap.release()