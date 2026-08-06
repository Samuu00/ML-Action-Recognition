import cv2
import numpy as np
import mediapipe as mp
from typing import Optional
from src.domain.interfaces import ExtractorInterface
from src.domain.entities import FrameData

# Fallback per bypassare il mancato caricamento dinamico di mp.solutions su Windows
try:
    mp_pose = mp.solutions.pose
except AttributeError:
    import mediapipe.python.solutions.pose as mp_pose  # type: ignore

class MediaPipePoseExtractor(ExtractorInterface):
    def __init__(self, min_detection_confidence: float = 0.7, min_tracking_confidence: float = 0.7):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def extract(self, image: np.ndarray, timestamp: float) -> Optional[FrameData]:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            return None

        landmarks_arr = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
            dtype=np.float32
        )
        return FrameData(timestamp=timestamp, landmarks=landmarks_arr)

    def close(self) -> None:
        self.pose.close()