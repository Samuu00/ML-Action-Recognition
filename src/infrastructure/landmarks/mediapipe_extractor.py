import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional

class MediaPipePoseExtractor:
    """Adapter per l'estrazione dei landmark corporei tramite MediaPipe Pose."""

    def __init__(
            self,
            min_detection_confidence: float = 0.5,
            min_tracking_confidence: float = 0.5
    ):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def extract(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        annotated_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            return None, annotated_frame

        # Disegna lo scheletro sul frame
        self.mp_drawing.draw_landmarks(
            annotated_frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # Estrazione e flattening coordinate (33 landmarks * 3 = 99 features)
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        return np.array(landmarks, dtype=np.float32), annotated_frame

    def close(self) -> None:
        """Rilascia le risorse di MediaPipe."""
        self.pose.close()