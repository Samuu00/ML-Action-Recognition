import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional


class MediaPipePoseExtractor:
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
            model_complexity=0,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12

        # Warm-up pre-allocazione C++
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.pose.process(dummy_frame)

    def extract(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        annotated_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            return None, annotated_frame

        self.mp_drawing.draw_landmarks(
            annotated_frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )

        raw_lms = results.pose_landmarks.landmark

        # Array shape (33, 3)
        coords = np.array([[lm.x, lm.y, lm.z] for lm in raw_lms], dtype=np.float32)

        # Centramento: punto medio tra spalla sinistra e destra
        left_shoulder = coords[self.LEFT_SHOULDER]
        right_shoulder = coords[self.RIGHT_SHOULDER]
        center = (left_shoulder + right_shoulder) / 2.0
        centered_coords = coords - center

        # Normalizzazione scala: distanza interspallare
        shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
        scale_factor = shoulder_dist if shoulder_dist > 1e-4 else 1.0
        normalized_coords = centered_coords / scale_factor

        return normalized_coords.flatten(), annotated_frame

    def close(self) -> None:
        if hasattr(self, 'pose') and self.pose is not None:
            self.pose.close()