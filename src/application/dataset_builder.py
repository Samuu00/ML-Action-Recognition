import os
import cv2
import numpy as np
from typing import Dict
from src.infrastructure.extractors.mediapipe_extractor import MediaPipePoseExtractor
from src.domain.normalizer import PoseNormalizer
from src.utils.logger import setup_logger

logger = setup_logger("DatasetBuilder")

class DatasetBuilder:
    def __init__(self, raw_data_dir: str, output_path: str):
        self.raw_data_dir = raw_data_dir
        self.output_path = output_path
        self.extractor = MediaPipePoseExtractor()

    def build_dataset(self) -> None:
        x, y = [], []
        label_map: Dict[str, int] = {}
        current_label_id = 0

        for class_name in os.listdir(self.raw_data_dir):
            class_dir = os.path.join(self.raw_data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            label_map[class_name] = current_label_id
            logger.info(f"Processing '{class_name}' (ID: {current_label_id})")

            for video_name in os.listdir(class_dir):
                video_path = os.path.join(class_dir, video_name)
                cap = cv2.VideoCapture(video_path)
                frame_landmarks = []

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_data = self.extractor.extract(frame, timestamp=0.0)
                    if frame_data is not None:
                        norm_features = PoseNormalizer.normalize(frame_data.landmarks)
                        frame_landmarks.append(norm_features.flatten())

                cap.release()
                if len(frame_landmarks) > 0:
                    x.append(np.array(frame_landmarks))
                    y.append(current_label_id)

            current_label_id += 1

        self.extractor.close()
        np.savez_compressed(self.output_path, X=np.array(X, dtype=object), y=np.array(y))
        logger.info(f"Saved dataset to {self.output_path}")