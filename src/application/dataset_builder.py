import glob
import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import List, Optional

from src.infrastructure.landmarks.mediapipe_extractor import MediaPipePoseExtractor
from src.utils.logger import setup_logger

logger = setup_logger("DatasetBuilder")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DatasetBuilder:
    def __init__(
        self,
        config_path: Optional[str] = None,
        raw_data_dir: Optional[str] = None,
        output_path: Optional[str] = None
    ):
        config_file = Path(config_path) if config_path else PROJECT_ROOT / "config" / "config.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"Config non trovato: {config_file}")

        with open(config_file, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        cfg_ds = self.config["dataset"]
        cfg_fe = self.config["feature_extraction"]
        cfg_pipe = self.config["pipeline"]

        raw_dir = raw_data_dir or cfg_ds["raw_data_dir"]
        out_file = output_path or cfg_ds["processed_path"]

        self.raw_data_dir = (PROJECT_ROOT / raw_dir).resolve()
        self.output_path = (PROJECT_ROOT / out_file).resolve()

        self.classes = self.config["training"]["classes"]
        self.window_size = cfg_pipe["window_size"]

        self.extractor = MediaPipePoseExtractor(
            min_detection_confidence=cfg_fe["min_detection_confidence"],
            min_tracking_confidence=cfg_fe["min_tracking_confidence"]
        )

    def _process_video(self, video_path: str) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        features_sequence: List[np.ndarray] = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            landmarks, _ = self.extractor.extract(frame)
            if landmarks is not None:
                features_sequence.append(landmarks)

        cap.release()
        return features_sequence

    def build_dataset(self) -> None:
        logger.info(f"Avvio estrazione dataset da '{self.raw_data_dir}'...")
        X_list: List[np.ndarray] = []
        y_list: List[int] = []

        for class_idx, class_name in enumerate(self.classes):
            class_dir = self.raw_data_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Directory non trovata: {class_dir}")
                continue

            video_files = glob.glob(str(class_dir / "*.mp4")) + glob.glob(str(class_dir / "*.avi"))
            logger.info(f"Elaborazione '{class_name}' ({len(video_files)} video)...")

            for video_path in video_files:
                seq = self._process_video(video_path)

                if len(seq) >= self.window_size:
                    for i in range(len(seq) - self.window_size + 1):
                        window = np.array(seq[i : i + self.window_size], dtype=np.float32)
                        X_list.append(window)  # Shape: (T, 99)
                        y_list.append(class_idx)

        if not X_list:
            raise ValueError(f"Nessun dato estratto da '{self.raw_data_dir}'.")

        X = np.array(X_list, dtype=np.float32)  # Shape: (N, T, 99)
        y = np.array(y_list, dtype=np.int64)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.output_path, X=X, y=y)

        logger.info(f"Dataset salvato in '{self.output_path}'. Shape X: {X.shape}, Shape y: {y.shape}")
        self.extractor.close()

    def build(self) -> None:
        self.build_dataset()