# src/application/real_time_runner.py
import cv2
import time
import numpy as np
import yaml
from pathlib import Path

from src.infrastructure.landmarks.mediapipe_extractor import MediaPipePoseExtractor
from src.infrastructure.inference.onnx_classifier import ONNXGestureClassifier
from src.domain.pipeline.sliding_window_buffer import SlidingWindowBuffer
from src.utils.logger import setup_logger

logger = setup_logger("RealTimeRunner")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RealTimeRunner:
    def __init__(self, config_path: str = "config/config.yaml"):
        config_file = PROJECT_ROOT / config_path
        with open(config_file, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        cfg_pipe = self.config["pipeline"]
        cfg_fe = self.config["feature_extraction"]

        self.camera_index = cfg_pipe.get("camera_index", 0)
        self.window_size = cfg_pipe["window_size"]
        self.classes = self.config["training"]["classes"]

        onnx_path = PROJECT_ROOT / "data" / "gesture_model.onnx"
        self.classifier = ONNXGestureClassifier(str(onnx_path))
        self.extractor = MediaPipePoseExtractor(
            min_detection_confidence=cfg_fe["min_detection_confidence"],
            min_tracking_confidence=cfg_fe["min_tracking_confidence"]
        )
        self.buffer = SlidingWindowBuffer(window_size=self.window_size)

    def _init_camera(self) -> cv2.VideoCapture:
        indices_to_try = [self.camera_index, 0, 1, 2]
        indices_to_try = list(dict.fromkeys(indices_to_try))

        for idx in indices_to_try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    logger.info(f"Webcam agganciata sull'indice: {idx}")
                    return cap
                cap.release()

        raise RuntimeError(f"Nessuna webcam disponibile su indici: {indices_to_try}")

    def run(self) -> None:
        cap = self._init_camera()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        window_name = "Real-Time Gesture Recognition"

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)

        logger.info("Avvio della pipeline. Premi 'q' o clicca sulla 'X' per uscire.")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                landmarks, annotated_frame = self.extractor.extract(frame)

                current_gesture = "BUFFERING..."
                confidence = 0.0

                if landmarks is not None:
                    self.buffer.add(landmarks)

                    if self.buffer.is_ready():
                        window_data = self.buffer.get_window()
                        probs = self.classifier.predict(window_data)
                        class_idx = int(np.argmax(probs))
                        confidence = float(probs[class_idx])
                        current_gesture = self.classes[class_idx]

                # Overlay UI
                cv2.putText(
                    annotated_frame,
                    f"Gesture: {current_gesture} ({confidence:.2f})",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                cv2.imshow(window_name, annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.extractor.close()
            logger.info("Risorse rilasciate.")