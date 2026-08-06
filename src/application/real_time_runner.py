from pathlib import Path
from typing import List
import cv2
import yaml

from src.infrastructure.camera.opencv_camera import OpenCVCamera
from src.infrastructure.landmarks.mediapipe_extractor import MediaPipePoseExtractor
from src.infrastructure.inference.onnx_classifier import ONNXGestureClassifier
from src.domain.pipeline.sliding_window_buffer import SlidingWindowBuffer
from src.domain.pipeline.prediction_smoother import PredictionSmoother
from src.utils.logger import setup_logger

logger = setup_logger("RealTimeRunner")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RealTimeRunner:
    def __init__(self, config_path: str = "config/config.yaml"):
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = PROJECT_ROOT / config_file

        with open(config_file, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        camera_cfg = self.config.get("camera", {})
        device_id: int = camera_cfg.get("device_id", 0)
        width: int = camera_cfg.get("width", 1280)
        height: int = camera_cfg.get("height", 720)

        model_rel_path = self.config.get("model", {}).get("path", "data/gesture_model.onnx")
        model_path = PROJECT_ROOT / model_rel_path

        training_cfg = self.config.get("training", {})
        self.classes: List[str] = training_cfg.get("classes", ["idle", "wave", "fist", "swipe"])
        self.threshold: float = training_cfg.get("prediction_threshold", 0.7)

        pipeline_cfg = self.config.get("pipeline", {})
        window_size: int = pipeline_cfg.get("window_size", 30)

        self.camera = OpenCVCamera(device_id=device_id, width=width, height=height)
        self.extractor = MediaPipePoseExtractor(
            min_detection_confidence=self.config.get("feature_extraction", {}).get("min_detection_confidence", 0.5),
            min_tracking_confidence=self.config.get("feature_extraction", {}).get("min_tracking_confidence", 0.5)
        )
        self.classifier = ONNXGestureClassifier(model_path=str(model_path))
        self.buffer = SlidingWindowBuffer(window_size=window_size)
        self.smoother = PredictionSmoother(window_size=5)
        self.window_name = "Real-Time Gesture Recognition"

    def run(self) -> None:
        logger.info("Avvio della pipeline. Premi 'q' o chiudi la finestra per uscire.")
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        try:
            while True:
                # Controlla se la finestra è stata chiusa con la 'X'
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("Finestra chiusa dall'utente.")
                    break

                frame = self.camera.read()
                if frame is None:
                    break

                landmarks, annotated_frame = self.extractor.extract(frame)

                label = "NO POSE"
                confidence = 0.0

                if landmarks is not None:
                    self.buffer.add(landmarks)
                    if self.buffer.is_ready():
                        window_data = self.buffer.get_window()
                        probs = self.classifier.predict(window_data)
                        smoothed_probs = self.smoother.smooth(probs)

                        max_idx = int(smoothed_probs.argmax())
                        max_prob = float(smoothed_probs[max_idx])

                        if max_prob >= self.threshold:
                            label = self.classes[max_idx].upper()
                            confidence = max_prob
                        else:
                            label = "UNCERTAIN"
                else:
                    # Reset se la posa va lost per evitare che lo smoother mantenga lo stato precedente
                    self.buffer.clear()
                    self.smoother.reset()

                # Rendering HUD
                color = (0, 255, 0) if label not in ["NO POSE", "UNCERTAIN"] else (0, 0, 255)
                hud_text = f"Gesto: {label} ({confidence * 100:.1f}%)"
                cv2.putText(annotated_frame, hud_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

                cv2.imshow(self.window_name, annotated_frame)

                # Processa gli eventi UI Windows
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.camera.release()
            self.extractor.close()
            cv2.destroyAllWindows()
            logger.info("Risorse rilasciate.")