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

# Calcolo del percorso radice del progetto: 3 livelli sopra la posizione attuale del modulo
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RealTimeRunner:
    def __init__(self, config_path: str = "config/config.yaml"):
        # Risoluzione dinamica del percorso relativo al file di configurazione
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = PROJECT_ROOT / config_file

        if not config_file.exists():
            raise FileNotFoundError(f"File di configurazione non trovato in: {config_file}")

        with open(config_file, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Configurazione Camera
        camera_cfg = self.config.get("camera", {})
        device_id: int = camera_cfg.get("device_id", 0)
        width: int = camera_cfg.get("width", 1280)
        height: int = camera_cfg.get("height", 720)

        # Configurazione Infezione & Modello
        model_rel_path = self.config.get("model", {}).get("path", "data/gesture_model.onnx")
        model_path = PROJECT_ROOT / model_rel_path

        training_cfg = self.config.get("training", {})
        self.classes: List[str] = training_cfg.get("classes", ["idle", "wave", "fist", "swipe"])
        self.threshold: float = training_cfg.get("prediction_threshold", 0.7)

        # Configurazione Pipeline
        pipeline_cfg = self.config.get("pipeline", {})
        window_size: int = pipeline_cfg.get("window_size", 30)

        # Inizializzazione moduli della pipeline Clean Architecture
        self.camera = OpenCVCamera(device_id=device_id, width=width, height=height)
        self.extractor = MediaPipePoseExtractor(
            min_detection_confidence=self.config.get("feature_extraction", {}).get("min_detection_confidence", 0.5),
            min_tracking_confidence=self.config.get("feature_extraction", {}).get("min_tracking_confidence", 0.5)
        )
        self.classifier = ONNXGestureClassifier(model_path=str(model_path))
        self.buffer = SlidingWindowBuffer(window_size=window_size)
        self.smoother = PredictionSmoother(window_size=5)

    def run(self) -> None:
        logger.info("Avvio della pipeline di riconoscimento gesti in tempo reale. Premi 'q' per uscire.")

        try:
            while True:
                frame = self.camera.read()
                if frame is None:
                    logger.warning("Frame non ricevuto dalla fotocamera. Interruzione...")
                    break

                landmarks, annotated_frame = self.extractor.extract(frame)

                if landmarks is not None:
                    self.buffer.add(landmarks)
                else:
                    self.buffer.clear()

                label = "IDLE"
                confidence = 0.0

                if self.buffer.is_ready():
                    window_data = self.buffer.get_window()  # Shape: (1, 30, 99)
                    probs = self.classifier.predict(window_data)

                    smoothed_probs = self.smoother.smooth(probs)
                    max_idx = int(smoothed_probs.argmax())
                    max_prob = float(smoothed_probs[max_idx])

                    if max_prob >= self.threshold:
                        label = self.classes[max_idx].upper()
                        confidence = max_prob

                color = (0, 255, 0) if label != "IDLE" else (255, 255, 255)
                hud_text = f"Gesto: {label} ({confidence * 100:.1f}%)"
                cv2.putText(
                    annotated_frame,
                    hud_text,
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color,
                    2,
                    cv2.LINE_AA
                )

                cv2.imshow("Real-Time Gesture Recognition", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Interruzione manuale richiesta dall'utente.")
                    break

        finally:
            self.camera.release()
            self.extractor.close()
            cv2.destroyAllWindows()
            logger.info("Risorse rilasciate e finestra chiusa con successo.")