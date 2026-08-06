import time
import cv2
import yaml
from src.infrastructure.camera import ThreadedCamera
from src.infrastructure.extractors.mediapipe_extractor import MediaPipePoseExtractor
from src.infrastructure.classifiers.onnx_classifier import ONNXGestureClassifier
from src.infrastructure.pipeline.sliding_window import SlidingWindowBuffer, PredictionSmoother
from src.domain.normalizer import PoseNormalizer
from src.utils.logger import setup_logger

logger = setup_logger("RealTimeRunner")

class RealTimeRunner:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        cfg_pipe = self.config["pipeline"]
        cfg_train = self.config["training"]

        self.num_features = self.config["feature_extraction"]["num_landmarks"] * 3

        self.camera = ThreadedCamera(
            device_id=self.config["camera"]["device_id"],
            width=self.config["camera"]["width"],
            height=self.config["camera"]["height"]
        )
        self.extractor = MediaPipePoseExtractor(
            min_detection_confidence=self.config["feature_extraction"]["min_detection_confidence"],
            min_tracking_confidence=self.config["feature_extraction"]["min_tracking_confidence"]
        )
        self.classifier = ONNXGestureClassifier(
            model_path=cfg_train["model_onnx_path"],
            labels=cfg_train["classes"]
        )
        self.buffer = SlidingWindowBuffer(cfg_pipe["window_size"], self.num_features)
        self.smoother = PredictionSmoother(cfg_pipe["smoothing_window"], len(cfg_train["classes"]))

    def run(self) -> None:
        logger.info(f"Starting Real-Time Runner...")
        with self.camera:
            while True:
                frame = self.camera.read()
                if frame is None:
                    time.sleep(0.005)
                    continue

                timestamp = time.time()
                frame_data = self.extractor.extract(frame, timestamp)

                display_text = "Tracking..."
                if frame_data is not None:
                    norm_features = PoseNormalizer.normalize(frame_data.landmarks).flatten()
                    self.buffer.append(norm_features)

                    window = self.buffer.get_window()
                    if window is not None:
                        raw_pred = self.classifier.predict(window)
                        smooth_pred = self.smoother.smooth(raw_pred, self.config["training"]["classes"])

                        if smooth_pred.confidence >= self.config["pipeline"]["prediction_threshold"]:
                            display_text = f"Gesture: {smooth_pred.label} ({smooth_pred.confidence:.2f})"
                        else:
                            display_text = "Gesture: IDLE"

                # Overlay GUI
                cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow("Real-Time Gesture Recognition", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.extractor.close()
        cv2.destroyAllWindows()