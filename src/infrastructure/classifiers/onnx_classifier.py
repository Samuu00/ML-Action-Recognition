import onnxruntime as ort
import numpy as np
from typing import List
from src.domain.interfaces import ClassifierInterface
from src.domain.entities import GesturePrediction


class ONNXGestureClassifier(ClassifierInterface):
    def __init__(self, model_path: str, labels: List[str]):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.labels = labels

    def predict(self, window_data: np.ndarray) -> GesturePrediction:
        input_tensor = window_data.astype(np.float32)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})[0]

        probs = np.exp(outputs[0]) / np.sum(np.exp(outputs[0]))
        max_idx = int(np.argmax(probs))

        return GesturePrediction(
            label=self.labels[max_idx],
            confidence=float(probs[max_idx]),
            probabilities=probs.tolist()
        )