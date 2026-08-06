import numpy as np
import onnxruntime as ort
from typing import List

class ONNXGestureClassifier:
    def __init__(self, model_path: str, labels: List[str]):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.labels = labels

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def predict(self, window: np.ndarray) -> np.ndarray:
        """
        Esegue l'inferenza ONNX.

        :param window: Array con forma (1, sequence_length, feature_dim)
        :return: Vettore probabilità 1D con forma (num_classes,)
        """
        raw_outputs = self.session.run(
            [self.output_name],
            {self.input_name: window.astype(np.float32)}
        )[0]

        probs = self._softmax(raw_outputs)
        return np.squeeze(probs, axis=0)