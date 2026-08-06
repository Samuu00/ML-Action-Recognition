import onnxruntime as ort
import numpy as np

class ONNXGestureClassifier:
    """Inference engine basato su ONNX Runtime."""

    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, window_data: np.ndarray) -> np.ndarray:
        """
        Esegue l'inferenza sul buffer di input.

        Args:
            window_data: array NumPy con shape (1, sequence_length, num_features)

        Returns:
            np.ndarray: Probabilità Softmax con shape (1, num_classes)
        """
        if window_data.dtype != np.float32:
            window_data = window_data.astype(np.float32)

        raw_outputs = self.session.run([self.output_name], {self.input_name: window_data})[0]

        # Applica Softmax sui logit grezzi
        exp_preds = np.exp(raw_outputs - np.max(raw_outputs, axis=-1, keepdims=True))
        probabilities = exp_preds / np.sum(exp_preds, axis=-1, keepdims=True)
        return probabilities[0]