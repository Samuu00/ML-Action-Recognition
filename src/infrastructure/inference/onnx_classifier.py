from pathlib import Path
import numpy as np
import onnxruntime as ort
import logging

logger = logging.getLogger("ONNXGestureClassifier")


class ONNXGestureClassifier:
    def __init__(self, model_path: str):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Modello ONNX non trovato in: {path.resolve()}")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, window_data: np.ndarray) -> np.ndarray:
        """
        Args:
            window_data: np.ndarray con shape (1, 99, T)
        Returns:
            np.ndarray: Probabilità per classe (4,)
        """
        # Guardrail sulle dimensioni di input
        if window_data.ndim != 3 or window_data.shape[1] != 99:
            raise ValueError(f"Shape input non valida: {window_data.shape}. Atteso (1, 99, T)")

        outputs = self.session.run([self.output_name], {self.input_name: window_data})
        logits = outputs[0][0]

        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        return probabilities