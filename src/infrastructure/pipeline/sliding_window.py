from collections import deque
import numpy as np
from typing import List, Optional
from src.domain.entities import GesturePrediction

class SlidingWindowBuffer:
    def __init__(self, window_size: int, num_features: int):
        self.window_size = window_size
        self.num_features = num_features
        self.buffer: deque = deque(maxlen=window_size)

    def append(self, frame_features: np.ndarray) -> None:
        self.buffer.append(frame_features)

    def is_full(self) -> bool:
        return len(self.buffer) == self.window_size

    def get_window(self) -> Optional[np.ndarray]:
        if not self.is_full():
            return None
        return np.expand_dims(np.array(self.buffer, dtype=np.float32), axis=0)

class PredictionSmoother:
    def __init__(self, window_size: int, num_classes: int):
        self.history: deque = deque(maxlen=window_size)
        self.num_classes = num_classes

    def smooth(self, prediction: GesturePrediction, labels: List[str]) -> GesturePrediction:
        self.history.append(prediction.probabilities)
        avg_probs = np.mean(self.history, axis=0)
        max_idx = int(np.argmax(avg_probs))

        return GesturePrediction(
            label=labels[max_idx],
            confidence=float(avg_probs[max_idx]),
            probabilities=avg_probs.tolist()
        )