from collections import deque
from typing import Optional, List
import numpy as np
from src.domain.entities import PredictionResult

class SlidingWindowBuffer:
    """Buffer temporale ad anello (sliding window) per raccogliere le sequenze di landmark."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def add(self, landmarks: np.ndarray) -> None:
        """Aggiunge un frame di landmark (shape: 99,) al buffer."""
        self.buffer.append(landmarks)

    def is_ready(self) -> bool:
        """Ritorna True se la finestra contiene esattamente window_size frame."""
        return len(self.buffer) == self.window_size

    def get_window(self) -> np.ndarray:
        """
        Restituisce il contenuto corrente del buffer formattato per l'inferenza ONNX.
        Returns:
            np.ndarray: Array con shape (1, window_size, num_features) e dtype float32.
        """
        if not self.is_ready():
            raise ValueError(f"Buffer non pronto: {len(self.buffer)}/{self.window_size} frame presenti.")
        return np.expand_dims(np.array(self.buffer, dtype=np.float32), axis=0)

    def clear(self) -> None:
        """Svuota il buffer in caso di perdita del tracking."""
        self.buffer.clear()

