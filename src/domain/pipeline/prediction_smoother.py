import numpy as np
from collections import deque

class PredictionSmoother:
    """Filtro a media mobile temporale per stabilizzare le probabilità di classe ed evitare jittering."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def smooth(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Calcola la media mobile delle probabilità ricevute.
        Args:
            probabilities: Array 1D di probabilità (num_classes,)
        Returns:
            np.ndarray: Probabilità smussate 1D (num_classes,)
        """
        self.history.append(probabilities)
        return np.mean(self.history, axis=0)

    def reset(self) -> None:
        """Azzera la cronologia del filtro."""
        self.history.clear()