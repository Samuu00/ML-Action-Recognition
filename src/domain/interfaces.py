from typing import Protocol, Optional
import numpy as np
from src.domain.entities import FrameData, GesturePrediction

class ExtractorInterface(Protocol):
    def extract(self, image: np.ndarray, timestamp: float) -> Optional[FrameData]:
        """Estrae i landmark pose da un singolo frame RGB."""
        ...

    def close(self) -> None:
        """Rilascia le risorse dell'estrattore."""
        ...

class ClassifierInterface(Protocol):
    def predict(self, window_data: np.ndarray) -> GesturePrediction:
        """
         Input: Tensor/Array con shape (1, Window_Size, Features)
        Output: GesturePrediction contenente classe e probabilità
         """
        ...