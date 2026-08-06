from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float
    visibility: float

@dataclass(frozen=True)
class FrameData:
    timestamp: float
    landmarks: np.ndarray

@dataclass(frozen=True)
class GesturePrediction:
    label: str
    confidence: float
    probabilities: List[float]