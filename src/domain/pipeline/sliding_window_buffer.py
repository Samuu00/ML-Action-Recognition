import numpy as np
from collections import deque


class SlidingWindowBuffer:
    def __init__(self, window_size: int = 15):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def add(self, landmarks: np.ndarray) -> None:
        self.buffer.append(landmarks)

    def is_ready(self) -> bool:
        return len(self.buffer) == self.window_size

    def get_window(self) -> np.ndarray:
        if not self.is_ready():
            raise ValueError(f"Buffer non pronto: {len(self.buffer)}/{self.window_size}")

        arr = np.array(self.buffer, dtype=np.float32)
        tensor_data = np.transpose(arr, (1, 0))
        return np.expand_dims(tensor_data, axis=0)

    def clear(self) -> None:
        self.buffer.clear()