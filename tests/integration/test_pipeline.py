import pytest
import numpy as np
from src.infrastructure.pipeline.sliding_window import SlidingWindowBuffer

def test_sliding_window_buffer_fill():
    window_size = 30
    num_features = 99
    buffer = SlidingWindowBuffer(window_size=window_size, num_features=num_features)

    assert not buffer.is_full()
    assert buffer.get_window() is None

    dummy_frame = np.ones(num_features, dtype=np.float32)
    for _ in range(window_size):
        buffer.append(dummy_frame)

    assert buffer.is_full()
    window = buffer.get_window()
    assert window is not None
    assert window.shape == (1, window_size, num_features)