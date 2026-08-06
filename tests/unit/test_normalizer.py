import pytest
import numpy as np
from src.domain.normalizer import PoseNormalizer


def test_pose_normalizer_centering():
    mock_landmarks = np.zeros((33, 4), dtype=np.float32)
    mock_landmarks[23] = np.array([10.0, 5.0, 0.0, 1.0])
    mock_landmarks[24] = np.array([12.0, 5.0, 0.0, 1.0])
    mock_landmarks[11] = np.array([10.0, 15.0, 0.0, 1.0])
    mock_landmarks[12] = np.array([12.0, 15.0, 0.0, 1.0])

    normalized = PoseNormalizer.normalize(mock_landmarks)

    mid_hip_normalized = (normalized[23] + normalized[24]) / 2.0
    np.testing.assert_almost_equal(mid_hip_normalized, np.array([0.0, 0.0, 0.0]))


def test_pose_normalizer_insufficient_landmarks():
    invalid_landmarks = np.zeros((10, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        PoseNormalizer.normalize(invalid_landmarks)