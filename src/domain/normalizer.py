import numpy as np

class PoseNormalizer:
    @staticmethod
    def normalize(landmarks: np.ndarray) -> np.ndarray:
        """
            Input shape: (K, 4) dove colonne sono [x, y, z, visibility]
            Output shape: (K, 3) coordinate normalizzate [x_norm, y_norm, z_norm]
        """
        if landmarks.shape[0] < 25:
            raise ValueError('landmarks too small')

        coords = landmarks[:, :3].copy()
        hip_center = (coords[23] + coords[24]) / 2.0
        coords -= hip_center

        shoulder_dist = np.linalg.norm(coords[11] - coords[12])
        scale_factor = shoulder_dist if shoulder_dist > 1e-6 else 1.0

        normalized_coords = coords / scale_factor
        return normalized_coords.astype(np.float32)