import torch
from torch.utils.data import Dataset
import numpy as np


class GestureDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X ha shape (N, T, C) -> e.g. (N, 15, 99)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx]  # Shape: (T, 99)

        if x.shape[0] != 99 and x.shape[1] == 99:
            x = x.permute(1, 0)

        return x, self.y[idx]