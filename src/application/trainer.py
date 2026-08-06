import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List
from src.utils.logger import setup_logger

logger = setup_logger("Trainer")

class TemporalGestureCNN(nn.Module):
    def __init__(self, input_features: int, sequence_length: int, num_classes: int):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=input_features, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.squeeze(-1)
        return self.fc(x)

def export_to_onnx(model: nn.Module, dummy_input: torch.Tensor, onnx_path: str) -> None:
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    logger.info(f"Exported ONNX model to {onnx_path}")