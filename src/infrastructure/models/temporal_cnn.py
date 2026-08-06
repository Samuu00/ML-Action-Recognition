# src/infrastructure/models/temporal_cnn.py
import torch
import torch.nn as nn


class TemporalCNN(nn.Module):
    def __init__(
            self,
            num_features: int = 99,
            num_classes: int = 4,
            sequence_length: int = 15
    ):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.sequence_length = sequence_length

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)  # Shape: (B, 128, 1)
        feat = feat.squeeze(-1)  # Shape: (B, 128)
        return self.classifier(feat)