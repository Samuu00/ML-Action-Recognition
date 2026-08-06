import torch
import torch.nn as nn

class TemporalGestureCNN(nn.Module):
    def __init__(self, num_features: int = 99, sequence_length: int = 30, num_classes: int = 4):
        super().__init__()
        self.num_features = num_features
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

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)

        feat = self.features(x)
        feat = feat.squeeze(-1)
        return self.classifier(feat)