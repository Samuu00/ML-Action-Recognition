import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import yaml
from pathlib import Path
from typing import Tuple

from src.infrastructure.models.temporal_cnn import TemporalGestureCNN
from src.utils.logger import setup_logger

logger = setup_logger("ModelTrainer")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ModelTrainer:
    def __init__(self, config_path: str = "config/config.yaml"):
        config_file = PROJECT_ROOT / config_path
        with open(config_file, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        cfg_ds = self.config["dataset"]
        cfg_tr = self.config["training"]
        cfg_pipe = self.config["pipeline"]
        cfg_fe = self.config["feature_extraction"]

        self.dataset_path = PROJECT_ROOT / cfg_ds["processed_path"]
        self.onnx_output_path = PROJECT_ROOT / "data" / "gesture_model.onnx"
        self.classes = cfg_tr["classes"]
        self.num_classes = len(self.classes)
        self.window_size = cfg_pipe["window_size"]
        self.num_features = cfg_fe["num_landmarks"] * 3

        self.epochs = cfg_tr.get("epochs", 40)
        self.batch_size = cfg_tr.get("batch_size", 32)
        self.lr = cfg_tr.get("lr", 0.001)

    def _load_data(self) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
        data = np.load(self.dataset_path)
        X, y = data["X"], data["y"]

        # Stratified Split per preservare la distribuzione delle classi
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader, X_val, y_val

    def train(self) -> None:
        train_loader, val_loader, X_val, y_val = self._load_data()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = TemporalGestureCNN(
            num_features=self.num_features,
            sequence_length=self.window_size,
            num_classes=self.num_classes
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        logger.info(f"Avvio addestramento su {device} ({self.epochs} epoche)...")
        for epoch in range(1, self.epochs + 1):
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)

            train_loss /= len(train_loader.dataset)

            # Validation Loop
            model.eval()
            val_loss, correct = 0.0, 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_X.size(0)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == batch_y).sum().item()

            val_loss /= len(val_loader.dataset)
            acc = correct / len(val_loader.dataset)

            if epoch % 10 == 0 or epoch == self.epochs:
                logger.info(
                    f"Epoch {epoch:02d}/{self.epochs} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc * 100:.2f}%"
                )

        # Classification Metrics
        model.eval()
        with torch.no_grad():
            val_preds = model(torch.tensor(X_val, dtype=torch.float32).to(device)).argmax(dim=1).cpu().numpy()

        logger.info("\n" + classification_report(y_val, val_preds, target_names=self.classes))

        # Clean ONNX Export (TorchScript Exporter con Opset 18)
        model.eval().to("cpu")
        dummy_input = torch.randn(1, self.window_size, self.num_features)
        self.onnx_output_path.parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            model,
            dummy_input,
            str(self.onnx_output_path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            dynamo=False  # Mantiene l'exporter stabile basato su TorchScript senza fallback C API
        )
        logger.info(f"Modello esportato pulito in ONNX: {self.onnx_output_path}")