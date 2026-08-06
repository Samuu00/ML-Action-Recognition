import cv2
import time
import yaml
from pathlib import Path
from src.utils.logger import setup_logger

logger = setup_logger("DatasetRecorder")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

class DatasetRecorder:
    def __init__(self, config_path: str = "config/config.yaml"):
        config_file = PROJECT_ROOT / config_path
        with open(config_file, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.classes = self.config["training"]["classes"]
        self.raw_data_dir = PROJECT_ROOT / self.config["dataset"]["raw_data_dir"]

    def record_samples(self, samples_per_class: int = 5, duration_sec: int = 3) -> None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Impossibile accedere alla webcam per la registrazione.")

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        for class_name in self.classes:
            class_dir = self.raw_data_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            for sample_idx in range(1, samples_per_class + 1):
                logger.info(f"Prepararsi per la classe '{class_name}' - Campione {sample_idx}/{samples_per_class}")

                # Countdown prima della registrazione
                for countdown in range(3, 0, -1):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cv2.putText(
                        frame, f"GET READY [{class_name.upper()}]: {countdown}",
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2
                    )
                    cv2.imshow("Dataset Recorder", frame)
                    cv2.waitKey(1000)

                output_path = class_dir / f"{class_name}_{int(time.time())}_{sample_idx}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

                start_time = time.time()
                while int(time.time() - start_time) < duration_sec:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    out.write(frame)
                    cv2.putText(
                        frame, f"RECORDING [{class_name.upper()}]...",
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
                    )
                    cv2.imshow("Dataset Recorder", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                out.release()
                logger.info(f"Salvato: {output_path}")

        cap.release()
        cv2.destroyAllWindows()