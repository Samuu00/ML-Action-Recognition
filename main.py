import argparse
from src.utils.logger import setup_logger

logger = setup_logger("Main")

def main():
    parser = argparse.ArgumentParser(description="Real-Time Gesture Recognition Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        default="run",
        choices=["run", "record", "build-dataset", "train"],
        help="Modalità di esecuzione della pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path al file di configurazione YAML"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Numero di campioni video per classe (usato solo in mode='record')"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=3,
        help="Durata in secondi di ciascun campione (usato solo in mode='record')"
    )

    args = parser.parse_args()

    try:
        if args.mode == "run":
            from src.application.real_time_runner import RealTimeRunner
            runner = RealTimeRunner(config_path=args.config)
            runner.run()

        elif args.mode == "record":
            from src.application.dataset_recorder import DatasetRecorder
            recorder = DatasetRecorder(config_path=args.config)
            recorder.record_samples(samples_per_class=args.samples, duration_sec=args.duration)

        elif args.mode == "build-dataset":
            from src.application.dataset_builder import DatasetBuilder
            builder = DatasetBuilder(config_path=args.config)
            builder.build_dataset()

        elif args.mode == "train":
            from src.application.trainer import ModelTrainer
            trainer = ModelTrainer(config_path=args.config)
            trainer.train()

    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)

if __name__ == "__main__":
    main()