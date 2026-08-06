import argparse
import sys
from src.application.real_time_runner import RealTimeRunner
from src.application.dataset_builder import DatasetBuilder
from src.utils.logger import setup_logger

logger = setup_logger("Main")


def main():
    parser = argparse.ArgumentParser(description="Gesture Recognition CLI")
    parser.add_argument("--mode", type=str, choices=["run", "build-dataset"], default="run",
                        help="Modalità di esecuzione: 'run' per inferenza live, 'build-dataset' per estrarre landmark.")
    parser.add_argument("--config", type=str, default="config/settings.yaml",
                        help="Path al file di configurazione YAML.")

    args = parser.parse_args()

    if args.mode == "run":
        try:
            runner = RealTimeRunner(config_path=args.config)
            runner.run()
        except KeyboardInterrupt:
            logger.info("Manual interruption")
        except Exception as e:
            logger.error(f"Error during the execution: {e}", exc_info=True)
            sys.exit(1)

    elif args.mode == "build-dataset":
        builder = DatasetBuilder(raw_data_dir="data/raw", output_path="data/processed/dataset.npz")
        builder.build_dataset()


if __name__ == "__main__":
    main()