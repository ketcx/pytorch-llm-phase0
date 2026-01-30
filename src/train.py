"""Training script example using config and logging."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import Config, parse_cli_args, load_config, RunManager


def main():
    """Main training function."""
    # Parse CLI arguments
    args = parse_cli_args()

    # Load configuration
    config = load_config(args)

    # Create run manager
    run_manager = RunManager(config, tag=args.tag)

    # Log configuration
    run_manager.logger.info("Configuration:")
    run_manager.logger.info(str(config))

    # Simulate training loop
    run_manager.logger.info("Starting training...")

    for epoch in range(config.training.epochs):
        for step in range(100):  # Dummy steps
            # Simulate metrics
            metrics = {
                "loss": 1.0 / (step + 1),
                "accuracy": step / 100,
            }

            run_manager.log_metrics(metrics, step=epoch * 100 + step)

        run_manager.logger.info(f"Epoch {epoch + 1}/{config.training.epochs} completed")

    run_manager.logger.info("Training finished!")
    run_manager.logger.info(f"Results saved to: {run_manager.run_dir}")


if __name__ == "__main__":
    main()
