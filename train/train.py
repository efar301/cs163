import argparse

from trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a super-resolution model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file.')
    args = parser.parse_args()

    trainer = Trainer(config_dir=args.config)
    trainer.train()


if __name__ == "__main__":
    main()
