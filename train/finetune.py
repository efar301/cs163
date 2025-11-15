import argparse

from finetuner import Finetuner


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune a super-resolution model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file.')
    args = parser.parse_args()

    finetuner = Finetuner(config_dir=args.config)
    finetuner.finetune()


if __name__ == "__main__":
    main()
