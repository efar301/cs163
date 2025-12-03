import argparse

from warmstarter import WarmStarter


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm start a super-resolution model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file.')
    args = parser.parse_args()

    warmstarter = WarmStarter(config_dir=args.config)
    warmstarter.warmstart()


if __name__ == "__main__":
    main()
