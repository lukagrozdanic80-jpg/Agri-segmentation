import argparse
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Train a segmentation model.")
    parser.add_argument("--config", default="configs/baseline_unet_resnet50.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    print("Loaded training config:")
    print(yaml.safe_dump(config, sort_keys=False))
    print("Training loop will be implemented in the next milestone.")


if __name__ == "__main__":
    main()
