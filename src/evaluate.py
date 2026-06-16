import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a segmentation model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/baseline_unet_resnet50.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Evaluation placeholder for checkpoint: {args.checkpoint}")
    print(f"Using config: {args.config}")


if __name__ == "__main__":
    main()
