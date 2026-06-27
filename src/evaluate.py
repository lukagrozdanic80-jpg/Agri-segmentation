import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from dataset import AgricultureVisionDataset
from losses import CombinedSegmentationLoss
from metrics import mean_iou
from models import build_model


COLORS = np.array(
    [
        [0, 0, 0],
        [230, 25, 75],
        [60, 180, 75],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a segmentation checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/baseline_unet_resnet50.yaml")
    parser.add_argument("--data-root", default=None, help="Override dataset root from config.")
    parser.add_argument("--output-dir", default=None, help="Directory for evaluation outputs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override validation batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers.")
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--save-samples", type=int, default=8)
    return parser.parse_args()


def move_batch_to_device(batch, device):
    images = batch["image"].to(device, non_blocking=True)
    masks = batch["mask"].to(device, non_blocking=True)
    valid_masks = batch.get("valid_mask")

    if valid_masks is not None:
        valid_masks = valid_masks.to(device, non_blocking=True)

    return images, masks, valid_masks


def build_val_loader(config, batch_size=None, num_workers=None):
    data_config = config["data"]
    training_config = config["training"]
    val_dataset = AgricultureVisionDataset(
        root=data_config["root"],
        split="val",
        use_nir=data_config.get("use_nir", False),
        return_valid_mask=data_config.get("return_valid_mask", True),
        image_mean=data_config.get("image_mean"),
        image_std=data_config.get("image_std"),
    )

    return DataLoader(
        val_dataset,
        batch_size=batch_size or training_config["batch_size"],
        shuffle=False,
        num_workers=(
            num_workers
            if num_workers is not None
            else training_config.get("num_workers", 2)
        ),
        pin_memory=torch.cuda.is_available(),
    )


def build_checkpoint_model(config, checkpoint_path, device):
    model_config = config["model"]
    data_config = config["data"]
    model = build_model(
        architecture=model_config["architecture"],
        encoder=model_config["encoder"],
        encoder_weights=None,
        in_channels=model_config.get("in_channels", 3),
        num_classes=data_config["num_classes"],
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def colorize_mask(mask):
    return COLORS[mask.clip(min=0, max=len(COLORS) - 1)]


def image_to_numpy(image_tensor, image_mean=None, image_std=None):
    image = image_tensor.detach().cpu().float().numpy()
    image = np.transpose(image[:3], (1, 2, 0))

    if image_mean is not None and image_std is not None:
        mean = np.asarray(image_mean, dtype=np.float32)
        std = np.asarray(image_std, dtype=np.float32)
        image = image * std + mean

    return np.clip(image, 0.0, 1.0)


def save_prediction_sample(image, target, prediction, sample_id, output_dir, image_mean=None, image_std=None):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_to_numpy(image, image_mean=image_mean, image_std=image_std))
    axes[0].set_title("Image")
    axes[1].imshow(colorize_mask(target.detach().cpu().numpy()))
    axes[1].set_title("Ground truth")
    axes[2].imshow(colorize_mask(prediction.detach().cpu().numpy()))
    axes[2].set_title("Prediction")

    for axis in axes:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / f"{sample_id}.png", dpi=150)
    plt.close(fig)


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device,
    num_classes,
    output_dir,
    save_samples,
    image_mean=None,
    image_std=None,
    limit_batches=None,
):
    model.eval()
    total_loss = 0.0
    total_miou = 0.0
    num_batches = 0
    saved_samples = 0
    samples_dir = output_dir / "samples"

    progress = tqdm(dataloader, desc="Evaluate", leave=False)
    for batch_index, batch in enumerate(progress, start=1):
        images, masks, valid_masks = move_batch_to_device(batch, device)
        logits = model(images)
        loss = criterion(logits, masks, valid_mask=valid_masks)
        miou = mean_iou(logits, masks, num_classes=num_classes, valid_mask=valid_masks)
        predictions = torch.argmax(logits, dim=1)

        total_loss += loss.item()
        total_miou += miou.item()
        num_batches += 1
        progress.set_postfix(loss=f"{loss.item():.4f}", miou=f"{miou.item():.4f}")

        if saved_samples < save_samples:
            sample_ids = batch["id"]
            for sample_index, sample_id in enumerate(sample_ids):
                if saved_samples >= save_samples:
                    break

                save_prediction_sample(
                    images[sample_index],
                    masks[sample_index],
                    predictions[sample_index],
                    sample_id,
                    samples_dir,
                    image_mean=image_mean,
                    image_std=image_std,
                )
                saved_samples += 1

        if limit_batches is not None and batch_index >= limit_batches:
            break

    return total_loss / max(num_batches, 1), total_miou / max(num_batches, 1), saved_samples


def write_results(output_dir, checkpoint_path, val_loss, val_miou, saved_samples, checkpoint):
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "metrics.txt"
    lines = [
        f"checkpoint={checkpoint_path}",
        f"checkpoint_epoch={checkpoint.get('epoch', 'unknown')}",
        f"checkpoint_best_miou={checkpoint.get('best_miou', 'unknown')}",
        f"val_loss={val_loss:.4f}",
        f"val_miou={val_miou:.4f}",
        f"saved_samples={saved_samples}",
    ]
    results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results_path


def main():
    args = parse_args()
    config_path = Path(args.config)

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if args.data_root:
        config["data"]["root"] = args.data_root

    output_dir = Path(args.output_dir or "outputs/evaluation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_loader = build_val_loader(
        config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model, checkpoint = build_checkpoint_model(config, args.checkpoint, device)

    loss_config = config["loss"]
    criterion = CombinedSegmentationLoss(
        ce_weight=loss_config.get("ce_weight", 1.0),
        dice_weight=loss_config.get("dice_weight", 1.0),
        class_weights=loss_config.get("class_weights"),
    ).to(device)

    val_loss, val_miou, saved_samples = evaluate(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
        num_classes=config["data"]["num_classes"],
        output_dir=output_dir,
        save_samples=args.save_samples,
        image_mean=config["data"].get("image_mean"),
        image_std=config["data"].get("image_std"),
        limit_batches=args.limit_batches,
    )
    results_path = write_results(
        output_dir=output_dir,
        checkpoint_path=args.checkpoint,
        val_loss=val_loss,
        val_miou=val_miou,
        saved_samples=saved_samples,
        checkpoint=checkpoint,
    )

    print(f"Device: {device}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation mIoU: {val_miou:.4f}")
    print(f"Saved samples: {saved_samples}")
    print(f"Results written to: {results_path}")


if __name__ == "__main__":
    main()
