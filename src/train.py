import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from dataset import AgricultureVisionDataset
from losses import CombinedSegmentationLoss
from metrics import mean_iou
from models import build_model
from transforms import build_train_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Train a segmentation model.")
    parser.add_argument("--config", default="configs/baseline_unet_resnet50.yaml")
    parser.add_argument("--data-root", default=None, help="Override dataset root from config.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--resume", default=None, help="Path to a checkpoint to resume from.")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device):
    images = batch["image"].to(device, non_blocking=True)
    masks = batch["mask"].to(device, non_blocking=True)
    valid_masks = batch.get("valid_mask")

    if valid_masks is not None:
        valid_masks = valid_masks.to(device, non_blocking=True)

    return images, masks, valid_masks


def train_one_epoch(model, dataloader, criterion, optimizer, device, limit_batches=None):
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress = tqdm(dataloader, desc="Train", leave=False)
    for batch_index, batch in enumerate(progress, start=1):
        images, masks, valid_masks = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks, valid_mask=valid_masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")

        if limit_batches is not None and batch_index >= limit_batches:
            break

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, dataloader, criterion, device, num_classes, limit_batches=None):
    model.eval()
    total_loss = 0.0
    total_miou = 0.0
    num_batches = 0

    progress = tqdm(dataloader, desc="Val", leave=False)
    for batch_index, batch in enumerate(progress, start=1):
        images, masks, valid_masks = move_batch_to_device(batch, device)

        logits = model(images)
        loss = criterion(logits, masks, valid_mask=valid_masks)
        miou = mean_iou(logits, masks, num_classes=num_classes, valid_mask=valid_masks)

        total_loss += loss.item()
        total_miou += miou.item()
        num_batches += 1
        progress.set_postfix(loss=f"{loss.item():.4f}", miou=f"{miou.item():.4f}")

        if limit_batches is not None and batch_index >= limit_batches:
            break

    return total_loss / num_batches, total_miou / num_batches


def build_dataloaders(config):
    data_config = config["data"]
    training_config = config["training"]
    train_transforms = build_train_transforms(config.get("augmentations"))

    train_dataset = AgricultureVisionDataset(
        root=data_config["root"],
        split="train",
        transforms=train_transforms,
        use_nir=data_config.get("use_nir", False),
        return_valid_mask=data_config.get("return_valid_mask", True),
        image_mean=data_config.get("image_mean"),
        image_std=data_config.get("image_std"),
    )
    val_dataset = AgricultureVisionDataset(
        root=data_config["root"],
        split="val",
        use_nir=data_config.get("use_nir", False),
        return_valid_mask=data_config.get("return_valid_mask", True),
        image_mean=data_config.get("image_mean"),
        image_std=data_config.get("image_std"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config.get("num_workers", 2),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=training_config.get("num_workers", 2),
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_miou, config):
    checkpoint = {
        "epoch": epoch,
        "best_miou": best_miou,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
    }
    torch.save(checkpoint, path)


def build_model_from_config(config, device):
    model_config = config["model"]
    data_config = config["data"]
    extra_model_args = {
        key: value
        for key, value in model_config.items()
        if key
        not in {
            "architecture",
            "encoder",
            "encoder_weights",
            "in_channels",
            "checkpoint_name",
        }
    }
    return build_model(
        architecture=model_config["architecture"],
        encoder=model_config["encoder"],
        encoder_weights=model_config.get("encoder_weights", "imagenet"),
        in_channels=model_config.get("in_channels", 3),
        num_classes=data_config["num_classes"],
        **extra_model_args,
    ).to(device)


def load_checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_miou = checkpoint.get("best_miou", 0.0)
    return start_epoch, best_miou


def main():
    args = parse_args()
    config_path = Path(args.config)

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if args.data_root:
        config["data"]["root"] = args.data_root

    if args.epochs:
        config["training"]["epochs"] = args.epochs

    output_dir = Path(args.output_dir or config["training"].get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(config)

    model_config = config["model"]
    data_config = config["data"]
    model = build_model_from_config(config, device)

    loss_config = config["loss"]
    criterion = CombinedSegmentationLoss(
        ce_weight=loss_config.get("ce_weight", 1.0),
        dice_weight=loss_config.get("dice_weight", 1.0),
        class_weights=loss_config.get("class_weights"),
    ).to(device)

    training_config = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_config["epochs"],
    )

    start_epoch = 1
    best_miou = 0.0
    best_checkpoint_path = output_dir / model_config.get(
        "checkpoint_name",
        "best_unet_resnet50.pth",
    )

    if args.resume:
        start_epoch, best_miou = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            device,
        )
        print(f"Resumed from checkpoint: {args.resume}")
        print(f"Starting epoch: {start_epoch}")
        print(f"Best validation mIoU so far: {best_miou:.4f}")

    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    end_epoch = start_epoch + training_config["epochs"] - 1

    for epoch in range(start_epoch, end_epoch + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            limit_batches=args.limit_train_batches,
        )
        val_loss, val_miou = validate(
            model,
            val_loader,
            criterion,
            device,
            num_classes=data_config["num_classes"],
            limit_batches=args.limit_val_batches,
        )
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{end_epoch} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_miou={val_miou:.4f}"
        )

        if val_miou > best_miou:
            best_miou = val_miou
            save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_miou,
                config,
            )
            print(f"Saved best checkpoint: {best_checkpoint_path}")

    print(f"Best validation mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
