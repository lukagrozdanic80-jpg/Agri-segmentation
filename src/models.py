import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from transformers import AutoConfig, AutoModel


class DinoViTSegmentationModel(nn.Module):
    def __init__(
        self,
        model_name,
        num_classes,
        in_channels=3,
        pretrained=True,
        freeze_backbone=True,
        decoder_channels=256,
        trust_remote_code=True,
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError("DINO ViT backbones expect RGB input with 3 channels.")

        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            )
        else:
            config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            )
            self.backbone = AutoModel.from_config(config, trust_remote_code=trust_remote_code)

        self.freeze_backbone = freeze_backbone
        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Could not infer hidden_size from the DINO backbone config.")

        half_channels = max(decoder_channels // 2, num_classes)
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(hidden_size, decoder_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, half_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(half_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(half_channels, num_classes, kernel_size=1),
        )

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _run_backbone(self, images):
        try:
            return self.backbone(
                pixel_values=images,
                interpolate_pos_encoding=True,
            )
        except TypeError:
            return self.backbone(pixel_values=images)

    def _tokens_to_feature_map(self, tokens, image_height, image_width):
        patch_size = getattr(self.backbone.config, "patch_size", 16)
        if isinstance(patch_size, (tuple, list)):
            patch_height, patch_width = patch_size
        else:
            patch_height = patch_width = patch_size

        grid_height = image_height // patch_height
        grid_width = image_width // patch_width
        expected_tokens = grid_height * grid_width

        if tokens.shape[1] == expected_tokens + 1:
            tokens = tokens[:, 1:, :]
        elif tokens.shape[1] != expected_tokens:
            if tokens.shape[1] > expected_tokens:
                tokens = tokens[:, -expected_tokens:, :]
            else:
                grid_size = int(math.sqrt(tokens.shape[1]))
                grid_height = grid_width = grid_size
                expected_tokens = grid_height * grid_width
                tokens = tokens[:, :expected_tokens, :]

        batch_size, _, hidden_size = tokens.shape
        features = tokens.transpose(1, 2).reshape(
            batch_size,
            hidden_size,
            grid_height,
            grid_width,
        )
        return features

    def forward(self, images):
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self._run_backbone(images)
        else:
            outputs = self._run_backbone(images)

        tokens = getattr(outputs, "last_hidden_state", None)
        if tokens is None and isinstance(outputs, dict):
            tokens = outputs.get("last_hidden_state")
        if tokens is None:
            raise ValueError("DINO backbone did not return last_hidden_state tokens.")

        features = self._tokens_to_feature_map(tokens, images.shape[-2], images.shape[-1])
        logits = self.segmentation_head(features)
        return F.interpolate(
            logits,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


def build_model(
    architecture,
    encoder,
    num_classes,
    encoder_weights="imagenet",
    in_channels=3,
    **kwargs,
):
    architecture = architecture.lower()

    if architecture == "unet":
        return smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )

    if architecture in {"dino_vit", "dinov3_vit"}:
        pretrained = encoder_weights not in {None, "none", False}
        return DinoViTSegmentationModel(
            model_name=encoder,
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=pretrained,
            freeze_backbone=kwargs.get("freeze_backbone", True),
            decoder_channels=kwargs.get("decoder_channels", 256),
            trust_remote_code=kwargs.get("trust_remote_code", True),
        )

    raise ValueError(f"Unsupported architecture: {architecture}")
