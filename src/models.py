import segmentation_models_pytorch as smp


def build_model(architecture, encoder, num_classes, encoder_weights="imagenet"):
    architecture = architecture.lower()

    if architecture == "unet":
        return smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )

    raise ValueError(f"Unsupported architecture: {architecture}")
