import albumentations as A


def build_train_transforms(config):
    if not config or not config.get("enabled", False):
        return None

    return A.Compose(
        [
            A.HorizontalFlip(p=config.get("horizontal_flip_p", 0.5)),
            A.VerticalFlip(p=config.get("vertical_flip_p", 0.5)),
            A.RandomRotate90(p=config.get("rotate90_p", 0.5)),
            A.RandomBrightnessContrast(
                brightness_limit=config.get("brightness_limit", 0.2),
                contrast_limit=config.get("contrast_limit", 0.2),
                p=config.get("brightness_contrast_p", 0.3),
            ),
            A.GaussNoise(p=config.get("gauss_noise_p", 0.2)),
        ]
    )
