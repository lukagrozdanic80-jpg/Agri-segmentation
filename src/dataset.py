from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


CLASS_NAMES = [
    "background",
    "double_plant",
    "planter_skip",
    "standing_water",
    "waterway",
    "weed_cluster",
]

LABEL_DIR_NAMES = {
    "double_plant": "double_plant",
    "planter_skip": "planter_skip",
    "standing_water": "water",
    "waterway": "waterway",
    "weed_cluster": "weed_cluster",
}


class AgricultureVisionDataset(Dataset):
    """Agriculture Vision dataset for multiclass semantic segmentation."""

    def __init__(
        self,
        root,
        split="train",
        transforms=None,
        use_nir=False,
        return_valid_mask=False,
        image_mean=None,
        image_std=None,
    ):
        self.root = Path(root)
        self.split = split
        self.transforms = transforms
        self.use_nir = use_nir
        self.return_valid_mask = return_valid_mask
        self.image_mean = image_mean
        self.image_std = image_std
        self.image_paths = sorted((self.root / split / "images" / "rgb").glob("*.jpg"))

        if not self.image_paths:
            rgb_dir = self.root / split / "images" / "rgb"
            raise FileNotFoundError(f"No RGB images found in: {rgb_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = self._load_image(image_path)
        height, width = image.shape[:2]
        mask = self._load_mask(image_path.stem, shape=(height, width))
        valid_mask = self._load_valid_mask(image_path.stem, shape=(height, width))

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask, masks=[valid_mask])
            image = augmented["image"]
            mask = augmented["mask"]
            valid_mask = augmented["masks"][0]

        sample = {
            "image": self._to_image_tensor(image),
            "mask": torch.as_tensor(mask, dtype=torch.long),
            "id": image_path.stem,
        }

        if self.return_valid_mask:
            sample["valid_mask"] = torch.as_tensor(valid_mask > 0, dtype=torch.bool)

        return sample

    def _load_image(self, image_path):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not self.use_nir:
            return image

        nir_path = self.root / self.split / "images" / "nir" / f"{image_path.stem}.jpg"
        nir = cv2.imread(str(nir_path), cv2.IMREAD_GRAYSCALE)
        if nir is None:
            raise FileNotFoundError(f"Could not read NIR image: {nir_path}")

        return np.dstack([image, nir])

    def _load_mask(self, sample_id, shape):
        """Merge class-specific binary masks into one multiclass mask."""
        mask = np.zeros(shape, dtype=np.uint8)
        labels_dir = self.root / self.split / "labels"

        for class_index, class_name in enumerate(CLASS_NAMES[1:], start=1):
            label_dir_name = LABEL_DIR_NAMES[class_name]
            class_mask_path = labels_dir / label_dir_name / f"{sample_id}.png"
            if not class_mask_path.exists():
                continue

            class_mask = cv2.imread(str(class_mask_path), cv2.IMREAD_GRAYSCALE)
            mask[class_mask > 0] = class_index

        return mask

    def _load_valid_mask(self, sample_id, shape):
        mask_path = self.root / self.split / "masks" / f"{sample_id}.png"
        valid_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if valid_mask is None:
            return np.ones(shape, dtype=np.uint8)

        return valid_mask

    def _to_image_tensor(self, image):
        image = image.astype(np.float32) / 255.0

        if self.image_mean is not None and self.image_std is not None:
            mean = np.asarray(self.image_mean, dtype=np.float32)
            std = np.asarray(self.image_std, dtype=np.float32)
            image = (image - mean) / std

        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image)
