from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset


CLASS_NAMES = [
    "background",
    "cloud_shadow",
    "double_plant",
    "planter_skip",
    "standing_water",
    "waterway",
    "weed_cluster",
]


class AgricultureVisionDataset(Dataset):
    """Dataset skeleton for Agriculture Vision semantic segmentation."""

    def __init__(self, root, split="train", transforms=None):
        self.root = Path(root)
        self.split = split
        self.transforms = transforms
        self.image_paths = sorted((self.root / split / "images").glob("*.jpg"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = self._load_mask(image_path.stem)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask

    def _load_mask(self, sample_id):
        """Merge class-specific binary masks into one multiclass mask."""
        mask = np.zeros((512, 512), dtype=np.uint8)
        labels_dir = self.root / self.split / "labels"

        for class_index, class_name in enumerate(CLASS_NAMES[1:], start=1):
            class_mask_path = labels_dir / class_name / f"{sample_id}.png"
            if not class_mask_path.exists():
                continue

            class_mask = cv2.imread(str(class_mask_path), cv2.IMREAD_GRAYSCALE)
            mask[class_mask > 0] = class_index

        return mask
