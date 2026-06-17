# Agriculture Vision Semantic Segmentation

Project for semantic segmentation of agricultural field anomalies using the Agriculture Vision dataset.

## Goal

Train and compare semantic segmentation models that classify each pixel into one of the target classes:

- background / normal yield
- cloud shadow
- double plant
- planter skip
- standing water
- waterway
- weed cluster

## Planned Models

- U-Net with ResNet-50 encoder as the baseline
- U-Net with EfficientNet-B3 encoder
- DINO/ViT-based segmentation model for comparison

## Project Structure

```text
configs/      Training and experiment configuration files
data/         Local dataset directory, ignored by Git
notebooks/    Exploration notebooks
outputs/      Local training outputs, ignored by Git
scripts/      Utility scripts
src/          Main project source code
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

The dataset is not committed to Git. Download Agriculture Vision separately and keep it under `data/`.

Dataset: https://huggingface.co/datasets/shi-labs/Agriculture-Vision

Expected local layout:

```text
data/agriculture-vision/
  train/
    images/
    labels/
      cloud_shadow/
      double_plant/
      planter_skip/
      standing_water/
      waterway/
      weed_cluster/
    masks/
    nir/              Optional, used only when use_nir=true
  val/
    images/
    labels/
    masks/
    nir/
```

`src/dataset.py` merges the class-specific binary masks into a single multiclass target mask with class IDs `0..6`. The valid-region mask can also be returned so loss and mIoU ignore pixels outside the evaluated area.

## First Milestones

1. Prepare dataset loading and mask merging.
2. Add exploratory data analysis for class distribution.
3. Train the U-Net + ResNet-50 baseline.
4. Add augmentations and compare metrics.
5. Evaluate with mean IoU.
