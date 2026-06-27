# Experiment Log

## Baseline: U-Net + ResNet-50

Config: `configs/baseline_unet_resnet50.yaml`

Dataset: Agriculture Vision 2021  
Input: RGB images, 512x512  
Classes: background, double_plant, planter_skip, standing_water/water, waterway, weed_cluster  
Loss: Cross-Entropy + Dice Loss  
Optimizer: AdamW  
Scheduler: CosineAnnealingLR  
Batch size: 8  

| Epoch | Train Loss | Val Loss | Val mIoU | Notes |
|---:|---:|---:|---:|---|
| 1 | 0.6991 | 0.3476 | 0.4892 | Best checkpoint |
| 2 | 0.5795 | 0.3448 | 0.4860 | Resumed from epoch 1 checkpoint |

Best checkpoint: epoch 1  
Best validation mIoU: 0.4892

## U-Net + ResNet-50 with Augmentations

Config: `configs/unet_resnet50_aug.yaml`

| Epoch | Train Loss | Val Loss | Val mIoU | Notes |
|---:|---:|---:|---:|---|
| 1 | 0.7567 | 0.4040 | 0.5331 | Best checkpoint |
| 2 | 0.6191 | 0.3338 | 0.4865 | Lower mIoU than epoch 1 |

Evaluation:

| Validation Loss | Validation mIoU | Saved Samples |
|---:|---:|---:|
| 0.4040 | 0.5331 | 8 |

Best checkpoint: epoch 1  
Best validation mIoU: 0.5331  
Improvement over baseline: +0.0439

Conclusion: full augmentations improved validation mIoU compared with the baseline.

## U-Net + ResNet-50 with Light Augmentations

Config: `configs/unet_resnet50_aug_light.yaml`

| Epoch | Train Loss | Val Loss | Val mIoU | Notes |
|---:|---:|---:|---:|---|
| 1 | 0.7297 | 0.3741 | 0.4611 | Saved checkpoint |
| 2 | 0.5675 | 0.3245 | 0.4751 | Best checkpoint |

Evaluation:

| Validation Loss | Validation mIoU | Saved Samples |
|---:|---:|---:|
| 0.3245 | 0.4751 | 8 |

Best checkpoint: epoch 2  
Best validation mIoU: 0.4751  
Difference from baseline: -0.0141  
Difference from full augmentations: -0.0580

Conclusion: light augmentations did not improve over the baseline, while the stronger augmentation setup produced the best ResNet-50 result so far.

## Class Distribution Analysis

Subset: first 2000 training samples
Valid pixels only: yes

| Class ID | Class Name | Pixel Count | Percentage |
|---:|---|---:|---:|
| 0 | background | 470758456 | 89.8525% |
| 1 | double_plant | 9249713 | 1.7655% |
| 2 | planter_skip | 750730 | 0.1433% |
| 3 | standing_water | 9461312 | 1.8059% |
| 4 | waterway | 4570764 | 0.8724% |
| 5 | weed_cluster | 29132900 | 5.5605% |

Conclusion: the dataset is highly imbalanced. The rarest class is `planter_skip`, so class-weighted cross-entropy was tested as a follow-up experiment.

## U-Net + ResNet-50 with Augmentations and Class-Weighted Loss

Config: `configs/unet_resnet50_aug_weighted.yaml`
Setup: partial experiment with 1000 train batches and 300 validation batches

Class weights:

| Class ID | Weight |
|---:|---:|
| 0 | 0.1103 |
| 1 | 0.7869 |
| 2 | 2.7620 |
| 3 | 0.7780 |
| 4 | 1.1194 |
| 5 | 0.4434 |

| Epoch | Train Loss | Val Loss | Val mIoU | Notes |
|---:|---:|---:|---:|---|
| 1 | 1.5507 | 0.9322 | 0.4007 | Partial experiment |

Best validation mIoU: 0.4007

Conclusion: full inverse-sqrt class weights were stable enough to train, but the preliminary mIoU was lower than the standard augmentation setup.

## U-Net + ResNet-50 with Augmentations and Soft Class-Weighted Loss

Config: `configs/unet_resnet50_aug_weighted_soft.yaml`
Setup: partial experiment with 1000 train batches and 300 validation batches

Soft class weights:

| Class ID | Weight |
|---:|---:|
| 0 | 0.5552 |
| 1 | 0.8935 |
| 2 | 1.8810 |
| 3 | 0.8890 |
| 4 | 1.0597 |
| 5 | 0.7217 |

| Epoch | Train Loss | Val Loss | Val mIoU | Notes |
|---:|---:|---:|---:|---|
| 1 | 1.1191 | 0.6044 | 0.4107 | Partial experiment |

Best validation mIoU: 0.4107

Conclusion: soft class weights slightly improved over the full class-weighted setup, but still underperformed the standard augmentation result in preliminary testing.

## U-Net + ResNet-50 with Augmentations and ImageNet Normalization

Config: `configs/unet_resnet50_aug.yaml`
Setup: partial experiment with ImageNet mean/std normalization enabled

Normalization:

| Channel | Mean | Std |
|---|---:|---:|
| R | 0.485 | 0.229 |
| G | 0.456 | 0.224 |
| B | 0.406 | 0.225 |

| Epoch | Train Loss | Val Loss | Val mIoU | Notes |
|---:|---:|---:|---:|---|
| 1 | 1.0233 | 0.5533 | 0.4509 | Partial experiment |

Best validation mIoU: 0.4509

Conclusion: ImageNet normalization was added as correct preprocessing for ImageNet-pretrained encoders. The initial partial result was lower than the best full augmentation run, so a full normalization-only run was not prioritized before testing EfficientNet-B3.

## Current Comparison

| Experiment | Best Val mIoU |
|---|---:|
| Baseline U-Net + ResNet-50 | 0.4892 |
| U-Net + ResNet-50 with light augmentations | 0.4751 |
| U-Net + ResNet-50 with augmentations | 0.5331 |
| U-Net + ResNet-50 with weighted loss, partial | 0.4007 |
| U-Net + ResNet-50 with soft weighted loss, partial | 0.4107 |
| U-Net + ResNet-50 with ImageNet normalization, partial | 0.4509 |
