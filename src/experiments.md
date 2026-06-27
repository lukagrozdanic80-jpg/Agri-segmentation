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

## Current Comparison

| Experiment | Best Val mIoU |
|---|---:|
| Baseline U-Net + ResNet-50 | 0.4892 |
| U-Net + ResNet-50 with light augmentations | 0.4751 |
| U-Net + ResNet-50 with augmentations | 0.5331 |
