# Log 

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

## U-Net + ResNet-50 sa augmentacijama

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

Zakljucak: pune augmentacije su poboljsale validation mIoU u odnosu na baseline.

## U-Net + ResNet-50 sa light augmentacijama

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

Zakljucak: light augmentacije nisu poboljsale rezultat u odnosu na baseline, dok je jaci setup augmentacija dao najbolji ResNet-50 rezultat do tada.

## Analiza raspodjele klasa

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

Zakljucak: dataset je veoma neuravnotezen. Najrjedja klasa je `planter_skip`, pa je class-weighted cross-entropy testiran kao sljedeci eksperiment.

## U-Net + ResNet-50 sa augmentacijama i class-weighted loss

Config: `configs/unet_resnet50_aug_weighted.yaml`
Setup: parcijalni eksperiment sa 1000 train batch-eva i 300 validation batch-eva

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

Zakljucak: puni inverse-sqrt class weights su bili dovoljno stabilni za trening, ali je pocetni mIoU bio nizi nego kod standardnog augmentacionog setupa.

## U-Net + ResNet-50 sa augmentacijama i soft class-weighted loss

Config: `configs/unet_resnet50_aug_weighted_soft.yaml`
Setup: parcijalni eksperiment sa 1000 train batch-eva i 300 validation batch-eva

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

Zakljucak: soft class weights su malo poboljsali rezultat u odnosu na puni weighted setup, ali su i dalje bili slabiji od standardnog augmentacionog rezultata u pocetnom testiranju.

## U-Net + ResNet-50 sa augmentacijama i ImageNet normalizacijom

Config: `configs/unet_resnet50_aug.yaml`
Setup: parcijalni eksperiment sa ukljucenom ImageNet mean/std normalizacijom

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

Zakljucak: ImageNet normalizacija je dodata kao ispravan preprocessing za ImageNet-pretrained encodere. Pocetni parcijalni rezultat bio je nizi od najboljeg full augmentation eksperimenta, pa full run samo sa normalizacijom nije bio prioritet prije testiranja EfficientNet-B3 modela.

## U-Net + EfficientNet-B3 sa augmentacijama

Config: `configs/unet_efficientnet_b3_aug.yaml`
Setup: full eksperiment sa 2 epohe na Agriculture Vision train/val splitu
Input: RGB images

| Epoch | Train Loss | Val Loss | Val mIoU | Notes |
|---:|---:|---:|---:|---|
| 1 | 0.6917 | 0.3327 | 0.4535 | Saved checkpoint |
| 2 | 0.5081 | 0.2673 | 0.5587 | Best checkpoint |

Evaluation:

| Validation Loss | Validation mIoU | Saved Samples |
|---:|---:|---:|
| 0.2673 | 0.5587 | 8 |

Best checkpoint: epoch 2  
Best validation mIoU: 0.5587  
Improvement over best ResNet-50 augmentation result: +0.0256

Zakljucak: EfficientNet-B3 je postigao najbolji validation mIoU do tada i poboljsao rezultat u odnosu na prethodni ResNet-50 augmentacioni eksperiment.

## DINOv3 ViT-S/16 parcijalni trening

Config: `configs/dino_v3_vits16_seg.yaml`
Checkpoint: `outputs_dino_v3_partial/best_dino_v3_vits16.pth`
Setup: parcijalni trening sa 1000 train batch-eva i full validation evaluacijom
Input: RGB images
Backbone: pretrained `facebook/dinov3-vits16-pretrain-lvd1689m`

Training result:

| Train Loss | Val Loss | Val mIoU | Notes |
|---:|---:|---:|---|
| 0.5846 | 0.4154 | 0.7298 | Partial training validation during training |

Full validation evaluation:

| Validation Loss | Validation mIoU | Saved Samples |
|---:|---:|---:|
| 0.3453 | 0.7169 | 8 |

Best validation mIoU: 0.7169
Improvement over EfficientNet-B3 result: +0.1582

Zakljucak: DINOv3 je dao najbolji rezultat do tada. Cak i sa parcijalnim treningom, nadmasio je prethodni EfficientNet-B3 eksperiment na full validation evaluaciji.

## DINOv3 ViT-S/16 full 1 epoch

Config: `configs/dino_v3_vits16_seg.yaml`
Checkpoint: `outputs_dino_v3_full_1ep/best_dino_v3_vits16.pth`
Setup: full 1 epoch trening na Agriculture Vision train/val splitu
Input: RGB images
Backbone: pretrained `facebook/dinov3-vits16-pretrain-lvd1689m`

Training result:

| Epoch | Train Loss | Val Loss | Val mIoU | Notes |
|---:|---:|---:|---:|---|
| 1 | 0.4237 | 0.2861 | 0.6943 | Best checkpoint |

Evaluation:

| Validation Loss | Validation mIoU | Saved Samples |
|---:|---:|---:|
| 0.2861 | 0.6943 | 8 |

Best validation mIoU: 0.6943
Improvement over EfficientNet-B3 result: +0.1356

Zakljucak: full 1 epoch DINOv3 run je potvrdio da DINOv3 arhitektura nadmasuje CNN eksperimente. Rezultat je bio malo nizi od parcijalnog DINOv3 checkpointa evaluiranog na full validation skupu, ali je ostao najbolji potpuno trenirani single-epoch eksperiment.

## Trenutno poredjenje

| Experiment | Best Val mIoU |
|---|---:|
| Baseline U-Net + ResNet-50 | 0.4892 |
| U-Net + ResNet-50 with light augmentations | 0.4751 |
| U-Net + ResNet-50 with augmentations | 0.5331 |
| U-Net + ResNet-50 with weighted loss, partial | 0.4007 |
| U-Net + ResNet-50 with soft weighted loss, partial | 0.4107 |
| U-Net + ResNet-50 with ImageNet normalization, partial | 0.4509 |
| U-Net + EfficientNet-B3 with augmentations | 0.5587 |
| DINOv3 ViT-S/16 partial training, full validation | 0.7169 |
| DINOv3 ViT-S/16 full 1 epoch | 0.6943 |
