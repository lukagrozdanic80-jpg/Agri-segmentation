import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class CombinedSegmentationLoss(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.cross_entropy = nn.CrossEntropyLoss()
        self.dice = smp.losses.DiceLoss(mode="multiclass")

    def forward(self, logits, targets):
        ce_loss = self.cross_entropy(logits, targets.long())
        dice_loss = self.dice(logits, targets.long())
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
