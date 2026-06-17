import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class CombinedSegmentationLoss(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0, ignore_index=255):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=ignore_index)

    def forward(self, logits, targets, valid_mask=None):
        if valid_mask is not None:
            targets = targets.clone()
            targets[~valid_mask.bool()] = self.ignore_index

        ce_loss = self.cross_entropy(logits, targets.long())
        dice_loss = self.dice(logits, targets.long())
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
