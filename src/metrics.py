import torch


def mean_iou(logits, targets, num_classes, eps=1e-7):
    predictions = torch.argmax(logits, dim=1)
    ious = []

    for class_id in range(num_classes):
        pred_mask = predictions == class_id
        target_mask = targets == class_id

        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()

        if union > 0:
            ious.append((intersection + eps) / (union + eps))

    if not ious:
        return torch.tensor(0.0, device=logits.device)

    return torch.stack(ious).mean()
