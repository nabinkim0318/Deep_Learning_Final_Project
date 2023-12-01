import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, predictions, targets, M):
        present_count = torch.sum(M, axis=(1, 2))
        total_values = M.shape[1] * M.shape[2]
        missing_count = total_values - present_count
        missing_ratio = missing_count / total_values
        sample_weights = 1 - missing_ratio
        losses = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none', pos_weight=self.pos_weight)
        weighted_losses = losses * sample_weights
        return weighted_losses.mean()