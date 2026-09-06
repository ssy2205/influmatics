"""Vectorized distance regression and within-batch pairwise ranking losses."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class PairwiseRankingLoss(nn.Module):
    """Mean hinge ranking loss over all ordered pairs i != j.

    Inputs are floating tensors of shape (B,) on the same device. Equal true
    distances are included exactly as specified: sign=0 gives a constant
    `margin` contribution with zero prediction gradient for that pair.
    B=1 has no pairs and returns a differentiable zero. B=0 is rejected.
    Both time and intermediate memory are O(B**2).
    """

    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = float(margin)
        if not math.isfinite(self.margin) or self.margin < 0:
            raise ValueError("margin must be finite and nonnegative")

    def forward(self, pred, true):
        if pred.ndim != 1 or pred.shape != true.shape:
            raise ValueError("pred and true must have matching shape (B,)")
        if not pred.is_floating_point() or not true.is_floating_point():
            raise TypeError("pred and true must be floating-point tensors")
        if pred.device != true.device:
            raise ValueError("pred and true must be on the same device")
        batch_size = pred.numel()
        if batch_size == 0:
            raise ValueError("The batch must be nonempty")
        if batch_size == 1:
            return pred.sum() * 0.0

        pred_diff = pred[:, None] - pred[None, :]
        target_sign = torch.sign(true[:, None] - true[None, :])
        pair_losses = F.relu(self.margin - target_sign * pred_diff)
        diagonal = torch.eye(batch_size, dtype=torch.bool, device=pred.device)
        pair_losses = pair_losses.masked_fill(diagonal, 0.0)
        # (i,j) and (j,i) have the same loss. Their mean is also the i<j mean.
        return pair_losses.sum() / (batch_size * (batch_size - 1))


class CombinedDistanceLoss(nn.Module):
    """SmoothL1Loss(pred, true) + alpha * PairwiseRankingLoss(pred, true).

    Each component uses mean reduction; SmoothL1 uses PyTorch's beta=1.0.
    Returns a scalar tensor suitable for loss.backward().
    """

    def __init__(self, alpha=0.3, margin=0.1):
        super().__init__()
        self.alpha = float(alpha)
        if not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        self.regression_loss = nn.SmoothL1Loss()
        self.ranking_loss = PairwiseRankingLoss(margin=margin)

    def forward(self, pred, true):
        # Validate shapes before regression to prevent accidental broadcasting.
        ranking = self.ranking_loss(pred, true)
        regression = self.regression_loss(pred, true)
        return regression + self.alpha * ranking


if __name__ == "__main__":
    torch.manual_seed(42)
    pred = torch.rand(32, requires_grad=True)
    true = torch.rand(32)
    criterion = CombinedDistanceLoss(alpha=0.3, margin=0.1)
    loss = criterion(pred, true)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    print(f"B=32 | SmoothL1={criterion.regression_loss(pred, true).item():.6f} | "
          f"Ranking={criterion.ranking_loss(pred, true).item():.6f} | "
          f"Combined={loss.item():.6f}")
    print("Forward and backward passed; all prediction gradients are finite.")
