import torch
import torch.nn as nn

class CombinedDistanceLoss(nn.Module):
    """
    Combined loss for distance regression preserving rank ordering.
    Loss = SmoothL1Loss(pred, true) + alpha * PairwiseRankingLoss(pred, true)
    """
    def __init__(self, alpha=0.3, margin=0.1):
        super().__init__()
        self.alpha = alpha
        self.margin = margin
        self.smooth_l1 = nn.SmoothL1Loss()
        
    def forward(self, pred, true):
        # 1. Absolute distance loss
        l1_loss = self.smooth_l1(pred, true)
        
        # 2. Pairwise ranking loss over all O(B^2) pairs in the batch
        # pred and true shape: (B,)
        
        # Compute pairwise differences
        pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)  # (B, B)
        true_diff = true.unsqueeze(1) - true.unsqueeze(0)  # (B, B)
        
        # target_sign: 1 if true_i > true_j, -1 if true_i < true_j, 0 if equal
        target_sign = torch.sign(true_diff)
        
        # margin_loss: max(0, -target_sign * pred_diff + margin)
        # We only care about pairs where true values are strictly ordered (target_sign != 0)
        # For target_sign == 0, the loss should be 0.
        
        ranking_loss_matrix = torch.clamp(-target_sign * pred_diff + self.margin, min=0.0)
        
        # Mask out diagonal (i==j) and pairs with identical true distances
        mask = (target_sign != 0).float()
        
        # Compute mean over valid pairs
        valid_pairs = mask.sum()
        if valid_pairs > 0:
            ranking_loss = (ranking_loss_matrix * mask).sum() / valid_pairs
        else:
            ranking_loss = torch.tensor(0.0, device=pred.device)
            
        return l1_loss + self.alpha * ranking_loss

# Verification
if __name__ == '__main__':
    batch_size = 32
    torch.manual_seed(42)
    pred_distances = torch.rand(batch_size, requires_grad=True)
    true_distances = torch.rand(batch_size)
    
    criterion = CombinedDistanceLoss(alpha=0.3, margin=0.1)
    loss = criterion(pred_distances, true_distances)
    
    print(f"Pred distances shape: {pred_distances.shape}")
    print(f"True distances shape: {true_distances.shape}")
    print(f"Combined Loss: {loss.item():.4f}")
    
    loss.backward()
    print("Backward pass successful. Gradients computed.")
