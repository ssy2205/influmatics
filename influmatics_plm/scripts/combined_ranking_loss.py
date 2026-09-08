import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedRankingLoss(nn.Module):
    def __init__(self, lambda_param: float = 0.3, margin: float = 0.1):
        super(CombinedRankingLoss, self).__init__()
        self.lambda_param = lambda_param
        self.margin = margin
        self.smooth_l1 = nn.SmoothL1Loss()
        
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        # Ensure 1D tensors
        pred = pred.view(-1)
        true = true.view(-1)
        
        # 1. SmoothL1Loss
        l1_loss = self.smooth_l1(pred, true)
        
        # 2. PairwiseRankingLoss (Vectorized O(B^2))
        true_diff = true.unsqueeze(1) - true.unsqueeze(0)
        pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
        
        target_sign = torch.sign(true_diff)
        
        # Exclude pairs where true_i == true_j
        mask = (target_sign != 0).float()
        
        # Margin ranking loss: max(0, -target_sign * pred_diff + margin)
        ranking_loss_matrix = F.relu(-target_sign * pred_diff + self.margin)
        
        # Apply mask and average over valid pairs
        valid_pairs_count = mask.sum()
        if valid_pairs_count > 0:
            ranking_loss = (ranking_loss_matrix * mask).sum() / valid_pairs_count
        else:
            ranking_loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
            
        # Combine
        total_loss = l1_loss + self.lambda_param * ranking_loss
        return total_loss

if __name__ == '__main__':
    # Synthetic Data Validation
    batch_size = 32
    torch.manual_seed(42)
    
    pred = torch.randn(batch_size, requires_grad=True)
    true = torch.randn(batch_size)
    
    # Induce a tie to test mask
    true[0] = true[1]
    
    criterion = CombinedRankingLoss(lambda_param=0.3, margin=0.1)
    
    # Forward Pass
    loss = criterion(pred, true)
    print(f"Forward pass successful. Loss: {loss.item():.4f}")
    
    # Backward Pass
    loss.backward()
    print("Backward pass successful.")
    print(f"Gradients computed. Norm: {pred.grad.norm().item():.4f}")
