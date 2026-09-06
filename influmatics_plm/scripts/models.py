import torch
import torch.nn as nn

class DistanceHead(nn.Module):
    def __init__(self, emb_dim=1280, hidden_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, u, v):
        diff = torch.abs(u - v)
        mult = u * v
        x = torch.cat([u, v, diff, mult], dim=-1)
        return self.mlp(x).squeeze(-1)
