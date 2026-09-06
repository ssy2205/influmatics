import torch
import torch.nn as nn

# Classical H3N2 HA1 Antigenic Site Coordinates (1-based numbering)
KOEL_7_SITES = {145, 155, 156, 158, 159, 189, 193}
EPITOPE_A = {122, 124, 126, 131, 133, 135, 137, 142, 143, 144, 145, 146}
EPITOPE_B = {155, 156, 158, 159, 160, 186, 187, 188, 189, 190, 192, 193, 196, 197}
EPITOPE_C = {45, 46, 47, 48, 50, 51, 53, 54, 275, 276, 278, 279, 280, 297, 299, 300, 305, 307, 308, 309, 310, 311, 312}
EPITOPE_D = {96, 102, 103, 117, 121, 167, 170, 171, 172, 173, 174, 175, 176, 177, 179, 182, 201, 203, 207, 208, 209, 212, 213, 214, 215, 216, 217, 218, 219, 226, 227, 228, 229, 230, 238, 240, 242, 244, 246, 247, 248}
EPITOPE_E = {57, 62, 63, 67, 75, 78, 80, 81, 82, 83, 86, 87, 88, 91, 92, 94, 109, 260, 261, 262, 265}
ALL_EPITOPES = EPITOPE_A | EPITOPE_B | EPITOPE_C | EPITOPE_D | EPITOPE_E

class EpitopeWeightedPooling(nn.Module):
    """
    Bio-informed Domain Weighted Pooling module for H3 HA1 embeddings.
    Assigns:
      - 3.5x weight to the 7 key cluster-transition sites (Koel et al., 2013)
      - 2.0x weight to canonical epitope residues (Sites A-E)
      - 1.0x weight to structural framework residues
    """
    def __init__(self, seq_len=329, w_koel=3.5, w_epitope=2.0, w_framework=1.0):
        super().__init__()
        weights = torch.full((seq_len,), w_framework, dtype=torch.float32)
        for pos in range(1, seq_len + 1):
            if pos in KOEL_7_SITES:
                weights[pos - 1] = w_koel
            elif pos in ALL_EPITOPES:
                weights[pos - 1] = w_epitope
                
        # Register buffer so it moves with the model across CPU/MPS/CUDA
        self.register_buffer("weights", weights.unsqueeze(-1)) # (L, 1)

    def forward(self, token_embeddings):
        """
        token_embeddings: Tensor of shape (B, L, D) or (L, D)
        returns: (B, D) or (D,)
        """
        if token_embeddings.ndim == 2:
            l, d = token_embeddings.shape
            w = self.weights[:l]
            return (token_embeddings * w).sum(dim=0) / w.sum()
        elif token_embeddings.ndim == 3:
            b, l, d = token_embeddings.shape
            w = self.weights[:l].unsqueeze(0) # (1, L, 1)
            return (token_embeddings * w).sum(dim=1) / w.sum()
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {token_embeddings.shape}")

if __name__ == '__main__':
    pooler = EpitopeWeightedPooling(seq_len=329)
    dummy_tokens = torch.randn(329, 1280)
    pooled = pooler(dummy_tokens)
    print("Dummy token input shape:", dummy_tokens.shape)
    print("Epitope-weighted pooled output shape:", pooled.shape)
    print("Total epitope residues weighted:", len(ALL_EPITOPES))
    print("Koel key residues weighted:", len(KOEL_7_SITES))
