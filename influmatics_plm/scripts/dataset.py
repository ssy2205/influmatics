import torch
from torch.utils.data import Dataset

class DistanceDataset(Dataset):
    """
    Pairwise distance dataset with strict key validation.
    No silent zero-vector fallbacks.
    """
    def __init__(self, df, embeddings):
        self.df = df.reset_index(drop=True)
        self.embeddings = embeddings
        
        # Validate that all required keys exist
        v1_set = set(self.df['virus1'])
        v2_set = set(self.df['virus2'])
        all_v = v1_set.union(v2_set)
        missing = [v for v in all_v if v not in self.embeddings]
        if missing:
            raise KeyError(f"DistanceDataset found {len(missing)} missing virus keys in embeddings. Example: {missing[:5]}")
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        v1, v2 = row['virus1'], row['virus2']
        dist = float(row['distance'])
        
        emb1 = self.embeddings[v1]
        emb2 = self.embeddings[v2]
        
        return emb1, emb2, torch.tensor(dist, dtype=torch.float32)
