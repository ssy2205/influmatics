import torch
from torch.utils.data import Dataset
import pandas as pd

class DistanceDataset(Dataset):
    def __init__(self, df, embeddings):
        self.virus1 = df['virus1'].values
        self.virus2 = df['virus2'].values
        self.distance = df['distance'].values
        self.embeddings = embeddings
        
    def __len__(self):
        return len(self.distance)
        
    def __getitem__(self, idx):
        v1 = self.virus1[idx]
        v2 = self.virus2[idx]
        
        emb1 = self.embeddings.get(v1, torch.zeros(1280))
        emb2 = self.embeddings.get(v2, torch.zeros(1280))
        dist = torch.tensor(self.distance[idx], dtype=torch.float32)
        
        return emb1, emb2, dist
