import torch
import os
from torch.utils.data import Dataset


class DQLCTDataset(Dataset):
    def __init__(self, processed_dir, split='train'):
        """
        Args:
            processed_dir: Path to 'data/processed'
            split: 'train' (uses both healthy/pd folders)
                   (In a real scenario, you'd split files into train/test folders)
        """
        self.root = processed_dir
        self.files = []
        self.labels = []

        # Load Healthy (Class 0)
        healthy_path = os.path.join(self.root, 'healthy')
        if os.path.exists(healthy_path):
            for f in os.listdir(healthy_path):
                if f.endswith('.pt'):
                    self.files.append(os.path.join(healthy_path, f))
                    self.labels.append(0)  # 0 = Healthy

        # Load PD (Class 1)
        pd_path = os.path.join(self.root, 'pd')
        if os.path.exists(pd_path):
            for f in os.listdir(pd_path):
                if f.endswith('.pt'):
                    self.files.append(os.path.join(pd_path, f))
                    self.labels.append(1)  # 1 = Parkinson's

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load Tensor: Shape [4, Time, Freq]
        data = torch.load(self.files[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Optional: Normalize? (Spectrograms often benefit from Log scaling)
        # For now, let's keep it raw to test the pipeline
        return data, label