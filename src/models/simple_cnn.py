import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDQLCTNet(nn.Module):
    def __init__(self, input_channels=4, num_classes=2):
        super().__init__()

        # 1. Feature Extractor (Convolutional Layers)
        # Input: [Batch, 4, Time, Freq]
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)  # Downsample by 2x

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # 2. Classifier (Global Average Pooling + Linear)
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Smashes (64, H, W) -> (64, 1, 1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: [Batch, 4, Time, Freq]

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # [Batch, 64]

        # Classification
        x = self.fc(x)
        return x