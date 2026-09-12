"""
AdvRoNIDS Model A: Clean 1D-CNN Classifier (Section 6.2.2)

Architecture Specification:
- Input: Standardized feature vector reshaped to (batch_size, channels=1, length=71)
- Conv1D Block 1: 32 filters, kernel=3, stride=1, padding='same', ReLU, BatchNorm1d(32)
- Conv1D Block 2: 64 filters, kernel=3, stride=1, padding='same', ReLU, BatchNorm1d(64)
- MaxPool1D: pool_size=2
- Conv1D Block 3: 128 filters, kernel=3, stride=1, padding='same', ReLU, BatchNorm1d(128)
- Global Average Pooling: AdaptiveAvgPool1d(1) -> 128-dim vector
- Flatten
- FC1: 128 -> 64, ReLU, Dropout(p=0.3)
- FC2 (output): 64 -> 15 (raw logits for multi-class cross entropy)
"""

import torch
import torch.nn as nn


class AdvRoNIDS_CNN(nn.Module):
    """
    1D Convolutional Neural Network for Network Intrusion Detection
    as defined in AdvRoNIDS Section 6.2.2.
    """

    def __init__(self, num_classes=15, in_channels=1, input_features=71, dropout_rate=0.3):
        super(AdvRoNIDS_CNN, self).__init__()
        self.num_classes = num_classes
        self.input_features = input_features

        # Conv1D Block 1: (batch, 1, 71) -> (batch, 32, 71)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding="same")
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()

        # Conv1D Block 2: (batch, 32, 71) -> (batch, 64, 71)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding="same")
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()

        # MaxPool1D: (batch, 64, 71) -> (batch, 64, 35)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Conv1D Block 3: (batch, 64, 35) -> (batch, 128, 35)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding="same")
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()

        # Global Average Pooling: (batch, 128, 35) -> (batch, 128, 1)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classifier head
        self.fc1 = nn.Linear(128, 64)
        self.relu_fc1 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 71) or (batch_size, 1, 71).
        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, num_classes).
        """
        # Ensure 3D shape (batch_size, channels=1, length=71)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # Pool
        x = self.pool1(x)

        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # Global Average Pooling & Flatten
        x = self.gap(x)
        x = x.squeeze(-1)  # (batch, 128)

        # Fully Connected Classifier
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout(x)
        logits = self.fc2(x)  # (batch, num_classes)

        return logits
