"""A deliberately small convolutional network for 28 x 28 grayscale digit images."""

import torch
from torch import nn
from torch.nn import functional


class SmallConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_convolution = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)  # 28 -> 26
        self.second_convolution = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)  # 13 -> 11
        self.classifier = nn.Linear(in_features=16 * 5 * 5, out_features=10)  # 11 -> 5 after pooling

    def forward(self, images):
        features = functional.max_pool2d(functional.relu(self.first_convolution(images)), kernel_size=2)
        features = functional.max_pool2d(functional.relu(self.second_convolution(features)), kernel_size=2)
        return self.classifier(features.flatten(start_dim=1))
