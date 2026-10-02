"""
Neural network model architectures for ZKP Federated Evaluation.
Includes CNN models for MNIST and HAR datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_MNIST(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification.
    Architecture: 2 convolutional layers followed by 2 fully connected layers.
    """
    
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        """Forward pass of the network."""
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN_HAR(nn.Module):
    """
    Convolutional Neural Network for Human Activity Recognition.
    Architecture: 1D convolutions for temporal sensor data processing.
    """
    
    def __init__(self):
        super(CNN_HAR, self).__init__()
        self.conv1 = nn.Conv1d(9, 64, kernel_size=5)  # 9 input channels (sensor features)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 29, 128)  # 29 is the size after convolutions
        self.fc2 = nn.Linear(128, 6)  # 6 activity classes

    def forward(self, x):
        """Forward pass of the network."""
        x = x.view(-1, 9, 128)  # Reshape to (batch_size, channels, time_steps)
        x = F.relu(F.max_pool1d(self.conv1(x), 2))
        x = F.relu(F.max_pool1d(self.conv2(x), 2))
        x = x.view(-1, 128 * 29)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLP_HAR(nn.Module):
    """Classifier for the official HAR 561-dimensional feature vectors."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(561, 256), nn.ReLU(), nn.Dropout(0.5),
                                    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
                                    nn.Linear(128, 6))

    def forward(self, x):
        return self.layers(x)


def get_model(dataset_name):
    """Helper function to get the appropriate model."""
    if dataset_name.lower() == 'mnist':
        return CNN_MNIST()
    elif dataset_name.lower() == 'har':
        return MLP_HAR()
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
