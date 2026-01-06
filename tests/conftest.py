"""
Pytest fixtures for torch-fx-optimizer tests.

Provides reusable test models and inputs for testing optimization passes.
"""

import pytest
import torch
import torch.nn as nn


@pytest.fixture
def simple_mlp():
    """Simple 2-layer MLP for basic testing.

    Architecture:
        - Input: 10 features
        - Hidden: 20 features (ReLU activation)
        - Output: 10 features

    Returns:
        nn.Module: SimpleMLP model instance
    """
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 10)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    return SimpleMLP()


@pytest.fixture
def test_input_mlp():
    """Standard test input for MLP (batch_size=2).

    Returns:
        torch.Tensor: Random tensor of shape (2, 10)
    """
    return torch.randn(2, 10)


@pytest.fixture
def redundant_model():
    """Model with duplicate operations for testing redundancy elimination.

    This model intentionally performs the same operation multiple times
    to test redundant operation elimination passes.

    Returns:
        nn.Module: Model with redundant operations
    """
    class RedundantModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)

        def forward(self, x):
            # Intentionally redundant operations
            y1 = torch.relu(x)
            y2 = torch.relu(x)  # Duplicate of y1
            z = y1 + y2  # Could be optimized to 2 * y1
            return self.fc(z)

    return RedundantModel()


@pytest.fixture
def test_input_redundant():
    """Test input for redundant model.

    Returns:
        torch.Tensor: Random tensor of shape (4, 10)
    """
    return torch.randn(4, 10)


@pytest.fixture
def cnn_model():
    """Simple CNN model with Conv2d + BatchNorm + ReLU.

    Architecture:
        - Conv2d: 3 -> 16 channels, 3x3 kernel
        - BatchNorm2d: 16 channels
        - ReLU activation
        - Conv2d: 16 -> 32 channels, 3x3 kernel
        - AdaptiveAvgPool2d: output size (1, 1)
        - Flatten + Linear: 32 -> 10 classes

    Returns:
        nn.Module: Simple CNN model
    """
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(32, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = torch.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = torch.relu(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    return SimpleCNN()


@pytest.fixture
def test_input_cnn():
    """Test input for CNN model (small image).

    Returns:
        torch.Tensor: Random tensor of shape (2, 3, 32, 32) - 2 images, 32x32 RGB
    """
    return torch.randn(2, 3, 32, 32)


@pytest.fixture
def deep_model():
    """Deeper model for testing gradient checkpointing.

    A sequential model with multiple layers suitable for testing
    activation checkpointing optimizations.

    Returns:
        nn.Module: Deep sequential model
    """
    class DeepModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 20),
                nn.ReLU(),
                nn.Linear(20, 20),
                nn.ReLU(),
                nn.Linear(20, 20),
                nn.ReLU(),
                nn.Linear(20, 10),
            )

        def forward(self, x):
            return self.layers(x)

    return DeepModel()


@pytest.fixture
def test_input_deep():
    """Test input for deep model.

    Returns:
        torch.Tensor: Random tensor of shape (4, 10)
    """
    return torch.randn(4, 10)


@pytest.fixture
def device():
    """Get appropriate device for testing.

    Returns 'cpu' for consistent testing across platforms.
    For GPU testing, tests should explicitly use their own device logic.

    Returns:
        str: Device string ('cpu')
    """
    return 'cpu'


@pytest.fixture
def atol():
    """Absolute tolerance for tensor comparisons.

    Returns:
        float: Absolute tolerance value
    """
    return 1e-6


@pytest.fixture
def rtol():
    """Relative tolerance for tensor comparisons.

    Returns:
        float: Relative tolerance value
    """
    return 1e-5
