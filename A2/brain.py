from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class NNBrain:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, weights: Optional[np.ndarray]=None):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.2
        self.W2 = np.random.randn(hidden_size, hidden_size) * 0.2
        self.W3 = np.random.randn(hidden_size, output_size) * 0.2
        if weights is not None:
            self.set_weights(weights)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.tanh(np.dot(x, self.W1))
        x = np.tanh(np.dot(x, self.W2))
        return np.tanh(np.dot(x, self.W3))

    def get_weights(self) -> np.ndarray:
        return np.concatenate([self.W1.flatten(), self.W2.flatten(), self.W3.flatten()])

    def get_num_weights(self) -> int:
        return self.W1.size + self.W2.size + self.W3.size

    def set_weights(self, flat_weights: np.ndarray) -> None:
        offset = 0
        for W in [self.W1, self.W2, self.W3]:
            size = W.size
            W[:] = flat_weights[offset:offset + size].reshape(W.shape)
            offset += size

class TorchBrain(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, weights: Optional[np.ndarray]=None):
        super().__init__()
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        # Initialize weights if provided
        if weights is not None:
            self.set_weights(weights)

    def forward(self, x: torch.Tensor) -> np.ndarray:

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x.numpy()  # Scale outputs to [-pi/2, pi/2]

    def get_weights(self) -> np.ndarray:
        """Return all weights as a single 1D tensor."""
        params = []
        for p in self.parameters():
            params.append(p.data.view(-1))
        return torch.cat(params).numpy()

    def get_num_weights(self) -> int:
        """Return the total number of weights in the model."""
        return sum(p.numel() for p in self.parameters())

    def set_weights(self, flat_weights: np.ndarray) -> None:
        """Load weights from a 1D tensor or numpy array into the model."""
        flat_weights = torch.tensor(flat_weights, dtype=torch.float32)
        idx = 0
        for p in self.parameters():
            num_params = p.numel()
            p.data = flat_weights[idx:idx+num_params].view_as(p).clone()
            idx += num_params