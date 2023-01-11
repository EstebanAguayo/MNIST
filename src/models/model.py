import torch.nn.functional as F
from torch import nn, Tensor
import pytest

class MyAwesomeModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)

        self.output = nn.Linear(64, output_size)

        self.relu = nn.ReLU()

    def forward(self, x: Tensor)-> Tensor: #Hardcoded inputs, not good:
        if x.ndim != 2:
            raise ValueError('Expected input to a 2D tensor')
        if x.shape[1] != 784:
            raise ValueError('Expected each sample to have shape 784 input values')
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        x = F.log_softmax(self.output(x), dim=1)
        return x


