import torch
import torch.nn as nn

class MiniBatchDiscrimination(nn.Module):
    def __init__(self, num_kernels, kernel_dim):
        super(MiniBatchDiscrimination, self).__init__()
        self.num_kernels = num_kernels
        self.kernel_dim = kernel_dim
        self.W = nn.Parameter(torch.randn(num_kernels, kernel_dim))

    def forward(self, x):
        # x is of shape (batch_size, features)
        x_expanded = x.unsqueeze(1)  # Shape becomes (batch_size, 1, features)
        diff = x_expanded - x.unsqueeze(0)  # Broadcasting for all pairs
        distances = torch.sum(diff ** 2, dim=-1)  # Shape (batch_size, batch_size)
        return torch.mean(torch.exp(-distances))
