import torch.nn as nn
import torch

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(
            dim=-1,
            keepdim=True,
            unbiased=False
        )
        x = (x - mean) / torch.sqrt(variance + self.eps )
        return self.gamma * x + self.beta