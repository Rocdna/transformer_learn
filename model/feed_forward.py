import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        # [B, L, d_model]
        x = self.linear1(x)
        # [B, L, d_ff]
        x = self.activation(x)
        # [B, L, d_model]
        x = self.linear2(x)
        return x
