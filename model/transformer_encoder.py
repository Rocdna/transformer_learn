import torch.nn as nn

from .encoder_layer import TransformerEncoderLayer

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # 堆叠多个 Encoder Layer
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        # 依次通过每一层，串行
        for layer in self.layers:
            x = layer(x, mask)

        return x





















