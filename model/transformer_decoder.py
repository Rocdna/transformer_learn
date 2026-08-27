import torch.nn as nn
from .decoder_layer import TransformerDecoderLayer


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        dropout=0.1
    ):
        super().__init__()

        # 堆叠多个 Decoder Layer
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x,
        encoder_output,
        self_mask=None,
        cross_mask=None
    ):
        # 依次通过每个 Decoder Layer
        for layer in self.layers:
            x = layer(
                x,
                encoder_output,
                self_mask,
                cross_mask
            )
        return x