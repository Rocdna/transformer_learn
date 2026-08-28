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
        cross_mask=None,
        self_kv_caches=None
    ):
        # self_kv_caches: 各层缓存的列表 [None, (K,V), ...]，或 None
        new_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = None if self_kv_caches is None else self_kv_caches[i]
            out = layer(x, encoder_output, self_mask, cross_mask, layer_cache)

            if self_kv_caches is not None:
                # 带缓存：每层把 (输出, 更新后缓存) 分别取出来
                x, nc = out
                new_caches.append(nc)
            else:
                x = out

        if self_kv_caches is not None:
            return x, new_caches
        return x