import torch.nn as nn

from .masked_multi_head_attention import MaskedMultiHeadAttention
from .cross_attention import CrossAttention
from .feed_forward import FeedForward
from .layer_norm import LayerNorm


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # ① Masked Self-Attention
        self.self_attention = MaskedMultiHeadAttention(d_model, num_heads)

        # ② Cross-Attention
        self.cross_attention = CrossAttention(d_model, num_heads)

         # ③ FFN
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff)

        # 三个子模块各有一个 LayerNorm
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)


    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        # ==================================
        # 1. Masked Self-Attention
        # ==================================
        attention_output = self.self_attention(x, self_mask)

        # 残差连接 + LayerNorm
        x = self.norm1(x + self.dropout(attention_output))

        # ==================================
        # 2. Cross-Attention
        # ==================================
        attention_output = self.cross_attention(x, encoder_output, cross_mask)

        # 残差连接 + LayerNorm
        x = self.norm2(x + self.dropout(attention_output))

        # ==================================
        # 3. FFN
        # ==================================
        ffn_output = self.ffn(x)

        # 残差连接 + LayerNorm
        x = self.norm3(x + self.dropout(ffn_output))

        return x