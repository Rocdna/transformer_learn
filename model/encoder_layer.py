import torch.nn as nn

from .multi_head_attention import MultiHeadAttention
from .feed_forward import FeedForward
from .layer_norm import LayerNorm

# transformer 编码器
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)

        self.ffn = FeedForward(d_model, d_ff)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        # 随机失活
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # ==============================
        # 1. Self-Attention
        # ==============================
        attention_output = self.attention(x, mask)

        # 残差连接 + LayerNorm
        x = self.norm1(x + self.dropout(attention_output))
        # ==============================
        # 2. FFN
        # ==============================
        ffn_output = self.ffn(x)
        # 残差连接 + LayerNorm
        x = self.norm2(x + self.dropout(ffn_output))
        return x








