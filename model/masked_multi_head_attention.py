import math
import torch
import torch.nn as nn


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        batch_size, seq_len, _ = x.shape
        # X → Q/K/V
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # [B, L, D] → [B, H, L, d_k]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k)

        # [B, L, H, d_k] → [B, H, L, d_k]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # QKᵀ / √d_k
        scores = torch.matmul(Q, K.transpose(-2, -1))

        scores = scores / math.sqrt(self.d_k)

        # 屏蔽未来 Token（mask 为 None 表示不屏蔽）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Attention 权重
        attention_weights = torch.softmax(scores, dim=-1)

        # Attention x V
        context = torch.matmul(attention_weights, V)

        # [B, H, L, d_k] → [B, L, H, d_k]
        context = context.transpose(1 ,2).contiguous()

        # [B, L, H, d_k] → [B, L, D]
        context = context.view(batch_size, seq_len, self.d_model)

        # 输出投影
        output = self.W_O(context)
        return output


