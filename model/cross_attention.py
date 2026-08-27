import math
import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # Q 来自 Decoder
        self.W_Q = nn.Linear(
            d_model,
            d_model
        )

        # K、V 来自 Encoder
        self.W_K = nn.Linear(
            d_model,
            d_model
        )

        self.W_V = nn.Linear(
            d_model,
            d_model
        )

        self.W_O = nn.Linear(
            d_model,
            d_model
        )


    def forward(self, decoder_x, encoder_output, mask=None):
        # Decoder 提供 Q
        Q = self.W_Q(decoder_x)

        # Encoder 提供 K、V
        K = self.W_K(encoder_output)
        V = self.W_V(encoder_output)

        batch_size = decoder_x.size(0)
        target_len = decoder_x.size(1)
        source_len = encoder_output.size(1)

        # [B, target_len, D] → [B, H, target_len, d_k]
        Q = Q.view(batch_size, target_len, self.num_heads, self.d_k).transpose(1,2)
        K = K.view(batch_size, source_len, self.num_heads, self.d_k).transpose(1,2)
        V = V.view(batch_size, source_len, self.num_heads, self.d_k).transpose(1,2)

        # QKᵀ
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # 缩放
        scores = scores / math.sqrt(self.d_k)

        # 如果有 Mask，则屏蔽对应位置
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # 注意力权重
        attention_weights = torch.softmax(scores, dim=-1)
        # Attention × V
        context = torch.matmul(attention_weights, V)

        # [B, H, target_len, d_k] → [B, target_len, D]
        context = context.transpose(1 ,2).contiguous()
        context = context.view(batch_size, target_len, self.d_model)

        output = self.W_O(context)
        return output



