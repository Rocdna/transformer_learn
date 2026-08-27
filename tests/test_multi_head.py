import torch

from model.multi_head_attention import MultiHeadAttention


# ==================================================
# 参数
# ==================================================

batch_size = 2

seq_len = 5

d_model = 8

num_heads = 2


# ==================================================
# 创建 Multi-Head Attention
# ==================================================

attention = MultiHeadAttention(
    d_model=d_model,
    num_heads=num_heads
)


# ==================================================
# 模拟输入
#
# 假设这是：
#
# Embedding + Positional Encoding
# ==================================================

x = torch.randn(
    batch_size,
    seq_len,
    d_model
)


print("X:")
print(x.shape)


# ==================================================
# Multi-Head Attention
# ==================================================

output = attention(x)


print()

print("Output:")
print(output.shape)