import torch

from model.embedding import TokenEmbedding
from model.positional_encoding import PositionalEncoding


# ==================================================
# 参数
# ==================================================

vocab_size = 10

d_model = 4

seq_len = 3


# ==================================================
# Embedding
# ==================================================

embedding = TokenEmbedding(
    vocab_size=vocab_size,
    d_model=d_model
)


# ==================================================
# Position Encoding
# ==================================================

pos_encoding = PositionalEncoding(
    d_model=d_model,
    max_seq_len=10
)


# ==================================================
# 假设一句话：
#
# Token ID：
#
# 我 = 2
# 爱 = 5
# 你 = 7
# ==================================================

tokens = torch.tensor([
    [2, 5, 7]
])


print("Token IDs：")

print(tokens)

print()


# ==================================================
# Token ID
#
# ↓
#
# Embedding
# ==================================================

x = embedding(tokens)


print("Embedding：")

print(x)

print()

print("Embedding shape：")

print(x.shape)


# ==================================================
# 保存一份
# ==================================================

embedding_output = x.clone()


# ==================================================
# 加入位置编码
# ==================================================

x = pos_encoding(x)


print("加入 Position Encoding 后：")

print(x)

print()

print("最终 shape：")

print(x.shape)


pe = pos_encoding.pe[:, :seq_len, :]

print("================================")
print("Embedding")
print("================================")

print(embedding_output)


print()
print("================================")
print("Position Encoding")
print("================================")

print(pe)


print()
print("================================")
print("Embedding + Position")
print("================================")

print(embedding_output + pe)