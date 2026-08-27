import torch

from model.transformer import Transformer


# ==============================
# 模型参数
# ==============================

src_vocab_size = 20
tgt_vocab_size = 30

d_model = 16
num_heads = 4
d_ff = 64

num_layers = 2
max_seq_len = 20


# ==============================
# 创建模型
# ==============================

model = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,

    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,

    num_layers=num_layers,
    max_seq_len=max_seq_len
)


# ==============================
# 模拟输入
# ==============================

# Batch = 2
# Source 长度 = 5

src = torch.tensor([
    [1, 2, 3, 4, 5],
    [1, 2, 3, 6, 7]
])


# Batch = 2
# Target 长度 = 4

tgt = torch.tensor([
    [1, 2, 3, 4],
    [1, 2, 5, 6]
])


# ==============================
# Target Causal Mask
# ==============================

tgt_len = tgt.size(1)

tgt_mask = torch.tril(
    torch.ones(
        tgt_len,
        tgt_len
    )
)


# ==============================
# 前向传播
# ==============================

logits = model(
    src,
    tgt,
    tgt_mask=tgt_mask
)


print("src:", src.shape)
print("tgt:", tgt.shape)
print("logits:", logits.shape)