import torch
import torch.nn as nn

# 词嵌入
class TokenEmbedding(nn.Module):
    """
    Token Embedding。
    作用：
        Token ID
            ↓
        d_model 维向量
    例如：
        vocab_size = 100
        d_model = 512
    那么内部就是：
        [100, 512]
    的可训练矩阵。
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # ==========================================
        # Embedding 层
        # ==========================================
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model
        )

    def forward(self, x):
        """
        x: [batch_size, seq_len]
        例如：
        [
            [1, 20, 31, 42],
            [1, 52, 18, 2]
        ]
        ↓
        Embedding
        ↓
        [batch_size, seq_len, d_model]
        """
        return self.embedding(x)
