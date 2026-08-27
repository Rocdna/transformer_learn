import torch.nn as nn

from .embedding import TokenEmbedding
from .positional_encoding import PositionalEncoding
from .transformer_encoder import TransformerEncoder

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout=0.1):
        super().__init__()
        # Token → 向量
        self.embedding = TokenEmbedding(vocab_size, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # 多层 encoder
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        # Token -> Embedding
        x = self.embedding(src)

        # 加入位置编码
        x = self.pos_encoding(x)
        # 随机失活
        x = self.dropout(x)

        # Encoder Layers
        x = self.encoder(x, mask)

        return x
