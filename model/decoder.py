import torch.nn as nn
from .embedding import TokenEmbedding
from .positional_encoding import PositionalEncoding
from .transformer_decoder import TransformerDecoder


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout=0.1):
        super().__init__()
        # Token → 向量
        self.embedding = TokenEmbedding(vocab_size, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # 多层 decoder
        self.decoder = TransformerDecoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, encoder_output, self_mask=None, cross_mask=None):
        # Token → Embedding
        x = self.embedding(tgt)

        # 加入位置编码
        x = self.pos_encoding(x)

        # 随机失活
        x = self.dropout(x)

        # Decoder Layers
        x = self.decoder(x, encoder_output, self_mask, cross_mask)

        return x
