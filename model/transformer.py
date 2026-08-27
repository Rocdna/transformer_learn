import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder

class Transformer(nn.Module):
    def __init__(
            self, src_vocab_size, tgt_vocab_size, 
            d_model, num_heads, d_ff, num_layers, 
            max_seq_len,dropout=0.1):
        super().__init__()

        # ==============================
        # Encoder
        # ==============================
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            dropout=dropout
        )

        # ==============================
        # Decoder
        # ==============================
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        # ==============================
        # 输出层
        # d_model → 目标词表大小
        # ==============================
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)


    def forward(self, src, tgt, src_mask=None, tgt_mask=None, cross_mask=None):
        # ==============================
        # ① Encoder
        # ==============================
        encoder_output = self.encoder(src, src_mask)

        # ==============================
        # ② Decoder
        # ==============================
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, cross_mask)

        # ==============================
        # ③ 映射到目标词表
        # ==============================
        logits = self.output_projection(decoder_output)

        return logits













