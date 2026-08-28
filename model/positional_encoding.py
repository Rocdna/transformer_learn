import math

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Transformer 经典的 Sinusoidal Positional Encoding。
    给每一个 Token 添加位置信息。
    输入：

        [batch_size, seq_len, d_model]
    输出：
        [batch_size, seq_len, d_model]
    """

    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        # ==================================================
        # 创建位置编码矩阵
        # [max_seq_len, d_model]
        # 例如：[5000, 128]
        # ==================================================
        pe = torch.zeros(max_seq_len, d_model)

        # ==================================================
        # position
        # [0, 1, 2, 3, ...]
        # shape:
        # [max_seq_len, 1]
        # ==================================================
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        # ==================================================
        # 计算公式中的：
        # 1 / 10000^(2i / d_model)
        # ==================================================
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # ==================================================
        # 偶数维度使用 sin
        # 0, 2, 4, 6, ...
        # ==================================================
        pe[:, 0::2] = torch.sin(position * div_term)

        # ==================================================
        # 奇数维度使用 cos
        # 1, 3, 5, 7, ...
        # ==================================================
        pe[:, 1::2] = torch.cos(position * div_term)

        # ==================================================
        # 增加 batch 维度
        # [max_seq_len, d_model]
        #            ↓
        # [1, max_seq_len, d_model]
        # ==================================================
        pe = pe.unsqueeze(0)

        # ==================================================
        # register_buffer
        # ==================================================
        # 这不是模型参数。
        # 但是它需要跟着模型一起：
        # CPU → GPU
        # 保存到 state_dict
        # 所以使用 register_buffer。
        # ==================================================
        self.register_buffer("pe", pe)

    def forward(self, x, offset=0):
        """
        x: [batch_size, seq_len, d_model]
        offset: 位置偏移（KV cache 增量解码时，token 的绝对位置从 offset 开始）
        """
        seq_len = x.size(1)
        # ==============================================
        # 取出当前片段对应的位置编码
        # 整句解码时 offset=0 -> pe[:, :seq_len, :]
        # 增量解码时 offset=当前位置 -> pe[:, offset:offset+seq_len, :]
        # ==============================================
        x = x + self.pe[:, offset:offset + seq_len, :]
        return x
