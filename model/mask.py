import torch


# 填充掩码
def create_padding_mask(
    tokens,
    pad_id
):
    """
    创建 Padding Mask。
    tokens: [batch_size, seq_len]
    例如：
    [
        [1, 20, 30, 2, 0, 0],
        [1, 50, 60, 70, 2, 0]
    ]
    pad_id = 0
    返回：
    [
        [True, True, True, True, False, False],
        [True, True, True, True, True, False]
    ]
    True：
        这个位置是真实 Token，可以看。
    False：
        这个位置是 PAD，不应该看。
    """
    return tokens != pad_id

# 因果掩码
def create_causal_mask(seq_len):
    """
    创建 Causal Mask。
    防止 Decoder 看到未来 Token。
    seq_len = 4
    返回：
        True  False False False
        True  True  False False
        True  True  True  False
        True  True  True  True
    True：
        可以看到

    False：
        不允许看到
    """
    # 下三角矩阵
    return torch.tril(
        torch.ones(
            seq_len,
            seq_len,
            dtype=torch.bool
        )
    )



