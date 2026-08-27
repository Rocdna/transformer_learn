import torch

# 让每个句子的长度一直，填充 Padding
def collate_fn(batch, src_pad_id, tgt_pad_id):
    """
    把 Dataset 返回的多个样本
    组成一个 Batch。
    batch 里面类似：
    [
        {
            "src": tensor([...]),
            "tgt_input": tensor([...]),
            "labels": tensor([...])
        },
        {
            "src": tensor([...]),
            "tgt_input": tensor([...]),
            "labels": tensor([...])
        }
    ]
    最终变成：
    {
        "src": Tensor,
        "tgt_input": Tensor,
        "labels": Tensor
    }
    """

    # ==================================================
    # 1. 找出这个 Batch 中最长的 src
    # ==================================================

    max_src_len = max(
        len(item["src"])
        for item in batch
    )

    # ==================================================
    # 2. 找出这个 Batch 中最长的 tgt
    # ==================================================
    max_tgt_len = max(
        len(item["tgt_input"])
        for item in batch
    )

    # ==================================================
    # 3. 获取 PAD ID
    # ==================================================
    # 注意：
    # src 和 tgt 使用的是两个 tokenizer
    # 所以理论上它们都有自己的 PAD ID。
    # 我们这里分别使用。
    # src_pad_id = 0
    # tgt_pad_id = 0

    # ==================================================
    # 4. 创建 Batch Tensor
    # 先全部填充 PAD
    # ==================================================
    src_batch = torch.full(
        ( len(batch),max_src_len ),
        src_pad_id,
        dtype=torch.long
    )

    tgt_input_batch = torch.full(
        ( len(batch), max_tgt_len ),
        tgt_pad_id,
        dtype=torch.long
    )

    labels_batch = torch.full(
        ( len(batch), max_tgt_len ),
        tgt_pad_id,
        dtype=torch.long
    )

    # ==================================================
    # 5. 把真正的数据复制进去
    # ==================================================
    for i, item in enumerate(batch):
        src = item["src"]
        tgt_input = item["tgt_input"]
        labels = item["labels"]
        # ----------------------------------------------
        # src
        # ----------------------------------------------
        src_batch[i, :len(src)] = src

        # ----------------------------------------------
        # tgt_input
        # ----------------------------------------------
        tgt_input_batch[i, :len(tgt_input)] = tgt_input

        # ----------------------------------------------
        # labels
        # ----------------------------------------------
        labels_batch[i, :len(labels)] = labels

    # ==================================================
    # 6. 返回 Batch
    # ==================================================
    return {
        "src": src_batch,
        "tgt_input": tgt_input_batch,
        "labels": labels_batch
    }










