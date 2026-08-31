"""
nanoLearn · 第一步模型: BigramLanguageModel(热身)
照 Karpathy 视频《Let's build GPT from scratch》的第一个模型。

它"最蠢":只根据当前这一个字符预测下一个，不考虑任何左边历史。
价值：用最少代码跑通 数据→logits→损失→采样生成 整条链路，
      之后再逐步加入 位置编码 + 注意力，升级成完整 GPT。

组件你已经全会了（就是你翻译项目的 Decoder 嫁衣）：
  - nn.Embedding(V, V):每个字符ID -> 一串 logits（"下一个是什么"的打分）
  - view(B*T, C) 打平 + cross_entropy:和训练时 labels 错位一位同理
  - multinomial 采样生成：就是你 Generator._sample_one 的贪心/温度推理去掉
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # 唯一的可学参数：一个 [V, V] 表。第 i 行 = "字符 i 后面紧跟什么"的打分。
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """
        idx:     [B, T] 输入的一批 token ID
        targets: [B, T] 错一位的目标（可选，推理时没有）
        返回 (logits, loss)；loss 仅在训练(targets 有值)时算
        """
        logits = self.token_embedding_table(idx)      # [B, T, V]
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        # 打平成 [B*T, V] 和 [B*T]，对每个 token 位置算下一个字符的交叉熵
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """给定开头 idx [B,T]，自回归地再生成 max_new_tokens 个字符"""
        for _ in range(max_new_tokens):
            logits, _ = self(idx)                      # [B,T,V]
            logits = logits[:, -1, :]                  # 只看最后一个位置 [B,V]
            probs = F.softmax(logits, dim=-1)          # 转成概率
            idx_next = torch.multinomial(probs, num_samples=1)  # 采样（不是 argmax）
            idx = torch.cat([idx, idx_next], dim=1)    # 拼回上下文，继续
        return idx


if __name__ == "__main__":
    from get_batch import DataLoader
    import pickle, os

    # 读词表
    with open(os.path.join(os.path.dirname(__file__), "data", "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    itos = meta["itos"]

    # 没训练，随机初始化的模型直接生成 → 看看"纯乱猜"长什么样
    torch.manual_seed(1337)
    m = BigramLanguageModel(meta["vocab_size"])
    dl = DataLoader("train", batch_size=4, block_size=8)
    x, y = dl.get_batch("cpu")
    logits, loss = m(x, y)          # loss 出来很大(约 ln(65)≈4.17)，因为乱猜
    # 纯乱猜：65 个字符等概率 → 每步熵约 ln(65) ≈ 4.17
    import math
    print(f"未训练 loss: {loss.item():.4f}  (随机乱猜约为 ln(65)={math.log(65):.4f})")

    start = torch.zeros((1, 1), dtype=torch.long)   # 从 ID 0 开始
    ids = m.generate(start, 300)[0].tolist()
    print("随机生成(应是一堆乱码):")
    print("".join(itos[i] for i in ids))