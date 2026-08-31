"""
nanoLearn · 数据加载器
从 train.bin / val.bin 一维 ID 序列里,随机切出一个个
   输入块 x = data[i : i+block_size]
   目标块 y = data[i+1 : i+1+block_size]   (错一位，预测下一个字符)
返回形状都是 [batch_size, block_size] 的 int64 张量。
"""
import os
import torch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


class DataLoader:
    def __init__(self, split, batch_size=32, block_size=256, data_dir=None):
        self.split = split
        self.batch_size = batch_size
        self.block_size = block_size
        data_dir = data_dir or os.path.join(HERE, "data")
        # np.memmap:把 .bin 映射成一维数组，按需读，不整份进内存
        self.data = np.memmap(
            os.path.join(data_dir, f"{split}.bin"),
            dtype=np.uint16, mode="r",
        )

    def get_batch(self, device):
        """随机抽 batch_size 个起点，各取一段 block_size，输入错一位成目标"""
        data = self.data
        n = len(data)
        # 起点索引：保证 i + block_size <= n
        ix = torch.randint(n - self.block_size, (self.batch_size,))
        # 每个起点一段输入
        x = torch.stack([
            torch.from_numpy(data[i: i + self.block_size].astype(np.int64))
            for i in ix
        ])
        # 同一段错一位作为目标（位置 j 预测 j+1）
        y = torch.stack([
            torch.from_numpy(data[i + 1: i + 1 + self.block_size].astype(np.int64))
            for i in ix
        ])
        return x.to(device), y.to(device)


if __name__ == "__main__":
    dl = DataLoader("train", batch_size=4, block_size=8)
    x, y = dl.get_batch("cpu")
    print("x shape:", tuple(x.shape), " y shape:", tuple(y.shape))
    print("x[0]:", x[0].tolist())
    print("y[0]:", y[0].tolist())
    print("y 是 x 往后错一位?", (x[:, 1:] == y[:, :-1]).all().item())