"""
nanoLearn · 第一步:构建数据集
把原始文本 input.txt 预处理成字符级的 token ID 二进文件，供 GPT 训练用。

流程(照官方 nanoGPT/data/shakespeare_char/prepare.py):
  1. 读入整份文本
  2. 统计所有出现过的字符 → 建词表(字符→ID / ID→字符)
  3. 按 9:1 切成 train / val
  4. 每一个字符都映射成一个整数(ID)
  5. 存成 uint16 的 .bin(numpy 紧凑二进制),并 pickle 一份词表 meta 备用

产物(data/ 下): train.bin、val.bin、meta.pkl
"""
import os
import pickle
import numpy as np

# ---- 1. 读文本 ----
HERE = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(HERE, "data", "input.txt")

with open(input_path, "r", encoding="utf-8") as f:
    data = f.read()
print(f"数据集字符数: {len(data):,}")

# ---- 2. 建字符词表 ----
chars = sorted(list(set(data)))          # 去重 + 排序,得到"出现过哪些字符"
stoi = {ch: i for i, ch in enumerate(chars)}   # 字符 -> ID
itos = {i: ch for i, ch in enumerate(chars)}   # ID -> 字符
vocab_size = len(chars)
print(f"词表大小: {vocab_size}")
print("出现的字符:", "".join(chars))


def encode(s: str):
    """字符串 -> ID 列表"""
    return [stoi[c] for c in s]


def decode(l):
    """ID 列表 -> 字符串"""
    return "".join(itos[i] for i in l)


# ---- 3. 切分 9:1 ----
n = len(data)
cut = int(n * 0.9)
train_text, val_text = data[:cut], data[cut:]

# ---- 4. 编码成整数 ----
train_ids = np.array(encode(train_text), dtype=np.uint16)
val_ids = np.array(encode(val_text), dtype=np.uint16)
print(f"train token 数: {len(train_ids):,}")
print(f"val   token 数: {len(val_ids):,}")

# ---- 5. 导出:紧凑二进制 + 词表 ----
out_dir = os.path.join(HERE, "data")
train_ids.tofile(os.path.join(out_dir, "train.bin"))
val_ids.tofile(os.path.join(out_dir, "val.bin"))

with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
    pickle.dump({"vocab_size": vocab_size, "itos": itos, "stoi": stoi}, f)

print("完成:data/train.bin、val.bin、meta.pkl")