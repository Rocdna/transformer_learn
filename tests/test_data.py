import os
import torch
from datasets import load_dataset,load_from_disk
from data.tokenizer import CharTokenizer
from data.my_datasets import TranslationDataset
from torch.utils.data import DataLoader
from functools import partial
# 填充
from data.collate import collate_fn
# 掩码
from model.mask import create_padding_mask,create_causal_mask
# 词嵌入
from model.embedding import TokenEmbedding




# 告诉 datasets 库：禁止联网，只用本地缓存
os.environ["HF_DATASETS_OFFLINE"] = "1"

# 加载数据集
# 正常加载，数据会来自 ./data 缓存
# dataset = load_dataset("xyshyniaphy/cn2en_s", cache_dir="./data")

# # 将数据集保存为更高效的格式
# dataset.save_to_disk("./data/dataset_ready")

# 数据路径锚定到项目根目录，不依赖"当前运行目录"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset = load_from_disk(os.path.join(ROOT, "data", "dataset_ready"))
print('-' * 30)

train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]

def extract_pairs(rows):
    pairs=[]
    for row in rows:
        conv = row["conversations"]
        zh = conv[0]["content"].split("--", 1)[1].strip()
        en = conv[1]["content"]
        pairs.append((zh, en))
    return pairs

pairs = extract_pairs(train_data)

# print(f'处理后的数据：{pairs}')

# 构建中文 Tokenizer
src_texts = [
    pair[0]
    for pair in pairs
]
src_tokenizer = CharTokenizer(
    src_texts
)

# ==================================================
# 5. 建立英文 Tokenizer
# ==================================================
tgt_texts = [
    pair[1]
    for pair in pairs
]

tgt_tokenizer = CharTokenizer(
    tgt_texts
)

print("中文词表大小：", src_tokenizer.vocab_size)
print("英文词表大小：", tgt_tokenizer.vocab_size)

# ==================================================
# 6. 创建 TranslationDataset
# ==================================================

train_dataset = TranslationDataset(
    pairs,
    src_tokenizer,
    tgt_tokenizer
)
print("数据集大小：", len(train_dataset))

# ==================================================
# 7. 查看第 0 条
# ==================================================
# sample = train_dataset[0]
# print("\n===== src =====")
# print(sample["src"])
# print(sample["src"].shape)
# print("\n===== tgt_input =====")
# print(sample["tgt_input"])
# print(sample["tgt_input"].shape)
# print("\n===== labels =====")
# print(sample["labels"])
# print(sample["labels"].shape)


# 构建数据集
loader = DataLoader(
    train_dataset,
    batch_size=4,           # 一批四个句子
    shuffle=True,           # 打乱
    collate_fn=partial(     # 填充 pad
        collate_fn,
        src_pad_id=src_tokenizer.pad_id,
        tgt_pad_id=tgt_tokenizer.pad_id
    )
)
batch = next(iter(loader))

# print("src:")
# print(batch["src"])
# print()
# print("tgt_input:")
# print(batch["tgt_input"])
# print()
# print("labels:")
# print(batch["labels"])
# print("src shape:", batch["src"].shape)
# print(
#     "tgt_input shape:",
#     batch["tgt_input"].shape
# )
# print(
#     "labels shape:",
#     batch["labels"].shape
# )

# src_mask = create_padding_mask(
#     batch["src"],
#     src_tokenizer.pad_id
# )
# print("src:")
# print(batch["src"])
# print()
# print("src padding mask:")
# print(src_mask)


# tgt_len = batch["tgt_input"].shape[1]
# tgt_causal_mask = create_causal_mask(
#     tgt_len
# )
# print("tgt causal mask:")
# print(tgt_causal_mask)


embedding = TokenEmbedding(
    vocab_size=100,
    d_model=8
)

x = torch.tensor([
    [1, 20, 31, 42],
    [1, 52, 18, 2]
])

print("输入：")
print(x)
print()
print("输入 shape：")
print(x.shape)

# ==========================================
# Embedding
# ==========================================
output = embedding(x)
print()
print("Embedding 输出：")
print(output)
print()
print("输出 shape：")
print(output.shape)
