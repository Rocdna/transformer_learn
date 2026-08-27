import os
from datetime import datetime

import torch
import torch.nn as nn
from datasets import load_from_disk
from functools import partial
from torch.utils.data import DataLoader
from torch.optim import AdamW

from model.mask import create_padding_mask, create_causal_mask
from model.transformer import Transformer
from model.config import (
    D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, MAX_SEQ_LEN,
    BATCH_SIZE, LR, EPOCHS,
)
from data.tokenizer import CharTokenizer, WordTokenizer
from data.my_datasets import TranslationDataset
from data.collate import collate_fn
from tqdm import tqdm

# ==================================================
# Step 0. 加载数据
# ==================================================
ROOT = os.path.dirname(os.path.abspath(__file__))
dataset = load_from_disk(os.path.join(ROOT, "data", "dataset_clean"))
train_data = dataset["train"]


def extract_pairs(rows):
    pairs = []
    for row in rows:
        conv = row["conversations"]
        zh = conv[0]["content"].split("--", 1)[1].strip()
        en = conv[1]["content"]
        pairs.append((zh, en))
    return pairs


pairs = extract_pairs(train_data)

# 中 / 英两个 tokenizer（中文按字符，英文按单词）
src_tokenizer = CharTokenizer([p[0] for p in pairs])
tgt_tokenizer = WordTokenizer([p[1] for p in pairs])

# ==================================================
# Step 1. 定义 batch（DataLoader 自动凑 batch + padding）
# ==================================================
train_loader = DataLoader(
    TranslationDataset(pairs, src_tokenizer, tgt_tokenizer),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=partial(
        collate_fn,
        src_pad_id=src_tokenizer.pad_id,
        tgt_pad_id=tgt_tokenizer.pad_id,
    ),
)

# ==================================================
# Step 1b. 模型参数（结构参数统一从 model/config.py 取）
# ==================================================
SRC_VOCAB = src_tokenizer.vocab_size
TGT_VOCAB = tgt_tokenizer.vocab_size

model = Transformer(
    src_vocab_size=SRC_VOCAB,
    tgt_vocab_size=TGT_VOCAB,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    d_ff=D_FF,
    num_layers=NUM_LAYERS,
    max_seq_len=MAX_SEQ_LEN,
)

# ==================================================
# Step 3. 损失函数（ignore_index 让 padding 不计入损失）
# ==================================================
criterion = nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.pad_id)

# 优化器（学习率从 model/config.py 读取）
optimizer = AdamW(model.parameters(), lr=LR)

# 训练步数（每步取一个 batch），数值见 model/config.py 的 EPOCHS
epochs = EPOCHS

# ==================================================
# Step 4. 训练循环
# ==================================================
model.train()
data_iter = iter(train_loader)

# 指数滑动平均 (EMA) 平滑 loss，避免单 batch 波动
ema_loss = 0.0

# tqdm 进度条；set_postfix 让 loss 实时显示在进度条尾部
pbar = tqdm(range(1, epochs + 1), desc="训练")
for epoch in pbar:
    try:
        batch = next(data_iter)
    except StopIteration:
        # 一个 epoch 跑完了，重新开一轮
        data_iter = iter(train_loader)
        batch = next(data_iter)

    src = batch["src"]
    tgt_input = batch["tgt_input"]
    labels = batch["labels"]

    # ==========================================
    # Step 2. 生成两张 mask
    # ==========================================
    # 目标侧 causal mask（下三角，防止看到未来）
    tgt_mask = create_causal_mask(tgt_input.size(1))

    # 源侧 padding mask，unsqueeze 成 [B,1,1,src_len] 以匹配交叉注意力 4D scores
    src_mask = create_padding_mask(src, src_tokenizer.pad_id)
    src_mask = src_mask.unsqueeze(1).unsqueeze(1).float()

    # ==========================================
    # 前向
    # ==========================================
    logits = model(
        src,
        tgt_input,
        src_mask=src_mask,   # encoder 自注意力
        tgt_mask=tgt_mask,   # decoder causal 自注意力
        cross_mask=src_mask, # decoder 交叉注意力
    )
    # logits: [B, tgt_len, TGT_VOCAB]

    # ==========================================
    # Step 3. loss
    # ==========================================
    loss = criterion(
        logits.reshape(-1, TGT_VOCAB),
        labels.reshape(-1),
    )

    # ==========================================
    # 反向传播 + 更新
    # ==========================================
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 更新 EMA 平滑 loss
    ema_loss = 0.9 * ema_loss + 0.1 * loss.item()

    # 进度条尾部实时显示平滑后的 loss
    pbar.set_postfix(loss=f"{ema_loss:.4f}")

    # 每 100 步打印一次平滑 loss
    if epoch % 100 == 0:
        print(f"step {epoch:>5d}  loss = {ema_loss:.4f}")

# ==================================================
# 训练完成，保存模型（以「日_时_分_秒」命名，避免覆盖旧模型）
# ==================================================
timestamp = datetime.now().strftime("%d_%H_%M_%S")
ckpt_path = os.path.join(ROOT, f"transformer_{timestamp}.pt")
torch.save(model.state_dict(), ckpt_path)
print(f"训练完成，模型已保存到 {os.path.basename(ckpt_path)} ✅")