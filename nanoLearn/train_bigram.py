"""
nanoLearn · 训练 bigram(先把训练循环验证对)
在莎士比亚语料上训练 BigramLanguageModel。CPU 几百步即可，秒级完成。

目的：
  1) 验证"数据→前向→loss→backward→优化→生成"整条链路跑通
  2) 看到 loss 从 4.7 掉下来，生成从乱码变成"像单词但不含语法"
  3) 这套训练循环是模板——之后换成完整 GPT 时 training 代码一行不动
"""
import math
import os
import pickle
import time

import torch
from torch.nn import functional as F

from get_batch import DataLoader
from my_bigram import BigramLanguageModel

# ---------- 超参数 ----------
BATCH_SIZE = 32        # 一次喂几个样本(窗口)
BLOCK_SIZE = 256       # 一个窗口多少个 token
EVAL_ITERS = 100       # 评估 loss 时平均几步
LEARNING_RATE = 1e-3
MAX_ITERS = 3000       # 总训练步数
EVAL_INTERVAL = 200    # 每多少步评估+打印一次
SAVE_STEPS = 500       # 每多少步存一份 checkpoint
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_bigram")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HERE = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(1337)

# ---------- 数据 ----------
train_loader = DataLoader("train", BATCH_SIZE, BLOCK_SIZE)
val_loader = DataLoader("val", BATCH_SIZE, BLOCK_SIZE)

# ---------- 模型 / 优化器 ----------
with open(os.path.join(HERE, "data", "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
itos = meta["itos"]
vocab_size = meta["vocab_size"]

model = BigramLanguageModel(vocab_size).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


@torch.no_grad()
def estimate_loss():
    """在 train / val 上各平均 EVAL_ITERS 步的 loss，判断有没有过拟合"""
    out = {}
    model.eval()
    for split, loader in [("train", train_loader), ("val", val_loader)]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = loader.get_batch(DEVICE)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_ckpt(step, t, train_loss):
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({"model": model.state_dict(), "step": step, "loss": train_loss},
               os.path.join(OUT_DIR, f"bigram_step{step}.pt"))


# ---------- 训练循环 ----------
print(f"device={DEVICE}  vocab={vocab_size}")
ema_loss = None
t0 = time.time()
for step in range(MAX_ITERS):
    x, y = train_loader.get_batch(DEVICE)
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # 平滑 loss 显示(参考之前 EMA 的做法，避免单 batch 抖动)
    ema_loss = loss.item() if ema_loss is None else 0.9 * ema_loss + 0.1 * loss.item()

    if (step + 1) % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"step {step+1:5d} | 平滑train {ema_loss:.4f} | "
              f"估train {losses['train']:.4f} | val {losses['val']:.4f} | "
              f"{(time.time()-t0):.0f}s")

    if (step + 1) % SAVE_STEPS == 0:
        save_ckpt(step + 1, time.time() - t0, ema_loss)

print(f"训练完成，用时 {(time.time()-t0):.0f}s，checkpoint 在 {OUT_DIR}/")


# ---------- 采样生成 ----------
def generate(model, max_new_tokens=300):
    model.eval()
    with torch.no_grad():
        start = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
        ids = model.generate(start, max_new_tokens)[0].tolist()
    return "".join(itos[i] for i in ids)


print("\n===== 采样生成(训练后) =====")
print(generate(model))