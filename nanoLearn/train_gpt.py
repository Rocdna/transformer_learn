"""
nanoLearn · 训练 my_gpt(带位置编码 + 注意力)
训练循环和 train_bigram 完全一样——再次验证"训练模板可复用"。
区别只在模型从 bigram 换成了带注意力的 GPT,forward 内部变复杂,训练代码一行没动。
"""
import math
import os
import pickle
import time
import torch
from torch.nn import functional as F

from get_batch import DataLoader
from my_gpt import GPT, GPTConfig

# ---------- 超参数(官方 shakespeare_char 尺寸,CPU 上较慢,后台跑) ----------
BATCH_SIZE = 32
BLOCK_SIZE = 256
EVAL_ITERS = 30
LEARNING_RATE = 3e-4
MAX_ITERS = 4000
EVAL_INTERVAL = 500
SAVE_STEPS = 1000
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_gpt")
DEVICE = ""  # 空字符串 -> 无 GPU 强制 CPU(见下面),避免误判
HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cpu"
# 官方参考配置(6 层 6 头 192 维),够分量把 loss 压到 ~1.5 让文本变通顺
CFG = dict(n_layer=6, n_head=6, n_embd=192, dropout=0.0)

torch.manual_seed(1337)

# ---------- 数据 ----------
train_loader = DataLoader("train", BATCH_SIZE, BLOCK_SIZE)
val_loader = DataLoader("val", BATCH_SIZE, BLOCK_SIZE)

# ---------- 模型(从 meta 拿 vocab_size) ----------
with open(os.path.join(HERE, "data", "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
itos = meta["itos"]

config = GPTConfig(vocab_size=meta["vocab_size"], block_size=BLOCK_SIZE, **CFG)
model = GPT(config).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95))


@torch.no_grad()
def estimate_loss():
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
    torch.save({"model": model.state_dict(), "config": config, "step": step, "loss": train_loss},
               os.path.join(OUT_DIR, f"gpt_step{step}.pt"))


# ---------- 训练 ----------
print(f"device={DEVICE}  vocab={config.vocab_size}  params={model.get_num_params()/1e6:.2f}M")
ema_loss = None
t0 = time.time()
for step in range(MAX_ITERS):
    x, y = train_loader.get_batch(DEVICE)
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    ema_loss = loss.item() if ema_loss is None else 0.9 * ema_loss + 0.1 * loss.item()

    if (step + 1) % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"step {step+1:5d} | 平滑train {ema_loss:.4f} | 估train {losses['train']:.4f} | "
              f"val {losses['val']:.4f} | {(time.time()-t0):.0f}s")

    if (step + 1) % SAVE_STEPS == 0:
        save_ckpt(step + 1, time.time() - t0, ema_loss)

print(f"训练完成,用时 {(time.time()-t0):.0f}s,checkpoint 在 {OUT_DIR}/")


# ---------- 采样生成 ----------
@torch.no_grad()
def generate(model, max_new_tokens=400, temperature=1.0, top_k=None):
    model.eval()
    start = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    ids = model.generate(start, max_new_tokens, temperature=temperature, top_k=top_k)[0].tolist()
    return "".join(itos[i] for i in ids)


print(f"\n===== 采样生成(temperature=1.0) =====")
print(generate(model, 400))
print(f"\n===== 采样生成(temperature=0.8, top_k=50) =====")
print(generate(model, 400, temperature=0.8, top_k=50))