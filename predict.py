import os

import torch
from datasets import load_from_disk

from model.mask import create_padding_mask, create_causal_mask
from model.transformer import Transformer
from model.config import D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, MAX_SEQ_LEN
from data.tokenizer import CharTokenizer, WordTokenizer

# ==================================================
# 0. 加载数据（取 tokenizer，用于编 / 解码）
# ==================================================
ROOT = os.path.dirname(os.path.abspath(__file__))
dataset = load_from_disk(os.path.join(ROOT, "data", "dataset_clean"))


def extract_pairs(rows):
    pairs = []
    for row in rows:
        conv = row["conversations"]
        zh = conv[0]["content"].split("--", 1)[1].strip()
        en = conv[1]["content"]
        pairs.append((zh, en))
    return pairs


pairs = extract_pairs(dataset["train"])
src_tok = CharTokenizer([p[0] for p in pairs])
tgt_tok = WordTokenizer([p[1] for p in pairs])

# ==================================================
# 1. 建模型 + 加载训练好的参数
#    （模型结构和 train.py 完全一致，参数才能对上）
# ==================================================
model = Transformer(
    src_vocab_size=src_tok.vocab_size,
    tgt_vocab_size=tgt_tok.vocab_size,
    d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF,
    num_layers=NUM_LAYERS, max_seq_len=MAX_SEQ_LEN,
)
# 找最新的 transformer_*.pt（按「日_时_分_秒」命名，取时间最晚的一个）
import glob

ckpts = sorted(glob.glob(os.path.join(ROOT, "transformer_*.pt")))
if ckpts:
    ckpt = ckpts[-1]
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    print(f"已加载 {ckpt}")
else:
    print("⚠ 找不到 transformer_*.pt，请先运行 train.py 训练")
model.eval()  # 推理模式：关掉 dropout

# ==================================================
# 2. greedy_decode —— 逐个词贪心地生成
# ==================================================
def greedy_decode(src_ids, max_len=30):
    """
    src_ids: [1, src_len]，已含 <BOS><EOS>
    返回: 生成的目标 token ids 列表（以 <BOS> 开头）
    """
    # --- 源侧：走 encoder，得到"记忆" enc_out（只算一次）---
    src_mask = create_padding_mask(src_ids, src_tok.pad_id).unsqueeze(1).unsqueeze(1).float()
    enc_out = model.encoder(src_ids, src_mask)          # [1, src_len, d_model]

    # --- 目标侧：从 <BOS> 开始挤牙膏 ---
    tgt = torch.tensor([[tgt_tok.bos_id]])              # [1, 1]
    for _ in range(max_len):
        tgt_mask = create_causal_mask(tgt.size(1))      # [cur, cur] 下三角
        dec_out = model.decoder(tgt, enc_out, tgt_mask, src_mask)  # [1, cur, d_model]
        logits = model.output_projection(dec_out)       # [1, cur, tgt_vocab] 投影成词表分数
        next_id = logits[0, -1, :].argmax().item()      # 只看最后位置，取最像的词
        tgt = torch.cat([tgt, torch.tensor([[next_id]])], dim=1)
        if next_id == tgt_tok.eos_id:                   # 撞见 <EOS> 就停
            break
    return tgt[0].tolist()


# ==================================================
# 3. 翻几个中文句子
# ==================================================
samples = ["你好", "谢谢", "我想喝水", "今天天气很好", "早上好", "我饿了"]
for zh in samples:
    src_ids = torch.tensor([src_tok.encode(zh, add_bos=True, add_eos=True)])
    out_ids = greedy_decode(src_ids)
    en = " ".join(tgt_tok.decode(out_ids))
    print(f"中文: {zh}\n英文: {en}\n")