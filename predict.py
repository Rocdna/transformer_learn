import os
import torch
from datasets import load_from_disk
from model.mask import create_padding_mask
from model.transformer import Transformer
from model.generator import Generator
from model.config import D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, MAX_SEQ_LEN,TOP_K,TEMP
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
# ==================================================
# 2. 用 Generator 逐句生成（自带 KV cache + 温度/top-k 采样）
#    temperature=0 → 贪心（和旧 greedy_decode 等价）
# ==================================================
gen = Generator(model, src_tok, tgt_tok, max_len=30,
                temperature=TEMP, top_k=TOP_K)

# ==================================================
# 3. 翻几个中文句子
# ==================================================
samples = ["你好", "谢谢你", "我想喝水", "今天天气很好", "早上好", "我饿了", "你吃了吗？", "你想喝橙汁吗？"]
print('普通版\n')
for zh in samples:
    en = gen.translate(zh)
    print(f"中文: {zh}\n英文: {en}\n")

print('Beam Search\n')
for zh in samples:
    en = gen.translate_beam(zh)
    print(f"中文: {zh}\n英文: {en}\n")

