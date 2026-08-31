"""
nanoLearn · my_gpt.py —— 手写 GPT(Decoder-only)

在 bigram 基础上两个改动:
  ① token 嵌入 + 位置嵌入(wpe)相加 → 模型知道"第几个位置"
  ② 每层 Block 里的因果自注意力 → 每个位置能看到所有左边历史(有向图"看过去")

结构(由小到大,擦官方 nanoGPT/model.py):
  LayerNorm → CausalSelfAttention → MLP → Block → GPT(总装) → forward → generate
关键差异(和你的翻译项目比):
  - Pre-LN:  Block 里 x = x + attn(ln_1(x)) 先 LN 再残差外
  - c_attn:  一个 Linear(n_embd→3n_embd) 一次算 Q/K/V 再 split(工程写法)
  - wpe:     位置用可学习 Embedding(block_size, n_embd),不是正弦
  - weight tying: token 嵌入矩阵和输出 lm_head 绑成同一个 weight
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# ================= 基础小块 =================
class LayerNorm(nn.Module):
    """带可选 bias 的 LayerNorm(官方为了方便不吃掉 bias 写了一份)"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """多头因果自注意力。图视角:自回归 mask 把允许的边剪成"只看左边"。"""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # 一个 Linear 同时算 QKV,再 split 成三段(等价 3 个 Linear,省一次大矩阵)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # 因果掩码:bias[i,j]=1 表示 i 能看 j(只看 i 自己及左边)。存成下三角 buffer
        tril = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("tril", tril.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()                      # (batch, 序列长, n_embd)
        # 拆 Q/K/V,再按头拆分并搬到"头"在 batch 维
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))     # 边权重矩阵 (B,nh,T,T)
        att = att.masked_fill(self.tril[:, :, :T, :T] == 0, float("-inf"))  # 剪掉未来边
        att = F.softmax(att, dim=-1)                                # 归一化成权重
        att = self.attn_dropout(att)
        y = att @ v                                                 # 对能看的邻居的 V 加权平均
        y = y.transpose(1, 2).contiguous().view(B, T, C)            # 拼回头维
        y = self.resid_dropout(self.c_proj(y))                      # 输出投影
        return y


class MLP(nn.Module):
    """前馈:Linear → GELU → Linear(ratio 4n_embd,照 GPT-2)"""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Pre-LN 残差块:注意力 + MLP。LLM 的主流结构而非你翻译项目的 Post-LN。"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=False)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=False)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))    # 先 LN 再注意力,残差在外
        x = x + self.mlp(self.ln_2(x))
        return x


# ================= 配置 =================
@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65      # 运行时会从 meta.pkl 覆盖
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0


# ================= 总装 =================
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),   # token 嵌入 (V, n_embd)
            wpe=nn.Embedding(config.block_size, config.n_embd),   # 位置嵌入 (block_size, n_embd)
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # 若干 Block
            ln_f=LayerNorm(config.n_embd, bias=False),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying:输出投影和 token 嵌入共享权重
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # 残差投影用小一点 init,稳定训练
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        print(f"模型参数量: {self.get_num_params()/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)  # 0..t-1
        # ④ 位置编码:tok 嵌入 + pos 嵌入,逐位相加
        tok_emb = self.transformer.wte(idx)          # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)          # (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:             # ⑤ 若干 Block(注意力+MLP)
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)                 # (b, t, V)  训练要全部位置的 logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])     # 推理只对最后一位投影(省算力)
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """基于温度 + top-k 采样逐 token 自回归生成(你 Generator 那套)。"""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx