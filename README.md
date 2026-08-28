# 手撕 Transformer · 中英机器翻译

从零用 PyTorch 实现一个 **Transformer 序列到序列（seq2seq）** 模型，做**中文 → 英文**的句子翻译。项目以学习为主：把 Transformer 的每个模块拆开、手写、讲清楚，再组装训练。

---

## 一、项目结构

```
transformers/
├── train.py                      # 训练入口（数据→模型→反向传播→保存）
├── predict.py                    # 推理入口（加载模型，贪心解码英文句子）
├── README.md                     # 本说明书
├── README_old_教程笔记.md        # 早期的学习笔记（历史留档）
├── .gitignore                    # 忽略数据集缓存与模型 checkpoint
├── data/
│   ├── make_dataset.py           # 生成对齐中英数据集 dataset_clean
│   ├── tokenizer.py              # CharTokenizer(中文按字符) / WordTokenizer(英文按单词)
│   ├── my_datasets.py            # PyTorch Dataset：把句对编码成 src / tgt_input / labels
│   ├── collate.py                # 把多个样本凑成一个 batch（补齐 padding）
│   └── dataset_clean/            # 训练产物：生成好的数据集（686 句对）
├── model/
│   ├── config.py                 # ★ 所有超参数统一放这里（train/predict 共用）
│   ├── transformer.py            # ★ 总装：Encoder + Decoder + 输出投影层
│   ├── encoder.py                # Encoder 外壳：Embedding + 位置编码 + N 层
│   ├── decoder.py                # Decoder 外壳：Embedding + 位置编码 + N 层
│   ├── transformer_encoder.py    # 堆叠 n 个 EncoderLayer
│   ├── transformer_decoder.py    # 堆叠 n 个 DecoderLayer
│   ├── encoder_layer.py          # 单个 Encoder 层（自注意力 + FFN + 残差 + LayerNorm）
│   ├── decoder_layer.py          # 单个 Decoder 层（掩码自注意力 + 交叉注意力 + FFN）
│   ├── multi_head_attention.py   # 多头自注意力
│   ├── masked_multi_head_attention.py  # 带因果掩码的多头自注意力（Decoder 用）
│   ├── cross_attention.py        # 交叉注意力（Q 来自 Decoder，K/V 来自 Encoder）
│   ├── feed_forward.py           # 前馈网络 FFN(Linear→ReLU→Linear)
│   ├── layer_norm.py             # 手写 LayerNorm
│   ├── embedding.py              # TokenEmbedding（ID→向量）
│   ├── positional_encoding.py    # 正弦位置编码
│   └── mask.py                   # 生成 padding mask 与 causal mask
│   └── generator.py              # ★ 生成器类：KV cache 增量解码 + 温度/top-k 采样
├── tests/                        # 各模块的单元测试
└── image*.png                    # 参考示意图（非代码）
```

> `★` 标记的是组装与配置的「总入口」。

---

## 二、快速开始

需要环境：`torch`、`datasets`、`numpy`、`tqdm`（本项目用的是 conda 环境 `nplbase`）。

```bash
# ① 生成数据集（已生成过可跳过）
python data/make_dataset.py

# ② 训练（输出模型到 transformer_<日_时_分_秒>.pt）
python train.py

# ③ 预测（自动加载最新 checkpoint）
python predict.py
```

---

## 三、数据是怎么来的

### 3.1 数据集生成 `data/make_dataset.py`

- **手写句对（PAIRS）**：235 条日常会话，中英严格一一对应。
- **模板扩展（EXTRA_* 函数）**：用「已人工校对过的」中英片段组合出更多严格对齐句对，系统覆盖**动词×宾语、主语×时态**等组合（如 `我想喝咖啡`、`他在看书`、`她喜欢唱歌`…），目的是教模型**组合翻译**而不仅是背整句。
- 最终 **686 句对**，按 **8:1:1** 随机切分 train（548）/ validation（68）/ test（70）。
- **客套话扩充（`_polite_pairs`）**：短客套（谢谢/你好/不客气/再见…）原本每种只 1 条，被 `I want to` / `It is` 等大模板淹没导致翻错。新增 35 条**同锚点、不同形**的短句（如 `谢谢→Thank you.` / `非常感谢→Thank you very much.` / `多谢→Thanks a lot.`），给解码第一步的锚点补齐先验质量，**不删任何大模板**。
- 生成的格式沿用 `conversations`：`[{user: "translate ... -- 中文"}, {assistant: 英文}]`，存到 `data/dataset_clean/`。

### 3.2 分词 `data/tokenizer.py`

- **CharTokenizer**（中文源侧）：按**字符**切分，词表 = `<PAD><BOS><EOS><UNK>` + 所有出现过的字符。
- **WordTokenizer**（英文目标侧）：按**单词 + 少量标点**切分，词表 = 特殊 token + 排序后的所有单词。用词级而非字符级，能缓解字符解码的重复塌缩，也更容易学会 `<EOS>`。
- 两者都有 `encode`（文本→ID），`decode`（ID→词/字符），以及特殊 token 的 ID。

### 3.3 数据打包 `data/my_datasets.py` + `data/collate.py`

- `TranslationDataset.__getitem__` 把一句 `(中文, 英文)` 编码成：
  - `src`：中文 + `<BOS><EOS>`
  - `tgt_input`：`<BOS> 英文...`（去掉末尾 EOS，作为 Decoder 输入）
  - `labels`：`英文... <EOS>`（去掉开头 BOS，作为要预测的正确答案）——**错位一位**实现 teacher forcing。
- `collate_fn` 把 batch 里长度不同的句子**补齐 padding**，返回形状一致的 tensor。

---

## 四、Transformer 架构

### 4.0 总装 `model/transformer.py`

```
logits = output_projection( decoder( encoder(src) ) )
```

`Transformer.forward(src, tgt, src_mask, tgt_mask, cross_mask)`:
1. `src` 过 **Encoder** → 得到源句的上下文表示 `encoder_output`
2. `tgt` 和 `encoder_output` 一起过 **Decoder**
3. 最后 `nn.Linear(d_model, tgt_vocab_size)` 把 d_model 维投影成**目标词表打分 logits**

> `logits` 是 raw 分数；之后交给 `CrossEntropyLoss` 或 `argmax` 决策。

### 4.1 Encoder `model/encoder.py`

`TokenEmbedding(src_vocab, d_model)` → `+ PositionalEncoding` → `Dropout` → 若干 `TransformerEncoderLayer` → `[B, src_len, d_model]`

**单个 EncoderLayer** `model/encoder_layer.py`:
```
x = LayerNorm(x + Dropout(MultiHeadSelfAttention(x)))
x = LayerNorm(x + Dropout(FFN(x)))
```
（Post-LN 结构：残差加法之后做 LayerNorm）

### 4.2 Decoder `model/decoder.py`

`TokenEmbedding(tgt_vocab, d_model)` → `+ PositionalEncoding` → `Dropout` → 若干 `TransformerDecoderLayer` → `[B, tgt_len, d_model]`

**单个 DecoderLayer** `model/decoder_layer.py`（三段式）:
```
① x = LayerNorm(x + MaskedSelfAttention(x))       # 防止看到未来
② x = LayerNorm(x + CrossAttention(x, enc_out))   # 从源句“查资料”
③ x = LayerNorm(x + FFN(x))
```

### 4.3 注意力

- **`multi_head_attention.py`**：标准缩放点积注意力。`d_model` 拆成 `num_heads × d_k`，每个头独立算 `softmax(QKᵀ/√d_k)V`，再拼回、过 `W_O`。带可选的 padding mask。
- **`masked_multi_head_attention.py`**：Decoder 自注意力变体，**同样逻辑 + causal mask**，保证每个位置只能看到自己及左边。
- **`cross_attention.py`**：**Q 来自 Decoder，K/V 来自 Encoder**，让目标词去“关注”源句对应内容。

### 4.4 其他基础模块

- `feed_forward.py`：`Linear(d_model→d_ff) → ReLU → Linear(d_ff→d_model)`。
- `layer_norm.py`：手写 LayerNorm（`(x-mean)/√(var+eps) * gamma + beta`）。
- `embedding.py`：`nn.Embedding`，ID→向量。
- `positional_encoding.py`：经典正弦位置编码，`register_buffer` 随模型保存/搬移设备。
- `mask.py`：
  - `create_padding_mask(tokens, pad_id)`：标记真实 token（True=可看）。
  - `create_causal_mask(seq_len)`：下三角，True=可看，防看未来。
  - 注意：mask 在使用时要 reshape 成与 4D 注意力分数匹配的形状（`[B,1,1,src_len]`），见 train.py / predict.py。

---

## 五、训练 `train.py`

### 5.1 超参数 `model/config.py`

所有超参数**只改这一处**，train.py / predict.py 都从它读取，避免两处不一致：

| 参数 | 值 | 说明 |
|---|---|---|
| D_MODEL | 256 | 词向量维度 |
| NUM_HEADS | 8 | 注意力头数（需整除 D_MODEL） |
| D_FF | 1024 | 前馈层维度 |
| NUM_LAYERS | 4 | 编码器/解码器层数 |
| MAX_SEQ_LEN | 1024 | 最大序列长度 |
| BATCH_SIZE | 4 | 每 batch 句对数 |
| LR | 3e-4 | 学习率 |
| EPOCHS | 1000 | 训练步数（每步一个 batch） |

### 5.2 训练循环要点

- **Teacher Forcing**：训练时把「正确目标句」整体喂进 Decoder，同时对每一步位置算 loss。
- **损失**：`CrossEntropyLoss(ignore_index=pad_id)`——padding 位置不计 loss；`logits.reshape(-1, vocab)` 与 `labels.reshape(-1)`（`-1` 是自动推断维度）对齐后打分。
- **EMA 平滑**：`ema_loss = 0.9*ema_loss + 0.1*loss.item()`，进度条与每 100 步打印的都是平滑值，避免单 batch 抖动。
- **反向传播**：`zero_grad → loss.backward() → optimizer.step()`（AdamW）。
- **保存**：以 `transformer_<日_时_分_秒>.pt`（如 `transformer_27_18_16_23.pt`）命名，每次训练生成新文件，**不覆盖旧模型**。

### 5.3 两层 mask 怎么传

- **tgt_mask = causal mask**（Decoder 自注意力用）。
- **src_mask / cross_mask = padding mask**，且 `unsqueeze(1).unsqueeze(1).float()` 成 `[B,1,1,src_len]`，才能与交叉注意力 4D 分数广播。

---

## 六、预测 `predict.py` + 生成器 `model/generator.py`

- 自动在根目录找**最新的** `transformer_*.pt`（按文件名字典序取最后），`load_state_dict` 加载。
- 生成逻辑抽成了 `Generator` 类，自带两条现代 LLM 推理机制：

### 6.1 KV Cache（增量解码）

朴素 **greedy_decode** 每推一个新词，都把**整个目标序列**重新过一遍 Decoder——历史位置的 K/V 全被重算，白白浪费。KV cache 的做法：

- 推理时**只喂最新的一个 token**，把历史位置的 **K/V 缓存**起来。
- 注意力里 `new_K = cat(历史 K, 新 K)`，新 token 的 Query 去注意完整的过去 → 与整句前向**结果完全一致**（已验证 `误差≈4e-7`）。
- Q 从不缓存（每一步的 Query 都是新的）；位置编码要带 `offset`（否则单 token 会被当成第 0 位）。
- 改动的核心在 `masked_multi_head_attention.py`（拼 K/V）+ `decoder*.py`（透传 cache + start_pos）；训练时传 `None`，行为完全不变。

### 6.2 温度 + top-k 采样

`argmax` 永远取最可能的词 → 单调、易重复。`Generator._sample_one` 改为从概率分布采样：

- **温度 temperature**：`logits / temperature` 后再 softmax。`>1` 变平坦（更随机），`→0` 变尖（更接近贪心），`<=0` 即贪心。
- **top-k**：只保留得分最高的 k 个，其余设 `-inf`，避免随机乱选。
- 玩具模型上采样常「走神」，写实演示了 why 大模型要配大底数训练。

### 6.3 一句话用法

```python
from model.generator import Generator
gen = Generator(model, src_tok, tgt_tok, max_len=30, temperature=0.0, top_k=0)
en = gen.translate("你好")     # temperature=0 → 贪心
```

---

## 七、学习笔记 / 常见坑

- **loss 很低 ≠ 翻得准**：teacher forcing 训练 + 数据少，模型倾向于「背整句」。推理是自回归（自己生成的词再喂回），一旦某个词偏了就容易滑向另一句背过的**近邻**句子（如「我想喝水」错成「我想喝咖啡」）。样本里 `谢谢`这类**短句**之间区分度低，最容易串扰 → **数据侧解法**：靠 `_polite_pairs` 扩充客套话锚点的先验（见 §3.1），让它们在解码第一步就有足够质量，不再被大模板顶掉。
- **加层数/头数降不了 loss**：瓶颈往往不是容量，而是**数据量**和**训练稳定性**。数据只有几百句时，大模型反而更难训（学习率要更低、更易过拟合）。
- **模型变大要降学习率**：从 12 层/1e-3 降到 4 层/3e-4 后收敛明显更稳。
- **checkpoint 对不上**：train.py 与 predict.py 的模型结构必须一致。现在统一从 `model/config.py` 读参数，且 predict 自动取最新 checkpoint，避免手动同步出错。
- **positional 用 register_buffer**、**view 前先 contiguous**（transpose 后内存布局变了）、**mask 是 None 时不屏蔽**，这几处是手写 Transformer 最容易踩的细节。