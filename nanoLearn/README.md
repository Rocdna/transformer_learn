# nanoLearn · nanoGPT 学习工作区

这是**你自己的学习/重写空间**,不入库(已被 `../.gitignore` 忽略)。

## 怎么用
1. **先读参考源码** `../nanoGPT/model.py`,对照本项目 `../model/` 逐块映射:
   - `CausalSelfAttention` ← 你的 `masked_multi_head_attention`
   - `Block`(LayerNorm + attn + FFN) ← 你的 `decoder_layer`
   - GPT = 你的 Decoder 去掉 Encoder
2. **关掉原代码,在这重写一遍** `my_gpt.py`(自己从零组 nn.Module)。
3. 跑通后对比原版,标注差异。

## 学习清单(每项半天)
- [ ] 重写 `my_gpt.py`:Embedding + 位置编码 + Block + 输出投影
- [ ] 自注意力:因果 mask + 缩放点积
- [ ] Pre-LN vs Post-LN(RoPE / RMSNorm / SwiGLU 是附加项)

草稿和失败的写法都放这,不担心污染主仓库。