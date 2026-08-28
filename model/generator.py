import torch

from model.mask import create_padding_mask


class Generator:
    """
    把「生成」从外部脚本抽成模型的一种正式能力。

    自带两条现代 LLM 推理机制：
      1) KV Cache  —— 缓存历史位置的 K/V，增量解码，避免每步重算整句
      2) 温度 + top-k —— 从概率分布采样（而非永远取最可能的词），生成更多样
    """
    def __init__(self, model, src_tokenizer, tgt_tokenizer,
                 max_len=30, temperature=0.0, top_k=0):
        self.model = model
        self.src_tok = src_tokenizer
        self.tgt_tok = tgt_tokenizer
        self.max_len = max_len
        self.temperature = temperature   # 0 = 贪心；越大越随机
        self.top_k = top_k              # 0 = 不限制；>0 只从得分最高的 top_k 个里选
        self.model.eval()

    # ----------------------------------------------------------
    # 采样：给定最后一个位置的 logits，选下一个 token
    # ----------------------------------------------------------
    def _sample_one(self, logits):
        """logits: [1, 1, vocab_size]（当前这个新 token 的打分）"""
        logits = logits[0, -1, :]

        # temperature <= 0 → 贪心，永远取最可能的词（和旧 argmax 行为一致）
        if self.temperature <= 0.0:
            return logits.argmax().item()

        # 温度：把 logits 除以 temperature，再 softmax
        #   temp 越大 → 概率分布越平坦 → 采样越随机
        #   temp→0  → 分布越尖 → 越接近贪心
        logits = logits / self.temperature

        # top-k：只保留得分最高的 k 个，其余设为 -inf（softmax 后概率为 0）
        if self.top_k > 0:
            k = min(self.top_k, logits.size(-1))
            threshold, _ = torch.topk(logits, k)
            threshold = threshold[..., -1]                 # 第 k 大的得分
            logits = torch.where(
                logits < threshold,
                torch.full_like(logits, float("-inf")),
                logits,
            )

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    # ----------------------------------------------------------
    # 核心：增量解码（带 KV cache）
    # ----------------------------------------------------------
    def decode(self, src_ids):
        """
        src_ids: [1, src_len]，已含 <BOS><EOS>
        返回   : 生成的英文 token ids 列表（含开头 <BOS>）
        """
        device = next(self.model.parameters()).device
        src_ids = src_ids.to(device)

        with torch.no_grad():
            # ① 源句只编码一次，得到"记忆"
            src_mask = create_padding_mask(src_ids, self.src_tok.pad_id)
            src_mask = src_mask.unsqueeze(1).unsqueeze(1).float()
            enc_out = self.model.encoder(src_ids, src_mask)

            # ② 初始空 KV cache：每层一个 (K, V) = [1, H, 0, d_k]
            layers = self.model.decoder.decoder.layers
            heads = layers[0].self_attention.num_heads
            d_k = layers[0].self_attention.d_k
            empty = torch.empty(1, heads, 0, d_k, device=device)
            kv_cache = [(empty.clone(), empty.clone()) for _ in range(len(layers))]

            # ③ 从 <BOS> 逐个 token 生成
            tgt = torch.tensor([[self.tgt_tok.bos_id]], device=device, dtype=torch.long)
            for _ in range(self.max_len):
                # 当前这个 token 在完整序列里的绝对位置
                pos = tgt.size(1) - 1

                # mask：新 token 应能关注所有已生成的位置（过去全部可见）
                cur_mask = torch.ones(1, tgt.size(1), device=device, dtype=torch.bool)

                # 只喂「最新的一个 token」+ 历史 KV cache → 增量前向
                dec_out, kv_cache = self.model.decoder(
                    tgt[:, -1:], enc_out, cur_mask, src_mask,
                    kv_cache, start_pos=pos,
                )
                logits = self.model.output_projection(dec_out)   # [1, 1, tgt_vocab]

                next_id = self._sample_one(logits)
                tgt = torch.cat([tgt, torch.tensor([[next_id]], device=device)], dim=1)

                if next_id == self.tgt_tok.eos_id:               # 撞见 <EOS> 停止
                    break

            return tgt[0].tolist()

    # ----------------------------------------------------------
    # 构造一层的空 KV cache（beam 每条路径都要一份独立缓存）
    # ----------------------------------------------------------
    def _empty_kv_cache(self, device):
        layers = self.model.decoder.decoder.layers
        heads = layers[0].self_attention.num_heads
        d_k = layers[0].self_attention.d_k
        empty = torch.empty(1, heads, 0, d_k, device=device)
        return [(empty.clone(), empty.clone()) for _ in range(len(layers))]

    # ----------------------------------------------------------
    # Beam Search：同时保留 top-B 条路径，最终按整句联合概率选最优
    #   beam_width     : 每步保留几条路径（束宽）
    #   length_penalty : 长度惩罚。句子越长联合概率天然越低，
    #                    除以 len**penalty 打平衡，penalty=0 则完全不惩罚
    # ----------------------------------------------------------
    def beam_search(self, src_ids, beam_width=5, length_penalty=0.6, max_len=None):
        device = next(self.model.parameters()).device
        src_ids = src_ids.to(device)
        max_len = max_len or self.max_len
        beam_width = max(1, beam_width)

        with torch.no_grad():
            # ① 源句只编码一次，beam 所有路径共用 enc_out
            src_mask = create_padding_mask(src_ids, self.src_tok.pad_id)
            src_mask = src_mask.unsqueeze(1).unsqueeze(1).float()
            enc_out = self.model.encoder(src_ids, src_mask)

            # ② 每条路径 = (token序列, 累计对数概率, 该路径专属KV cache)
            bos = self.tgt_tok.bos_id
            beam = [(torch.tensor([[bos]], device=device, dtype=torch.long),
                     0.0,
                     self._empty_kv_cache(device))]
            completed = []

            for _ in range(max_len):
                # —— 对 beam 里每条还未结束的路径做一步解码 ——
                candidates = []
                for seq, log_prob, kv_cache in beam:
                    pos = seq.size(1) - 1
                    cur_mask = torch.ones(1, seq.size(1), device=device, dtype=torch.bool)
                    dec_out, new_kv = self.model.decoder(
                        seq[:, -1:], enc_out, cur_mask, src_mask,
                        kv_cache, start_pos=pos,
                    )
                    logits = self.model.output_projection(dec_out)      # [1,1,vocab]
                    log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)  # [vocab]

                    # 扩展：这条路径 × 每个词 = 新候选
                    for vocab_id in torch.argsort(log_probs, descending=True)[:beam_width * 2]:
                        wid = vocab_id.item()
                        new_seq = torch.cat([seq, torch.tensor([[wid]], device=device)], dim=1)
                        candidates.append((new_seq, log_prob + log_probs[wid].item(),
                                           new_kv, wid))

                # ③ 按累计对数概率排序，只保留 top-beam_width 条
                candidates.sort(key=lambda c: c[1], reverse=True)
                beam = candidates[:beam_width]

                # ④ 收走已生成 <EOS> 的路径，剩下的继续
                still_open = []
                for seq, lp, kv, wid in beam:
                    if wid == self.tgt_tok.eos_id:
                        completed.append((seq, lp))
                    else:
                        still_open.append((seq, lp, kv))
                beam = still_open

                if not beam and completed:
                    break

            # ⑤ 未结束就强制收尾（把未完成的也当作候选）
            if beam:
                for seq, lp, kv in beam:
                    completed.append((seq, lp))

            if not completed:
                return [bos]

            # ⑥ 选路：用「长度归一化的联合概率」打分，避免偏好短句
            def score(c):
                seq, lp = c
                length = seq.size(1)
                return lp / (length ** length_penalty) if length_penalty > 0 else lp

            best = max(completed, key=score)[0]
            return best[0].tolist()

    # ----------------------------------------------------------
    # 便捷接口：直接翻中文
    # ----------------------------------------------------------
    def translate(self, text):
        src_ids = torch.tensor([self.src_tok.encode(text, add_bos=True, add_eos=True)])
        out_ids = self.decode(src_ids)
        return " ".join(self.tgt_tok.decode(out_ids))

    def translate_beam(self, text):
        src_ids = torch.tensor([self.src_tok.encode(text, add_bos=True, add_eos=True)])
        out_ids = self.beam_search(src_ids)
        return " ".join(self.tgt_tok.decode(out_ids))