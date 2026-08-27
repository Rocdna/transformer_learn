import re


class WordTokenizer:
    """
    英文按单词 + 少量标点切分。
    适合 target 侧：词是离散单元，能缓解字符级解码的重复塌缩，
    也让 <EOS> 容易被学会（句子末尾就是空格词）。
    """
    _TOKEN = re.compile(r"[A-Za-z0-9']+|[.,!?;]")

    def __init__(self, texts):
        words = set()
        for t in texts:
            words.update(self._TOKEN.findall(t))

        specials = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
        self.vocab = specials + sorted(words)

        self.stoi = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.itos = {idx: tok for idx, tok in enumerate(self.vocab)}

        self.pad_id = self.stoi["<PAD>"]
        self.bos_id = self.stoi["<BOS>"]
        self.eos_id = self.stoi["<EOS>"]
        self.unk_id = self.stoi["<UNK>"]
        self.vocab_size = len(self.vocab)

    def encode(self, text, add_bos=False, add_eos=False):
        ids = [self.bos_id] if add_bos else []
        for w in self._TOKEN.findall(text):
            ids.append(self.stoi.get(w, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        """返回词列表（不含特殊 token）"""
        toks = []
        for i in ids:
            t = self.itos[i]
            if t in ("<PAD>", "<BOS>", "<EOS>"):
                continue
            toks.append(t)
        return toks


class CharTokenizer:
    def __init__(self, texts):
        """
        texts:
            用于创建词表的所有文本。
            例如：
                [
                    "你好",
                    "我爱你",
                    "Hello",
                    "I love you"
                ]   
        """
        # ==================================================
        # 1. 收集所有字符
        # ==================================================

        ## 去重
        chars = set()

        for text in texts:
            chars.update(text)

        # ==================================================
        # 2. 排序
        # ==================================================
        
        chars = sorted(chars)

        # ==================================================
        # 3. 加入特殊 Token
        # ==================================================

        special_tokens = [
            "<PAD>",
            "<BOS>",
            "<EOS>",
            "<UNK>"
        ]

        self.vocab = special_tokens + chars

        # ==================================================
        # 4. Token → ID   加索引
        # ==================================================
    
        self.stoi = {
            token: idx
            for idx, token in enumerate(self.vocab)
        }

        # ==================================================
        # 5. ID → Token
        # ==================================================

        self.itos = {
            idx: token
            for idx, token in enumerate(self.vocab)
        }

        # ==================================================
        # 6. 保存特殊 Token 的 ID
        # ==================================================

        self.pad_id = self.stoi["<PAD>"]

        self.bos_id = self.stoi["<BOS>"]

        self.eos_id = self.stoi["<EOS>"]

        self.unk_id = self.stoi["<UNK>"]


        # 词表大小
        self.vocab_size = len(self.vocab)

    def encode(self, text, add_bos=False, add_eos=False):
        """
        字符串 → Token IDs
        """

        ids = []

        if add_bos:
            ids.append(self.bos_id)

        for ch in text:
            # 如果字符存在
            # 就使用它对应的 ID
            if ch in self.stoi:
                ids.append(self.stoi[ch])
            else:
                # 不认识的字符
                # 使用 <UNK>
                ids.append(self.unk_id)

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids):
        """
        Token IDs → 字符串
        """

        tokens=[]

        for idx in ids:

            token = self.itos[idx]

            # 特殊 Token 直接不显示
            if token in [
                "<PAD>",
                "<BOS>",
                "<EOS>"
            ]:
                continue

            tokens.append(token)

        return tokens        







