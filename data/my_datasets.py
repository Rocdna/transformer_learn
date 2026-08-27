import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(
        self,
        data,
        src_tokenizer,
        tgt_tokenizer,
        src_key="chinese",
        tgt_key="english"
    ):
        """
        data:
            我们前面处理好的数据。
            每一条类似：
            (
                "你好",
                "Hello"
            )
            src_tokenizer:
                中文 Tokenizer
            tgt_tokenizer:
                英文 Tokenizer
            src_key / tgt_key:
                这里暂时保留这个参数，
                方便以后如果换 HuggingFace Dataset 使用。
        """
        self.data = data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self):
        """
        返回数据集大小。
        """         
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取第 idx 条数据。
        最终返回：
        {
            "src": ...,
            "tgt_input": ...,
            "labels": ...
        }
        """ 

        # ==========================================
        # 1. 取出一条中英文句对
        # ==========================================
        src_text, tgt_text = self.data[idx]

        # ==========================================
        # 2. 中文 -> Token IDs
        # src 不需要 BOS / EOS 也可以，
        # 我们暂时先加 EOS。
        # ==========================================
        src_ids = self.src_tokenizer.encode(
            src_text,
            add_bos=True,
            add_eos=True
        )

        # ==========================================
        # 3. 英文 -> Token IDs
        # 注意：
        # tgt_input：
        # <BOS> I love you
        # labels：
        # I love you <EOS>
        # ==========================================
        tgt_ids = self.tgt_tokenizer.encode(
            tgt_text,
            add_bos=True,
            add_eos=True
        )

        # ==========================================
        # 4. 构造 Decoder 输入
        # 去掉最后一个 EOS
        # [BOS, I, love, you, EOS]
        #           ↓
        # [BOS, I, love, you]
        # ==========================================
        tgt_input = tgt_ids[:-1]

        # ==========================================
        # 5. 构造正确答案
        # 去掉最前面的 BOS
        # [BOS, I, love, you, EOS]
        #          ↓
        # [I, love, you, EOS]
        # ==========================================
        labels = tgt_ids[1:]

        # ==========================================
        # 6. 转成 Tensor
        # ==========================================
        src = torch.tensor(
            src_ids,
            dtype=torch.long
        )

        tgt_input = torch.tensor(
            tgt_input,
            dtype=torch.long
        )

        labels = torch.tensor(
            labels,
            dtype=torch.long
        )   

        return {
            "src": src,
            "tgt_input": tgt_input,
            "labels": labels,
        }