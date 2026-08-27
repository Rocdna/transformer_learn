import torch
import math
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    Transformer 中的 Multi-Head Attention。
    输入： x: [batch_size, seq_len, d_model]
    输出：    [batch_size, seq_len, d_model]
    例如：
        batch_size = 2
        seq_len = 5
        d_model = 8
        num_heads = 2
    那么：
        输入： [2, 5, 8]
        输出： [2, 5, 8]
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        # ==================================================
        # Multi-Head Attention 要求：
        # d_model 必须能够被 num_heads 整除。
        # 例如：
        # d_model = 8
        # num_heads = 2
        # 每个 Head：
        # 8 / 2 = 4
        # ==================================================
        assert d_model % num_heads == 0, "d_model 必须能够被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads

        # 每个头的维度
        self.d_k = d_model // num_heads

        # ==================================================
        # Q、K、V 的线性变换
        # 注意：
        # 这里不是：
        # 每个 Head 单独一个 Linear
        # 而是先一次性把：
        # d_model → d_model
        # 然后再拆成多个 Head。
        # ==================================================
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        # 最后的输出 Linear
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        """
        x: [batch_size, seq_len, d_model]
        mask: Attention Mask。
        后面加入。
        """
        # ==================================================
        # 取得输入的 Shape
        # ==================================================
        batch_size = x.size(0)
        seq_len = x.size(1)

        # ==================================================
        # 第一步：
        # X → Q
        # X → K
        # X → V
        # 此时：
        # Q/K/V:
        # [batch_size, seq_len, d_model]
        # ==================================================
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        # ==================================================
        # 第二步：
        # 把 d_model 拆成：
        # num_heads × d_k
        # 例如：
        # [2, 5, 8]
        #     ↓
        # [2, 5, 2, 4]
        # 含义：
        # batch = 2
        # seq_len = 5
        # num_heads = 2
        # d_k = 4
        # ==================================================

        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k)

        # ==================================================
        # 第三步：
        # 调整维度顺序
        # 原来：
        # [batch, seq_len, heads, d_k]
        # 变成：
        # [batch, heads, seq_len, d_k]
        # 为什么？
        # 因为接下来我们希望：
        # 每个 Head 独立计算 Attention。
        # ==================================================
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        # ==================================================
        # 第四步：
        # Q × K^T
        # K:
        # [batch, heads, seq_len, d_k]
        # 转置最后两个维度：
        # [batch, heads, d_k, seq_len]
        # 所以：
        # [2, 2, 5, 4] × [2, 2, 4, 5] =  [2, 2, 5, 5]
        # 注意：
        # 每一个 Head 都有一个
        # [seq_len, seq_len] 的 Attention 矩阵。
        # ==================================================
        scores = torch.matmul(Q, K.transpose(-2, -1))
        # ==================================================
        # 第五步：
        # Scaled Dot Product Attention
        # 除以 sqrt(d_k)
        # ==================================================
        scores = scores / math.sqrt(self.d_k)

        # ==================================================
        # 第六步：Mask
        # 暂时保留接口。
        # 后面我们会专门讲：
        # Padding Mask
        # Causal Mask
        # ==================================================
        if mask is not None:
            scores = scores.masked_fill(
                mask == 0,
                float("-inf")
            )

        # ==================================================
        # 第七步：
        # Softmax
        # 最后一个维度进行 Softmax。
        # 也就是：
        # 对每个 Query：
        # “我应该关注哪些 Token？”
        # ==================================================
        attention_weights = torch.softmax(scores, dim=-1)

        # ==================================================
        # 第八步：
        # Attention Weights × V
        # [batch, heads, seq_len, seq_len]
        #               ×
        # [batch, heads, seq_len, d_k]
        #               =
        # [batch, heads, seq_len, d_k]
        # ==================================================
        context = torch.matmul(attention_weights, V)
        # ==================================================
        # 第九步：
        # 把多个 Head 拼回来。
        # 当前：
        # [batch, heads, seq_len, d_k]
        # 例如：    [2, 2, 5, 4]
        # 我们希望： [2, 5, 2, 4]
        # 然后把：2 × 4 合并成：8
        # 最终：
        # [2, 5, 8]
        # ==================================================
        context = context.transpose(1, 2)
        # ==================================================
        # contiguous()
        # transpose 后 Tensor 在内存中的布局发生了变化。
        # 后面使用 view() 前，
        # 通常需要 contiguous()。
        # ==================================================
        context = context.contiguous()
        # ==================================================
        # 合并：
        # [batch, seq_len, heads, d_k]
        #             ↓
        # [batch, seq_len, d_model]
        # ==================================================
        context = context.view(batch_size, seq_len, self.d_model)
        # ==================================================
        # 第十步：
        # 最后的 Linear
        # Concat 后：
        # [batch, seq_len, d_model]
        #           ↓
        #          W_O
        #           ↓
        # [batch, seq_len, d_model]
        # ==================================================
        output = self.W_O(context)
        return output
