"""Transformer 基础模块：多头自注意力、FFN、正弦位置编码和 Encoder 层。"""

import torch
import torch.nn as nn
import math
class MyMultiHeadAttention(nn.Module):
    """输入、输出均为 [B, T, d_model]；各头在不同投影子空间中聚合序列信息。"""

    def __init__(self,d_model,num_heads):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.head_dim=d_model//num_heads
        assert self.head_dim*num_heads==d_model,"d_model 必须能被 num_heads整除"
        self.W_q=nn.Linear(d_model,d_model)
        self.W_k=nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)


    def forward(self,x,mask=None):
        batch_size,seq_len,_=x.shape
        # 每个投影先保留时间顺序，再拆成 [B, num_heads, T, head_dim]。
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 每个查询位置与全部键位置比较，得到 [B, num_heads, T, T]。
        scores = torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(self.head_dim)
        if mask is not None:
            scores=scores.masked_fill(mask==0,-1e9)
        # 沿键位置归一化，再对 V 加权求和。
        attn_weights=torch.softmax(scores,dim=-1)
        context=torch.matmul(attn_weights,V)
        # 将同一时间位置的各个头拼接回 [B, T, d_model]。
        context=context.transpose(1,2).contiguous().view(batch_size,seq_len,self.d_model)
        output=self.W_o(context)
        return output

class PositionWiseFeedForward(nn.Module):
    """逐位置共享的两层 MLP：d_model → d_ff → d_model，不混合时间位置。"""

    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.linear1=nn.Linear(d_model,d_ff)
        self.dropout=nn.Dropout(dropout)
        self.linear2=nn.Linear(d_ff,d_model)
        self.relu=nn.ReLU()
    def forward(self,x):
        x=self.linear1(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.linear2(x)
        return x

class PositionalEncoding(nn.Module):
    """
    为 [B, T, d_model] 序列添加正弦位置编码，适用于音频帧等特征序列。
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建一个足够长的位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # pe (positional encoding) 的大小为 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # 偶数维度使用 sin, 奇数维度使用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 将 pe 注册为 buffer，这样它就不会被视为模型参数，但会随模型移动（例如 to(device)）
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.size(1) 是当前输入的序列长度
        # 将位置编码加到输入向量上
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    """自注意力和 FFN 各接残差与 LayerNorm，保持输入输出形状不变。"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MyMultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 1. 自注意力 + 残差连接 + 层归一化
        attn_output = self.self_attn(x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # 2. 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        return x
if __name__=="__main__":
    x=torch.randn(2,10,512)
    mha=MyMultiHeadAttention(d_model=512,num_heads=8)
    out=mha(x)
    print("输入：",x.shape)
    print("输出：",out.shape)
