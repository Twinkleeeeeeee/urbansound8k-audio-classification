"""以时间帧为 Token 的音频 Transformer，复用 transformer_layers 中的基础模块。

默认输入 [B, 1, 64, 126]，输出 [B, 10]；直接运行只检查随机输入的前向传播。
"""

import torch
import torch.nn as nn

from transformer_layers import PositionalEncoding, TransformerEncoderLayer


class AudioTransformer(nn.Module):
    """频带投影 → 位置编码 → 两层 Encoder → 时间平均 → 十分类。"""

    def __init__(self):
        super().__init__()

        # 每个时间帧的 64 个频带特征 → 128 维模型特征。
        self.input_proj = nn.Linear(64, 128)

        self.pos_encoding = PositionalEncoding(
            d_model=128,
            dropout=0.1,
        )

        # 两层结构相同，但分别创建，各自学习自己的参数。
        self.encoder1 = TransformerEncoderLayer(
            d_model=128,
            num_heads=4,
            d_ff=256,
            dropout=0.1,
        )

        self.encoder2 = TransformerEncoderLayer(
            d_model=128,
            num_heads=4,
            d_ff=256,
            dropout=0.1,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        """保留 batch 维，对每条音频输出十个未经 Softmax 的类别分数。"""
        # 输入：[B, 1, 64, 126]
        x = x.squeeze(1)             # [B, 64, 126]
        x = x.transpose(1, 2)        # [B, 126, 64]

        x = self.input_proj(x)       # [B, 126, 128]
        x = self.pos_encoding(x)     # [B, 126, 128]

        x = self.encoder1(x)         # [B, 126, 128]
        x = self.encoder2(x)         # [B, 126, 128]

        x = x.mean(dim=1)            # [B, 128]，沿时间维度取平均。
        logits = self.classifier(x)  # [B, 10]，十个类别的分数。
        return logits


# 独立前向检查，不读取音频、不更新参数。
if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.manual_seed(42)

    model = AudioTransformer()
    model.eval()

    # 用随机数据模拟两条音频的 Log-Mel；此处不读取真实音频。
    x = torch.randn(2, 1, 64, 126)

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (2, 10), "输出形状应为 [2, 10]"
    assert torch.isfinite(logits).all().item(), "输出中出现了 NaN 或 Inf"

    print("输入形状:", x.shape)
    print("输出形状:", logits.shape)
    print("前向检查通过：输出形状正确，所有数值均为有限值。")
