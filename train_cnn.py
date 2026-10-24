"""UrbanSound8K CNN 基线：训练、验证选优及测试评估。

直接运行会训练 20 轮，并将最佳权重和测试混淆矩阵写入当前工作目录。
"""

import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

from audio_dataset import UrbanSound8KDataset, AudioConfig


# 随机种子。
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# CNN 分类器。
class AudioCNN(nn.Module):
    """输入 [B, 1, 64, 126]，经过四个卷积块，输出类别分数 [B, n_classes]。"""

    def __init__(self, n_classes=10):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(1, 32),      # [B, 1, 64, 126]  → [B, 32, 32, 63]
            block(32, 64),     # → [B, 64, 16, 31]
            block(64, 128),    # → [B, 128, 8, 15]
            block(128, 128),   # → [B, 128, 4, 7]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)       # 默认 [B, 128, 4, 7]。
        x = self.pool(x)           # 每个通道的空间位置取平均：[B, 128, 1, 1]。
        x = x.flatten(1)           # 保留 batch 维，得到 [B, 128]。
        return self.classifier(x)  # [B, n_classes]，未经 Softmax 的 logits。


# 单轮训练。
def train_one_epoch(model, loader, optimizer, criterion, device):
    """逐 batch 更新参数，返回按样本加权的平均 Loss 和 Accuracy。"""
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, total_correct / total


# 固定参数评估。
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """不记录梯度，返回 Loss、Accuracy、Macro-F1、预测编号和真实编号。"""
    model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(y.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return total_loss / total, acc, macro_f1, preds, labels


# 训练入口。
def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    data_root = Path(r"D:\datasets\UrbanSound8K")
    csv_path = data_root / "metadata" / "UrbanSound8K.csv"
    audio_dir = data_root / "audio"

    df = pd.read_csv(csv_path)
    cfg = AudioConfig()

    train_df = df[df["fold"].isin(range(1, 9))].reset_index(drop=True)
    val_df = df[df["fold"] == 9].reset_index(drop=True)
    test_df = df[df["fold"] == 10].reset_index(drop=True)

    train_ds = UrbanSound8KDataset(train_df, audio_dir, cfg)
    val_ds = UrbanSound8KDataset(val_df, audio_dir, cfg)
    test_ds = UrbanSound8KDataset(test_df, audio_dir, cfg)

    # 训练集打乱；验证、测试保持元数据顺序。Windows 多进程需保留 main 入口保护。
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    model = AudioCNN(n_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    EPOCHS = 20
    best_val_f1 = 0.0
    ckpt_path = Path("best_cnn.pt")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, va_f1, _, _ = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0
        print(f"[{epoch:02d}/{EPOCHS}] "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} val_f1={va_f1:.4f} | "
              f"{dt:.1f}s")

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            torch.save(model.state_dict(), ckpt_path)

    print(f"\n最佳 val macro-F1 = {best_val_f1:.4f}")

    model.load_state_dict(torch.load(ckpt_path))
    te_loss, te_acc, te_f1, preds, labels = evaluate(model, test_loader, criterion, device)
    print(f"Test  acc={te_acc:.4f}  macro-F1={te_f1:.4f}")

    cm = confusion_matrix(labels, preds)
    np.save("cnn_confusion_matrix.npy", cm)

    print("\n混淆矩阵 (行=真实, 列=预测):")
    print(cm)


if __name__ == "__main__":
    main()
