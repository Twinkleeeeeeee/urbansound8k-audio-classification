"""标准化音频 Transformer（E1）：仅用训练集拟合固定全局均值和标准差。

直接运行训练 20 轮，只使用训练集和验证集；每次创建独立结果目录。
"""

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset

from audio_dataset import AudioConfig, UrbanSound8KDataset
from train_cnn import evaluate, set_seed, train_one_epoch
from audio_transformer import AudioTransformer
from train_audio_transformer import (
    DATA_ROOT, EPOCHS, SEED, NUM_WORKERS, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE,
    LEARNING_RATE, WEIGHT_DECAY, save_predictions, save_training_curves,
)

# 上级目录沿用历史名称，保留已有实验位置。
OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "step6_5_normalization"
EPSILON = 1e-8


class StandardizedDataset(Dataset):
    """用训练集固定统计量处理基础 Dataset，保持特征形状和类别标签不变。"""

    def __init__(self, base_dataset, mean, std, epsilon=EPSILON):
        self.base_dataset = base_dataset
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        log_mel, label = self.base_dataset[idx]
        log_mel = (log_mel - self.mean) / (self.std + self.epsilon)
        return log_mel, label


def fit_statistics(loader):
    """累计训练特征的和、平方和与元素数，计算全局均值和总体标准差。"""
    total, total_square = 0.0, 0.0
    count, samples = 0, 0
    for x, _ in loader:
        values = x.to(torch.float64)
        total += values.sum().item()
        total_square += values.square().sum().item()
        count += values.numel()
        samples += x.size(0)
    if count == 0:
        raise ValueError("训练集为空，无法计算均值和标准差")
    mean = total / count
    std = math.sqrt(max(total_square / count - mean * mean, 0.0))
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise ValueError("训练数据的均值或标准差无效")
    return {"mean": mean, "std": std, "samples": samples, "elements": count,
            "epsilon": EPSILON, "fit_folds": list(range(1, 9))}


def save_json(path, data):
    """将统计量、实验配置或指标写入 UTF-8 JSON 文件。"""
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main():
    # 不接受额外选项；检查入口位于 checks/，避免误启动完整训练。
    if len(sys.argv) > 1:
        raise SystemExit("本训练程序不接受命令行选项；直接运行本文件会开始训练。")
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)  # 新目录，避免覆盖旧实验。
    print("本次结果目录：", output_dir, flush=True)

    # 数据：Fold 1–8 训练，Fold 9 验证。
    df = pd.read_csv(DATA_ROOT / "metadata" / "UrbanSound8K.csv")
    train_df = df[df["fold"].isin(range(1, 9))].reset_index(drop=True)
    val_df = df[df["fold"] == 9].reset_index(drop=True)
    cfg = AudioConfig()
    raw_train_ds = UrbanSound8KDataset(train_df, DATA_ROOT / "audio", cfg)
    raw_val_ds = UrbanSound8KDataset(val_df, DATA_ROOT / "audio", cfg)
    class_names = df.groupby("classID")["class"].first().to_dict()

    # 仅用训练集拟合统计量；训练、验证共用，不在验证集重新拟合。
    print("正在统计训练集均值和标准差……", flush=True)
    stats_loader = DataLoader(raw_train_ds, batch_size=TRAIN_BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)
    stats = fit_statistics(stats_loader)
    save_json(output_dir / "normalization.json", stats)
    train_ds = StandardizedDataset(raw_train_ds, stats["mean"], stats["std"])
    val_ds = StandardizedDataset(raw_val_ds, stats["mean"], stats["std"])
    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS)

    # 在创建模型前重置种子，使训练集统计过程不影响初始化起点。
    set_seed(SEED)
    model = AudioTransformer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                   weight_decay=WEIGHT_DECAY)
    save_json(output_dir / "config.json", {
        "experiment": "E1_normalized", "data_root": str(DATA_ROOT),
        "seed": SEED, "epochs": EPOCHS, "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY, "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE, "num_workers": NUM_WORKERS,
        "train_folds": list(range(1, 9)), "val_fold": 9, "test_evaluated": False,
        "class_names": class_names, "normalization": stats,
        "train_val_shared_fsID": sorted(set(train_df.fsID) & set(val_df.fsID)),
    })

    # 每轮训练和验证，按验证 Macro-F1 同步保存权重、预测和混淆矩阵。
    history = []
    best_f1 = -1.0
    best_epoch = None
    best_validation = None
    checkpoint_path = output_dir / "best_transformer.pt"
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, va_f1, preds, labels = evaluate(model, val_loader, criterion, device)
        if not all(math.isfinite(v) for v in (tr_loss, tr_acc, va_loss, va_acc, va_f1)):
            raise RuntimeError("出现无效损失或指标，请检查输入数据和训练过程")
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "val_macro_f1": va_f1})
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        print(f"[{epoch}/{EPOCHS}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} val_f1={va_f1:.4f}", flush=True)
        if va_f1 > best_f1:
            best_f1 = va_f1
            best_epoch = epoch
            best_validation = {"loss": va_loss, "accuracy": va_acc, "macro_f1": va_f1}
            torch.save(model.state_dict(), checkpoint_path)
            save_predictions(val_df, preds, labels, output_dir / "val_predictions.csv", class_names)
            cm = confusion_matrix(labels, preds, labels=list(range(10)))
            np.save(output_dir / "val_confusion_matrix.npy", cm)

    # 固定最佳参数，用评估模式重新计算训练集指标，再保存汇总和曲线。
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    train_eval_loader = DataLoader(train_ds, batch_size=EVAL_BATCH_SIZE,
                                   shuffle=False, num_workers=NUM_WORKERS)
    loss, acc, f1, _, _ = evaluate(model, train_eval_loader, criterion, device)
    save_json(output_dir / "metrics.json", {
        "completed_epochs": EPOCHS, "best_epoch": best_epoch,
        "validation": best_validation,
        "train_eval_best_checkpoint": {"loss": loss, "accuracy": acc, "macro_f1": f1},
        "test_evaluated": False,
    })
    save_training_curves(output_dir / "history.csv", output_dir / "training_curves.png")
    print(f"完成。最佳轮次：{best_epoch}，验证 Macro-F1：{best_f1:.4f}", flush=True)


if __name__ == "__main__":
    main()
