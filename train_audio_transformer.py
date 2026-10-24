"""音频 Transformer 基线（E0）：使用未标准化的 Log-Mel 训练十分类模型。

直接运行训练 20 轮，再加载最佳验证权重评估测试集；--check-only 只做小批量检查。
"""

import argparse
import csv
import hashlib
import io
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from audio_dataset import AudioConfig, UrbanSound8KDataset
from train_cnn import evaluate, set_seed, train_one_epoch
from audio_transformer import AudioTransformer


PROJECT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(r"D:\datasets\UrbanSound8K")
EPOCHS = 20
SEED = 42
NUM_WORKERS = 4
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


def save_predictions(frame, preds, labels, path, class_names):
    """按未打乱的元数据行顺序保存预测；调用方需保持样本顺序一致。"""
    result = frame.copy()
    if not np.array_equal(result["classID"].to_numpy(), labels):
        raise RuntimeError("预测标签与 metadata 顺序不一致")
    result["true_label"] = labels
    result["pred_label"] = preds
    result["pred_class"] = [class_names[int(p)] for p in preds]
    result["correct"] = preds == labels
    result.to_csv(path, index=False, encoding="utf-8-sig")


def save_training_curves(history_path, output_path):
    """读取逐轮记录，保存训练与验证的 Loss、Accuracy 和验证 Macro-F1 曲线。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = pd.read_csv(history_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history.epoch, history.train_loss, label="train")
    axes[0].plot(history.epoch, history.val_loss, label="validation")
    axes[0].set_title("Cross-entropy loss")
    axes[1].plot(history.epoch, history.train_acc, label="train")
    axes[1].plot(history.epoch, history.val_acc, label="validation")
    axes[1].set_title("Accuracy")
    axes[2].plot(history.epoch, history.val_macro_f1, label="validation")
    axes[2].set_title("Macro-F1")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def check_one_batch(model, train_loader, val_loader, optimizer, criterion, device):
    """真实音频的前向、反向、参数更新、评估和权重读写检查；不运行完整训练。"""
    xb, yb = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    assert tuple(xb.shape) == (TRAIN_BATCH_SIZE, 1, 64, 126)
    assert tuple(yb.shape) == (TRAIN_BATCH_SIZE,)
    assert torch.isfinite(xb).all().item()
    model.eval()
    with torch.no_grad():
        logits = model(xb.to(device))
    assert tuple(logits.shape) == (TRAIN_BATCH_SIZE, 10)
    assert torch.isfinite(logits).all().item()

    before = model.input_proj.weight.detach().clone()
    loss, _ = train_one_epoch(model, [(xb, yb)], optimizer, criterion, device)
    assert math.isfinite(loss)
    assert not torch.equal(before, model.input_proj.weight.detach())
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all().item()
        for p in model.parameters() if p.requires_grad
    )
    va_loss, _, _, preds, labels = evaluate(model, [val_batch], criterion, device)
    assert math.isfinite(va_loss)
    assert np.array_equal(labels, val_batch[1].numpy())
    assert len(preds) == len(labels) == EVAL_BATCH_SIZE

    # 在内存中验证权重保存和加载，不生成实验权重文件。
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = AudioTransformer().to(device)
    restored.load_state_dict(torch.load(buffer, map_location=device, weights_only=True))
    restored.eval()
    with torch.no_grad():
        expected = model(val_batch[0].to(device))
        actual = restored(val_batch[0].to(device))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print(f"device: {device}", flush=True)
    print(f"真实训练 batch: x={tuple(xb.shape)}, y={tuple(yb.shape)}", flush=True)
    print(f"logits: {tuple(logits.shape)}", flush=True)
    print(f"单 batch train loss={loss:.6f}; validation loss={va_loss:.6f}", flush=True)
    print("检查通过：梯度有限、参数已更新、评估标签顺序正确、权重重载输出完全一致。", flush=True)
    print("这里只检查了一个训练 batch 和一个验证 batch，未评估测试集。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="训练 UrbanSound8K 音频 Transformer（E0：原始 Log-Mel）")
    parser.add_argument("--check-only", action="store_true", help="只检查真实 batch，不完整训练")
    args = parser.parse_args()
    set_seed(SEED)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据：沿用 CNN 的固定 fold 划分和 Log-Mel 处理。
    csv_path = DATA_ROOT / "metadata" / "UrbanSound8K.csv"
    df = pd.read_csv(csv_path)
    cfg = AudioConfig()
    train_df = df[df["fold"].isin(range(1, 9))].reset_index(drop=True)
    val_df = df[df["fold"] == 9].reset_index(drop=True)
    test_df = df[df["fold"] == 10].reset_index(drop=True)
    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise RuntimeError("训练集、验证集或测试集为空，请检查 metadata")

    audio_dir = DATA_ROOT / "audio"
    train_ds = UrbanSound8KDataset(train_df, audio_dir, cfg)
    val_ds = UrbanSound8KDataset(val_df, audio_dir, cfg)
    test_ds = UrbanSound8KDataset(test_df, audio_dir, cfg)
    train_loader = DataLoader(
        train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )

    # 模型与优化器：参数随机初始化，从头训练。
    model = AudioTransformer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    if args.check_only:
        check_one_batch(model, train_loader, val_loader, optimizer, criterion, device)
        return

    # 每次创建新目录；上级目录名沿用历史实验，避免改变已有结果位置。
    run_name = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = PROJECT_DIR / "results" / "step6_transformer" / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_dir / "best_transformer.pt"
    history_path = output_dir / "history.csv"
    log_path = output_dir / "training.log"

    def log(message):
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    class_names = df.groupby("classID")["class"].first().to_dict()
    source_files = [
        "audio_dataset.py", "train_cnn.py", "transformer_layers.py",
        "audio_transformer.py", "train_audio_transformer.py",
    ]
    config = {
        "experiment": "E0_baseline",
        "model": "AudioTransformer",
        "model_config": {
            "n_mels": 64, "d_model": 128, "num_heads": 4,
            "num_encoder_layers": 2, "d_ff": 256,
            "encoder_dropout": 0.1, "classifier_dropout": 0.3,
            "pooling": "mean_over_time", "n_classes": 10,
        },
        "audio_config": {
            name: getattr(cfg, name) for name in [
                "sample_rate", "duration", "target_len", "n_mels", "n_fft",
                "hop_length", "f_min", "f_max", "top_db",
            ]
        },
        "seed": SEED, "epochs": EPOCHS, "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "loss": "CrossEntropyLoss", "selection_metric": "validation_macro_f1",
        "train_folds": list(range(1, 9)), "val_fold": 9, "test_fold": 10,
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "train_batch_size": TRAIN_BATCH_SIZE, "eval_batch_size": EVAL_BATCH_SIZE,
        "num_workers": NUM_WORKERS, "torch_num_threads": torch.get_num_threads(),
        "device": str(device), "torch_version": torch.__version__,
        "data_root": str(DATA_ROOT), "class_names": class_names,
        "metadata_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256((PROJECT_DIR / name).read_bytes()).hexdigest()
            for name in source_files
        },
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    columns = [
        "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
        "val_macro_f1", "epoch_seconds",
    ]
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=columns).writeheader()

    log(f"device: {device}")
    if device.type == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(device)}")
    log(f"train/val/test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    log(f"parameters: {config['parameter_count']}")
    log(f"results: {output_dir}")
    started = time.perf_counter()

    # 每轮训练后评估验证集，按 Macro-F1 保存最佳权重及对应预测。
    best_val_f1 = -math.inf
    best_epoch = None
    best_validation = None
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        log(f"Epoch {epoch:02d}/{EPOCHS}: training...")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, va_f1, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device,
        )
        elapsed = time.perf_counter() - t0
        if not all(math.isfinite(v) for v in (tr_loss, tr_acc, va_loss, va_acc, va_f1)):
            raise RuntimeError(f"第 {epoch} 轮出现非有限损失或指标，训练已停止")
        row = dict(zip(columns, [epoch, tr_loss, tr_acc, va_loss, va_acc, va_f1, elapsed]))
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=columns).writerow(row)
        log(
            f"[{epoch:02d}/{EPOCHS}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} val_f1={va_f1:.4f} | {elapsed:.1f}s"
        )
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_epoch = epoch
            best_validation = {"loss": va_loss, "accuracy": va_acc, "macro_f1": va_f1}
            torch.save(model.state_dict(), checkpoint_path)
            save_predictions(
                val_df, val_preds, val_labels, output_dir / "val_predictions.csv", class_names,
            )
            np.save(
                output_dir / "val_confusion_matrix.npy",
                confusion_matrix(val_labels, val_preds, labels=list(range(10))),
            )
            log(f"Saved best validation model: epoch={epoch}, macro-F1={va_f1:.4f}")

    # 训练结束后恢复最佳验证权重，再评估测试集；不按测试结果选择权重。
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    log(f"Training finished. Best epoch={best_epoch}, val macro-F1={best_val_f1:.4f}")
    log("Evaluating the best model on Fold 10 once...")
    te_loss, te_acc, te_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device,
    )
    if not all(math.isfinite(v) for v in (te_loss, te_acc, te_f1)):
        raise RuntimeError("测试损失或指标包含非有限值")
    save_predictions(
        test_df, test_preds, test_labels, output_dir / "test_predictions.csv", class_names,
    )
    np.save(
        output_dir / "test_confusion_matrix.npy",
        confusion_matrix(test_labels, test_preds, labels=list(range(10))),
    )
    metrics = {
        "completed_epochs": EPOCHS, "best_epoch": best_epoch,
        "validation": best_validation,
        "test": {"loss": te_loss, "accuracy": te_acc, "macro_f1": te_f1},
        "total_seconds": time.perf_counter() - started,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    log(f"Test acc={te_acc:.4f} macro-F1={te_f1:.4f}")
    save_training_curves(history_path, output_dir / "training_curves.png")
    log(f"Complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
