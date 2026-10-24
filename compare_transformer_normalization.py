"""比较 CNN、原始 Transformer（E0）和标准化 Transformer（E1）的验证结果。

必要时补充 E0 最佳权重的训练集评估；不训练、不评估测试集。
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from audio_dataset import AudioConfig, UrbanSound8KDataset
from train_cnn import evaluate
from audio_transformer import AudioTransformer


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="对照已有的 CNN、E0、E1 验证结果")
    parser.add_argument("--e0-run", type=Path, default=ROOT / "results/step6_transformer/20261016_121144_056759")
    parser.add_argument("--e1-run", type=Path, default=ROOT / "results/step6_5_normalization/20261016_163311_346140")
    args = parser.parse_args()
    e0_dir, e1_dir = args.e0_run, args.e1_run
    e0 = json.loads((e0_dir / "metrics.json").read_text(encoding="utf-8"))
    e1 = json.loads((e1_dir / "metrics.json").read_text(encoding="utf-8"))
    cfg = json.loads((e1_dir / "config.json").read_text(encoding="utf-8"))
    stats = json.loads((e1_dir / "normalization.json").read_text(encoding="utf-8"))

    # Windows 的多进程 DataLoader 需要从保存的文件、main 保护下运行。
    cache = e1_dir / "baseline_E0_train_eval.json"
    checkpoint_hash = hashlib.sha256((e0_dir / "best_transformer.pt").read_bytes()).hexdigest()
    assert checkpoint_hash == e0["checkpoint_sha256"]
    if cache.exists():
        e0_train = json.loads(cache.read_text(encoding="utf-8"))
        assert e0_train["checkpoint_sha256"] == checkpoint_hash
    else:
        torch.set_num_threads(2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AudioTransformer().to(device)
        model.load_state_dict(torch.load(e0_dir / "best_transformer.pt", map_location=device, weights_only=True))
        meta = pd.read_csv(Path(cfg["data_root"]) / "metadata/UrbanSound8K.csv")
        frame = meta[meta.fold.isin(range(1, 9))].reset_index(drop=True)
        ds = UrbanSound8KDataset(frame, Path(cfg["data_root"]) / "audio", AudioConfig())
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
        print("Evaluating E0 best checkpoint on training folds in eval mode...", flush=True)
        loss, acc, f1, _, labels = evaluate(model, loader, torch.nn.CrossEntropyLoss(), device)
        assert np.array_equal(labels, frame.classID.to_numpy())
        e0_train = {
            "checkpoint_sha256": checkpoint_hash, "best_epoch": e0["best_epoch"],
            "samples": len(labels), "loss": loss, "accuracy": acc, "macro_f1": f1,
            "mode": "eval", "input": "original Log-Mel", "test_evaluated": False,
        }
        cache.write_text(json.dumps(e0_train, indent=2), encoding="utf-8")

    cnn_dir = ROOT / "results/step5_validation/20261015_211927_259155_a3ec280e"
    frames = {
        "CNN": pd.read_csv(cnn_dir / "predictions.csv"),
        "E0": pd.read_csv(e0_dir / "val_predictions.csv"),
        "E1": pd.read_csv(e1_dir / "val_predictions.csv"),
    }
    reference = frames["E0"].set_index("slice_file_name")
    summary = []
    per_class = []
    for name, frame in frames.items():
        ordered = frame.set_index("slice_file_name").loc[reference.index]
        assert len(ordered) == 816
        assert np.array_equal(ordered.true_label, reference.true_label)
        keep = ~ordered.fsID.isin(cfg["train_val_shared_fsID"])
        acc = accuracy_score(ordered.true_label, ordered.pred_label)
        f1 = f1_score(ordered.true_label, ordered.pred_label, average="macro")
        summary.append({
            "model": name, "accuracy": acc, "macro_f1": f1,
            "macro_f1_without_shared_source": f1_score(
                ordered.loc[keep, "true_label"], ordered.loc[keep, "pred_label"], average="macro",
            ),
        })
        p, r, f, support = precision_recall_fscore_support(
            ordered.true_label, ordered.pred_label, labels=list(range(10)), zero_division=0,
        )
        for i in range(10):
            per_class.append({"model": name, "class": cfg["class_names"][str(i)],
                              "precision": p[i], "recall": r[i], "f1": f[i], "support": int(support[i])})
    table = pd.DataFrame(summary)
    table.to_csv(e1_dir / "validation_comparison.csv", index=False)
    pd.DataFrame(per_class).to_csv(e1_dir / "per_class_comparison.csv", index=False)
    assert abs(table.loc[table.model == "E0", "macro_f1"].iloc[0] - e0["validation"]["macro_f1"]) < 1e-12
    assert abs(table.loc[table.model == "E1", "macro_f1"].iloc[0] - e1["validation"]["macro_f1"]) < 1e-12

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, directory in [("E0", e0_dir), ("E1", e1_dir)]:
        history = pd.read_csv(directory / "history.csv")
        axes[0].plot(history.epoch, history.val_macro_f1, label=name)
        axes[1].plot(history.epoch, history.val_loss, label=name)
    axes[0].set_title("Validation Macro-F1")
    axes[1].set_title("Validation cross-entropy")
    for ax in axes[:2]:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[2].bar(table.model, table.macro_f1)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Best-checkpoint validation Macro-F1")
    for i, value in enumerate(table.macro_f1):
        axes[2].text(i, value + 0.015, f"{value:.2%}", ha="center")
    fig.tight_layout()
    fig.savefig(e1_dir / "validation_comparison.png", dpi=160)
    plt.close(fig)

    delta = (e1["validation"]["macro_f1"] - e0["validation"]["macro_f1"]) * 100
    e1_train = e1["train_eval_best_checkpoint"]
    comparison = {"validation": summary, "E0_train_eval": e0_train, "E1_train_eval": e1_train,
                  "validation_macro_f1_delta_percentage_points": delta, "test_evaluated_by_comparison": False}
    (e1_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    report = f"""# E1：固定全局标准化实验

已完成 20 轮训练。E1 唯一新增的处理是对 Log-Mel 使用训练集固定全局标准化；模型和训练超参数沿用 E0。

## 验证结果

|模型|Accuracy|Macro-F1|
|---|---:|---:|
|CNN|{summary[0]['accuracy']:.2%}|{summary[0]['macro_f1']:.2%}|
|E0 原始 Transformer|{e0['validation']['accuracy']:.2%}|{e0['validation']['macro_f1']:.2%}|
|E1 标准化 Transformer|{e1['validation']['accuracy']:.2%}|{e1['validation']['macro_f1']:.2%}|

E1 最佳模型来自第 {e1['best_epoch']} 轮，验证 Macro-F1 比 E0 高 {delta:.2f} 个百分点。本次 E1 和本次对照均未评估测试集。

这是种子 42 的一次运行，只能支持“标准化在本次配置和运行中提高了最佳验证成绩”，不能据此声称稳定或显著提升，也不能认定初始化注意力过度集中是低分的唯一原因。

## 标准化做了什么

仅使用 Fold 1–8 的 {stats['samples']} 条音频、{stats['elements']} 个 Log-Mel 值拟合一个标量均值和标准差：

```python
log_mel = (log_mel - ({stats['mean']:.10f})) / ({stats['std']:.10f} + 1e-8)
```

统计包含补零波形对应的频谱区域，标准差使用总体定义（ddof=0）。训练与验证共用这组冻结统计量，不按音频或 batch 单独计算。输入仍为 `[B, 1, 64, 126]`，模型输出仍为 `[B, 10]`。

加载 `best_transformer.pt` 做推理时，必须同时读取 `normalization.json`，先按上述公式处理 Log-Mel；模型 state_dict 本身不包含这两个统计量。

## 更一致的训练/验证差距

以下两组训练指标均来自各自最佳验证权重，固定参数、eval 模式、关闭 Dropout、无增强：

|模型|训练 Accuracy|验证 Accuracy|差距|
|---|---:|---:|---:|
|E0|{e0_train['accuracy']:.2%}|{e0['validation']['accuracy']:.2%}|{(e0_train['accuracy']-e0['validation']['accuracy'])*100:.2f} 个百分点|
|E1|{e1_train['accuracy']:.2%}|{e1['validation']['accuracy']:.2%}|{(e1_train['accuracy']-e1['validation']['accuracy'])*100:.2f} 个百分点|

E1 后期训练 Loss 继续下降，验证 Loss 整体走高，过拟合问题仍存在。不能用最后一轮的在线训练 Accuracy 代替上述最佳权重的 eval 结果。

## 公平性与验证

- Fold 1–8 训练、Fold 9 验证；固定划分实验，不是十折交叉验证。
- 与 E0 相同的模型源码、2 层 Encoder、4 个头、128 维、FFN 256、274570 个参数。
- 同样的 seed 42、20 轮、AdamW、lr=0.001、weight_decay=0.0001、batch size 32/64、num_workers=4。
- 统计拟合和真实 batch 检查后重置随机种子、重新创建模型与优化器；初始化哈希已核对，检查中的参数更新不会带入正式训练。
- 已验证不同大小 batch 的流式统计、标准化公式、标签保持、真实 batch 前向/反向及权重重载。
- 已核对完整 20 轮、816 条验证预测的文件名/标签顺序、重算指标及混淆矩阵、权重和源码哈希；原有 CNN、E0 和基础模块未覆盖。
- 共享来源 fsID=180937 风险沿用原划分，并在 `validation_comparison.csv` 中报告排除该来源的辅助统计；该统计不用于选模型，也不是重新训练后的无泄漏结果。

## 主要文件

- `normalization.json`：训练集均值、标准差、统计范围。
- `best_transformer.pt`：最佳模型权重。
- `config.json`、`checks.json`、`source/`：配置、检查结果、训练源码存档。
- `history.csv`、`training.log`、`training_curves.png`：逐轮记录。
- `metrics.json`、`val_predictions.csv`、`val_confusion_matrix.npy`：最终验证结果。
- `validation_comparison.png`、`validation_comparison.csv`、`per_class_comparison.csv`：对照图和逐类指标。
- `baseline_E0_train_eval.json`：补充的 E0 最佳权重训练集评估；未重训。

## 运行

在项目目录运行 `train_audio_transformer_normalized.py` 可重做 E1，每次创建新结果目录；直接运行会开始统计与训练，不接受命令行选项。基线训练入口为 `train_audio_transformer.py`。`compare_transformer_normalization.py` 对照这里记录的 E0/E1 结果，重复执行时复用经过权重哈希核对的 E0 训练集评估缓存。
"""
    (e1_dir / "README.md").write_text(report, encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
    print(f"Comparison saved to: {e1_dir}", flush=True)


if __name__ == "__main__":
    main()
