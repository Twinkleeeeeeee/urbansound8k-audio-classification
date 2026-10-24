"""CNN 验证误差分析：保存逐条预测、混淆矩阵、同源统计和试听对照。

只使用 Fold 9；复用已有权重，不训练，每次创建独立分析目录。
"""

import hashlib
import html
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 保存图片，不打开阻塞程序的绘图窗口。
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from audio_dataset import AudioConfig, UrbanSound8KDataset
from train_cnn import AudioCNN


PROJECT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(r"D:\datasets\UrbanSound8K")
CHECKPOINT_PATH = PROJECT_DIR / "best_cnn.pt"
OUTPUT_ROOT = PROJECT_DIR / "results" / "step5_validation"


# 收集预测：调用方使用 shuffle=False，使输出与元数据行顺序对应。
@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    shapes = None
    for x, y in loader:
        logits = model(x.to(device))
        if shapes is None:
            shapes = {"input": list(x.shape), "logits": list(logits.shape)}
        assert logits.shape == (len(y), 10)
        assert torch.isfinite(logits).all(), "模型输出包含 NaN 或 Inf"
        all_labels.append(y)
        all_probs.append(logits.softmax(dim=1).cpu())
    labels = torch.cat(all_labels).numpy()
    probs = torch.cat(all_probs).numpy()
    return labels, probs, shapes


# 混淆矩阵：行是真实类别，列是预测类别；归一化时按行除。
def plot_confusion_matrices(cm, class_names, output_dir):
    support = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, support, out=np.zeros_like(cm, dtype=float), where=support != 0)
    for values, filename, title in [
        (cm, "confusion_counts.png", "Validation confusion matrix: counts"),
        (normalized, "confusion_row_normalized.png", "Validation confusion matrix: row fractions"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 10), layout="constrained")
        image = ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if values is normalized else None)
        ax.set_xticks(range(10), class_names, rotation=45, ha="right")
        ax.set_yticks(range(10), class_names)
        ax.set(xlabel="Predicted class", ylabel="True class", title=title)
        for i in range(10):
            for j in range(10):
                value = values[i, j]
                label = f"{value:.2f}" if values is normalized else str(int(value))
                color = "white" if value > values.max() / 2 else "black"
                ax.text(j, i, label, ha="center", va="center", color=color, fontsize=8)
        fig.colorbar(image, ax=ax, shrink=0.8)
        fig.savefig(output_dir / filename, dpi=140, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
    return normalized


# 选样：每组最多两个不同来源的误判，并加入两个类别的正确样本作对照。
def select_examples(records, error_pairs):
    selected = []
    for pair_id, pair in enumerate(error_pairs[:2], start=1):
        true_id, pred_id = int(pair["true_label"]), int(pair["pred_label"])
        used_sources = set()
        errors = records[(records.true_label == true_id) & (records.pred_label == pred_id)]
        # 优先选未标记同源风险的样本；同等条件下选最大 Softmax 值较高的误判。
        errors = errors.sort_values(
            ["source_overlap_flag", "max_softmax", "slice_file_name"],
            ascending=[True, False, True],
        )
        candidates = [("error", errors, 2)]
        for class_id, role in [(true_id, "correct_true_class"), (pred_id, "correct_predicted_class")]:
            correct = records[(records.true_label == class_id) & records.correct]
            correct = correct.sort_values(
                ["source_overlap_flag", "max_softmax", "slice_file_name"],
                ascending=[True, False, True],
            )
            candidates.append((role, correct, 1))
        for role, rows, limit in candidates:
            count = 0
            for _, row in rows.iterrows():
                source = int(row.fsID)
                if source in used_sources:
                    continue
                used_sources.add(source)
                item = row.to_dict()
                item.update(pair_id=pair_id, role=role, case_id=f"P{pair_id}_{role}_{count + 1}")
                selected.append(item)
                count += 1
                if count == limit:
                    break
    return pd.DataFrame(selected)


def load_model_waveform(row, dataset):
    """仅为绘图和试听重建波形，复用 Dataset 的重采样器及定长处理。"""
    data, sample_rate = sf.read(row.audio_path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T).contiguous().mean(dim=0, keepdim=True)
    resampler = dataset._get_resampler(sample_rate)
    if resampler is not None:
        wav = resampler(wav)
    return dataset._pad_or_trim(wav)


# 绘图：所有对照样本共用波形幅度范围和 Log-Mel 色标。
def plot_examples(examples, error_pairs, dataset, output_dir):
    audio_dir = output_dir / "audio"
    audio_dir.mkdir()
    waves, mels = {}, {}
    for row in examples.itertuples():
        wav = load_model_waveform(row, dataset)
        log_mel, label = dataset[int(row.sample_index)]
        assert label == row.true_label
        assert wav.shape == (1, dataset.cfg.target_len)
        waves[row.case_id] = wav.squeeze(0).numpy()
        mels[row.case_id] = log_mel.squeeze(0).numpy()
        # FLOAT WAV 保留模型实际输入，不做额外增益或幅度裁剪。
        sf.write(audio_dir / f"{row.case_id}.wav", waves[row.case_id], dataset.cfg.sample_rate, subtype="FLOAT")
    vmin = min(float(mel.min()) for mel in mels.values())
    vmax = max(float(mel.max()) for mel in mels.values())
    amplitude = max(max(float(np.abs(wav).max()) for wav in waves.values()), 1e-6)
    figures = []
    for pair_id, pair in enumerate(error_pairs[:2], start=1):
        group = examples[examples.pair_id == pair_id]
        fig, axes = plt.subplots(len(group), 2, figsize=(16, 3.2 * len(group)), squeeze=False, layout="constrained")
        for axis_row, row in zip(axes, group.itertuples()):
            wave, mel = waves[row.case_id], mels[row.case_id]
            risk = " | SOURCE OVERLAP FLAG" if row.source_overlap_flag else ""
            title = (
                f"{row.role}: {row.true_class} -> {row.pred_class}{risk}\n"
                f"{row.slice_file_name} | fsID={row.fsID} | pmax={row.max_softmax:.3f}"
            )
            times = np.arange(len(wave)) / dataset.cfg.sample_rate
            axis_row[0].plot(times, wave, linewidth=0.4)
            axis_row[0].set(title=title, xlabel="Time (s)", ylabel="Amplitude", xlim=(0, dataset.cfg.duration), ylim=(-amplitude, amplitude))
            axis_row[0].title.set_fontsize(9)
            hop_seconds = dataset.cfg.hop_length / dataset.cfg.sample_rate
            image = axis_row[1].imshow(
                mel, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax,
                extent=(-hop_seconds / 2, (mel.shape[1] - 0.5) * hop_seconds, 0, dataset.cfg.n_mels),
            )
            axis_row[1].set(title="Log-Mel: actual model input", xlabel="Time (s)", ylabel="Mel bin", xlim=(0, dataset.cfg.duration))
        fig.suptitle(f"Fold 9: {pair['true_class']} -> {pair['pred_class']} | {pair['count']} errors", fontsize=14)
        fig.colorbar(image, ax=axes[:, 1].tolist(), label="Power (dB, shared scale)", shrink=0.8)
        filename = f"pair_{pair_id}_{pair['true_class']}_to_{pair['pred_class']}.png"
        fig.savefig(output_dir / filename, dpi=130, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        figures.append(filename)
    return figures, {"log_mel_vmin": vmin, "log_mel_vmax": vmax, "waveform_abs_limit": amplitude}


def write_reading_notes(output_dir, metrics, error_pairs, examples, figures, error_sources):
    lines = [
        "# CNN 验证集误差分析", "",
        f"- 数据：Fold 9，{metrics['samples']} 条；错误 {metrics['errors']} 条。",
        f"- Accuracy：{metrics['accuracy']:.4%}；Macro-F1：{metrics['macro_f1']:.4%}。",
        f"- 权重 SHA256：`{metrics['checkpoint_sha256']}`。",
        "- 本次只进行验证集推理，没有训练，也没有重新评估测试集。", "",
        "## 怎么看结果", "",
        "1. `predictions.csv`：一行对应一条验证音频；correct=False 表示误判。",
        "2. `per_class_metrics.csv`：分别查看 Precision、Recall、F1 和真实样本数。",
        "3. 两张混淆矩阵：行是真实类别、列是预测类别；归一化版本每行和为 1。",
        "4. `selected_examples.csv` 和下方对照图：先听音，再观察波形和模型实际接收的 Log-Mel。",
        "5. 打开 `listening.html` 可试听选中样本的预处理波形。", "",
        "## 当前主要混淆", "",
        "|真实类别|预测类别|错误数|", "|---|---|---:|",
    ]
    lines += [f"|{p['true_class']}|{p['pred_class']}|{p['count']}|" for p in error_pairs[:8]]
    lines += ["", "## 主要混淆按原始录音分组", "",
              "多条切片可能来自同一段录音。错误切片数量不等于独立录音数量。", "",
              "|真实类别|预测类别|fsID|错误切片数|同源风险标记|", "|---|---|---:|---:|---|"]
    for pair in error_pairs[:2]:
        sources = error_sources[(error_sources.true_class == pair["true_class"]) & (error_sources.pred_class == pair["pred_class"])]
        for row in sources.itertuples():
            lines.append(f"|{row.true_class}|{row.pred_class}|{row.fsID}|{row.count}|{row.source_overlap_flag}|")
    lines += ["", "## 对照图", ""]
    for filename in figures:
        lines += [f"![{filename}]({filename})", ""]
    lines += [
        "## 解释边界", "",
        "- 选样规则：每组优先取不同 fsID、未标记同源风险的高分误判，再加入正确分类对照。目的性选样不代表整个类别的分布。",
        "- source_overlap_flag 表示按元数据，该录音 fsID 同时出现在训练集和验证集；不等于已证明波形完全重复。标记样本仍计入完整验证指标。",
        "- max_softmax 是模型最大 Softmax 输出，不是经过校准的正确概率。高分误判也需要检查。",
        "- 所有对照图共用色标和波形幅度范围；dB 来自数字音频特征，不是实测声压级。",
        "- 可观察时长、补零、时频结构、背景声差异；没有听音或对照实验支持的解释应标为假设。",
        "- 不能仅凭这些图断言 CNN 不适合时间结构，或 Transformer 一定更好。",
        "- 这份权重是根据同一个验证集选出的，验证分数用于分析和选模型，不能当作独立最终测试成绩。",
        "", "## 你的观察记录（先留空）", "",
        "|样本/对照组|听到了什么|图中观察到什么|可能原因（假设）|后续怎样验证|",
        "|---|---|---|---|---|", "| | | | | |", "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    cards = []
    for row in examples.itertuples():
        title = html.escape(f"{row.case_id}: {row.true_class} -> {row.pred_class}")
        detail = html.escape(f"{row.slice_file_name} | fsID={row.fsID} | source_overlap_flag={row.source_overlap_flag}")
        cards.append(f'<section><h3>{title}</h3><p>{detail}</p><audio controls preload="none" src="audio/{row.case_id}.wav"></audio></section>')
    images = "".join(f'<img alt="{html.escape(name)}" src="{html.escape(name)}">' for name in figures)
    page = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>CNN 验证集试听</title>'
        '<style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;background:#f5f7fa;color:#182230}'
        'section{background:white;padding:12px 20px;margin:12px 0;border:1px solid #dbe2ea;border-radius:8px}'
        'audio{width:100%;max-width:520px}img{width:100%;margin:20px 0}p{overflow-wrap:anywhere}</style>'
        '<h1>CNN 验证集样本对照</h1><p>音频为单声道、16 kHz、固定 4 秒的实际模型输入；没有额外音量归一化。'
        '原始标签与预测显示在各条上方。先记录听音和图像观察，再提出原因假设。</p>'
        + "".join(cards) + images + '</html>'
    )
    (output_dir / "listening.html").write_text(page, encoding="utf-8")


# 分析入口：读元数据 → 验证预测 → 保存记录 → 挑选样本 → 绘图。
def main():
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata_path = DATA_ROOT / "metadata" / "UrbanSound8K.csv"
    df = pd.read_csv(metadata_path)
    val_df = df[df["fold"] == 9].reset_index(drop=True)
    train_df = df[df["fold"].isin(range(1, 9))]
    class_names = df.groupby("classID")["class"].first().reindex(range(10)).tolist()
    shared_sources = sorted(set(train_df.fsID) & set(val_df.fsID))
    cfg = AudioConfig()
    dataset = UrbanSound8KDataset(val_df, DATA_ROOT / "audio", cfg)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    # 从同一份字节加载并计算哈希，保证记录的哈希对应本次使用的权重。
    checkpoint_bytes = CHECKPOINT_PATH.read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    model = AudioCNN(n_classes=10).to(device)
    model.load_state_dict(torch.load(io.BytesIO(checkpoint_bytes), map_location=device, weights_only=True), strict=True)
    print(f"Validation: fold=9 samples={len(dataset)} device={device}", flush=True)
    labels, probs, shapes = collect_predictions(model, loader, device)
    assert len(labels) == len(val_df)
    assert np.array_equal(labels, val_df.classID.to_numpy()), "预测和元数据未按行对齐"
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-5)
    preds = probs.argmax(axis=1)
    cm = confusion_matrix(labels, preds, labels=list(range(10)))
    assert cm.sum() == len(val_df)
    assert np.array_equal(cm.sum(axis=1), val_df.groupby("classID").size().reindex(range(10), fill_value=0))

    records = val_df[["slice_file_name", "fold", "fsID", "start", "end"]].copy()
    records.insert(0, "sample_index", np.arange(len(records)))
    records["true_label"], records["pred_label"] = labels, preds
    records["true_class"] = [class_names[i] for i in labels]
    records["pred_class"] = [class_names[i] for i in preds]
    records["correct"] = labels == preds
    records["max_softmax"] = probs.max(axis=1)
    records["metadata_duration"] = records["end"] - records["start"]
    records["source_overlap_flag"] = records.fsID.isin(shared_sources)
    records["audio_path"] = [str((DATA_ROOT / "audio" / "fold9" / filename).resolve()) for filename in records.slice_file_name]
    assert records.audio_path.map(lambda path: Path(path).is_file()).all()
    error_pairs = [
        {"true_label": i, "pred_label": j, "true_class": class_names[i], "pred_class": class_names[j], "count": int(cm[i, j])}
        for i in range(10) for j in range(10) if i != j and cm[i, j] > 0
    ]
    error_pairs.sort(key=lambda pair: (-pair["count"], pair["true_label"], pair["pred_label"]))
    error_sources = records[~records.correct].groupby(
        ["true_class", "pred_class", "fsID", "source_overlap_flag"], as_index=False
    ).size().rename(columns={"size": "count"}).sort_values("count", ascending=False)
    report = classification_report(labels, preds, labels=list(range(10)), target_names=class_names, output_dict=True, zero_division=0)
    metrics = {
        "split": "validation", "fold": 9, "samples": len(labels), "errors": int((labels != preds).sum()),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, labels=list(range(10)), average="macro", zero_division=0)),
        "checkpoint_sha256": checkpoint_sha256, "train_val_shared_fsID": [int(source) for source in shared_sources],
    }
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = OUTPUT_ROOT / f"{timestamp}_{checkpoint_sha256[:8]}"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "checkpoint_used.pt").write_bytes(checkpoint_bytes)
    records.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({name: report[name] for name in class_names}).T.rename_axis("class").to_csv(output_dir / "per_class_metrics.csv", encoding="utf-8-sig")
    pd.DataFrame(error_pairs).to_csv(output_dir / "error_pairs.csv", index=False, encoding="utf-8-sig")
    error_sources.to_csv(output_dir / "error_sources.csv", index=False, encoding="utf-8-sig")
    np.save(output_dir / "confusion_counts.npy", cm)
    normalized = plot_confusion_matrices(cm, class_names, output_dir)
    assert np.allclose(normalized.sum(axis=1), 1)
    np.save(output_dir / "confusion_row_normalized.npy", normalized)
    examples = select_examples(records, error_pairs)
    figures, plot_scale = [], {}
    if not examples.empty:
        examples.to_csv(output_dir / "selected_examples.csv", index=False, encoding="utf-8-sig")
        figures, plot_scale = plot_examples(examples, error_pairs, dataset, output_dir)
    else:
        print("No error pairs: no comparison examples were generated.", flush=True)
    config = {
        "data_root": str(DATA_ROOT), "checkpoint_source": str(CHECKPOINT_PATH),
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "audio_config": {key: getattr(cfg, key) for key in ["sample_rate", "duration", "n_mels", "n_fft", "hop_length", "f_min", "f_max", "top_db", "target_len"]},
        "class_names": class_names, "batch_size": 64, "shuffle": False, "num_workers": 0,
        "first_batch_shapes": shapes, "plot_scale": plot_scale,
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    write_reading_notes(output_dir, metrics, error_pairs, examples, figures, error_sources)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print("First batch shapes:", shapes, flush=True)
    print("Selected examples:", len(examples), flush=True)
    print("Output:", output_dir, flush=True)


if __name__ == "__main__":
    main()
