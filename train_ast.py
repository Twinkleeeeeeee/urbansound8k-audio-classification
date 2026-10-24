"""UrbanSound8K 预训练 AST 微调，数据、模型、训练和评估集中在本文件。

直接运行：训练 20 轮，Fold 1-8 训练、Fold 9 验证，不自动评估测试集。
只训练分类头：python train_ast.py --finetune head
预测一条 WAV：python train_ast.py --predict path/to/audio.wav
评估已有结果：python train_ast.py --eval-run results/ast/某次运行目录 --split validation
最终方案确定后，将 --split 改为 test 单独评估 Fold 10。
"""

import argparse
import json
import math
import random
import hashlib
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import transformers
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification


# 配置：数据位置、预训练来源、音频和训练参数。
DATA_ROOT = Path(r"D:\datasets\UrbanSound8K")
OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "ast"
CACHE_DIR = Path(__file__).resolve().parent / "models" / "huggingface"
PREDICTION_RUN = Path(__file__).resolve().parent / "results" / "ast" / "20261019_101342_538523"
MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
MODEL_REVISION = "f826b80d28226b62986cc218e5cec390b1096902"
CLASS_NAMES = (
    "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
    "engine_idling", "gun_shot", "jackhammer", "siren", "street_music",
)
SAMPLE_RATE = 16000
DURATION = 4.0
EPOCHS = 20
SEED = 42
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
ACCUMULATION_STEPS = 16  # 通常等效 batch size 为 2 x 16 = 32。
NUM_WORKERS = 0
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
USE_AMP = True  # CUDA 上使用混合精度；CPU 自动使用 float32。



# 数据集：波形预处理与 AST 配套特征。


def load_feature_extractor():
    """加载与预训练权重配套的特征配置；不拟合验证集或测试集统计量。"""
    return ASTFeatureExtractor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=str(CACHE_DIR),
    )


def extract_audio_features(path, feature_extractor, target_len, resamplers=None):
    """WAV → 单声道 → 16kHz → 定长 → 配套特征，默认返回 [1024, 128]。"""
    if resamplers is None:
        resamplers = {}
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T.copy()).mean(dim=0)  # [原始采样点数]
    if sample_rate != SAMPLE_RATE:
        if sample_rate not in resamplers:
            resamplers[sample_rate] = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
        wav = resamplers[sample_rate](wav)
    wav = wav[:target_len]
    if wav.numel() < target_len:
        wav = F.pad(wav, (0, target_len - wav.numel()))
    # 内部完成 Kaldi fbank、时间帧补齐/截断，以及预训练模型配套的标准化。
    inputs = feature_extractor(wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
    return inputs["input_values"].squeeze(0)


class UrbanSound8KASTDataset(Dataset):
    """按索引读取 WAV；默认返回特征 [1024, 128] 和 0–9 的整数标签。"""

    def __init__(self, df, audio_dir, feature_extractor, duration=DURATION):
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.feature_extractor = feature_extractor
        self.target_len = int(SAMPLE_RATE * duration)
        self._resamplers = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.audio_dir / f"fold{int(row['fold'])}" / row['slice_file_name']
        features = extract_audio_features(path, self.feature_extractor,
                                          target_len=self.target_len, resamplers=self._resamplers)
        return features, int(row['classID'])


# 模型：预训练编码器、十分类输出和本项目检查点的恢复。


class AudioAST(nn.Module):
    """不传 config 时加载预训练参数；传入 config 时仅按结构创建模型。"""

    def __init__(self, config=None):
        super().__init__()
        self.freeze_encoder = False
        if config is None:
            # 保留预训练编码器；原来的 527 类输出层重新初始化为 10 类。
            self.ast, self.loading_info = ASTForAudioClassification.from_pretrained(
                MODEL_ID, revision=MODEL_REVISION, cache_dir=str(CACHE_DIR),
                num_labels=10, id2label=dict(enumerate(CLASS_NAMES)),
                label2id={name: i for i, name in enumerate(CLASS_NAMES)},
                problem_type="single_label_classification",
                ignore_mismatched_sizes=True, output_loading_info=True,
                use_safetensors=True, attn_implementation="sdpa",
            )
        else:
            # 恢复自己的检查点时直接按保存的结构创建，无需再次下载预训练权重。
            config._attn_implementation = "sdpa"
            self.ast = ASTForAudioClassification(config)
            self.loading_info = None

    def forward(self, x):
        return self.ast(input_values=x).logits  # [B, 1024, 128] -> [B, 10]


def configure_finetuning(model, finetune):
    """选择全参数微调或只训练分类头，返回总参数量和可训练参数量。"""
    if finetune not in ("all", "head"):
        raise ValueError("finetune 必须为 all 或 head")
    model.freeze_encoder = finetune == "head"
    for p in model.parameters():
        p.requires_grad = True
    if model.freeze_encoder:
        for p in model.ast.audio_spectrogram_transformer.parameters():
            p.requires_grad = False
        model.ast.gradient_checkpointing_disable()
        model.ast.audio_spectrogram_transformer.eval()
    else:
        model.ast.gradient_checkpointing_enable()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_ast_run(run_dir, device):
    """按保存的结构创建模型，再装入该实验的 best_ast.pt；不重新训练。"""
    run_dir = Path(run_dir)
    config = ASTConfig.from_json_file(run_dir / "ast_config.json")
    model = AudioAST(config=config)
    parameters = torch.load(run_dir / "best_ast.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(parameters)
    return model.to(device)


@torch.no_grad()
def predict_file(audio_path, run_dir=PREDICTION_RUN):
    """恢复指定实验的最佳权重，预测一条音频；不需要 metadata 或真实标签。"""
    audio_path, run_dir = Path(audio_path), Path(run_dir)
    if not audio_path.is_file():
        raise FileNotFoundError(f"音频文件不存在：{audio_path}")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config["sample_rate"] != SAMPLE_RATE:
        raise ValueError(f"该实验需要 {config['sample_rate']} Hz，本程序使用 {SAMPLE_RATE} Hz")
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = ASTFeatureExtractor.from_pretrained(run_dir, local_files_only=True)
    x = extract_audio_features(audio_path, extractor, target_len=int(SAMPLE_RATE * config["duration"]))
    if not torch.isfinite(x).all().item():
        raise ValueError("音频提取的特征包含 NaN 或 Inf")
    x = x.unsqueeze(0).to(device)  # [1024, 128] → [1, 1024, 128]
    model = load_ast_run(run_dir, device)
    model.eval()
    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=config["mixed_precision"] and device.type == "cuda"):
        logits = model(x)  # [1, 10]
    if not torch.isfinite(logits).all().item():
        raise RuntimeError("预测分数包含 NaN 或 Inf")
    pred = logits.argmax(dim=1).item()
    score = torch.softmax(logits.float(), dim=1)[0, pred].item()
    result = {"audio_path": str(audio_path.resolve()), "model_run": str(run_dir.resolve()),
              "device": str(device), "input_shape": list(x.shape), "logits_shape": list(logits.shape),
              "pred_label": pred, "pred_class": config["class_names"][str(pred)], "softmax_score": score}
    print(f"音频：{result['audio_path']}")
    print(f"模型：{result['model_run']}")
    print(f"设备：{device}，输入：{result['input_shape']}，输出：{result['logits_shape']}")
    print(f"预测类别：{pred} / {result['pred_class']}")
    print(f"Softmax 分数：{score:.2%}（不是预测正确率）")
    return result


# 辅助函数：固定种子、保存配置、逐条预测记录和训练曲线。


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path, data):
    """将配置或指标写入 UTF-8 JSON 文件。"""
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


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
    """读取逐轮指标并保存曲线，不重新进行模型推理。"""
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


# 单轮训练与评估：梯度累积、混合精度、Accuracy 和 Macro-F1。


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, accumulation_steps):
    """累计各组样本的平均损失梯度，返回整轮平均 Loss 和 Accuracy。"""
    model.train()
    if isinstance(model, AudioAST) and model.freeze_encoder:
        # 外层 train() 会切换所有子模块；冻结编码器始终使用评估模式。
        model.ast.audio_spectrogram_transformer.eval()
    optimizer.zero_grad(set_to_none=True)
    total_loss, total_correct, total = 0.0, 0, 0
    amp_enabled = scaler.is_enabled()
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        # 每组开始时确定实际样本数；末组不足时仍按每条音频等权累计梯度。
        if batch_idx % accumulation_steps == 0:
            group_samples = min(loader.batch_size * accumulation_steps,
                                len(loader.dataset) - batch_idx * loader.batch_size)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(x)
            loss = criterion(logits, y)
        if not torch.isfinite(loss).item():
            raise RuntimeError("训练 Loss 出现 NaN 或 Inf，已停止训练")
        # 小批平均损失 × 小批条数 / 整组条数；scale 用于缓解低精度梯度下溢。
        scaler.scale(loss * x.size(0) / group_samples).backward()
        if (batch_idx + 1) % accumulation_steps == 0 or batch_idx + 1 == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
        if (batch_idx + 1) % 500 == 0:
            print(f"  batch {batch_idx + 1}/{len(loader)}，train_loss={total_loss / total:.4f}", flush=True)
    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=USE_AMP):
    """固定参数，返回平均 Loss、Accuracy、Macro-F1、预测编号和真实编号。"""
    model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=use_amp and device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)
        if not torch.isfinite(loss).item():
            raise RuntimeError("评估 Loss 出现 NaN 或 Inf")
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(y.cpu())
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return (total_loss / total, accuracy_score(labels, preds),
            f1_score(labels, preds, labels=list(range(10)), average="macro", zero_division=0), preds, labels)


# 训练入口：准备数据和模型，逐轮验证选优，保存实验结果。


def train(finetune="all"):
    set_seed(SEED)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = OUTPUT_ROOT if finetune == "all" else OUTPUT_ROOT.parent / "ast_head_only"
    output_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), output_dir / "train_ast.py")
    started_at = datetime.now().astimezone().isoformat()
    started_clock = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def log(message):
        print(message, flush=True)
        with (output_dir / "training.log").open("a", encoding="utf-8") as file:
            file.write(message + "\n")

    # 数据范围、划分和标签沿用现有实验；特征使用 AST 配套处理。
    df = pd.read_csv(DATA_ROOT / "metadata" / "UrbanSound8K.csv")
    train_df = df[df.fold.isin(range(1, 9))].reset_index(drop=True)
    val_df = df[df.fold == 9].reset_index(drop=True)
    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError("训练集或验证集为空，请检查 metadata")
    extractor = load_feature_extractor()
    train_ds = UrbanSound8KASTDataset(train_df, DATA_ROOT / "audio", extractor, duration=DURATION)
    val_ds = UrbanSound8KASTDataset(val_df, DATA_ROOT / "audio", extractor, duration=DURATION)
    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 两种方案都从原始预训练权重开始，不加载已微调的最佳权重。
    log(f"device={device}，train/val={len(train_ds)}/{len(val_ds)}，结果目录={output_dir}")
    log("加载 AudioSet 预训练 AST；首次运行需要下载约 350MB 权重。")
    model = AudioAST().to(device)
    total_parameters, trainable_parameters = configure_finetuning(model, finetune)
    head_hash = hashlib.sha256()
    for name, p in model.ast.classifier.named_parameters():
        head_hash.update(name.encode("utf-8"))
        head_hash.update(p.detach().cpu().contiguous().numpy().tobytes())
    log(f"finetune={finetune}，总参数={total_parameters}，可训练参数={trainable_parameters}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and device.type == "cuda", init_scale=1024)
    extractor.save_pretrained(output_dir)
    model.ast.config.to_json_file(output_dir / "ast_config.json")
    config = {
        "experiment": "E2_AST_AudioSet_finetuning" if finetune == "all" else "E3_AST_head_only",
        "model_id": MODEL_ID, "revision": MODEL_REVISION,
        "data_root": str(DATA_ROOT), "sample_rate": SAMPLE_RATE, "duration": DURATION,
        "train_folds": list(range(1, 9)), "val_fold": 9, "test_fold": 10,
        "train_samples": len(train_ds), "val_samples": len(val_ds), "seed": SEED, "epochs": EPOCHS,
        "train_batch_size": TRAIN_BATCH_SIZE, "eval_batch_size": EVAL_BATCH_SIZE,
        "accumulation_steps": ACCUMULATION_STEPS, "effective_batch_size": TRAIN_BATCH_SIZE * ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "num_workers": NUM_WORKERS,
        "mixed_precision": scaler.is_enabled(), "gradient_checkpointing": finetune == "all",
        "finetune": "all_parameters" if finetune == "all" else "classifier_only",
        "total_parameters": total_parameters, "trainable_parameters": trainable_parameters,
        "initial_classifier_sha256": head_hash.hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_at": started_at,
        "selection_metric": "validation_macro_f1", "class_names": dict(enumerate(CLASS_NAMES)),
        "train_val_shared_fsID": sorted(set(train_df.fsID) & set(val_df.fsID)),
        "preprocessing": extractor.to_dict(), "test_evaluated": False,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
    }
    save_json(output_dir / "config.json", config)

    # 每轮训练和验证；最佳权重、验证预测和混淆矩阵来自同一轮。
    history, best_f1, best_epoch, best_validation = [], -1.0, None, None
    for epoch in range(1, EPOCHS + 1):
        epoch_clock = time.perf_counter()
        log(f"Epoch {epoch}/{EPOCHS}：开始训练")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, ACCUMULATION_STEPS)
        va_loss, va_acc, va_f1, preds, labels = evaluate(model, val_loader, criterion, device)
        if not all(math.isfinite(v) for v in (tr_loss, tr_acc, va_loss, va_acc, va_f1)):
            raise RuntimeError("训练或验证出现无效指标")
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "val_macro_f1": va_f1,
                        "epoch_seconds": time.perf_counter() - epoch_clock})
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        log(f"[{epoch}/{EPOCHS}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} val_f1={va_f1:.4f}")
        if va_f1 > best_f1:
            best_f1, best_epoch = va_f1, epoch
            best_validation = {"loss": va_loss, "accuracy": va_acc, "macro_f1": va_f1}
            torch.save(model.state_dict(), output_dir / "best_ast.pt")
            save_predictions(val_df, preds, labels, output_dir / "val_predictions.csv", dict(enumerate(CLASS_NAMES)))
            np.save(output_dir / "val_confusion_matrix.npy", confusion_matrix(labels, preds, labels=list(range(10))))

    # 恢复最佳权重并评估训练集，使训练、验证指标对应同一套参数。
    model.load_state_dict(torch.load(output_dir / "best_ast.pt", map_location=device, weights_only=True))
    train_eval_loader = DataLoader(train_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    loss, acc, f1, _, _ = evaluate(model, train_eval_loader, criterion, device)
    save_json(output_dir / "metrics.json", {
        "completed_epochs": EPOCHS, "best_epoch": best_epoch, "validation": best_validation,
        "train_eval_best_checkpoint": {"loss": loss, "accuracy": acc, "macro_f1": f1}, "test_evaluated": False,
        "finetune": config["finetune"], "trainable_parameters": trainable_parameters,
        "started_at": started_at, "finished_at": datetime.now().astimezone().isoformat(),
        "elapsed_seconds": time.perf_counter() - started_clock,
        "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_cuda_memory_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
    })
    save_training_curves(output_dir / "history.csv", output_dir / "training_curves.png")
    log(f"完成。最佳轮次={best_epoch}，验证 Macro-F1={best_f1:.4f}。本次未评估测试集。")


# 独立评估：恢复已有实验的模型与特征处理设置。


def evaluate_run(run_dir, split="validation"):
    """读取已保存的结构、特征配置和权重；不重新训练。"""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    data_root = Path(config["data_root"])
    df = pd.read_csv(data_root / "metadata" / "UrbanSound8K.csv")
    if split == "train":
        frame = df[df.fold.isin(config["train_folds"])]
    else:
        fold = config["val_fold"] if split == "validation" else config["test_fold"]
        frame = df[df.fold == fold]
    frame = frame.reset_index(drop=True)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = ASTFeatureExtractor.from_pretrained(run_dir, local_files_only=True)
    dataset = UrbanSound8KASTDataset(frame, data_root / "audio", extractor, duration=config["duration"])
    loader = DataLoader(dataset, batch_size=config["eval_batch_size"], shuffle=False, num_workers=0)
    model = load_ast_run(run_dir, device)
    loss, acc, f1, preds, labels = evaluate(
        model, loader, torch.nn.CrossEntropyLoss(), device, use_amp=config["mixed_precision"],
    )
    output_dir = run_dir / "evaluations" / f"{split}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output_dir.mkdir(parents=True, exist_ok=False)
    save_json(output_dir / "metrics.json", {"split": split, "samples": len(frame),
              "loss": loss, "accuracy": acc, "macro_f1": f1, "checkpoint": str(run_dir / "best_ast.pt"),
              "test_evaluated": split == "test"})
    save_predictions(frame, preds, labels, output_dir / "predictions.csv", dict(enumerate(CLASS_NAMES)))
    np.save(output_dir / "confusion_matrix.npy", confusion_matrix(labels, preds, labels=list(range(10))))
    print(f"{split}: Accuracy={acc:.4f}, Macro-F1={f1:.4f}，结果目录：{output_dir}")


# 命令行入口：无参数训练，--eval-run 评估，--predict 预测一条音频。


def main():
    parser = argparse.ArgumentParser(description="训练或评估 UrbanSound8K 预训练 AST")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--eval-run", type=Path, help="只评估这个实验目录的最佳权重")
    mode.add_argument("--predict", type=Path, help="预测一条 WAV 音频，不训练或评估整个数据集")
    parser.add_argument("--model-run", type=Path, help="--predict 使用的实验目录，默认使用已确定的全参数最佳实验")
    parser.add_argument("--split", choices=["train", "validation", "test"], help="评估划分，默认 validation")
    parser.add_argument("--finetune", choices=["all", "head"], default=None,
                        help="训练方式：all 全参数微调（默认），head 只训练分类头")
    args = parser.parse_args()
    if args.predict is not None:
        if args.split is not None or args.finetune is not None:
            parser.error("--predict 不能与 --split 或 --finetune 同时使用")
        predict_file(args.predict, run_dir=args.model_run or PREDICTION_RUN)
    elif args.eval_run is not None:
        if args.model_run is not None:
            parser.error("--model-run 只能与 --predict 一起使用")
        if args.finetune is not None:
            parser.error("--finetune 不能与 --eval-run 同时使用")
        evaluate_run(args.eval_run, split=args.split or "validation")
    else:
        if args.model_run is not None:
            parser.error("--model-run 只能与 --predict 一起使用")
        if args.split is not None:
            parser.error("--split 必须与 --eval-run 一起使用")
        train(finetune=args.finetune or "all")


if __name__ == "__main__":
    main()
