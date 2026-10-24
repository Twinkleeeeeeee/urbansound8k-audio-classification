"""CNN 和自建 Transformer 共用的 UrbanSound8K Log-Mel 数据集。

默认单条特征为 [1, 64, 126]，标签为 0–9 的整数；AST 使用独立的配套特征处理。
"""

import torch
import torchaudio
import torch.nn.functional as F
import soundfile as sf
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


# 音频和 Log-Mel 配置。
class AudioConfig:
    """集中保存采样率、时长和频谱参数，不在创建配置时读取音频。"""

    sample_rate = 16000
    duration = 4.0
    n_mels = 64
    n_fft = 1024
    hop_length = 512
    f_min = 0.0
    f_max = 8000.0
    top_db = 80.0

    @property
    def target_len(self):
        """目标采样点数；默认 16000 Hz × 4 秒 = 64000。"""
        return int(self.sample_rate * self.duration)


# 按索引读取音频并生成特征。
class UrbanSound8KDataset(Dataset):
    """接收元数据表、audio 目录和配置；取样时返回 (Log-Mel Tensor, 类别编号)。"""

    def __init__(self, df, audio_dir, cfg):
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.cfg = cfg

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=cfg.top_db
        )
        # 按原采样率缓存重采样器对象；不缓存某条音频的处理结果。
        self._resamplers = {}

    def __len__(self):
        return len(self.df)

    def _get_resampler(self, orig_sr):
        """采样率相同时返回 None，否则创建或复用重采样器。"""
        if orig_sr == self.cfg.sample_rate:
            return None
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=self.cfg.sample_rate
            )
        return self._resamplers[orig_sr]

    def _pad_or_trim(self, wav):
        """沿最后一维截取或右侧补零，保持声道维不变。"""
        target = self.cfg.target_len
        length = wav.size(-1)
        if length > target:
            return wav[..., :target]
        if length < target:
            return F.pad(wav, (0, target - length))
        return wav

    def __getitem__(self, idx):
        """读取第 idx 行对应的 WAV，返回 float32 特征及整数标签。"""
        row = self.df.iloc[idx]
        fold = int(row["fold"])
        filename = row["slice_file_name"]
        label = int(row["classID"])

        path = self.audio_dir / f"fold{fold}" / filename

        # 读取波形：[采样点数, 声道数] → [声道数, 采样点数]。
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T).contiguous()   # [C, T]

        # 按声道取平均，得到 [1, 采样点数]。
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # 重采样改变采样点密度，尽量保持声音时长。
        rs = self._get_resampler(sr)
        if rs is not None:
            wav = rs(wav)

        # 重采样后统一到目标时长，默认 [1, 64000]。
        wav = self._pad_or_trim(wav)

        # 默认得到 [1, 64, 126]；三维依次是声道、Mel 频带、时间帧。
        log_mel = self.to_db(self.mel(wav))   # [1, n_mels, time]

        return log_mel, label


# 直接运行时只检查样本和一个 batch，不训练模型。
if __name__ == "__main__":
    torch.manual_seed(42)

    data_root = Path(r"D:\datasets\UrbanSound8K")
    csv_path = data_root / "metadata" / "UrbanSound8K.csv"
    audio_dir = data_root / "audio"

    df = pd.read_csv(csv_path)
    print("总样本数:", len(df))
    print("每折样本数:")
    print(df.groupby("fold").size())

    cfg = AudioConfig()

    # 使用数据集提供的 fold：本项目选 1–8 训练、9 验证、10 测试，并非十折交叉验证。
    train_df = df[df["fold"].isin(range(1, 9))].reset_index(drop=True)
    val_df = df[df["fold"] == 9].reset_index(drop=True)
    test_df = df[df["fold"] == 10].reset_index(drop=True)

    print("train/val/test:", len(train_df), len(val_df), len(test_df))

    train_ds = UrbanSound8KDataset(train_df, audio_dir, cfg)
    val_ds = UrbanSound8KDataset(val_df, audio_dir, cfg)
    test_ds = UrbanSound8KDataset(test_df, audio_dir, cfg)

    # 先验证一个样本
    x, y = train_ds[0]
    print("单个样本:", x.shape, "label =", y)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    xb, yb = next(iter(train_loader))
    print("一个 batch:", xb.shape, yb.shape)
    print("label 分布:", torch.bincount(yb, minlength=10))
