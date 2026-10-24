"""查看类别分布、音频时长、各类波形和重点类别的 Log-Mel 图。

直接运行会生成 waveforms.png、focus_spectrograms.png 并显示图片；不调用分类模型。
"""

import numpy as np
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

data_root = Path(r"D:\datasets\UrbanSound8K")
df = pd.read_csv(data_root / "metadata" / "UrbanSound8K.csv")

# 各类别样本数。
print("=== 每类样本数 ===")
class_names = df.groupby("classID")["class"].first()
counts = df.groupby("classID").size()
for cid, cnt in counts.items():
    print(f"{cid}  {class_names[cid]:20s} {cnt}")

# 根据元数据计算各类别的时长范围。
print("\n=== 每类时长范围（秒）===")
df["dur"] = df["end"] - df["start"]
for cid in sorted(df["classID"].unique()):
    sub = df[df["classID"] == cid]
    print(f"{cid}  {class_names[cid]:20s} "
          f"min={sub['dur'].min():.2f} max={sub['dur'].max():.2f} "
          f"mean={sub['dur'].mean():.2f}")

# 读取原始采样率下的单声道波形。
def load_audio(row):
    path = data_root / "audio" / f"fold{int(row['fold'])}" / row["slice_file_name"]
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T).contiguous()
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, sr

# 每类选一条音频，绘制原始时长的波形。
print("\n绘图 waveforms...")
fig, axes = plt.subplots(5, 2, figsize=(14, 12))
axes = axes.flatten()

for cid in sorted(df["classID"].unique()):
    row = df[df["classID"] == cid].iloc[0]
    wav, sr = load_audio(row)
    ax = axes[cid]
    t = np.arange(wav.size(-1)) / sr
    ax.plot(t, wav.squeeze().numpy(), linewidth=0.5)
    ax.set_title(f"{cid}: {class_names[cid]}  ({wav.size(-1)/sr:.1f}s)")
    ax.set_xlabel("时间 (秒)")
    ax.set_ylabel("幅度")

plt.tight_layout()
plt.savefig("waveforms.png", dpi=100)
print("已保存 waveforms.png")

# 重点类别各选三条，按 16 kHz、4 秒生成 Log-Mel 对照图。
focus_classes = {0: "air_conditioner", 4: "drilling", 5: "engine_idling"}

mel_tf = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=1024, hop_length=512, n_mels=64,
    f_min=0.0, f_max=8000.0, power=2.0,
)
to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

fig, axes = plt.subplots(3, 3, figsize=(15, 9))

for row_i, cid in enumerate(focus_classes.keys()):
    rows = df[df["classID"] == cid].head(3)
    for col_i, (_, row) in enumerate(rows.iterrows()):
        wav, sr = load_audio(row)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        if wav.size(-1) > 64000:
            wav = wav[..., :64000]
        elif wav.size(-1) < 64000:
            wav = F.pad(wav, (0, 64000 - wav.size(-1)))

        log_mel = to_db(mel_tf(wav)).squeeze().numpy()
        ax = axes[row_i, col_i]
        ax.imshow(log_mel, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(f"{focus_classes[cid]} #{col_i+1}")
        ax.set_xlabel("时间帧")
        ax.set_ylabel("Mel 频带")

plt.tight_layout()
plt.savefig("focus_spectrograms.png", dpi=100)
print("已保存 focus_spectrograms.png")

plt.show()
