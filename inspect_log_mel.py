"""读取一条 WAV，逐步检查单声道、重采样、定长和 Log-Mel 特征。"""

import torch
import torchaudio
import torch.nn.functional as F
import soundfile as sf
from pathlib import Path

# 音频与特征配置。
SAMPLE_RATE = 16000      # 统一重采样到 16kHz
DURATION = 4.0           # 统一到 4 秒
TARGET_LEN = int(SAMPLE_RATE * DURATION)   # 64000

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
F_MIN = 0.0
F_MAX = 8000.0
TOP_DB = 80.0

# 读取一条音频；相对路径以程序当前工作目录为基准。
data_root = Path("data/UrbanSound8K")
audio_dir = data_root / "audio"
wav_path = sorted(audio_dir.glob("fold1/*.wav"))[0]

data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
waveform = torch.from_numpy(data.T).contiguous()   # [C, T]
print("原始:", waveform.shape, "sr =", sr)

# 对声道求平均，保留长度为 1 的声道维。
if waveform.size(0) > 1:
    waveform = waveform.mean(dim=0, keepdim=True)   # [C, T] -> [1, T]
print("单声道:", waveform.shape)

# 先重采样，再按目标采样率统一波形长度。
if sr != SAMPLE_RATE:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
    waveform = resampler(waveform)
print("重采样后:", waveform.shape)

# 截取开头 4 秒；不足时在右侧补零。
length = waveform.size(-1)
print("len:",length)
if length > TARGET_LEN:
    waveform = waveform[..., :TARGET_LEN]
elif length < TARGET_LEN:
    waveform = F.pad(waveform, (0, TARGET_LEN - length))
print("统一长度后:", waveform.shape)

# 计算 Mel 功率谱，再转换为 dB；默认输出 [1, 64, 126]。
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    f_min=F_MIN,
    f_max=F_MAX,
    power=2.0,
)
to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=TOP_DB)

mel = mel_transform(waveform)   # [1, n_mels, time]
log_mel = to_db(mel)
print("mel 原始形状:", mel.shape)
print("log_mel 形状:", log_mel.shape)
print("log_mel min/max/mean:",
      log_mel.min().item(), log_mel.max().item(), log_mel.mean().item())
