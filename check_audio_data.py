"""检查 PyTorch、CUDA、数据目录，并读取一条 WAV 查看采样率和形状。"""

import torch
import torchaudio
import soundfile as sf
from pathlib import Path

print("torch:", torch.__version__)
print("torchaudio:", torchaudio.__version__)
print("cuda:", torch.cuda.is_available())

data_root = Path(r"D:\datasets\UrbanSound8K")
csv_path = data_root / "metadata" / "UrbanSound8K.csv"
audio_dir = data_root / "audio"

print("csv exists:", csv_path.exists())
print("audio dir exists:", audio_dir.exists())

wavs = sorted(audio_dir.glob("fold1/*.wav"))
print("fold1 wav count:", len(wavs))

if wavs:
    wav_path = wavs[0]

    # soundfile 返回 NumPy 数组 [采样点数, 声道数]，不依赖 torchcodec。
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)

    # 转为 Tensor [声道数, 采样点数]，与 torchaudio 的波形维度约定一致。
    wav = torch.from_numpy(data.T).contiguous()

    print("file:", wav_path)
    print("wav shape:", wav.shape)
    print("sample rate:", sr)
