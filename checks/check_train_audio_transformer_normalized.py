r"""标准化训练程序的集成检查：10 条训练音频、10 条验证音频、1 个训练 batch；不使用测试集。

从项目目录运行：.\.venv\Scripts\python.exe -B checks/check_train_audio_transformer_normalized.py
参数只在本次检查进程中临时替换，不修改训练文件。结果保存在独立的检查目录。
"""

import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_audio_transformer_normalized as module


def main():
    # 大小不同的两个 batch，防止误把 batch 均值直接平均。
    x = torch.arange(18, dtype=torch.float64).reshape(3, 1, 2, 3)
    y = torch.tensor([0, 1, 2])
    stats = module.fit_statistics([(x[:2], y[:2]), (x[2:], y[2:])])
    assert stats["samples"] == 3 and stats["elements"] == 18
    assert math.isclose(stats["mean"], x.mean().item(), abs_tol=1e-12)
    assert math.isclose(stats["std"], x.std(unbiased=False).item(), abs_tol=1e-12)
    ds = module.StandardizedDataset(TensorDataset(x, y), stats["mean"], stats["std"])
    actual = torch.stack([ds[i][0] for i in range(3)])
    expected = (x - x.mean()) / (x.std(unbiased=False) + module.EPSILON)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    assert [ds[i][1].item() for i in range(3)] == y.tolist()

    # 旧检查选项必须明确退出，不能意外启动完整训练。
    with patch.object(sys, "argv", ["train_audio_transformer_normalized.py", "--check-only"]):
        try:
            module.main()
        except SystemExit:
            pass
        else:
            raise AssertionError("旧版选项未被拒绝")

    metadata_path = module.DATA_ROOT / "metadata" / "UrbanSound8K.csv"
    read_csv = pd.read_csv
    full = read_csv(metadata_path)
    train = full[full.fold.isin(range(1, 9))].groupby("classID").head(1)
    val = full[full.fold == 9].groupby("classID").head(1)
    assert len(train) == len(val) == 10
    small = pd.concat([train, val], ignore_index=True)

    def small_metadata(path, *args, **kwargs):
        if Path(path).resolve() == metadata_path.resolve():
            return small.copy()
        return read_csv(path, *args, **kwargs)

    original_train = module.train_one_epoch
    shapes = []

    def train_one_batch(model, loader, optimizer, criterion, device):
        assert len(loader) == 1
        before = model.input_proj.weight.detach().clone()
        hook = model.register_forward_hook(
            lambda m, inputs, output: shapes.append([list(inputs[0].shape), list(output.shape)])
        )
        try:
            result = original_train(model, loader, optimizer, criterion, device)
        finally:
            hook.remove()
        assert not torch.equal(before, model.input_proj.weight)
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters() if p.requires_grad)
        return result

    output_root = ROOT / "results" / "checks_audio_transformer_normalized"
    with (patch.object(module.pd, "read_csv", side_effect=small_metadata),
          patch.object(module, "EPOCHS", 1), patch.object(module, "NUM_WORKERS", 0),
          patch.object(module, "OUTPUT_ROOT", output_root),
          patch.object(module, "train_one_epoch", side_effect=train_one_batch),
          patch.object(sys, "argv", ["train_audio_transformer_normalized.py"])):
        module.main()

    run = max(output_root.iterdir(), key=lambda path: path.name)
    stats = json.loads((run / "normalization.json").read_text(encoding="utf-8"))
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert stats["samples"] == 10 and stats["elements"] == 10 * 64 * 126
    assert metrics["completed_epochs"] == metrics["best_epoch"] == 1
    assert metrics["test_evaluated"] is False
    assert shapes == [[[10, 1, 64, 126], [10, 10]]]
    raw_train = module.UrbanSound8KDataset(train, module.DATA_ROOT / "audio", module.AudioConfig())
    values = torch.stack([raw_train[i][0] for i in range(10)]).to(torch.float64)
    assert math.isclose(stats["mean"], values.mean().item(), abs_tol=1e-10)
    assert math.isclose(stats["std"], values.std(unbiased=False).item(), abs_tol=1e-10)

    raw_val = module.UrbanSound8KDataset(val, module.DATA_ROOT / "audio", module.AudioConfig())
    val_ds = module.StandardizedDataset(raw_val, stats["mean"], stats["std"])
    loader = DataLoader(val_ds, batch_size=10, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = module.AudioTransformer().to(device)
    model.load_state_dict(torch.load(run / "best_transformer.pt", map_location=device, weights_only=True))
    loss, acc, f1, preds, labels = module.evaluate(model, loader, torch.nn.CrossEntropyLoss(), device)
    saved = read_csv(run / "val_predictions.csv")
    assert saved.slice_file_name.tolist() == val.slice_file_name.tolist()
    assert preds.tolist() == saved.pred_label.tolist()
    assert labels.tolist() == saved.true_label.tolist()
    assert math.isclose(loss, metrics["validation"]["loss"], abs_tol=1e-6)
    assert math.isclose(acc, metrics["validation"]["accuracy"], abs_tol=1e-12)
    assert math.isclose(f1, metrics["validation"]["macro_f1"], abs_tol=1e-12)
    assert len(read_csv(run / "history.csv")) == 1
    assert (run / "training_curves.png").stat().st_size > 0
    print("检查通过：统计、标准化、真实 batch 训练、验证、权重恢复和结果保存。")
    print("输入 [10, 1, 64, 126]，输出 [10, 10]。检查结果：", run)


if __name__ == "__main__":
    main()
