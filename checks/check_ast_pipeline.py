r"""AST 集成检查：10 条训练音频、10 条验证音频、1 轮；不评估测试集。

从项目目录运行：.\.venv\Scripts\python.exe -B checks/check_ast_pipeline.py
检查结果单独保存到 results/checks_ast/，不会覆盖正式实验。
"""

import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from torch.utils.data import DataLoader, TensorDataset
from transformers import ASTFeatureExtractor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_ast as training
from train_ast import CACHE_DIR, MODEL_REVISION, UrbanSound8KASTDataset, load_ast_run


def check_accumulation():
    # 对照直接以 6 条、1 条为 batch 的参考更新，检查最后不足一组时的梯度权重。
    torch.manual_seed(42)
    x, y = torch.randn(7, 4), torch.arange(7) % 10
    actual, expected = torch.nn.Linear(4, 10), torch.nn.Linear(4, 10)
    expected.load_state_dict(actual.state_dict())
    optimizer = torch.optim.SGD(actual.parameters(), lr=0.02)
    criterion = torch.nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    training.train_one_epoch(actual, loader, optimizer, criterion, torch.device("cpu"), scaler, 3)
    reference_optimizer = torch.optim.SGD(expected.parameters(), lr=0.02)
    for start in (0, 6):
        reference_optimizer.zero_grad()
        criterion(expected(x[start:start + 6]), y[start:start + 6]).backward()
        reference_optimizer.step()
    for a, b in zip(actual.parameters(), expected.parameters()):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)


def main():
    torch.set_num_threads(2)
    check_accumulation()
    read_csv = pd.read_csv
    metadata_path = training.DATA_ROOT / "metadata" / "UrbanSound8K.csv"
    full = read_csv(metadata_path)
    train = full[full.fold.isin(range(1, 9))].groupby("classID").head(1)
    val = full[full.fold == 9].groupby("classID").head(1)
    assert len(train) == len(val) == 10
    small = pd.concat([train, val], ignore_index=True)

    def small_metadata(path, *args, **kwargs):
        if Path(path).resolve() == metadata_path.resolve():
            return small.copy()
        return read_csv(path, *args, **kwargs)

    report = {"test_evaluated": False, "full_training": False, "gradient_accumulation_reference": "passed"}
    shapes = {}
    original_train = training.train_one_epoch

    def checked_train(model, loader, optimizer, criterion, device, scaler, accumulation_steps):
        report["loading_info"] = model.loading_info
        assert not model.loading_info.get("unexpected_keys"), model.loading_info
        assert not model.loading_info.get("missing_keys"), model.loading_info
        mismatched = model.loading_info.get("mismatched_keys", [])
        assert len(mismatched) == 2, model.loading_info
        assert all("classifier.dense." in str(key) for key in mismatched), mismatched
        patch_layer = model.ast.audio_spectrogram_transformer.embeddings.patch_embeddings
        key = "audio_spectrogram_transformer.embeddings.patch_embeddings.projection.weight"
        path = CACHE_DIR / "models--MIT--ast-finetuned-audioset-10-10-0.4593" / "snapshots" / MODEL_REVISION / "model.safetensors"
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            torch.testing.assert_close(patch_layer.projection.weight.detach().cpu(), checkpoint.get_tensor(key), rtol=0, atol=0)
        report["pretrained_patch_weights_verified"] = True
        before = patch_layer.projection.weight.detach().clone()
        first_attention = model.ast.audio_spectrogram_transformer.layers[0].attention
        gradient_checks = []
        gradient_hook = first_attention.q_proj.weight.register_hook(
            lambda grad: gradient_checks.append(torch.isfinite(grad).all().item())
        )
        hooks = []
        for name, layer in [("patch_embeddings", patch_layer),
                            ("with_special_tokens", model.ast.audio_spectrogram_transformer.embeddings)]:
            hooks.append(layer.register_forward_hook(
                lambda module, inputs, output, name=name: shapes.update({name: list(output.shape)})
            ))
        try:
            result = original_train(model, loader, optimizer, criterion, device, scaler, accumulation_steps)
        finally:
            gradient_hook.remove()
            for hook in hooks:
                hook.remove()
        assert gradient_checks and all(gradient_checks)
        assert not torch.equal(before, patch_layer.projection.weight.detach()), "编码器权重未更新"
        report["encoder_gradient_finite"] = True
        report["encoder_weights_updated"] = True
        if device.type == "cuda":
            report["peak_cuda_memory_gib"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
        return result

    output_root = ROOT / "results" / "checks_ast"
    with (patch.object(training.pd, "read_csv", side_effect=small_metadata),
          patch.object(training, "OUTPUT_ROOT", output_root), patch.object(training, "EPOCHS", 1),
          patch.object(training, "train_one_epoch", side_effect=checked_train),
          patch.object(sys, "argv", ["train_ast.py"])):
        training.main()
    run = max(output_root.iterdir(), key=lambda p: p.name)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["completed_epochs"] == metrics["best_epoch"] == 1
    assert metrics["test_evaluated"] is False
    assert len(read_csv(run / "history.csv")) == 1
    assert (run / "training_curves.png").stat().st_size > 0
    matrix = np.load(run / "val_confusion_matrix.npy")
    assert matrix.shape == (10, 10) and matrix.sum() == 10

    # 使用保存的结构、预处理和权重离线恢复，重算验证指标和逐条预测。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_ast_run(run, device)
    extractor = ASTFeatureExtractor.from_pretrained(run, local_files_only=True)
    ds = UrbanSound8KASTDataset(val, training.DATA_ROOT / "audio", extractor)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    features, labels = next(iter(loader))
    assert features.shape == (2, 1024, 128) and labels.shape == (2,)
    assert torch.isfinite(features).all()
    with torch.no_grad():
        output = model(features.to(device))
    assert output.shape == (2, 10) and torch.isfinite(output).all()
    loss, acc, f1, preds, labels = training.evaluate(model, loader, torch.nn.CrossEntropyLoss(), device)
    saved = read_csv(run / "val_predictions.csv")
    assert saved.slice_file_name.tolist() == val.slice_file_name.tolist()
    assert preds.tolist() == saved.pred_label.tolist() and labels.tolist() == saved.true_label.tolist()
    assert math.isclose(loss, metrics["validation"]["loss"], rel_tol=1e-5, abs_tol=1e-5)
    assert acc == metrics["validation"]["accuracy"] and f1 == metrics["validation"]["macro_f1"]
    for name in ["best_ast.pt", "ast_config.json", "preprocessor_config.json", "config.json", "training.log"]:
        assert (run / name).is_file()

    with (patch.object(training.pd, "read_csv", side_effect=small_metadata),
          patch.object(sys, "argv", ["train_ast.py", "--eval-run", str(run), "--split", "validation"])):
        training.main()
    report.update({"input_shape": list(features.shape), "logits_shape": list(output.shape),
                   "internal_shapes": shapes, "checkpoint_reload": "passed", "evaluation_entry": "passed",
                   "train_samples": 10, "validation_samples": 10})
    (run / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("AST 检查通过。仅完成小样本检查，指标不代表正式实验性能。", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)
    print("检查结果：", run, flush=True)


if __name__ == "__main__":
    main()
