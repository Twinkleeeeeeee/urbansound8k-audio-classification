"""只训练分类头的集成检查：7 条训练音频、5 条验证音频、1 轮。"""

import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_ast as training


def parameter_hash(module):
    digest = hashlib.sha256()
    for name, p in module.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    torch.set_num_threads(2)
    # 按历史源码和相同种子重建初始分类头；历史实验没有保存初始权重。
    source = ROOT / "results/ast/20261019_101342_538523/train_ast.py"
    spec = importlib.util.spec_from_file_location("historical_ast", source)
    historical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(historical)
    historical.CACHE_DIR = training.CACHE_DIR
    historical.set_seed(42)
    historical_model = historical.AudioAST()
    reference_head_hash = parameter_hash(historical_model.ast.classifier)
    del historical_model
    gc.collect()

    read_csv = pd.read_csv
    metadata_path = training.DATA_ROOT / "metadata/UrbanSound8K.csv"
    full = read_csv(metadata_path)
    train = full[full.fold.isin(range(1, 9))].groupby("classID").head(1).head(7)
    val = full[full.fold == 9].groupby("classID").head(1).head(5)
    small = pd.concat([train, val], ignore_index=True)

    def small_metadata(path, *args, **kwargs):
        if Path(path).resolve() == metadata_path.resolve():
            return small.copy()
        return read_csv(path, *args, **kwargs)

    original_train = training.train_one_epoch
    report = {"test_evaluated": False, "full_training": False,
              "initial_classifier_matches_historical_code_reconstruction": False}

    def checked_train(model, loader, optimizer, criterion, device, scaler, accumulation_steps):
        encoder = model.ast.audio_spectrogram_transformer
        assert all(not p.requires_grad for p in encoder.parameters())
        assert sum(p.numel() for p in model.parameters()) == 86196490
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 9226
        assert not model.ast.is_gradient_checkpointing
        optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        assert optimizer_ids == {id(p) for p in model.ast.classifier.parameters()}
        head_before = parameter_hash(model.ast.classifier)
        assert head_before == reference_head_hash
        report["initial_classifier_matches_historical_code_reconstruction"] = True
        encoder_before = parameter_hash(encoder)
        gradients = []
        handles = [
            model.register_forward_pre_hook(
                lambda module, args: report.update({"input_shape": list(args[0].shape)})),
            model.register_forward_hook(
                lambda module, args, output: report.update({"logits_shape": list(output.shape)})),
            model.ast.classifier.dense.weight.register_hook(
                lambda grad: gradients.append(bool(torch.isfinite(grad).all().item()))),
        ]
        try:
            result = original_train(model, loader, optimizer, criterion, device, scaler, accumulation_steps)
        finally:
            for handle in handles:
                handle.remove()
        assert report["input_shape"] == [1, 1024, 128]  # 7 条音频最后一小批为 1 条。
        assert report["logits_shape"] == [1, 10]
        assert not encoder.training
        assert all(p.grad is None for p in encoder.parameters())
        assert parameter_hash(encoder) == encoder_before
        assert parameter_hash(model.ast.classifier) != head_before
        assert gradients and all(gradients)
        report.update({"encoder_all_parameters_bitwise_unchanged": True,
                       "encoder_gradients_absent": True, "encoder_eval_during_training": True,
                       "classifier_updated": True, "classifier_gradients_finite": True,
                       "total_parameters": 86196490, "trainable_parameters": 9226,
                       "initial_classifier_sha256": head_before})
        return result

    check_root = ROOT / "results/checks_ast_head_only"
    with (patch.object(training.pd, "read_csv", side_effect=small_metadata),
          patch.object(training, "OUTPUT_ROOT", check_root / "all"),
          patch.object(training, "EPOCHS", 1),
          patch.object(training, "train_one_epoch", side_effect=checked_train),
          patch.object(sys, "argv", ["train_ast.py", "--finetune", "head"])):
        training.main()
    run = max((check_root / "ast_head_only").iterdir(), key=lambda p: p.name)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["completed_epochs"] == metrics["best_epoch"] == 1
    assert metrics["test_evaluated"] is False
    assert metrics["finetune"] == "classifier_only"
    saved = read_csv(run / "val_predictions.csv")
    assert len(saved) == 5
    assert np.load(run / "val_confusion_matrix.npy").sum() == 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    restored = training.load_ast_run(run, device)
    extractor = training.ASTFeatureExtractor.from_pretrained(run, local_files_only=True)
    dataset = training.UrbanSound8KASTDataset(val, training.DATA_ROOT / "audio", extractor)
    loader = training.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    loss, acc, f1, preds, labels = training.evaluate(restored, loader, torch.nn.CrossEntropyLoss(), device)
    assert preds.tolist() == saved.pred_label.tolist()
    assert labels.tolist() == saved.true_label.tolist()
    assert acc == metrics["validation"]["accuracy"] and f1 == metrics["validation"]["macro_f1"]
    assert abs(loss - metrics["validation"]["loss"]) < 1e-5
    assert (run / "training_curves.png").stat().st_size > 0
    report.update({"checkpoint_reload_predictions_match": True,
                   "train_samples": 7, "validation_samples": 5,
                   "peak_cuda_memory_allocated_bytes": metrics["peak_cuda_memory_allocated_bytes"]})
    (run / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("HEAD_ONLY_CHECK_PASSED", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print("检查结果：", run, flush=True)


if __name__ == "__main__":
    main()
