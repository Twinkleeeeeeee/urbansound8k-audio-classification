"""只复算已保存的预测，不读取 WAV、不加载模型、不再次评估测试集。"""

import csv
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parent


def read_predictions(name):
    with (ROOT / "predictions" / name).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    labels = [int(row["true_label"]) for row in rows]
    predictions = [int(row["pred_label"]) for row in rows]
    return rows, labels, predictions


def verify(name, expected, fold):
    rows, labels, predictions = read_predictions(name)
    assert len(rows) == expected["samples"], name
    assert all(int(row["fold"]) == fold for row in rows), name
    assert all(0 <= label < 10 for label in labels + predictions), name
    assert len({r["slice_file_name"] for r in rows}) == len(rows), name
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, labels=list(range(10)), average="macro")
    assert abs(accuracy - expected["accuracy"]) < 1e-12, name
    assert abs(macro_f1 - expected["macro_f1"]) < 1e-12, name
    print(f"{name}: n={len(rows)}, accuracy={accuracy:.10f}, macro_f1={macro_f1:.10f}")
    return rows


def main():
    metrics = json.loads((ROOT / "validation_metrics.json").read_text(encoding="utf-8"))
    validation_rows = []
    for entry in metrics:
        validation_rows.append(verify(f'{entry["model"]}_validation.csv', entry, fold=9))
    # 各方案必须按同样的样本和真实标签比较。
    identities = [[(r["slice_file_name"], r["true_label"]) for r in rows]
                  for rows in validation_rows]
    assert all(identity == identities[0] for identity in identities)
    test = json.loads((ROOT / "ast_test_metrics.json").read_text(encoding="utf-8"))
    verify("ast_full_test.csv", test, fold=10)
    print("SAVED_RESULTS_VERIFIED: 5 validation sets and 1 saved test prediction set")


if __name__ == "__main__":
    main()
