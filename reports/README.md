# 实验记录与误差分析

这里是从已完成的本地实验导出的精简记录，不包含音频、权重、用户目录或启动日志。精确来源见 `evidence_sources.json`；完整历史实验在作者本地保留。

## 总体结果

`validation_comparison.csv`、`validation_metrics.json` 记录 CNN、原始输入 Transformer、标准化 Transformer、AST 全参数、AST 仅分类头的验证指标。统一在 Fold 9 的 816 条音频上比较。

- E0 → E1：验证 Macro-F1 从 67.82% 到 72.86%，提高 5.04 个百分点。标准化的统计量仅来自训练集，数值见 `transformer_normalization.json`。这支持在本次设置下改善输入尺度有帮助，未证明适用于所有模型和设置。
- AST 全参数 → 仅分类头：验证 Macro-F1 从 91.80% 到 87.46%。预训练编码器冻结后仍可配合新分类头完成较好的分类；在相同学习率 `1e-5` 和 20 轮预算下，全参数方案更好。
- 逐条配对比较：两方案均正确 690 条，仅全参数正确 51 条，仅分类头正确 9 条，均错 66 条。总体更好不代表每条都更好。每类对比见 `ast_validation_per_class.csv`。
- 两种 AST 方案同预训练来源、特征、划分、种子与批次设置；不是分别充分调参后的最佳能力比较。历史全参数实验没有同口径显存和逐轮耗时记录，不报告精确加速/显存下降倍数。

## 固定模型后的测试

先根据验证结果固定全参数 AST 第 3 轮检查点，再执行一次 Fold 10 测试，未据此调参。837 条中正确 750 条、错误 87 条；Accuracy 89.6057347670%，Macro-F1 90.4645516289%。检查点 SHA256 见 `ast_test_metrics.json`。

`ast_test_per_class.csv` 给出各类 Precision、Recall、F1 与样本数；`ast_test_confusions.csv` 给出错误方向。较明显的混淆有：

| 真实类别 → 预测类别 | 条数 | 占该真实类别 |
|---|---:|---:|
| siren → dog_bark | 20 | 20/83 = 24.10% |
| drilling → street_music | 13 | 13/100 = 13.00% |
| air_conditioner → children_playing | 9 | 9/100 = 9.00% |
| air_conditioner → engine_idling | 6 | 6/100 = 6.00% |

混淆矩阵只能确定错误分布，不能独自证明模型依赖什么声音线索。试听中曾观察到某些机械声在持续性、敲击感上相似，这属于待验证解释，不能替代系统的消融或声学分析。

## 来源重叠与评估范围

`fsID=180937` 跨训练与验证：训练 Fold 1 有 79 条 jackhammer，验证 Fold 9 有 16 条 drilling，其中 4 条验证片段的元数据时间范围与训练片段重叠。已核查的波形相关性没有确证完全重复，不能直接认定重复音频、错标或量化指标受影响幅度。本次保留原划分并公开记录风险。

本地元数据未发现训练/测试、验证/测试 fsID 重叠；未证实与 AudioSet 预训练数据完全无重叠。当前为单种子、固定 8/1/1 folds 协议，不是官方十折均值。AST 与 CNN 的预训练、特征、规模和优化设置不同，结果不能单独隔离架构的作用。

## 如何复核

```powershell
python -B reports/verify_saved_results.py
```

脚本检查五组验证记录对应同样的样本与真实标签，并从预测编号复算 Accuracy、Macro-F1；同时复算已保存测试预测的指标。它不运行模型，不再次接触 WAV，不产生新预测。CSV 不含完整 logits/概率，不能据此重算交叉熵损失。

`predictions/` 仅含文件标识、fsID、fold 和真假类别，类别编号对应根目录 `train_ast.py` 中的 `CLASS_NAMES`，也保存在 AST 协议 JSON 中。元数据标识来源为 UrbanSound8K，请遵守其上游许可和引用要求。

`history/` 是已有逐轮记录的副本，`figures/` 是已有曲线的副本。公开协议 JSON 只导出与复现实验相关的字段；没有改写原实验结果。检查点未在仓库分发，预测 CSV 不构成模型的独立重新推理复现。
