# UrbanSound8K：预训练 AST

这是独立的完整训练流程：全参数微调标识为 E2，只训练分类头的对照实验标识为 E3。AST 的数据处理、模型、训练、评估和保存功能集中在 `train_ast.py`，不依赖其他项目 Python 文件。项目文件导航见 `README.md`；精简实验记录与预测见 `reports/`。

## 文件与入口

| 文件 | 职责 |
|---|---|
| `train_ast.py` | 配置、Dataset、预训练模型、训练验证、最佳权重保存、曲线、指定划分评估和单条音频预测 |
| `checks/check_ast_pipeline.py` | 独立的小样本集成检查，不完整训练、不评估测试集 |

在项目根目录、激活自己的 Python 环境后运行（下面 `.venv` 命令适用于 Windows 本地虚拟环境）：

```powershell
# 正式训练：20 轮，不自动评估测试集
.\.venv\Scripts\python.exe train_ast.py

# 冻结原始预训练编码器，只训练分类头：20 轮，不自动评估测试集
.\.venv\Scripts\python.exe train_ast.py --finetune head

# 历史核验：额外需要作者本地旧源码快照，详见根目录 README
# 小样本检查冻结范围、分类头更新、保存与恢复
.\.venv\Scripts\python.exe -B checks/check_ast_head_only.py

# 小样本检查：10 条训练音频、10 条验证音频、1 轮
.\.venv\Scripts\python.exe checks/check_ast_pipeline.py

# 重新评估已有实验的验证集，将下面目录替换为真实运行目录
.\.venv\Scripts\python.exe train_ast.py --eval-run results/ast/某次运行目录 --split validation

# 最终方案确定后，用同一最佳检查点评估 Fold 10
.\.venv\Scripts\python.exe train_ast.py --eval-run results/ast/某次运行目录 --split test

# 单条 WAV 预测：默认加载已确定的全参数微调最佳权重
.\.venv\Scripts\python.exe train_ast.py --predict "D:\datasets\UrbanSound8K\audio\fold9\101729-0-0-1.wav"

# 需要使用其他已保存实验时，显式指定目录
.\.venv\Scripts\python.exe train_ast.py --predict "D:\my_audio.wav" --model-run results/ast/某次运行目录
```

仅导入文件不会开始训练。无参数运行时全参数微调；`--finetune head` 仅训练分类头；`--eval-run` 只评估已有检查点；`--predict` 只预测指定音频。未知选项或互相冲突的模式会报错，不会意外开始训练。训练配置集中在文件顶部。

### 单条预测如何使用

仓库不分发训练后的检查点；新克隆需先训练，并用 `--model-run` 指向自己的运行目录。以下默认目录仅适用于作者本地原始实验。

默认模型目录是项目中的 `results/ast/20261019_101342_538523/`，恢复其中 `best_ast.pt`，不会重新下载预训练权重或开始训练。默认目录相对于脚本所在位置确定，因此也可以从其他工作目录运行。

输入只需要 WAV 路径，不需要 UrbanSound8K 元数据或真实标签。与 Dataset 共用 `extract_audio_features()`：单声道 → 16kHz → 开头 4 秒/右侧补零 → 保存的 AST 配套特征；模型输入 `[1,1024,128]`，输出 `[1,10]`。预测使用 eval 和 no_grad，并沿用实验的设备可用性及混合精度设置。

程序打印类别编号、类别名称和 Softmax 分数。Softmax 分数不是预测正确率；模型只从已训练的十类中选择，不具备未知类别拒识能力。超过 4 秒的录音当前只分析开头 4 秒，不代表分析了整段录音。预测不自动写入文件或改变现有实验结果。

本地历史结果及源码备份不随仓库发布；当前接口保持已有检查点的参数名和结果格式。

## 数据流与形状

```text
WAV -> 单声道 -> 16kHz -> 截取开头 4 秒/右侧补零
    -> 波形 [64000]
    -> ASTFeatureExtractor（Kaldi fbank，128 个 Mel 频带）
    -> 时间帧补齐/截断到 1024，按预训练配置标准化
    -> 单条 [1024,128]，batch [B,1024,128]
    -> 16x16 图块，频率和时间步幅均为 10
    -> 12x101 = 1212 个图块，每块投影到 768 维
    -> 加两个特殊 Token 和位置嵌入：[B,1214,768]
    -> 12 层 Encoder，12 个注意力头，每头 64 维
    -> 两个特殊 Token 的最终表示取平均：[B,768]
    -> LayerNorm + Linear(768,10) -> logits [B,10]
```

4 秒波形在该 fbank 设置下通常产生 398 个时间帧，再在特征时间轴上补到 1024。4 秒波形补齐与特征补齐是两个不同操作。AST 的 Mel、窗长、步幅、取对数和标准化方式均与 `audio_dataset.py` 不同，不能直接替换 Tensor 形状后复用其旧特征。

标准化采用预训练权重配套的固定值：`(fbank - (-4.2677393)) / (2 * 4.5689974)`。不再叠加 E1 的训练集标准化，也不拟合验证/测试集统计量。

## 训练与评估协议

- Fold 1-8 训练、Fold 9 验证；按验证 Macro-F1 保存最佳权重。Fold 10 仅在显式调用测试评估入口时使用。
- seed 42，20 轮；AdamW，学习率 `1e-5`，weight decay `1e-4`。
- 训练/评估物理 batch size 为 2；训练累积 16 次，通常有效 batch size 为 32，末组按实际样本数加权。
- CUDA 使用 float16 混合精度与梯度缩放；全参数微调使用梯度检查点节省显存。CPU 自动使用 float32。默认全部参数参与微调，`--finetune head` 仅训练分类头。
- 不添加数据增强和学习率调度，先建立预训练 AST 基线。新十分类输出层需要在 UrbanSound8K 上训练，刚加载时的十分类结果不代表模型效果。
- 训练结束后恢复最佳权重，使用 eval 模式评估训练集，用于与同一模型的验证结果比较。
- 与 CNN/自实现 Transformer 的对比同时涉及预训练数据、模型大小、特征与优化设置差异，不能把分数差只归因于架构。
- 原划分共享来源 `fsID=180937` 的风险仍在，记录于配置中；本次没有更改划分。

## 结果与恢复

全参数正式实验每次创建 `results/ast/<时间戳>/`，只训练分类头创建 `results/ast_head_only/<时间戳>/`；两种方式不会覆盖彼此的结果。检查分别使用 `results/checks_ast/` 和 `results/checks_ast_head_only/`。分别记录：

- `best_ast.pt`：最佳验证 Macro-F1 对应的模型参数。
- `ast_config.json`、`preprocessor_config.json`：模型结构与预处理配置，离线恢复需要它们。
- `config.json`：数据划分、预训练来源及固定 revision、训练设置、依赖版本。
- 新实验还记录总参数量、可训练参数量、初始分类头哈希和源码快照；逐轮记录耗时，结束记录 CUDA 分配与预留显存峰值。
- `history.csv`、`training.log`、`training_curves.png`：训练历史与曲线。
- `metrics.json`、`val_predictions.csv`、`val_confusion_matrix.npy`：最佳模型结果。
- 独立评估写入该实验的 `evaluations/<划分_时间戳>/`，不覆盖训练时保存的结果。

首次运行下载约 350MB 权重到项目的 `models/huggingface/`。权重固定为 MIT 仓库 revision `f826b80d28226b62986cc218e5cec390b1096902`；后续可复用缓存。自己的检查点恢复不需要再次下载。

## 已完成的验证

2026-10-19 在本机 RTX 3050 Laptop 4GB 上完成独立小样本检查：10 条训练音频、10 条验证音频、1 轮。

- 输入 `[2,1024,128]`，图块嵌入 `[2,1212,768]`，加特殊 Token 后 `[2,1214,768]`，输出 `[2,10]`。
- 核对预训练图块投影权重与官方 safetensors 完全一致；只有分类输出层的两项参数需要按 527 类到 10 类的尺寸变化重新初始化。首次加载的这两项 MISMATCH 提示符合预期。
- 编码器注意力梯度有限，编码器参数实际更新；梯度累积与包含不足一组样本的参考更新一致。
- 最佳模型保存与离线恢复成功，恢复后的验证预测和指标与保存结果一致，单独评估入口通过验证。
- 小样本检查显存峰值约 1.63GiB。这不是完整训练的性能或显存保证；检查没有评估测试集。

随后完成正式 20 轮微调，结果保存在 `results/ast/20261019_101342_538523/`：

| 项目 | 结果 |
|---|---:|
| 最佳轮次 | 3 |
| 最佳验证 Accuracy | 90.81% |
| 最佳验证 Macro-F1 | 91.80% |
| 最佳权重重新评估训练集 Accuracy | 99.87% |

上述数值来自该实验训练阶段的 `metrics.json`；训练阶段未评估测试集，验证分数不代表测试结果。该路径是本地记录，公开精简分析见 [实验报告](reports/README.md)。

### 固定模型后的测试评估

2026-10-23，固定全参数微调方案及验证 Macro-F1 最高的第 3 轮权重后，使用已有评估入口完成一次 Fold 10 测试。没有重新训练或根据测试结果选参数。

- 测试集 837 条，正确 750 条、错误 87 条；Accuracy 89.61%，Macro-F1 90.46%。
- 预测、元数据顺序、混淆矩阵及重算指标核对一致；最佳权重 SHA256 在测试前后保持不变。
- 结果保存在 `results/ast/20261019_101342_538523/evaluations/test_20261023_152455_410142/`。
- 完整本地报告在 `results/ast_test_analysis/20261023_152344_0760196/`；公开指标、每类指标、混淆与保存的预测在 `reports/`。
- 原训练 `metrics.json` 保留训练结束时的 `test_evaluated=false`；本次独立评估的 `metrics.json` 记录 `test_evaluated=true`。

本地元数据中训练/测试、验证/测试的 `fsID` 均未发现重叠；这不能证明不存在其他录音重复或预训练数据重叠。训练/验证共享 `fsID=180937` 的已知风险仍然保留。当前成绩属于固定划分，不是官方十折交叉验证均值。

## 分类头对照实验的解释范围

E3 从原始 AudioSet 预训练权重开始，冻结整个 `audio_spectrogram_transformer`，仅训练 `classifier` 中的 LayerNorm 与 Linear。总参数量为 86,196,490，可训练参数量为 9,226；冻结编码器仍进行前向计算，但保持 eval 模式，不使用梯度检查点。E2 的原有默认行为保留。

第一轮对照保持 seed 42、学习率 1e-5、20 轮、物理 batch size 2、累积 16 次和相同验证选优规则。它比较相同学习率与轮数下的两种训练方式，不代表分别充分调参后的最优表现。历史 E2 没有保存初始分类头；检查会用历史源码和相同种子重建初始化进行核对，不能把重建称为直接验证历史初始权重文件。

历史 E2 没有同口径记录完整训练的显存峰值与逐轮秒数，因此新实验的资源记录不能直接用来宣称精确加速倍数或显存降低比例。

E3 已完成 20 轮，最佳第 19 轮，验证 Accuracy 85.66%、Macro-F1 87.46%；全参数高 4.34 个百分点。对照及逐类记录见 [实验报告](reports/README.md)。

## 来源

- 模型与配套配置：https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593
- AST 官方实现：https://github.com/YuanGongND/ast
- Transformers AST 文档：https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer
