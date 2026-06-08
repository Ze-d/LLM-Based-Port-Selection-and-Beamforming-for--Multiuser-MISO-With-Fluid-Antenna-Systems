# Stage 1：Hard Inference 与实验元数据阶段总结

## 阶段目标

本阶段对应完整复现路线 A 的第一阶段，目标是先修正评估可信度，再继续扩大实验规模。

具体目标：

- Proposed 模型训练阶段继续使用可微的 soft Gumbel-Sinkhorn。
- Proposed 模型验证和测试阶段改为论文要求的 hard activated ports。
- 评估结果文件 `results.csv` 写入完整实验元数据，便于后续多 seed、baseline 和 sweep 汇总。
- 保持现有 MVP 训练流程可运行，并刷新当前 `train=1000, test=200, epochs=20` 的结果。

## 主要改动

### 1. Proposed 模型支持 hard port inference

修改文件：

- `llm_fas/models.py`
- `tests/test_models.py`

新增 `selection_mode` 参数：

```python
model(H, tau=..., training=..., selection_mode="soft" | "hard" | None)
```

默认行为：

- `training=True` 且未显式指定时，使用 `soft`。
- `training=False` 且未显式指定时，使用 `hard`。
- 非法模式会抛出 `ValueError`。

hard 模式下新增输出：

- `selection_mode`
- `selection`
- `ports`
- `selection_hard`

其中 `ports` 是每个样本的唯一激活端口索引，`selection_hard` 是 one-hot 端口选择矩阵，`selection` 是实际用于计算 `H_eff` 和 sum rate 的选择矩阵。

### 2. 验证和评估统一使用 hard selection

修改文件：

- `llm_fas/train.py`
- `tests/test_train.py`

改动点：

- `_evaluate_model_rate(...)` 增加 `selection_mode` 参数。
- `train_proposed(...)` 的 validation rate 改为 hard selection。
- `evaluate_checkpoint(...)` 的 Proposed 测试结果改为 hard selection。
- Random baseline 在结果表中也标记为 `selection_mode=hard`，因为 Random 本身就是离散端口集合。

### 3. `results.csv` 增加完整元数据

当前结果字段从原来的简化字段扩展为：

```text
method, selection_mode, K, Nx, Ny, N, n_active,
W_lambda_x, W_lambda_y, Pmax_dBm, distance_km, seed,
train_samples, val_samples, test_samples, epochs, batch_size,
gpt2_layers, d_mha, mha_heads, test_sum_rate
```

这为后续 Stage 2 的多 seed 稳定性实验和 Stage 3 之后的 baseline 对比提供统一数据格式。

## 提交记录

本阶段相关提交：

```text
6a43201 feat: add hard inference to proposed model
85efda6 feat: write hard inference evaluation metadata
a728b4a exp: refresh mvp hard inference results
```

另外，为隔离实施 worktree 增加了忽略规则：

```text
611512c chore: ignore local worktrees
```

## 验证结果

实施前基线：

```text
19 passed
```

Task 1 后模型测试：

```text
4 passed
```

Task 2 后训练/评估测试：

```text
3 passed
```

合并回 `mvp-reproduction` 后完整测试：

```text
21 passed in 17.29s
```

最终 worktree 清理后，主工作区保持干净。

## 刷新的 MVP 实验结果

结果文件：

```text
outputs/mvp_train1000_test200_e20/results.csv
```

当前配置：

| 项目 | 值 |
|---|---:|
| `K` | 3 |
| `Nx x Ny` | 4 x 4 |
| `N` | 16 |
| `n_active` | 4 |
| `W` | 2.0 lambda x 2.0 lambda |
| `Pmax` | 20 dBm |
| `distance` | 0.2 km |
| `seed` | 20260606 |
| `train_samples` | 1000 |
| `val_samples` | 200 |
| `test_samples` | 200 |
| `epochs` | 20 |
| `batch_size` | 16 |
| `gpt2_layers` | 2 |
| `d_mha` | 128 |
| `mha_heads` | 4 |

测试 sum rate：

| Method | Selection mode | Test sum rate |
|---|---|---:|
| Random | hard | 19.177417755126953 |
| Proposed | hard | 19.26773422241211 |

Proposed 相对 Random：

| 指标 | 值 |
|---|---:|
| 绝对提升 | 0.09031646728515597 bps/Hz |
| 相对提升 | 0.4709521815626635% |

## 训练动态变化

由于 validation 现在使用 hard selection，新的 `val_loss` 与之前 relaxed validation 的历史结果不再完全可比。

当前 best validation epoch：

| 项目 | 值 |
|---|---:|
| Best epoch | 6 |
| Best validation loss | -19.053684692382813 |
| Best validation sum rate | 19.053684692382813 |

最后一轮：

| 项目 | 值 |
|---|---:|
| Epoch | 20 |
| Train loss | -19.025504440307618 |
| Validation loss | -19.050069274902345 |

模型仍然在前几轮快速提升，随后稳定在约 `19.05 bps/Hz` 的验证 sum rate 附近。

## 当前结论

Stage 1 已完成预期目标：

- 评估路径已改为论文一致的 hard port selection。
- Proposed 在当前 MVP+ 设置下仍略高于 Random。
- 结果 CSV 已具备后续多 seed 和多方法对比所需的基础元数据。
- 单元测试和脚本级 smoke run 均通过。

这说明后续可以进入 Stage 2：在当前小规模配置下做多 seed 稳定性实验，而不是立即扩大到论文完整 `10000` 训练样本和 6 层 GPT-2。

## 剩余风险

- 当前只有显式 `selection_mode="hard"` 的单元测试；隐式 `training=False, selection_mode=None` 默认 hard 路径由实现逻辑覆盖，但还没有单独测试。
- Proposed 相对 Random 的提升仍很小，需要 Stage 2 多 seed 判断是否稳定。
- 当前 LoRA 仍是 PEFT `c_attn` 近似实现，不是论文严格的 Q/V-only LoRA。
- 当前模型仍是 MVP 尺度：GPT-2 2 层、`d_mha=128`、batch size 16。

## 下一步建议

Stage 2 应优先做：

1. 增加多 seed runner，避免手工改配置和输出目录。
2. 在当前 `train=1000, val=200, test=200, epochs=20` 设置下跑至少 3 个 seed。
3. 汇总 Random 和 Proposed 的 mean/std。
4. 如果 hard Proposed 不能稳定优于 Random，先诊断端口选择和训练行为；如果稳定，再进入 Transformer baseline。
