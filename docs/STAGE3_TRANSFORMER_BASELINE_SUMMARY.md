# Stage 3：Transformer Baseline 接入总结

## 阶段目标

本阶段目标是将论文中的 Transformer baseline 接入现有 MVP 复现实验链路，使其与 Proposed 共用同一套 CSI 预处理、端口选择、功率因子预测、hard inference、beamforming 和 CSV schema。

## 实现范围

- 新增 `TransformerBaselineModel`。
- 保留 `ProposedLLMFASModel` 的 GPT-2 + LoRA backbone。
- 抽出共享 joint FAS 模型前后处理逻辑。
- 新增 `train_method()` 和 `evaluate_checkpoints()`，支持在同一个 `results.csv` 中写入 `random`、`transformer`、`proposed`。
- 新增 Stage 3 CLI：`scripts/run_stage3_transformer.py`。

## Smoke 实验设置

| 项目 | 值 |
|---|---:|
| Config | `configs/mvp.yaml` |
| Seeds | `20260606` |
| Methods | `transformer, proposed` |
| Selection mode | `hard` |
| Output root | `outputs/stage3_transformer_smoke` |

## Smoke 结果

| Method | Selection mode | Test sum rate |
|---|---|---:|
| random | hard | `19.177417755126953` |
| transformer | hard | `19.37671905517578` |
| proposed | hard | `19.428171768188477` |

## 结果文件

- `outputs/stage3_transformer_smoke/all_results.csv`
- `outputs/stage3_transformer_smoke/summary.csv`
- `outputs/stage3_transformer_smoke/seed_20260606/results.csv`
- `outputs/stage3_transformer_smoke/seed_20260606/transformer_train_history.csv`
- `outputs/stage3_transformer_smoke/seed_20260606/proposed_train_history.csv`

## 结论

本阶段完成的是软件链路验证：Transformer baseline 已能训练、保存 checkpoint，并与 Random/Proposed 共同写入同 schema 结果。

在 `seed=20260606` 的 smoke run 中，Transformer 和 Proposed 都高于 Random，且 Proposed 略高于 Transformer。但由于本次只运行 1 个 seed，这个结果不能作为方法优劣结论。

下一步应运行至少 3 个 seed 的 Stage 3 比较，再判断 Transformer 与 Proposed 的稳定差异。

## 后续动作

- 运行 `scripts/run_stage3_transformer.py --seeds 20260606,20260607,20260608` 获得三 seed 结果。
- 若 Proposed 仍不稳定优于 Random 或 Transformer，优先诊断端口选择分布、训练动态和模型容量，再进入论文完整配置。
