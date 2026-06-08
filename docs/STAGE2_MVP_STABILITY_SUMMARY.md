# Stage 2：MVP+ 多 Seed 稳定性实验总结

## 阶段目标

本阶段目标是在不扩大模型和数据规模的前提下，验证 hard-inference Proposed 相对 Random 的结果是否具有 seed 稳定性。

Stage 1 已经将 Proposed 的验证和测试路径改为 hard port selection。本阶段在该基础上新增多 seed runner，并用相同 MVP+ 配置运行 3 个随机种子，判断当前结果是否足够稳定，可以支撑继续进入 Transformer baseline。

## 实验设置

| 项目 | 值 |
|---|---:|
| Seeds | `20260606, 20260607, 20260608` |
| Train samples | `1000` |
| Val samples | `200` |
| Test samples | `200` |
| Epochs | `20` |
| Batch size | `16` |
| GPT-2 layers | `2` |
| d_mha | `128` |
| MHA heads | `4` |
| Selection mode | `hard` |
| Output root | `outputs/stage2_mvp_stability` |

## 新增工具

本阶段新增：

- `llm_fas/experiments.py`
- `scripts/run_stage2_seeds.py`
- `tests/test_experiments.py`

主要能力：

- 解析逗号分隔 seed 列表。
- 为每个 seed 克隆配置，并设置独立输出目录。
- 为每个 seed 保存 `config_snapshot.yaml`。
- 依次训练 Proposed、评估 Random 和 Proposed。
- 汇总所有 seed 的 `results.csv` 到 `all_results.csv`。
- 按 `method` 和 `selection_mode` 聚合 mean/std/min/max/wins。

## 结果文件

- `outputs/stage2_mvp_stability/all_results.csv`
- `outputs/stage2_mvp_stability/summary.csv`

每个 seed 还包含独立目录：

- `outputs/stage2_mvp_stability/seed_20260606/`
- `outputs/stage2_mvp_stability/seed_20260607/`
- `outputs/stage2_mvp_stability/seed_20260608/`

## 单 Seed 结果

| Seed | Random | Proposed | Proposed - Random | Proposed wins |
|---:|---:|---:|---:|---|
| `20260606` | `19.177417755126953` | `19.26773422241211` | `0.09031646728515625` | Yes |
| `20260607` | `19.102201461791992` | `19.305120010375976` | `0.20291854858398438` | Yes |
| `20260608` | `19.456623077392578` | `19.398284454345703` | `-0.058338623046875` | No |

## 聚合结果

| Method | Selection mode | Num seeds | Mean test sum rate | Std | Min | Max | Wins vs Random |
|---|---|---:|---:|---:|---:|---:|---:|
| Proposed | hard | `3` | `19.323712895711264` | `0.06723178045609907` | `19.26773422241211` | `19.398284454345703` | `2` |
| Random | hard | `3` | `19.24541409810384` | `0.18673858036009944` | `19.102201461791992` | `19.456623077392578` | `0` |

平均提升：

| 指标 | 值 |
|---|---:|
| Proposed mean - Random mean | `0.07829879760742513 bps/Hz` |
| Relative mean gain | `0.4068500630904392%` |

## 验证命令

新增代码验证：

```powershell
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest -q
```

结果：

```text
33 passed in 18.14s
```

CLI help 验证：

```powershell
uv run --with torch --with numpy --with pyyaml --with transformers --with peft python scripts/run_stage2_seeds.py --help
```

结果：命令退出码为 `0`，输出包含 `--config`、`--seeds`、`--output-root`。

真实三 seed 实验命令：

```powershell
uv run --with torch --with numpy --with pyyaml --with transformers --with peft python scripts/run_stage2_seeds.py --config configs/mvp.yaml --seeds 20260606,20260607,20260608 --output-root outputs/stage2_mvp_stability
```

结果：命令完成并写出 `all_results.csv` 和 `summary.csv`。运行过程中 HuggingFace 输出未登录限速提示，但模型加载成功，实验完成。

## 结论

当前 MVP+ 设置下，Proposed 的 3-seed 平均 sum rate 高于 Random：

```text
Proposed mean = 19.323712895711264
Random mean   = 19.24541409810384
```

但是 Proposed 只在 3 个 seed 中赢了 2 个，`seed=20260608` 上低于 Random：

```text
Random   = 19.456623077392578
Proposed = 19.398284454345703
```

因此，Stage 2 的结论不是“稳定通过”，而是：

> 当前 hard-inference Proposed 在 MVP+ 设置下表现出小幅平均优势，但 seed 稳定性不足。直接进入更大规模论文配置或新增复杂 baseline 前，应优先诊断训练和端口选择稳定性，或至少在 Stage 3 中保留该风险并继续用多 seed 评价。

## 后续动作

建议下一步不要立刻扩大到 `train=10000` 和 GPT-2 6 层，而是先做以下诊断或稳健性增强：

1. 检查 `seed=20260608` 的训练曲线和端口选择结果，确认 Proposed 输给 Random 是训练不足、端口选择波动，还是 Random 抽样较强。
2. 导出每个 seed 的 selected ports，用固定测试样本观察 Proposed 是否学到稳定端口模式。
3. 增加轻量配置下的训练轮数或 batch size 对照，判断当前波动是否来自训练不足。
4. 若继续 Stage 3 Transformer baseline，应要求 Transformer 和 Proposed 都使用相同多 seed 评估，不再只看单 seed。

## 风险记录

- 3 个 seed 的样本量仍较小，不能作为最终论文级统计结论。
- 当前 Proposed 平均优势只有约 `0.41%`，对随机性敏感。
- 当前仍是 MVP 模型容量：GPT-2 2 层、`d_mha=128`、batch size 16。
- 当前 LoRA 仍为 PEFT `c_attn` 近似实现，不是论文严格 Q/V-only LoRA。
