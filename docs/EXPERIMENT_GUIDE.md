# 实验操作指南

在 GPU 机器上拉取代码后，按以下步骤执行实验。

## 前置条件

- Python 3.12+（当前开发环境 3.14，建议 3.12）
- [uv](https://docs.astral.sh/uv/) 包管理器
- GPU 推荐 >= 8GB 显存（代码也支持 CPU，但会非常慢）

## 1. 拉取代码

```bash
git clone <repo-url>
cd "LLM-Based Port Selection and Beamforming for  Multiuser MISO With Fluid Antenna Systems"
git checkout mvp-reproduction
```

## 2. 环境验证（5 分钟）

在 GPU 机器上，不要直接依赖 `uv run --with torch` 做论文实验；该写法可能解析到 CPU-only PyTorch。先创建项目虚拟环境，并安装 CUDA-enabled PyTorch：

```bash
uv venv
uv pip install numpy scipy pyyaml matplotlib tqdm pytest transformers peft
# 按 GPU 机器的 CUDA/驱动版本选择 PyTorch 官方 CUDA wheel index；下面以 cu128 为例。
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

然后确认当前 `torch` 能看到 GPU：

```bash
uv run python -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('torch_cuda=', torch.version.cuda); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

预期：`cuda_available=True`，并打印 GPU 名称。

如果这里是 `False`，后续 `device: auto` 会自动退回 CPU；下面命令使用 `--device cuda`，会在 CUDA 不可用时直接失败，避免静默跑 CPU。

```bash
uv run python -m pytest tests/ -q
```

预期：`45 passed`

## 3. 小规模 Smoke（10-20 分钟 CPU / 2-5 分钟 GPU）

```bash
uv run python scripts/run_stage3_transformer.py --config configs/paper_smoke.yaml --methods proposed --output-root outputs/paper_smoke_test --seeds 20260606 --device cuda
```

检查 `outputs/paper_smoke_test/all_results.csv` 有 `random` 和 `proposed` 两行。
训练启动时会打印 `requested_device=cuda resolved_device=cuda` 和 GPU 名称；每个 seed 输出目录也会写入 `device_info.json`。

## 4. 论文主配置训练

### 4.1 单点验证（最先跑）

```bash
uv run python scripts/run_stage3_transformer.py --config configs/paper_default.yaml --methods proposed --output-root outputs/paper_default_seed20260606 --seeds 20260606 --device cuda
```

配置：d_mha=768, 6层GPT-2, LoRA rank=4, batch=100, train=10000, 200 epochs。
预计：GPU 1-3h, CPU ~30h。

### 4.2 Fig.5 收敛曲线

```bash
uv run python scripts/run_fig5_convergence.py --config configs/paper_default.yaml --output-root outputs/fig5_convergence --seeds 20260606 --device cuda
```

依次训练 bs=50/100/200 各 200 epochs。预计 GPU 4-8h, CPU ~107h。
先用 `--dry-run` 预览不启动训练。

### 4.3 多 seed 验证

```bash
uv run python scripts/run_stage3_extended.py --config configs/paper_default.yaml --methods transformer,proposed --seed-count 5 --output-root outputs/paper_5seed --device cuda
```

## 5. 结果文件

每个实验输出目录结构：

```
outputs/<name>/
  all_results.csv          # 汇总所有 seed 的结果
  summary.csv              # 按方法聚合的均值和标准差
  seed_<seed>/
    config_snapshot.yaml   # 本 seed 实际配置
    device_info.json       # requested/resolved device 和 CUDA 信息
    proposed.pt            # best-val checkpoint (gitignore 排除)
    proposed_train_history.csv  # epoch, train_loss, val_loss
    results.csv            # 本 seed 结果
```

`all_results.csv` 字段：method, selection_mode, K, Nx, Ny, N, n_active, W_lambda_x, W_lambda_y, Pmax_dBm, distance_km, seed, train_samples, val_samples, test_samples, epochs, batch_size, gpt2_layers, d_mha, mha_heads, test_sum_rate

## 6. 快速分析

```bash
uv run python -c "
import csv, statistics
from pathlib import Path
rows = list(csv.DictReader(Path('outputs/<exp>/all_results.csv').open()))
by = {}
for r in rows:
    by.setdefault(r['method'], []).append(float(r['test_sum_rate']))
for m, v in by.items():
    print(f'{m}: mean={statistics.mean(v):.4f} std={statistics.stdev(v) if len(v)>1 else 0:.4f} n={len(v)}')
"
```

## 7. 注意事项

- 首次运行会下载 GPT-2 模型（~500MB），需网络
- `.pt` checkpoint 被 gitignore，新机器需重新训练
- 数据集基于 seed 确定性生成，同 config+seed 跨机器可复现
- LoRA 挂在 GPT-2 `c_attn`（QKV 合并），非论文严格 Q/V-only 版
- CNN 和 LLM-sequential baseline 尚未实现（Stage 5）
- `paper_default.yaml` 未配置 early stopping（当前为固定 200 epochs）
