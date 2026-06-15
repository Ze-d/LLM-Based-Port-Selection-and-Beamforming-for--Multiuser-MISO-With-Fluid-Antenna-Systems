# 实验操作指南

## 修复内容回顾

基于与论文 [Guo et al., 2026] 的逐项对比，修复了 4 个差异点：

| # | 修复项 | 论文描述 | 修复前 | 修复后 |
|---|--------|---------|--------|--------|
| 1 | 位置编码 | Eq.(13)-(14): `H_em = H_fc2 + H_pe` | 无 PE | `_preprocess` 中加入正弦 PE |
| 2 | 输入投影 | FC1: `h_real ∈ R^{KN} → R^{K×dmha}` | 逐端口 Linear(N, dmha) | flatten 后 FC1(K*N, K*dmha) |
| 3 | Early Stopping | patience=10, 监控 val loss | 无，训练满 200 epochs | 10 epoch 无提升即停止 |
| 4 | Random 基线 | Random 端口 + MLP 学功率分配 | 均匀功率 p=q=Pmax/K | MLP(256→128→2K) 学 p,q |

---

## 环境准备

### 硬件要求

- **GPU**: NVIDIA GPU with >= 8GB VRAM（推荐 RTX 4060 或更高）
  - Small-scale 实验 (MVP)：4GB VRAM 足够
  - Full-scale 实验 (Paper)：需要 8GB+ VRAM
- **CPU**: 任意现代多核 CPU
- **Disk**: ~5GB（模型权重 + 数据集缓存）

### 软件依赖

```bash
pip install torch transformers peft pyyaml pytest accelerate safetensors
```

### 验证安装

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
python -m pytest tests/ -v --tb=short
```

预期输出：46 个测试全部通过。

---

## 配置文件说明

项目提供 5 个预设配置，按数据规模和用途分类：

| 配置文件 | 用途 | 数据量(train/val/test) | Epochs | Batch | GPT-2层 | d_mha | 预计时长 |
|---------|------|----------------------|--------|-------|---------|-------|---------|
| `mvp.yaml` | 快速开发/调试 | 1000/200/200 | 20 | 16 | 2 | 128 | ~2分钟 |
| `paper_smoke.yaml` | 论文规模冒烟测试 | 2000/400/400 | 5 | 16 | 6 | 768 | ~5分钟 |
| `paper_training_smoke.yaml` | 最简训练冒烟 | 64/32/32 | 2 | 8 | 6 | 768 | ~30秒 |
| `paper_default.yaml` | **完整论文复现** | 10000/2000/1000 | 200 | 100 | 6 | 768 | ~2-4小时 |
| `fig5_smoke.yaml` | Fig.5收敛冒烟 | 200/40/40 | 5 | 8 | 6 | 768 | ~1分钟 |

**推荐实验顺序：冒烟测试 -> 小规模验证 -> 完整复现**

---

## 实验流程

### Step 1: 最简冒烟测试（验证修复生效）

目的是快速验证代码修改后能正常运行。使用 `paper_training_smoke.yaml`。

```bash
python scripts/run_stage3_extended.py \
  --config configs/paper_training_smoke.yaml \
  --seeds 20260606 \
  --methods random,proposed \
  --output-root outputs/smoke_test_fixed
```

预期输出文件：

```
outputs/smoke_test_fixed/
├── all_results.csv          # 所有 seed 的详细结果
├── summary.csv               # 汇总统计
└── seed_20260606/
    ├── config_snapshot.yaml
    ├── device_info.json
    ├── random.pt             # Random 基线 MLP checkpoint
    ├── random_train_history.csv
    ├── proposed.pt           # Proposed 模型 checkpoint
    ├── proposed_train_history.csv
    ├── train_history.csv
    └── results.csv
```

查看结果：

```bash
cat outputs/smoke_test_fixed/summary.csv
```

**关键检查项：**
- [ ] `random` 行存在且 `test_sum_rate` 合理（应 > 0）
- [ ] `proposed` 行存在且 `test_sum_rate` 合理
- [ ] 训练在 2 epochs 内完成，无报错

---

### Step 2: 小规模 3-Seed 稳定性验证

使用 `mvp.yaml`（small-scale），3 个种子，包含 Random、Transformer、Proposed 三种方法。

```bash
python scripts/run_stage3_extended.py \
  --config configs/mvp.yaml \
  --seed-count 3 \
  --seeds 20260606,20260607,20260608 \
  --methods random,transformer,proposed \
  --output-root outputs/stage3_fixed_3seed
```

查看汇总结果：

```bash
cat outputs/stage3_fixed_3seed/summary.csv
```

**关键检查项：**
- [ ] `proposed.mean_test_sum_rate` > `random.mean_test_sum_rate`

**对比修复前 baseline：**

| 指标 | 修复前 (stage3_transformer_3seed) | 修复后 (预期) |
|------|----------------------------------|--------------|
| Proposed vs Random 胜场 | 2/3 | >= 2/3，均值差距应增大 |
| Proposed 均值 | 19.31 | 应更高 |

---

### Step 3: 完整论文规模复现

使用 `paper_default.yaml`（full-scale），5 个种子，对比三种方法。

```bash
python scripts/run_stage3_extended.py \
  --config configs/paper_default.yaml \
  --seed-count 5 \
  --seeds 20260606,20260607,20260608,20260609,20260610 \
  --methods random,transformer,proposed \
  --output-root outputs/paper_fixed_5seed
```

**预计时长：** 每个种子约 1-2 小时（RTX 4060），总计 5-10 小时。

后台运行：

```bash
# Windows PowerShell
Start-Process python -ArgumentList "scripts/run_stage3_extended.py --config configs/paper_default.yaml --seed-count 5 --methods random,transformer,proposed --output-root outputs/paper_fixed_5seed" -NoNewWindow -RedirectStandardOutput outputs/paper_fixed_5seed.log
```

查看结果：

```bash
cat outputs/paper_fixed_5seed/summary.csv
```

**关键检查项：**
- [ ] `proposed.mean_test_sum_rate` 在 5 种子平均下显著优于 Random
- [ ] `proposed.wins_vs_random` >= 4（5 个种子中至少赢 4 个）
- [ ] Early Stopping 可能在 200 epochs 前停止（正常行为）

---

### Step 4: Fig.5 收敛曲线对比（可选）

```bash
python scripts/run_fig5_convergence.py \
  --config configs/paper_default.yaml \
  --seeds 20260606 \
  --output-root outputs/fig5_fixed \
  --device cuda
```

---

### Step 5: Fig.6-10 参数扫描（完整论文复现）

论文各图需要扫描不同的系统参数。由于实验框架从配置文件读取参数，建议为每组参数值创建独立配置文件，然后运行。

以 **Fig.7 (Pmax 扫描)** 为例：

```bash
# 1. 创建参数配置文件
for pmax in 10 15 20 25 30; do
  sed "s/Pmax_dBm: 20.0/Pmax_dBm: ${pmax}.0/" configs/paper_default.yaml > configs/paper_pmax${pmax}.yaml
  # 修改 output_dir
done

# 2. 运行每个参数
for pmax in 10 15 20 25 30; do
  python scripts/run_stage3_extended.py \
    --config configs/paper_pmax${pmax}.yaml \
    --seeds 20260606 \
    --methods random,transformer,proposed \
    --output-root outputs/fig7_Pmax/Pmax_${pmax}
done
```

**各图参数取值：**

| Figure | 参数 | 取值列表 |
|--------|------|---------|
| Fig.6 (Wx) | system.W_lambda_x, system.W_lambda_y | [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] |
| Fig.7 (Pmax) | system.Pmax_dBm | [10, 15, 20, 25, 30] |
| Fig.8 (n) | system.n_active | [3, 4, 5, 6] |
| Fig.9 (d) | system.distance_km | [0.1, 0.15, 0.2, 0.25, 0.3] |
| Fig.10 (N) | system.Nx x system.Ny | [(3,3), (4,4), (5,5), (6,6)] |

---

## 新增功能：Random 基线 MLP 训练

修复后，Random 基线可以使用 MLP 学习功率分配。

### 单独训练 Random 基线

```bash
python scripts/run_stage3_extended.py \
  --config configs/mvp.yaml \
  --seeds 20260606 \
  --methods random \
  --output-root outputs/random_mlp_test
```

### 同时训练 Random + Proposed

```bash
python scripts/run_stage3_extended.py \
  --config configs/mvp.yaml \
  --seeds 20260606 \
  --methods random,proposed \
  --output-root outputs/random_proposed_test
```

当 `--methods` 包含 `random` 时，evaluation 会自动使用训练好的 MLP。如果 `--methods` 不包含 `random`，则回退到均匀功率分配（向后兼容）。

---

## 结果文件格式

### results.csv（单种子）

```csv
method,selection_mode,K,Nx,Ny,N,n_active,...,test_sum_rate
random,hard,3,4,4,16,4,...,19.244
transformer,hard,3,4,4,16,4,...,19.324
proposed,hard,3,4,4,16,4,...,19.350
```

### summary.csv（多种子汇总）

```csv
method,selection_mode,num_seeds,mean_test_sum_rate,std_test_sum_rate,min_test_sum_rate,max_test_sum_rate,wins_vs_random
proposed,hard,5,19.350,0.15,19.100,19.500,4
random,hard,5,19.200,0.12,19.050,19.350,0
transformer,hard,5,19.280,0.10,19.120,19.400,3
```

### train_history.csv（训练曲线）

```csv
epoch,train_loss,val_loss
1,-15.13,-18.93
2,-18.49,-19.12
...
```

列含义：
- `train_loss` = 负 sum_rate（训练集），越小越好（即 sum_rate 越大越好）
- `val_loss` = 负 sum_rate（验证集），使用 hard 端口选择

---

## 常见问题与排查

### 1. CUDA Out of Memory

**症状：** `RuntimeError: CUDA out of memory`

**解决：**
- 减少 `batch_size`：修改配置文件中的 `train.batch_size`
- 使用 MVP 配置（2层 GPT-2, 128维，仅需 4GB VRAM）
- 使用 CPU 训练：`--device cpu`

### 2. GPT-2 模型下载失败

**症状：** `OSError: Can't load tokenizer for 'gpt2'`

**解决：**
- 确保网络连接正常
- PowerShell: `$env:HF_ENDPOINT="https://hf-mirror.com"`
- Bash: `export HF_ENDPOINT=https://hf-mirror.com`

### 3. Loss 为 NaN 或 Infinity

**症状：** `RuntimeError: Non-finite training loss`

**解决：**
- 确认 learning rate = 1e-6（论文默认值）
- 尝试 `--device cpu` 排除 GPU 问题
- 检查 Sinkhorn 迭代是否数值稳定

### 4. Early Stopping 过早触发

**症状：** 训练在 epoch 11 就停止了

**分析：** 正常行为。如果模型在 10 个 epoch 内没有改善，训练自动停止。这是论文指定的行为（patience=10）。

### 5. 与论文结果差异仍然存在

如果修复后结果仍然不理想：

| 调整项 | 当前值 | 建议尝试 |
|--------|--------|---------|
| LoRA rank | 4 | 8 或 16 |
| tau_min | 0.1 | 0.01 |
| lr | 1e-6 | 5e-6 或 1e-5 |
| gpt2_layers | 6 | 8 或全部 12 层 |
| lora_target | ["c_attn"] | ["c_attn","c_fc","c_proj"] |
| 端口空间 N | 16 (4x4) | 36 (6x6) 或 49 (7x7) |

---

## 快速参考

```bash
# 冒烟测试（最快，约30秒）
python scripts/run_stage3_extended.py --config configs/paper_training_smoke.yaml --seeds 20260606 --methods random,proposed --output-root outputs/quick_test

# 小规模实验（约5分钟，三种方法对比）
python scripts/run_stage3_extended.py --config configs/mvp.yaml --seed-count 3 --methods random,transformer,proposed --output-root outputs/small_test

# 完整论文复现（约5-10小时，5 seeds）
python scripts/run_stage3_extended.py --config configs/paper_default.yaml --seed-count 5 --methods random,transformer,proposed --output-root outputs/full_paper

# 单种子训练+评估
python scripts/train_mvp.py --config configs/paper_default.yaml --device cuda
python scripts/evaluate_mvp.py --config configs/paper_default.yaml --checkpoint outputs/paper_default_seed20260606/proposed.pt --device cuda

# 运行全部测试
python -m pytest tests/ -v

# 查看帮助
python scripts/run_stage3_extended.py --help
python scripts/train_mvp.py --help
python scripts/evaluate_mvp.py --help
```
