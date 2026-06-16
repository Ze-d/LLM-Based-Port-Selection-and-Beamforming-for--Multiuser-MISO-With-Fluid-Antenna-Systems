# Paper-Aligned GPU 实验指南

这份文档用于在 GPU 机器上复现论文 `LLM-Based Port Selection and Beamforming for Multiuser MISO With Fluid Antenna Systems` 的实验。当前代码已经按论文描述重新对齐，旧 checkpoint 和旧 `outputs` 结果不能和新结果混用。

建议所有新实验都写入带 `paper_aligned` 的目录，例如：

```text
outputs/paper_aligned_smoke
outputs/paper_aligned_mvp_5methods
outputs/paper_aligned_3seed_main
outputs/paper_aligned_5seed_5methods
outputs/fig7_Pmax_paper_aligned
```

## 1. 当前实现状态

| 项目 | 当前实现 |
|---|---|
| 信道模型 | Jakes 空间相关 + 路损 `128.1 + 37.6log10(d)` |
| 输入预处理 | 复数 CSI 拆成 real/imag，`FC1 -> MHA -> FC2 -> PE` |
| GPT-2 主干 | 使用 GPT-2 前 `NL=6` 层，`d_llm=768` |
| LoRA | GPT-2 attention 的 Q/V-only LoRA，rank `r=4` |
| 端口选择 | Gumbel-Sinkhorn soft training，hard inference 使用行 argmax 并修复重复端口 |
| 功率输出 | `xpower -> sigmoid -> row-wise softmax -> Pmax` |
| Beamforming | 使用论文 Eq.18 的 model-based optimal beamforming structure |
| Loss | 无监督训练，直接最小化 `-sum_rate` |
| Early stopping | 监控 validation sum rate，`patience=10` |
| 默认数据划分 | `8000 train / 2000 val / 1000 test`，对应论文 10000 channel samples 中 20% validation |

支持的方法名：

```text
random           Random port selection + MLP power allocation
cnn              CNN sequential baseline
transformer      Transformer backbone parallel baseline
llm_sequential   LLM port selection + sequential CNN power allocation
proposed         GPT-2 + LoRA parallel port/power output
```

命令行中 `llm-sequential` 会自动映射为 `llm_sequential`。

说明：论文没有公开 CNN、Transformer、LLM-sequential baseline 的完整层数和所有工程细节。当前实现对论文明确写出的部分做了对齐，未公开部分保持合理补全。

## 2. 环境准备

### 2.1 推荐硬件

| 实验 | 推荐显存 | 说明 |
|---|---:|---|
| smoke | 4 GB+ | 只验证流程 |
| mvp 五方法 | 6-8 GB+ | 小规模调试 |
| paper 默认三方法 | 12 GB+ | `random,transformer,proposed` |
| paper 默认五方法 | 16-24 GB+ | `random,cnn,transformer,llm_sequential,proposed` |
| Fig.7-10 多 seed | 24 GB+ | 推荐 RTX 4090 / A5000 / A6000 / A100 |

如果 OOM，优先降低 `batch_size`；只有在必须快速验证流程时，再降低 `gpt2_layers`。

### 2.2 同步项目

推荐只同步代码、配置、脚本、文档和论文原文，不同步旧输出：

```bash
rsync -av \
  --exclude ".git" \
  --exclude ".venv" \
  --exclude "__pycache__" \
  --exclude ".pytest_cache" \
  --exclude "outputs" \
  ./ user@gpu-host:/path/to/llm_fas_repro/
```

如果通过 Git 同步，确认以下改动已包含：

```text
llm_fas/models.py
llm_fas/sinkhorn.py
llm_fas/baselines.py
configs/paper_default.yaml
scripts/run_fig7_pmax.py
scripts/run_fig8_active_ports.py
scripts/run_fig9_distance.py
scripts/run_fig10_ports.py
docs/EXPERIMENT_GUIDE.md
```

## 3. 安装依赖

### 3.1 创建虚拟环境

Linux:

```bash
cd /path/to/llm_fas_repro
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

Windows PowerShell:

```powershell
cd C:\path\to\llm_fas_repro
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

### 3.2 安装 PyTorch 和项目依赖

先查看 GPU 和驱动：

```bash
nvidia-smi
```

先按 GPU 机器的 CUDA/驱动安装对应的 PyTorch GPU 版本，然后安装其它依赖：

```bash
pip install transformers peft numpy scipy pyyaml matplotlib tqdm pytest accelerate safetensors
```

如果你已经确认当前环境有 GPU 版 PyTorch，也可以直接：

```bash
pip install -r requirements.txt
```

### 3.3 Hugging Face 缓存

第一次运行会下载 `gpt2`。服务器网络慢时建议设置缓存目录：

```bash
export HF_HOME=/data/hf_cache
export TRANSFORMERS_CACHE=/data/hf_cache
```

需要镜像时：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Windows PowerShell:

```powershell
$env:HF_HOME="D:\hf_cache"
$env:TRANSFORMERS_CACHE="D:\hf_cache"
$env:HF_ENDPOINT="https://hf-mirror.com"
```

## 4. 安装验证

确认 CUDA 可用：

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

先做语法检查：

```bash
python -m compileall llm_fas tests scripts
```

再跑核心测试：

```bash
python -m pytest tests/test_sinkhorn.py tests/test_physics.py tests/test_beamforming.py -q
python -m pytest tests/test_data_baselines.py tests/test_models.py tests/test_train.py -q
```

## 5. 配置文件

| 配置 | 用途 | train/val/test | epochs | batch | layers | d_mha |
|---|---|---:|---:|---:|---:|---:|
| `configs/paper_training_smoke.yaml` | 最小流程验证 | 64/32/32 | 2 | 8 | 6 | 768 |
| `configs/paper_smoke.yaml` | 论文结构冒烟 | 2000/400/400 | 5 | 16 | 6 | 768 |
| `configs/mvp.yaml` | 小规模五方法调试 | 1000/200/200 | 20 | 16 | 2 | 128 |
| `configs/paper_default.yaml` | 论文默认配置 | 8000/2000/1000 | 200 | 100 | 6 | 768 |

`paper_default.yaml` 的默认系统参数：

```text
K = 3
Nx = Ny = 4, N = 16
n_active = 4
W = 2lambda x 2lambda
Pmax = 20 dBm
d = 0.2 km
noise PSD = -174 dBm/Hz
bandwidth = 10 MHz
carrier = 2 GHz
```

## 6. 推荐训练顺序

脚本名 `run_stage3_extended.py` 是历史命名，当前仍是通用多 seed / 多方法训练入口。

### Step 1: 最小冒烟

目的：确认依赖、CUDA、GPT-2 下载、训练和评估流程都正常。

```bash
python scripts/run_stage3_extended.py \
  --config configs/paper_training_smoke.yaml \
  --seeds 20260606 \
  --methods random,proposed \
  --device cuda \
  --output-root outputs/paper_aligned_smoke
```

检查：

```bash
cat outputs/paper_aligned_smoke/summary.csv
cat outputs/paper_aligned_smoke/seed_20260606/device_info.json
```

期望：

```text
resolved_device = cuda
random/proposed 都有正的 test_sum_rate
```

### Step 2: 五方法 MVP

目的：确认 `random,cnn,transformer,llm_sequential,proposed` 都能训练和评估。

```bash
python scripts/run_stage3_extended.py \
  --config configs/mvp.yaml \
  --seeds 20260606 \
  --methods random,cnn,transformer,llm_sequential,proposed \
  --device cuda \
  --output-root outputs/paper_aligned_mvp_5methods
```

查看：

```bash
cat outputs/paper_aligned_mvp_5methods/summary.csv
```

### Step 3: 论文默认三方法 3-seed

目的：快速判断 paper-aligned 实现下 proposed 是否开始稳定优于 random / transformer。

```bash
python scripts/run_stage3_extended.py \
  --config configs/paper_default.yaml \
  --seeds 20260606,20260607,20260608 \
  --methods random,transformer,proposed \
  --device cuda \
  --output-root outputs/paper_aligned_3seed_main
```

查看：

```bash
cat outputs/paper_aligned_3seed_main/summary.csv
```

建议初步判断：

```text
proposed.mean_test_sum_rate > random.mean_test_sum_rate
proposed.mean_test_sum_rate > transformer.mean_test_sum_rate
proposed.wins_vs_random >= 2
```

### Step 4: 论文默认五方法 5-seed

目的：生成更接近论文主对比的结果。

```bash
python scripts/run_stage3_extended.py \
  --config configs/paper_default.yaml \
  --seeds 20260606,20260607,20260608,20260609,20260610 \
  --methods random,cnn,transformer,llm_sequential,proposed \
  --device cuda \
  --output-root outputs/paper_aligned_5seed_5methods
```

长任务建议用 `nohup`：

```bash
mkdir -p logs
nohup python scripts/run_stage3_extended.py \
  --config configs/paper_default.yaml \
  --seeds 20260606,20260607,20260608,20260609,20260610 \
  --methods random,cnn,transformer,llm_sequential,proposed \
  --device cuda \
  --output-root outputs/paper_aligned_5seed_5methods \
  > logs/paper_aligned_5seed_5methods.log 2>&1 &
```

监控：

```bash
tail -f logs/paper_aligned_5seed_5methods.log
watch -n 5 nvidia-smi
```

## 7. Fig.7-10 参数扫描

所有扫描都基于 `configs/paper_default.yaml`，每个参数点都会重新训练对应方法。

通用参数：

```text
--config       基础配置
--seeds        逗号分隔 seeds
--methods      逗号分隔方法
--device       cuda / cuda:0 / cpu / auto
--output-root  输出目录
--dry-run      只打印扫描矩阵，不启动训练
```

先 dry run：

```bash
python scripts/run_fig7_pmax.py --dry-run --device cuda --output-root outputs/fig7_Pmax_paper_aligned
python scripts/run_fig8_active_ports.py --dry-run --device cuda --output-root outputs/fig8_active_ports_paper_aligned
python scripts/run_fig9_distance.py --dry-run --device cuda --output-root outputs/fig9_distance_paper_aligned
python scripts/run_fig10_ports.py --dry-run --device cuda --output-root outputs/fig10_ports_paper_aligned
```

### 7.1 Fig.7: Pmax

扫描：

```text
Pmax = 10, 15, 20, 25, 30 dBm
```

训练：

```bash
python scripts/run_fig7_pmax.py \
  --config configs/paper_default.yaml \
  --values 10,15,20,25,30 \
  --seeds 20260606,20260607,20260608 \
  --methods random,cnn,transformer,llm_sequential,proposed \
  --device cuda \
  --output-root outputs/fig7_Pmax_paper_aligned
```

画图：

```bash
python scripts/plot_fig7.py \
  --output-root outputs/fig7_Pmax_paper_aligned \
  --output-prefix fig7_Pmax_paper_aligned
```

### 7.2 Fig.8: 激活端口数

扫描：

```text
n = 3, 4, 5, 6
```

训练：

```bash
python scripts/run_fig8_active_ports.py \
  --config configs/paper_default.yaml \
  --values 3,4,5,6 \
  --seeds 20260606,20260607,20260608 \
  --methods random,cnn,transformer,llm_sequential,proposed \
  --device cuda \
  --output-root outputs/fig8_active_ports_paper_aligned
```

画图：

```bash
python scripts/plot_fig8.py \
  --output-root outputs/fig8_active_ports_paper_aligned \
  --output-prefix fig8_active_ports_paper_aligned
```

### 7.3 Fig.9: 用户距离

扫描：

```text
d = 0.1, 0.15, 0.2, 0.25, 0.3 km
```

训练：

```bash
python scripts/run_fig9_distance.py \
  --config configs/paper_default.yaml \
  --values 0.1,0.15,0.2,0.25,0.3 \
  --seeds 20260606,20260607,20260608 \
  --methods random,cnn,transformer,llm_sequential,proposed \
  --device cuda \
  --output-root outputs/fig9_distance_paper_aligned
```

画图：

```bash
python scripts/plot_fig9.py \
  --output-root outputs/fig9_distance_paper_aligned \
  --output-prefix fig9_distance_paper_aligned
```

### 7.4 Fig.10: 总端口数

扫描：

```text
(Nx, Ny) = (3,3), (4,4), (5,5), (6,6)
N = 9, 16, 25, 36
```

训练：

```bash
python scripts/run_fig10_ports.py \
  --config configs/paper_default.yaml \
  --grids 3x3,4x4,5x5,6x6 \
  --seeds 20260606,20260607,20260608 \
  --methods random,cnn,transformer,llm_sequential,proposed \
  --device cuda \
  --output-root outputs/fig10_ports_paper_aligned
```

画图：

```bash
python scripts/plot_fig10.py \
  --output-root outputs/fig10_ports_paper_aligned \
  --output-prefix fig10_ports_paper_aligned
```

### 7.5 扫描输出结构

例如 Fig.7：

```text
outputs/fig7_Pmax_paper_aligned/
├── _configs/
│   ├── Pmax_10.yaml
│   └── ...
├── Pmax_10/
│   ├── all_results.csv
│   ├── summary.csv
│   └── seed_20260606/
└── Pmax_30/
```

画图脚本读取每个参数点的 `summary.csv`。多 seed 时会使用 `std_test_sum_rate` 画误差棒。

## 8. Fig.5 收敛曲线

Fig.5 关注不同 batch size 的训练/验证 loss 收敛。先按脚本 dry run 或直接运行：

```bash
python scripts/run_fig5_convergence.py \
  --config configs/paper_default.yaml \
  --seeds 20260606 \
  --output-root outputs/fig5_paper_aligned \
  --device cuda
```

画图：

```bash
python scripts/plot_fig5.py \
  --output-root outputs/fig5_paper_aligned \
  --output-prefix fig5_paper_aligned \
  --seed 20260606
```

预期现象：

```text
train_loss 和 val_loss 前 5-10 epoch 快速下降
之后逐渐平稳
train-val gap 不应明显扩大
```

## 9. 结果文件说明

单 seed 目录：

```text
outputs/<experiment>/seed_<seed>/
├── config_snapshot.yaml
├── device_info.json
├── random.pt
├── random_train_history.csv
├── cnn.pt
├── cnn_train_history.csv
├── transformer.pt
├── transformer_train_history.csv
├── llm_sequential.pt
├── llm_sequential_train_history.csv
├── proposed.pt
├── proposed_train_history.csv
├── train_history.csv
└── results.csv
```

汇总目录：

```text
outputs/<experiment>/
├── all_results.csv
└── summary.csv
```

`summary.csv` 关键列：

```text
method
selection_mode
num_seeds
mean_test_sum_rate
std_test_sum_rate
min_test_sum_rate
max_test_sum_rate
wins_vs_random
```

## 10. 结果检查

每次训练完成后先看：

```bash
cat outputs/<experiment>/summary.csv
```

重点检查：

```text
1. 每个方法都有结果。
2. test_sum_rate 是有限正数。
3. 多 seed 时 std_test_sum_rate 不应异常大。
4. proposed 是否在 mean_test_sum_rate 和 wins_vs_random 上领先。
```

查看单个 seed：

```bash
cat outputs/<experiment>/seed_20260606/results.csv
cat outputs/<experiment>/seed_20260606/proposed_train_history.csv
```

## 11. 日志和环境快照

训练前建议保存环境信息：

```bash
mkdir -p logs
{
  date
  python --version
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
  python -c "import torch; print('cuda available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
  nvidia-smi
  git status --short
} > logs/env_snapshot_paper_aligned.txt
```

训练后保存 summary：

```bash
cp outputs/paper_aligned_3seed_main/summary.csv logs/paper_aligned_3seed_main_summary.csv
cp outputs/fig7_Pmax_paper_aligned/Pmax_30/summary.csv logs/fig7_Pmax30_paper_aligned_summary.csv
```

## 12. 常见问题

### 12.1 CUDA 不可用

检查：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

如果 `nvidia-smi` 正常但 PyTorch 显示 `False`，通常是 PyTorch 安装成 CPU 版，需要重新安装 GPU 版 PyTorch。

### 12.2 OOM

优先顺序：

1. 降低 `train.batch_size`，例如 `100 -> 50 -> 25 -> 16`。
2. 先跑三方法：`random,transformer,proposed`。
3. 用 `configs/paper_smoke.yaml` 验证流程。
4. 只在快速验证时临时降低 `model.gpt2_layers`。

### 12.3 GPT-2 下载失败

设置缓存和镜像：

```bash
export HF_HOME=/data/hf_cache
export TRANSFORMERS_CACHE=/data/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
```

也可以在联网机器上预先下载 `gpt2` 缓存，再复制到 GPU 服务器。

### 12.4 旧 checkpoint 加载失败

这是正常现象。当前实现已经改过 `FC2`、LoRA 和 hard inference，旧 checkpoint 参数名或形状可能不匹配。请重新训练。

### 12.5 proposed 没有明显领先

先确认：

```text
1. 输出目录是 paper_aligned 新目录，不是旧 outputs。
2. 每个参数点都重新训练，而不是复用旧 checkpoint。
3. random baseline 是 trained random baseline。
4. 至少跑了 3 seeds。
5. Fig.7-10 每个横轴点都单独训练。
```

如果仍然差距很小，再尝试每次只改一个变量：

| 参数 | 默认 | 可尝试 |
|---|---:|---:|
| `train.batch_size` | 100 | 50 或 200 |
| `model.lora_rank` | 4 | 8 |
| `train.lr` | 1e-6 | 5e-6 |
| `train.tau_min` | 0.1 | 0.05 或 0.01 |
| `model.gpt2_layers` | 6 | 8 或 12 |

## 13. 快速命令汇总

```bash
# 环境检查
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# 最小冒烟
python scripts/run_stage3_extended.py --config configs/paper_training_smoke.yaml --seeds 20260606 --methods random,proposed --device cuda --output-root outputs/paper_aligned_smoke

# 五方法 MVP
python scripts/run_stage3_extended.py --config configs/mvp.yaml --seeds 20260606 --methods random,cnn,transformer,llm_sequential,proposed --device cuda --output-root outputs/paper_aligned_mvp_5methods

# 默认参数三方法 3-seed
python scripts/run_stage3_extended.py --config configs/paper_default.yaml --seeds 20260606,20260607,20260608 --methods random,transformer,proposed --device cuda --output-root outputs/paper_aligned_3seed_main

# 默认参数五方法 5-seed
python scripts/run_stage3_extended.py --config configs/paper_default.yaml --seeds 20260606,20260607,20260608,20260609,20260610 --methods random,cnn,transformer,llm_sequential,proposed --device cuda --output-root outputs/paper_aligned_5seed_5methods

# Fig.7-10 三 seed
python scripts/run_fig7_pmax.py --config configs/paper_default.yaml --values 10,15,20,25,30 --seeds 20260606,20260607,20260608 --methods random,cnn,transformer,llm_sequential,proposed --device cuda --output-root outputs/fig7_Pmax_paper_aligned
python scripts/run_fig8_active_ports.py --config configs/paper_default.yaml --values 3,4,5,6 --seeds 20260606,20260607,20260608 --methods random,cnn,transformer,llm_sequential,proposed --device cuda --output-root outputs/fig8_active_ports_paper_aligned
python scripts/run_fig9_distance.py --config configs/paper_default.yaml --values 0.1,0.15,0.2,0.25,0.3 --seeds 20260606,20260607,20260608 --methods random,cnn,transformer,llm_sequential,proposed --device cuda --output-root outputs/fig9_distance_paper_aligned
python scripts/run_fig10_ports.py --config configs/paper_default.yaml --grids 3x3,4x4,5x5,6x6 --seeds 20260606,20260607,20260608 --methods random,cnn,transformer,llm_sequential,proposed --device cuda --output-root outputs/fig10_ports_paper_aligned

# 画图
python scripts/plot_fig7.py --output-root outputs/fig7_Pmax_paper_aligned --output-prefix fig7_Pmax_paper_aligned
python scripts/plot_fig8.py --output-root outputs/fig8_active_ports_paper_aligned --output-prefix fig8_active_ports_paper_aligned
python scripts/plot_fig9.py --output-root outputs/fig9_distance_paper_aligned --output-prefix fig9_distance_paper_aligned
python scripts/plot_fig10.py --output-root outputs/fig10_ports_paper_aligned --output-prefix fig10_ports_paper_aligned

# 查看结果
cat outputs/paper_aligned_3seed_main/summary.csv
cat outputs/paper_aligned_5seed_5methods/summary.csv
cat outputs/fig7_Pmax_paper_aligned/Pmax_30/summary.csv
```
