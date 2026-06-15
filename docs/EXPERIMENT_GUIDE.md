# GPU 训练操作指南

这份文档用于把当前项目迁移到另一台有 GPU 的机器上训练，并按论文 `LLM-Based Port Selection and Beamforming for Multiuser MISO With Fluid Antenna Systems` 复现实验。

重要说明：当前代码已经修改过模型结构和 baseline。旧 checkpoint、旧 `outputs/fig7_Pmax/*` 结果不能直接代表新实现，建议新环境中重新训练并把输出写到新的目录。

---

## 1. 当前代码状态

相对最早版本，当前实现已经对齐或补齐了以下论文要点：

| # | 内容 | 当前状态 |
|---|---|---|
| 1 | 位置编码 `H_em = H_fc2 + H_pe` | 已加入正弦 PE |
| 2 | FC1 输入投影 `KN -> K*d_mha` | 已按 flatten CSI 实现 |
| 3 | Early stopping | `patience=10`，监控 validation rate |
| 4 | Random baseline | Random port + MLP 学习 `p,q` |
| 5 | 验证选择模式 | 训练/验证用 soft Gumbel-Sinkhorn |
| 6 | 温度退火 | `tau=max(0.1, 0.95^epoch)`，1-indexed |
| 7 | FC2 | 已改为论文式 `2K*d_mha -> N*n*d_llm` |
| 8 | Hard inference | 已从 Sinkhorn 输出硬化，不再 raw top-k |
| 9 | Power head | 已按论文改为 sigmoid 后再 softmax |
| 10 | Baselines | 已支持 `random,cnn,transformer,llm_sequential,proposed` |

方法名说明：

```text
random           Random port selection + MLP power allocation
cnn              CNN sequential baseline
transformer      Transformer backbone parallel baseline
llm_sequential   LLM port selection + sequential CNN power allocation
proposed         GPT-2 + LoRA parallel port/power output
```

命令行里 `llm-sequential` 也会自动映射为 `llm_sequential`。

---

## 2. GPU 机器准备

### 2.1 推荐硬件

| 实验类型 | 推荐显存 | 说明 |
|---|---:|---|
| `paper_training_smoke.yaml` | 4 GB+ | 只验证流程 |
| `mvp.yaml` | 6-8 GB+ | 小规模稳定性测试 |
| `paper_default.yaml` 三方法 | 12 GB+ | `random,transformer,proposed` |
| `paper_default.yaml` 五方法 | 16-24 GB+ | 增加 `cnn,llm_sequential` |
| 完整 Fig.7 多 seed | 24 GB+ 更稳 | 推荐 RTX 4090 / A5000 / A6000 / A100 |

注意：当前 `FC2` 已改成论文式完整映射，显存占用会比旧版更高。如果 OOM，先降低 `batch_size`，再考虑减少 `gpt2_layers`。

### 2.2 拷贝项目

推荐只同步代码、配置、脚本和文档，不同步旧输出：

```bash
rsync -av \
  --exclude ".git" \
  --exclude ".venv" \
  --exclude "__pycache__" \
  --exclude "outputs" \
  ./ user@gpu-host:/path/to/llm_fas_repro/
```

如果你用 Git，同步前请确认这些文件也包含在新环境中：

```text
llm_fas/models.py
llm_fas/train.py
llm_fas/experiments.py
scripts/run_stage2_seeds.py
scripts/run_stage3_transformer.py
scripts/run_stage3_extended.py
scripts/plot_fig7.py
tests/test_models.py
tests/test_experiments.py
docs/EXPERIMENT_GUIDE.md
```

---

## 3. 环境安装

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

### 3.2 安装 PyTorch 和依赖

先看 GPU 和驱动：

```bash
nvidia-smi
```

建议先按 GPU 机器的 CUDA/驱动情况安装对应的 PyTorch GPU 版本，再安装其它依赖。例如：

```bash
# 先安装与你的 CUDA/驱动匹配的 torch 版本
# 具体命令以 PyTorch 官网 Start Locally 页面为准

pip install transformers peft numpy scipy pyyaml matplotlib tqdm pytest accelerate safetensors
```

如果你的环境已经装好 GPU 版 PyTorch，可以直接：

```bash
pip install -r requirements.txt
```

### 3.3 Hugging Face 下载设置

第一次运行会下载 `gpt2`。如果服务器网络慢，可以设置缓存目录：

```bash
export HF_HOME=/data/hf_cache
export TRANSFORMERS_CACHE=/data/hf_cache
```

如果需要镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Windows PowerShell:

```powershell
$env:HF_HOME="D:\hf_cache"
$env:TRANSFORMERS_CACHE="D:\hf_cache"
$env:HF_ENDPOINT="https://hf-mirror.com"
```

---

## 4. 安装验证

确认 CUDA 可用：

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

先跑轻量测试：

```bash
python -m py_compile llm_fas/models.py llm_fas/train.py llm_fas/experiments.py
python -m pytest tests/test_sinkhorn.py tests/test_physics.py tests/test_beamforming.py -q
```

如果依赖齐全，再跑模型相关测试：

```bash
python -m pytest tests/test_models.py tests/test_experiments.py tests/test_train.py -q
```

---

## 5. 配置文件说明

| 配置文件 | 用途 | 数据量 train/val/test | Epochs | Batch | GPT-2 层 | d_mha |
|---|---|---:|---:|---:|---:|---:|
| `paper_training_smoke.yaml` | 最小流程验证 | 64/32/32 | 2 | 8 | 6 | 768 |
| `paper_smoke.yaml` | 论文结构冒烟 | 2000/400/400 | 5 | 16 | 6 | 768 |
| `mvp.yaml` | 小规模调试 | 1000/200/200 | 20 | 16 | 2 | 128 |
| `paper_default.yaml` | 论文默认设置 | 10000/2000/1000 | 200 | 100 | 6 | 768 |
| `fig5_smoke.yaml` | Fig.5 快速曲线 | 200/40/40 | 5 | 8 | 6 | 768 |

`paper_default.yaml` 默认参数：

```text
K=3, Nx=Ny=4, N=16, n_active=4
W=2lambda x 2lambda
Pmax=20 dBm
distance=0.2 km
noise PSD=-174 dBm/Hz
bandwidth=10 MHz
carrier=2 GHz
```

---

## 6. 推荐训练顺序

### Step 1: 最小冒烟测试

目的：确认依赖、CUDA、GPT-2 下载、训练/评估流程都能跑通。

```bash
python scripts/run_stage3_extended.py \
  --config configs/paper_training_smoke.yaml \
  --seeds 20260606 \
  --methods random,proposed \
  --device cuda \
  --output-root outputs/gpu_smoke
```

检查：

```bash
cat outputs/gpu_smoke/summary.csv
cat outputs/gpu_smoke/seed_20260606/device_info.json
```

期望：

```text
resolved_device = cuda
random/proposed 都有正的 test_sum_rate
```

### Step 2: 五方法小规模测试

目的：确认新增 `cnn` 和 `llm_sequential` baseline 可以正常训练。

```bash
python scripts/run_stage3_extended.py \
  --config configs/mvp.yaml \
  --seeds 20260606 \
  --methods random,cnn,transformer,llm_sequential,proposed \
  --device cuda \
  --output-root outputs/gpu_mvp_5methods_seed20260606
```

检查：

```bash
cat outputs/gpu_mvp_5methods_seed20260606/summary.csv
```

如果这一步 OOM，优先修改 `configs/mvp.yaml`：

```yaml
train:
  batch_size: 8
```

### Step 3: 默认参数 3-seed 验证

目的：在论文默认系统参数下，看 proposed 是否稳定优于 random/transformer。

```bash
python scripts/run_stage3_extended.py \
  --config configs/paper_default.yaml \
  --seeds 20260606,20260607,20260608 \
  --methods random,transformer,proposed \
  --device cuda \
  --output-root outputs/paper_default_3seed_main
```

查看结果：

```bash
cat outputs/paper_default_3seed_main/summary.csv
```

建议判断：

```text
proposed.mean_test_sum_rate > random.mean_test_sum_rate
proposed.mean_test_sum_rate > transformer.mean_test_sum_rate
proposed.wins_vs_random >= 2
```

### Step 4: 默认参数五方法 5-seed

目的：更接近论文的完整 baseline 对比。

```bash
python scripts/run_stage3_extended.py \
  --config configs/paper_default.yaml \
  --seeds 20260606,20260607,20260608,20260609,20260610 \
  --methods random,cnn,transformer,llm_sequential,proposed \
  --device cuda \
  --output-root outputs/paper_default_5seed_5methods
```

这一步耗时较长。建议用 `tmux` 或 `nohup`：

```bash
mkdir -p logs
nohup python scripts/run_stage3_extended.py \
  --config configs/paper_default.yaml \
  --seeds 20260606,20260607,20260608,20260609,20260610 \
  --methods random,cnn,transformer,llm_sequential,proposed \
  --device cuda \
  --output-root outputs/paper_default_5seed_5methods \
  > logs/paper_default_5seed_5methods.log 2>&1 &
```

实时查看：

```bash
tail -f logs/paper_default_5seed_5methods.log
watch -n 5 nvidia-smi
```

---

## 7. Fig.7-10 参数扫描脚本

当前已经提供独立脚本：

| Figure | 训练脚本 | 画图脚本 | 默认输出目录 |
|---|---|---|---|
| Fig.7 | `scripts/run_fig7_pmax.py` | `scripts/plot_fig7.py` | `outputs/fig7_Pmax` |
| Fig.8 | `scripts/run_fig8_active_ports.py` | `scripts/plot_fig8.py` | `outputs/fig8_active_ports` |
| Fig.9 | `scripts/run_fig9_distance.py` | `scripts/plot_fig9.py` | `outputs/fig9_distance` |
| Fig.10 | `scripts/run_fig10_ports.py` | `scripts/plot_fig10.py` | `outputs/fig10_ports` |

所有 `run_fig*.py` 都支持：

```bash
--config       基础配置，默认 configs/paper_default.yaml
--seeds        逗号分隔 seed
--methods      逗号分隔方法
--device       cuda / cuda:0 / cpu / auto
--output-root  输出目录
--dry-run      只打印扫描矩阵，不启动训练
```

建议先用 `--dry-run` 检查命令：

```bash
python scripts/run_fig7_pmax.py --dry-run --device cuda
python scripts/run_fig8_active_ports.py --dry-run --device cuda
python scripts/run_fig9_distance.py --dry-run --device cuda
python scripts/run_fig10_ports.py --dry-run --device cuda
```

---

### 7.1 Fig.7: Pmax 扫描

论文 Fig.7 扫描：

```text
Pmax = 10, 15, 20, 25, 30 dBm
```

单 seed 快速版：

```bash
python scripts/run_fig7_pmax.py \
  --config configs/paper_default.yaml \
  --seeds 20260606 \
  --methods random,cnn,transformer,llm_sequential,proposed \
  --device cuda \
  --output-root outputs/fig7_Pmax_new
```

多 seed 稳定版：

```bash
python scripts/run_fig7_pmax.py \
  --config configs/paper_default.yaml \
  --seeds 20260606,20260607,20260608 \
  --methods random,cnn,transformer,llm_sequential,proposed \
  --device cuda \
  --output-root outputs/fig7_Pmax_new_3seed
```

画图：

```bash
python scripts/plot_fig7.py \
  --output-root outputs/fig7_Pmax_new \
  --output-prefix fig7_Pmax_new
```

输出：

```text
outputs/fig7_Pmax_new.pdf
outputs/fig7_Pmax_new.png
```

---

### 7.2 Fig.8: 激活端口数扫描

论文 Fig.8 扫描：

```text
n = 3, 4, 5, 6
```

训练：

```bash
python scripts/run_fig8_active_ports.py \
  --config configs/paper_default.yaml \
  --seeds 20260606,20260607,20260608 \
  --methods random,cnn,transformer,llm_sequential,proposed \
  --device cuda \
  --output-root outputs/fig8_active_ports_new
```

画图：

```bash
python scripts/plot_fig8.py \
  --output-root outputs/fig8_active_ports_new \
  --output-prefix fig8_active_ports_new
```

---

### 7.3 Fig.9: 用户距离扫描

论文 Fig.9 扫描：

```text
d = 0.1, 0.15, 0.2, 0.25, 0.3 km
```

训练：

```bash
python scripts/run_fig9_distance.py \
  --config configs/paper_default.yaml \
  --seeds 20260606,20260607,20260608 \
  --methods random,cnn,transformer,llm_sequential,proposed \
  --device cuda \
  --output-root outputs/fig9_distance_new
```

画图：

```bash
python scripts/plot_fig9.py \
  --output-root outputs/fig9_distance_new \
  --output-prefix fig9_distance_new
```

---

### 7.4 Fig.10: 总端口数扫描

论文 Fig.10 扫描：

```text
(Nx, Ny) = (3,3), (4,4), (5,5), (6,6)
N = 9, 16, 25, 36
```

训练：

```bash
python scripts/run_fig10_ports.py \
  --config configs/paper_default.yaml \
  --seeds 20260606,20260607,20260608 \
  --methods random,cnn,transformer,llm_sequential,proposed \
  --device cuda \
  --output-root outputs/fig10_ports_new
```

画图：

```bash
python scripts/plot_fig10.py \
  --output-root outputs/fig10_ports_new \
  --output-prefix fig10_ports_new
```

---

### 7.5 输出结构

每个参数点都会生成一个子目录，例如：

```text
outputs/fig7_Pmax_new/
├── _configs/
│   ├── Pmax_10.yaml
│   └── ...
├── Pmax_10/
│   ├── all_results.csv
│   ├── summary.csv
│   └── seed_20260606/
└── Pmax_30/
```

画图脚本优先读取每个子目录的 `summary.csv`。如果是多 seed，会自动使用 `std_test_sum_rate` 画误差棒。

---

## 8. Fig.5 收敛曲线

```bash
python scripts/run_fig5_convergence.py \
  --config configs/paper_default.yaml \
  --seeds 20260606 \
  --output-root outputs/fig5_new \
  --device cuda
```

画图：

```bash
python scripts/plot_fig5.py
```

预期现象：

```text
train_loss 和 val_loss 前 5-10 epoch 快速下降
之后逐渐平稳
train-val gap 不应明显扩大
```

---

## 9. 结果文件

单 seed：

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

汇总：

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

---

## 10. 推荐记录方式

每次训练建议记录：

```bash
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvidia-smi
git status --short
```

保存到日志：

```bash
mkdir -p logs
{
  date
  python --version
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
  nvidia-smi
  git status --short
} > logs/env_snapshot.txt
```

训练完成后建议保存：

```bash
cp outputs/<experiment>/summary.csv logs/<experiment>_summary.csv
```

---

## 11. 常见问题

### 11.1 CUDA 不可用

检查：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

如果 `nvidia-smi` 正常但 PyTorch 显示 `False`，通常是 PyTorch 安装成 CPU 版，重新安装 GPU 版 PyTorch。

### 11.2 OOM

优先顺序：

1. 降低 `train.batch_size`，比如 `100 -> 50 -> 25 -> 16`。
2. 先跑三方法：`random,transformer,proposed`。
3. 用 `paper_smoke.yaml` 确认流程后再扩大。
4. 如果仍不行，临时降低 `gpt2_layers`。

### 11.3 GPT-2 下载失败

设置缓存和镜像：

```bash
export HF_HOME=/data/hf_cache
export TRANSFORMERS_CACHE=/data/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
```

也可以在联网机器上预先下载 `gpt2` 缓存，再复制到 GPU 服务器。

### 11.4 旧 checkpoint 加载失败

这是正常的。当前模型从 `embed_agg/embed_token_proj` 改成了完整 `fc2`，参数名和形状已经变化。请重新训练。

### 11.5 Fig.7 没有明显优于 Random

先确认：

```text
是否使用了新代码
是否包含 trained random baseline
是否至少 3 seeds
是否 Pmax 每个点都重新训练
是否输出目录不是旧 outputs/fig7_Pmax
```

如果仍然差距很小，再尝试：

| 参数 | 默认 | 可尝试 |
|---|---:|---:|
| `train.batch_size` | 100 | 50 或 200 |
| `model.lora_rank` | 4 | 8 |
| `train.lr` | 1e-6 | 5e-6 |
| `train.tau_min` | 0.1 | 0.05 或 0.01 |
| `model.gpt2_layers` | 6 | 8 或 12 |

每次只改一个变量，避免无法判断原因。

---

## 12. 快速命令汇总

```bash
# 环境检查
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# 最小冒烟
python scripts/run_stage3_extended.py --config configs/paper_training_smoke.yaml --seeds 20260606 --methods random,proposed --device cuda --output-root outputs/gpu_smoke

# 五方法小规模
python scripts/run_stage3_extended.py --config configs/mvp.yaml --seeds 20260606 --methods random,cnn,transformer,llm_sequential,proposed --device cuda --output-root outputs/gpu_mvp_5methods_seed20260606

# 默认参数三方法 3-seed
python scripts/run_stage3_extended.py --config configs/paper_default.yaml --seeds 20260606,20260607,20260608 --methods random,transformer,proposed --device cuda --output-root outputs/paper_default_3seed_main

# 默认参数五方法 5-seed
python scripts/run_stage3_extended.py --config configs/paper_default.yaml --seeds 20260606,20260607,20260608,20260609,20260610 --methods random,cnn,transformer,llm_sequential,proposed --device cuda --output-root outputs/paper_default_5seed_5methods

# Fig.7 单 seed
python scripts/run_fig7_pmax.py --config configs/paper_default.yaml --seeds 20260606 --methods random,cnn,transformer,llm_sequential,proposed --device cuda --output-root outputs/fig7_Pmax_new

# Fig.8-10 单 seed
python scripts/run_fig8_active_ports.py --config configs/paper_default.yaml --seeds 20260606 --methods random,cnn,transformer,llm_sequential,proposed --device cuda --output-root outputs/fig8_active_ports_new
python scripts/run_fig9_distance.py --config configs/paper_default.yaml --seeds 20260606 --methods random,cnn,transformer,llm_sequential,proposed --device cuda --output-root outputs/fig9_distance_new
python scripts/run_fig10_ports.py --config configs/paper_default.yaml --seeds 20260606 --methods random,cnn,transformer,llm_sequential,proposed --device cuda --output-root outputs/fig10_ports_new

# 画图
python scripts/plot_fig7.py --output-root outputs/fig7_Pmax_new --output-prefix fig7_Pmax_new
python scripts/plot_fig8.py --output-root outputs/fig8_active_ports_new --output-prefix fig8_active_ports_new
python scripts/plot_fig9.py --output-root outputs/fig9_distance_new --output-prefix fig9_distance_new
python scripts/plot_fig10.py --output-root outputs/fig10_ports_new --output-prefix fig10_ports_new

# 查看结果
cat outputs/paper_default_3seed_main/summary.csv
cat outputs/fig7_Pmax_new/Pmax_30/summary.csv
```
