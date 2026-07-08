# LLM-Based Port Selection and Beamforming for Multiuser MISO-FAS

本项目用于复现和扩展论文 **LLM-Based Port Selection and Beamforming for Multiuser MISO With Fluid Antenna Systems**。代码实现了多用户 MISO-FAS 信道仿真、端口选择、功率分配、model-based beamforming，以及 `Random / CNN / Transformer / LLM-sequential / Proposed` 等方法的训练和评估流程。

当前仓库重点维护的是 paper-aligned 复现实验：使用 GPT-2 前若干层作为 LLM backbone，通过 LoRA 微调，并以无监督 sum-rate objective 联合学习端口选择与功率分配。

## 项目结构

```text
.
├── configs/                         # 实验配置
├── docs/                            # 架构、实验、复现和报告文档
├── llm_fas/                         # 核心 Python 包
├── mid_report/                      # 中期报告用整理图
├── outputs/                         # 实验输出、summary、all_results 和原始图
├── raw/                             # 论文原文或外部原始材料
├── scripts/                         # 训练、评估、参数扫描和绘图脚本
├── tests/                           # 单元测试
├── requirements.txt                 # Python 依赖
└── README.md
```

更完整的文档导航见 [docs/README.md](docs/README.md)，图片索引见 [docs/FIGURES.md](docs/FIGURES.md)。

## 核心实现

| 模块 | 当前实现 |
|---|---|
| 信道模型 | Jakes 空间相关 + 路损 `128.1 + 37.6log10(d)` |
| 输入预处理 | 复数 CSI 拆分为 real/imag，经过 `FC1 -> MHA -> FC2 -> positional embedding` |
| LLM backbone | GPT-2 前 `NL` 层，paper-aligned 默认 `NL=6` |
| 参数高效微调 | GPT-2 attention Q/V-only LoRA，默认 rank `4` |
| 端口选择 | Gumbel-Sinkhorn soft training，hard inference 修复重复端口 |
| 功率输出 | sigmoid 后 row-wise softmax，并按 `Pmax` 归一化 |
| Beamforming | 论文 Eq.18 对应的 model-based optimal beamforming structure |
| 训练目标 | 无监督最大化 sum-rate，即最小化 `-sum_rate` |
| Sequential baselines | CNN / LLM-sequential 两阶段训练：先训练端口选择器，再冻结端口选择器并训练功率分配器 |

支持方法：

```text
random
cnn
transformer
llm_sequential
proposed
```

命令行中 `llm-sequential` 会自动映射为 `llm_sequential`。

## 环境安装

建议使用 Python 3.12 或更新版本，并在虚拟环境中安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果使用 `uv`，可以直接在命令前加依赖：

```powershell
uv run --with torch --with numpy --with scipy --with pyyaml --with matplotlib --with transformers --with peft python -m pytest -q
```

## 快速验证

运行单元测试：

```powershell
python -m pytest -q
```

运行 MVP 训练：

```powershell
python scripts/train_mvp.py --config configs/mvp.yaml
```

评估 MVP checkpoint：

```powershell
python scripts/evaluate_mvp.py --config configs/mvp.yaml --checkpoint outputs/mvp/proposed.pt
```

## Paper-Aligned 实验

默认 paper-aligned 配置在 [configs/paper_default.yaml](configs/paper_default.yaml)。推荐先阅读 [docs/EXPERIMENT_GUIDE.md](docs/EXPERIMENT_GUIDE.md)，其中包含 GPU 复现实验、五方法主实验、Fig.7-Fig.10 参数扫描和结果汇总命令。

五 seed、五方法主实验示例：

```powershell
python scripts/run_stage3_extended.py `
  --config configs/paper_default.yaml `
  --seeds 20260606,20260607,20260608,20260609,20260610 `
  --methods random,cnn,transformer,llm_sequential,proposed `
  --device cuda `
  --output-root outputs/paper_aligned_5seed_5methods
```

生成去掉 `LLM-sequential` 的中期报告图：

```powershell
python scripts/plot_paper_aligned_no_llmseq.py
```

该脚本会读取：

```text
outputs/paper_aligned_5seed_5methods/all_results.csv
```

并输出图片到：

```text
outputs/paper_aligned_5seed_5methods/figures/
```

## 当前报告图片

`mid_report/` 保存了当前整理后的报告图，每张图都有 `png / pdf / svg` 三种格式：

| 图片 | 含义 |
|---|---|
| `mean_sum_rate_no_llmseq.*` | 去掉 LLM-sequential 后，`Random / CNN / Transformer / Proposed` 的平均 test sum-rate |
| `gain_vs_random_no_llmseq.*` | `CNN / Transformer / Proposed` 相对 Random 的平均增益 |
| `per_seed_sum_rate_no_llmseq.*` | 五个 seed 下四种方法的 test sum-rate 曲线 |
| `per_seed_gain_no_llmseq.*` | Proposed 相对各 baseline 的逐 seed 配对增益，作为诊断图 |

正式汇报优先使用前三组图；`per_seed_gain_no_llmseq.*` 更适合用于解释每个 seed 的配对差异。

## 主要结果文件

主实验结果目录：

```text
outputs/paper_aligned_5seed_5methods/
├── all_results.csv
├── summary.csv
├── figures/
└── seed_20260606 ... seed_20260610/
```

其中：

- `all_results.csv` 保存每个 seed、每种方法的测试结果。
- `summary.csv` 保存跨 seed 均值、标准差和 wins_vs_random。
- `figures/` 保存原始论文风格图和当前重绘图。

## 常用绘图脚本

| 脚本 | 用途 |
|---|---|
| `scripts/plot_paper_aligned_no_llmseq.py` | 绘制中期报告图，去掉 LLM-sequential |
| `scripts/plot_fig7.py` | 绘制发射功率扫描图 |
| `scripts/plot_fig8.py` | 绘制激活端口数扫描图 |
| `scripts/plot_fig9.py` | 绘制距离扫描图 |
| `scripts/plot_fig10.py` | 绘制端口网格扫描图 |
| `scripts/plot_sweep_common.py` | 参数扫描图的公共绘图逻辑 |

## 备注

- `outputs/` 中包含大量实验产物，重新训练前不要随意删除旧目录。
- checkpoint 文件通常较大，不建议提交到 Git。
- 论文未公开部分 baseline 的全部工程细节，当前实现对公开内容做了 paper-aligned 对齐，对未公开细节采用合理补全。
- Sequential baseline 与 Random 采样语义修复后，旧 `outputs/` 中的同名结果仍是历史产物；正式对比请重新训练相关实验。
