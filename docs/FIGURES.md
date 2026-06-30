# 图片索引

本项目图片主要分为两类：`mid_report/` 中的最终汇报图，以及 `outputs/` 中的实验原始图。

## 中期报告图

目录：[../mid_report](../mid_report)

`mid_report/` 是当前推荐用于汇报的图片集合。每张图均提供 `png / pdf / svg` 三种格式。

| 文件名 | 推荐用途 |
|---|---|
| `mean_sum_rate_no_llmseq.*` | 展示 `Random / CNN / Transformer / Proposed` 的五 seed 平均 test sum-rate |
| `gain_vs_random_no_llmseq.*` | 展示 `CNN / Transformer / Proposed` 相对 Random 的平均增益 |
| `per_seed_sum_rate_no_llmseq.*` | 展示五个 seed 下四种方法的 test sum-rate 变化 |
| `per_seed_gain_no_llmseq.*` | 诊断 Proposed 相对各 baseline 的逐 seed 配对增益 |

正式报告建议优先使用：

```text
mid_report/mean_sum_rate_no_llmseq.pdf
mid_report/gain_vs_random_no_llmseq.pdf
mid_report/per_seed_sum_rate_no_llmseq.pdf
```

## 主实验原始图

目录：[../outputs/paper_aligned_5seed_5methods/figures](../outputs/paper_aligned_5seed_5methods/figures)

| 文件名 | 内容 |
|---|---|
| `mean_sum_rate.svg` | 原始五方法平均 sum-rate 图，包含 LLM-sequential |
| `gain_vs_random.svg` | 原始五方法 gain vs random 图 |
| `per_seed_sum_rate.svg` | 原始五方法逐 seed sum-rate 图 |
| `mean_sum_rate_no_llmseq.*` | 去掉 LLM-sequential 后的平均 sum-rate 图 |
| `gain_vs_random_no_llmseq.*` | 去掉 LLM-sequential 后的 gain vs random 图 |
| `per_seed_sum_rate_no_llmseq.*` | 去掉 LLM-sequential 后的逐 seed sum-rate 图 |
| `per_seed_gain_no_llmseq.*` | Proposed 逐 seed 配对增益诊断图 |
| `proposed_gain_no_llmseq.*` | 旧版诊断图，不建议用于正式报告 |

## 参数扫描图

这些图位于 `outputs/` 根目录，来自 paper-aligned 参数扫描实验。

| 文件名 | 内容 |
|---|---|
| `fig7_Pmax_paper_aligned.png/pdf` | 最大发射功率 `Pmax` 扫描 |
| `fig8_active_ports_paper_aligned.png/pdf` | 激活端口数 `n` 扫描 |

对应数据目录：

```text
outputs/fig7_Pmax_paper_aligned/
outputs/fig8_active_ports_paper_aligned/
outputs/fig9_distance_paper_aligned/
```

## 重新生成图片

生成中期报告图：

```powershell
python scripts/plot_paper_aligned_no_llmseq.py
```

生成 Fig.7-Fig.10 参数扫描图：

```powershell
python scripts/plot_fig7.py --output-root outputs/fig7_Pmax_paper_aligned --output-prefix fig7_Pmax_paper_aligned
python scripts/plot_fig8.py --output-root outputs/fig8_active_ports_paper_aligned --output-prefix fig8_active_ports_paper_aligned
python scripts/plot_fig9.py --output-root outputs/fig9_distance_paper_aligned --output-prefix fig9_distance_paper_aligned
python scripts/plot_fig10.py --output-root outputs/fig10_ports_paper_aligned --output-prefix fig10_ports_paper_aligned
```

如果更新了 `outputs/paper_aligned_5seed_5methods/all_results.csv`，请重新运行绘图脚本，并同步更新 `mid_report/`。
