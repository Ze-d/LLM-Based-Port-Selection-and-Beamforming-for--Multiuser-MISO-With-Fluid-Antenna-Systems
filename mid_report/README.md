# Mid Report Figures

本目录保存当前中期报告推荐使用的图片。所有图片均由：

```powershell
python scripts/plot_paper_aligned_no_llmseq.py
```

从以下结果文件生成：

```text
outputs/paper_aligned_5seed_5methods/all_results.csv
```

## 推荐使用

| 文件 | 用途 |
|---|---|
| `mean_sum_rate_no_llmseq.pdf` | 四方法平均 test sum-rate 对比 |
| `gain_vs_random_no_llmseq.pdf` | `CNN / Transformer / Proposed` 相对 Random 的增益 |
| `per_seed_sum_rate_no_llmseq.pdf` | 四方法在五个 seed 下的 test sum-rate |

## 诊断图

| 文件 | 用途 |
|---|---|
| `per_seed_gain_no_llmseq.pdf` | Proposed 相对 Random/CNN/Transformer 的逐 seed 配对增益 |

每张图同时提供 `.png`、`.pdf` 和 `.svg` 版本。报告排版优先使用 `.pdf`，网页或幻灯片预览可使用 `.png`。
