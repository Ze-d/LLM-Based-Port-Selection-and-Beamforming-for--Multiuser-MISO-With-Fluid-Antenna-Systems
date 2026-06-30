# 文档索引

本目录保存项目架构、实验复现、阶段总结和报告材料。建议从根目录 [README.md](../README.md) 开始阅读，再按需求进入下面的文档。

## 推荐阅读顺序

1. [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)  
   Paper-aligned GPU 实验指南，包含环境、训练、评估、Fig.7-Fig.10 扫描和常用命令。

2. [exp/reproduction_report.md](exp/reproduction_report.md)  
   论文复现报告，适合作为汇报或论文复现说明的主体材料。

3. [PAPER_ALIGNMENT_TECHNICAL_NOTES.md](PAPER_ALIGNMENT_TECHNICAL_NOTES.md)  
   Paper alignment 过程中的技术笔记，记录当前实现与论文描述的对齐情况、差异和排查结论。

4. [FIGURES.md](FIGURES.md)  
   图片索引，说明 `mid_report/` 和 `outputs/` 中各组图片的用途。

## 架构与设计

| 文档 | 内容 |
|---|---|
| [full_arch.md](full_arch.md) | 完整实验架构流程图 |
| [mvp_arch.md](mvp_arch.md) | MVP 版本架构说明 |
| [PLAN.md](PLAN.md) | 项目早期计划 |
| [MVP_REPRODUCTION_PLAN.md](MVP_REPRODUCTION_PLAN.md) | MVP 复现计划 |

## 实验结果与阶段总结

| 文档 | 内容 |
|---|---|
| [MVP_EXPERIMENT_RESULTS.md](MVP_EXPERIMENT_RESULTS.md) | MVP 实验结果 |
| [STAGE1_HARD_INFERENCE_SUMMARY.md](STAGE1_HARD_INFERENCE_SUMMARY.md) | Stage 1 hard inference 总结 |
| [STAGE2_MVP_STABILITY_SUMMARY.md](STAGE2_MVP_STABILITY_SUMMARY.md) | Stage 2 MVP 稳定性总结 |
| [STAGE3_TRANSFORMER_BASELINE_SUMMARY.md](STAGE3_TRANSFORMER_BASELINE_SUMMARY.md) | Stage 3 Transformer baseline 总结 |

## 报告与过程记录

| 路径 | 内容 |
|---|---|
| [exp/1.md](exp/1.md) | 实验过程记录 1 |
| [exp/2.md](exp/2.md) | 实验过程记录 2 |
| [exp/3.md](exp/3.md) | 实验过程记录 3 |
| [exp/plan.pdf](exp/plan.pdf) | 实验计划 PDF |
| [exp/reproduction_report.md](exp/reproduction_report.md) | 完整复现报告 |

## 自动化计划与规格文档

这些文件记录了阶段性实现计划和规格，主要用于追溯开发过程：

| 路径 | 内容 |
|---|---|
| [superpowers/plans/2026-06-08-stage1-hard-inference-metadata.md](superpowers/plans/2026-06-08-stage1-hard-inference-metadata.md) | Stage 1 计划 |
| [superpowers/plans/2026-06-08-stage2-mvp-stability.md](superpowers/plans/2026-06-08-stage2-mvp-stability.md) | Stage 2 计划 |
| [superpowers/plans/2026-06-08-stage3-transformer-baseline.md](superpowers/plans/2026-06-08-stage3-transformer-baseline.md) | Stage 3 baseline 计划 |
| [superpowers/plans/2026-06-09-stage3-best-checkpoints-extended-seeds.md](superpowers/plans/2026-06-09-stage3-best-checkpoints-extended-seeds.md) | 多 seed 扩展计划 |
| [superpowers/specs/2026-06-08-full-reproduction-design.md](superpowers/specs/2026-06-08-full-reproduction-design.md) | 完整复现设计规格 |

## 维护建议

- 新增正式实验结果时，优先更新 [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md) 和 [FIGURES.md](FIGURES.md)。
- 新增汇报图时，将最终版复制到 `mid_report/`，并在 [../mid_report/README.md](../mid_report/README.md) 中说明用途。
- 过程性记录可以继续放在 `docs/exp/`，但根 README 只保留稳定入口。
