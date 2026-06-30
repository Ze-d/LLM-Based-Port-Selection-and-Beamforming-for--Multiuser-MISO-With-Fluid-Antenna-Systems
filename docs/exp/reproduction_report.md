# 《LLM-Based Port Selection and Beamforming for Multiuser MISO With Fluid Antenna Systems》论文复现报告

## 1. 复现目标

本文复现对象为 Guo 等人在 2026 年提出的论文 **LLM-Based Port Selection and Beamforming for Multiuser MISO With Fluid Antenna Systems**。原论文面向多用户下行 MISO-FAS 系统，研究在有限 RF chain 条件下如何联合优化 FAS 端口选择与多用户波束赋形，从而提升系统和速率。

本次复现的主要目标包括：

1. 搭建论文中多用户 MISO-FAS 的信道、端口选择、功率分配和波束赋形仿真链路。
2. 复现 Proposed LLM-FAS 方法的核心结构，即 GPT-2 前若干层 + LoRA 微调 + 并行端口/功率输出。
3. 实现论文中的主要对比方法，包括 Random、CNN、Transformer、LLM-sequential 和 Proposed。
4. 在默认系统参数和关键变量扫描下，比较各方法的测试 sum-rate，并分析当前复现结果与论文预期趋势的一致性和差异。

## 2. 论文方法概述

原论文考虑一个基站配备二维 Fluid Antenna System 的多用户 MISO 下行系统。基站共有 \(N\) 个候选端口，但由于 RF chain 数量有限，每次只能激活 \(n\) 个端口服务 \(K\) 个单天线用户。因此系统需要同时解决两个耦合问题：

- 离散端口选择：从 \(N\) 个候选端口中选择 \(n\) 个激活端口；
- 连续波束赋形：在选定端口上为多个用户设计发射波束。

论文提出的 LLM-FAS 框架将 CSI 作为输入 token，经由 FC、MHA、位置编码和 GPT-2 backbone 提取高维信道特征，再通过 LoRA 进行参数高效微调。模型输出包括端口选择矩阵和功率分配因子，随后结合 model-based optimal beamforming structure 计算最终波束赋形矩阵。训练目标为无监督最大化 sum-rate，即最小化负 sum-rate。

## 3. 复现实现情况

当前仓库已完成 paper-aligned 版本的主要复现链路，核心实现包括：

| 模块 | 当前实现 |
|---|---|
| 信道模型 | Jakes 空间相关 + 路损模型 \(128.1 + 37.6\log_{10}(d)\) |
| CSI 输入 | 复数 CSI 拆分为 real/imag 分量 |
| 输入预处理 | `FC1 -> MHA -> FC2 -> positional encoding` |
| LLM 主干 | GPT-2 前 6 层，隐藏维度 768 |
| LoRA | GPT-2 attention Q/V-only LoRA，rank = 4 |
| 端口选择 | Gumbel-Sinkhorn soft training，hard inference 使用 argmax 并修复重复端口 |
| 功率输出 | sigmoid 后 row-wise softmax，再按 \(P_{max}\) 归一化 |
| 波束赋形 | 使用论文 Eq.18 对应的 model-based optimal beamforming structure |
| 损失函数 | 无监督训练，直接优化负 sum-rate |
| Early stopping | 监控 validation sum-rate，patience = 10 |

已实现并参与实验的方法包括：

| 方法 | 说明 |
|---|---|
| Random | 随机端口选择 + 学习式功率分配 |
| CNN | CNN sequential baseline |
| Transformer | Transformer backbone parallel baseline |
| LLM-sequential | LLM 端口选择 + sequential CNN 功率输出 |
| Proposed | GPT-2 + LoRA 并行端口和功率输出 |

需要说明的是，论文未完全公开 CNN、Transformer 和 LLM-sequential baseline 的全部工程细节，因此当前实现对论文明确给出的部分进行了对齐，对未公开部分采用了合理补全。

## 4. 实验设置

默认 paper-aligned 实验配置如下：

| 参数 | 值 |
|---|---:|
| 用户数 \(K\) | 3 |
| 端口网格 | \(4 \times 4\) |
| 候选端口数 \(N\) | 16 |
| 激活端口数 \(n\) | 4 |
| 天线区域 | \(2\lambda \times 2\lambda\) |
| 最大发射功率 \(P_{max}\) | 20 dBm |
| 用户距离 \(d\) | 0.2 km |
| 训练 / 验证 / 测试样本 | 8000 / 2000 / 1000 |
| GPT-2 层数 | 6 |
| LoRA rank | 4 |
| 端口推理方式 | hard selection |
| 随机种子 | 20260606-20260610 |

主要结果目录为：

- `outputs/paper_aligned_5seed_5methods`
- `outputs/paper_aligned_3seed_main`
- `outputs/fig7_Pmax_paper_aligned`
- `outputs/fig8_active_ports_paper_aligned`
- `outputs/fig9_distance_paper_aligned`

## 5. 默认五方法复现结果

在默认参数下，五方法 5-seed 结果如下：

| 方法 | 平均 sum-rate | 标准差 | 最小值 | 最大值 | 相对 Random 胜出次数 |
|---|---:|---:|---:|---:|---:|
| CNN | 19.2549 | 0.1079 | 19.1669 | 19.4378 | 3 |
| LLM-sequential | 19.3380 | 0.0989 | 19.2009 | 19.4701 | 5 |
| Proposed | 19.3380 | 0.0989 | 19.2009 | 19.4701 | 5 |
| Random | 19.2502 | 0.0771 | 19.1683 | 19.3763 | 0 |
| Transformer | 19.1936 | 0.0584 | 19.1069 | 19.2571 | 2 |

从平均结果看，Proposed 达到 `19.3380 bps/Hz`，高于 Random 的 `19.2502 bps/Hz`，绝对提升约 `0.0879 bps/Hz`，相对提升约 `0.46%`。Proposed 也高于 Transformer，绝对提升约 `0.1444 bps/Hz`。

但是，Proposed 与 LLM-sequential 的结果几乎完全一致，二者平均 sum-rate 只相差约 `2.9e-6 bps/Hz`。这说明当前复现已经验证了 LLM 类方法相对 Random/Transformer 的一定优势，但尚未复现出论文中 Proposed 并行结构相对 LLM-sequential 明显领先的现象。

## 6. 发射功率扫描结果

对 \(P_{max}=10,15,20,25,30\) dBm 进行 3-seed 扫描，结果如下：

| \(P_{max}\) | Random | Transformer | LLM-sequential | Proposed | Proposed - Random |
|---:|---:|---:|---:|---:|---:|
| 10 dBm | 10.2098 | 10.2116 | 10.2052 | 10.2052 | -0.0045 |
| 15 dBm | 14.5092 | 14.5145 | 14.5785 | 14.5668 | +0.0576 |
| 20 dBm | 19.2070 | 19.2228 | 19.2971 | 19.2971 | +0.0901 |
| 25 dBm | 24.0857 | 24.1050 | 24.1813 | 24.1813 | +0.0957 |
| 30 dBm | 29.0333 | 29.0541 | 29.1313 | 29.1313 | +0.0980 |

该扫描显示，随着发射功率增加，所有方法的 sum-rate 均单调提升，符合通信系统基本规律。Proposed 在 15 dBm 及以上均优于 Random，且增益随功率提高略有扩大；但在 10 dBm 低功率区域，Proposed 略低于 Random，说明当前模型在低 SNR 下的端口选择和功率输出仍不够稳定。

## 7. 激活端口数扫描结果

对激活端口数 \(n=3,4,5,6\) 进行 3-seed 扫描，结果如下：

| 激活端口数 \(n\) | Random | Transformer | CNN | LLM-sequential | Proposed | Proposed - Random |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 15.4823 | 15.4733 | 15.4529 | 15.5227 | 15.5231 | +0.0408 |
| 4 | 19.2070 | 19.2228 | 19.1962 | 19.2971 | 19.2971 | +0.0901 |
| 5 | 21.3212 | 21.2679 | 21.3486 | 21.2442 | 21.2442 | -0.0770 |
| 6 | 22.7355 | 22.7427 | 22.6769 | 22.7213 | 22.7219 | -0.0136 |

结果表明，随着激活端口数增加，系统 sum-rate 整体上升，这是因为更多激活端口带来了更高空间自由度。然而 Proposed 仅在 \(n=3,4\) 时优于 Random，在 \(n=5,6\) 时反而低于 Random。这说明当前模型在端口数增大时的组合选择泛化能力不足，可能需要更充分训练、更强正则、更合理的 Sinkhorn 温度退火，或针对不同 \(n\) 单独调参。

## 8. 用户距离扫描结果

当前已生成完整 summary 的距离点包括 \(d=0.1\) km 和 \(d=0.15\) km：

| 用户距离 \(d\) | Random | Transformer | CNN | LLM-sequential | Proposed | Proposed - Random |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 km | 30.3432 | 30.3642 | 30.3323 | 30.4415 | 30.4415 | +0.0983 |
| 0.15 km | 23.7880 | 23.8073 | 23.7770 | 23.8835 | 23.8835 | +0.0955 |

从已有结果看，距离增大导致 sum-rate 明显下降，符合路损增强的物理规律。Proposed 在已完成的两个距离点上均优于 Random 和 Transformer。

但 `outputs/fig9_distance_paper_aligned/d_0p2` 目录目前缺少完整 `summary.csv`，因此距离扫描尚未完整复现，不能给出 Fig.9 全曲线结论。

## 9. 结果分析

总体来看，本次复现已经取得以下正向结果：

1. 端到端训练、验证、测试流程已跑通，且 hard port inference 能稳定输出合法端口集合。
2. 在默认 5-seed 五方法实验中，Proposed 平均性能高于 Random、CNN 和 Transformer。
3. 在 \(P_{max}\ge 15\) dBm 的发射功率扫描中，Proposed 均优于 Random。
4. 在已完成的距离扫描点中，Proposed 保持领先，且距离增大导致 sum-rate 下降的趋势正确。
5. 训练曲线在早期快速收敛，后期进入平台区，说明无监督负 sum-rate 训练目标是有效的。

同时，当前结果也暴露出几个限制：

1. Proposed 相对 Random 的平均增益较小，默认 5-seed 下约为 `0.46%`。
2. Proposed 与 LLM-sequential 几乎重合，尚未体现并行 LLM-FAS 结构的明显优势。
3. 激活端口数增加到 \(n=5,6\) 后，Proposed 未能稳定优于 Random。
4. 低功率 \(P_{max}=10\) dBm 下，Proposed 略低于 Random。
5. 距离扫描尚未完整生成所有参数点 summary。
6. 由于原论文 baseline 细节未完全公开，CNN、Transformer、LLM-sequential 的实现可能与论文存在工程差异。

## 10. 与论文预期的一致性和差异

与论文预期一致的部分包括：

- Sum-rate 随发射功率升高而增加。
- Sum-rate 随用户距离增大而下降。
- 增加激活端口数通常能提高系统空间自由度和整体速率。
- LLM 类方法在默认设置下能超过 Random 和普通 Transformer baseline。

与论文预期存在差异的部分包括：

- Proposed 的领先幅度明显偏小。
- Proposed 未与 LLM-sequential 拉开差距。
- 在部分参数区间，尤其是低功率和较大激活端口数下，Proposed 不稳定。

这些差异可能来自以下原因：

1. 原论文未公开全部训练细节和 baseline 结构，导致复现存在合理但不可避免的实现偏差。
2. 当前训练样本和随机种子规模仍有限，部分参数点只有 3 seeds。
3. Gumbel-Sinkhorn 离散端口选择对温度、学习率和训练轮数敏感。
4. Proposed 与 LLM-sequential 的模块差异在当前实现中可能没有充分放大。
5. 随机端口 baseline 在小规模 \(N=16,n=4\) 设置下并不弱，使得学习方法的可提升空间有限。

## 11. 复现结论

本次复现基本完成了论文 LLM-FAS 的主要算法链路和实验框架，并在 paper-aligned 默认配置下得到合理、物理趋势一致的结果。Proposed 方法在 5-seed 默认五方法实验中取得最高或并列最高平均 sum-rate，并相对 Random 获得约 `0.46%` 的平均提升。

不过，当前复现结果还不能完全等同于论文结论。最关键的差距在于：Proposed 相对 LLM-sequential 几乎没有优势，且在部分扫描条件下不稳定。因此，当前复现应被判断为：

> 已成功复现论文方法的主体流程和部分性能趋势，但尚未完全复现论文中 Proposed 方法显著领先所有 baseline 的强结论。

## 12. 后续改进建议

后续建议优先开展以下工作：

1. 补齐 Fig.9 距离扫描缺失的 `d=0.2` 及后续距离点 summary。
2. 对 \(n=5,6\) 的激活端口数实验单独调参，检查端口选择重复修复、Sinkhorn 温度退火和功率归一化是否限制性能。
3. 对 Proposed 与 LLM-sequential 的输出端口分布和功率分配进行对比，确认二者结果重合的原因。
4. 增加 seed 数量，例如从 5 seeds 扩展到 10 seeds，以降低随机性影响。
5. 尝试更大的 LoRA rank、更高学习率或更长训练轮数，观察 Proposed 是否能进一步超过 sequential baseline。
6. 增加端口选择可视化和 per-user rate 分析，使结果不仅停留在平均 sum-rate 层面。
7. 若继续扩展论文方向，可进一步引入 imperfect CSI、uncertainty token 和 robust loss，将当前复现作为后续鲁棒 LLM-FAS 研究的基础平台。

