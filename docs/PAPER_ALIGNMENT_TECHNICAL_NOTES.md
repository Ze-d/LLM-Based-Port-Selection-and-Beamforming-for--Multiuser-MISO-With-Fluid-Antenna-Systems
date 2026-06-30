# Paper Alignment Technical Notes

这份文档记录当前代码与论文方法的技术对应关系、仍然存在的近似实现，以及后续复现实验异常时的定位路径。它用于排查问题，不替代 `docs/EXPERIMENT_GUIDE.md` 的训练命令。

## 1. 快速结论

当前代码已经对齐论文主框架：

```text
CSI -> real/imag preprocessing -> FC1 -> MHA -> FC2 -> positional encoding
    -> GPT-2 + Q/V-only LoRA
    -> port logits + power logits
    -> Gumbel-Sinkhorn port selection
    -> sigmoid + row-wise softmax power factors
    -> Eq.18-style beamforming
    -> sum-rate loss
```

但它不是作者源码的完全等价复刻。最需要关注的技术近似是：

1. 数据集划分是独立生成 `train/val/test`，不是先生成 10000 training samples 再切 20% validation。
2. LoRA 是在 HuggingFace GPT-2 `c_attn` 的 Q/V slice 上实现，不是逐 head 显式矩阵实现。
3. `n x N` 矩形 Gumbel-Sinkhorn 是 partial Sinkhorn 近似，论文没有公开矩形细节。
4. GPT-2 输出聚合使用 flatten 后接 `port_head/power_head`，论文没有明确 aggregation 细节。
5. CNN、Transformer、LLM-sequential、Random baseline 的隐藏网络结构没有完全公开，当前是合理补全。

这些点可以解释当前实验中 `proposed` 与 `llm_sequential` 几乎重合，而论文声称 `proposed` 应明显优于 `llm_sequential`。

## 2. 代码模块地图

| 模块 | 职责 | 重点检查点 |
|---|---|---|
| `llm_fas/physics.py` | 生成空间相关信道、路径损耗 | Jakes 相关矩阵、`128.1 + 37.6log10(d)` |
| `llm_fas/data.py` | 构建 train/val/test | seed 切分方式、样本数 |
| `llm_fas/models.py` | proposed、CNN、Transformer、LLM-sequential | preprocessing、LoRA、heads、sequential/parallel 差异 |
| `llm_fas/sinkhorn.py` | Gumbel-Sinkhorn 和 hard port selection | partial Sinkhorn、重复端口修复 |
| `llm_fas/beamforming.py` | Eq.18-style beamforming 和 sum-rate | `p/q` 维度、噪声功率、SINR |
| `llm_fas/baselines.py` | Random baseline | 随机端口、MLP power allocation |
| `llm_fas/train.py` | 训练、early stopping、评估 | soft training / hard testing、checkpoint、loss |
| `configs/paper_default.yaml` | 论文默认设置 | `8000/2000/1000`、`batch=100`、`layers=6` |

## 3. 与论文已对齐的部分

### 3.1 系统参数

`configs/paper_default.yaml` 当前默认：

```text
K = 3
Nx = Ny = 4
N = 16
n_active = 4
W = 2lambda x 2lambda
Pmax = 20 dBm
d = 0.2 km
noise PSD = -174 dBm/Hz
bandwidth = 10 MHz
carrier = 2 GHz
```

训练参数：

```text
epochs = 200
batch_size = 100
lr = 1e-6
tau = max(0.1, 0.95^epoch)
sinkhorn_iters = 10
early stopping patience = 10
```

### 3.2 Proposed 主干

当前 `ProposedLLMFASModel` 使用：

```text
GPT-2 first 6 layers
d_llm = 768
d_mha = 768
MHA heads = 8
LoRA rank = 4
```

论文对应描述：

```text
lightweight GPT-2
initial NL=6 layers
dllm=768
dmha=768
M=8
r=4
```

### 3.3 Power 后处理

论文：

```text
xpower -> sigmoid -> reshape to 2 x K -> row-wise softmax -> scale by Pmax
```

代码：

```python
power_factors = torch.sigmoid(power_logits)
p = softmax(power_factors[:, 0, :]) * Pmax
q = softmax(power_factors[:, 1, :]) * Pmax
```

该逻辑在 proposed、CNN、Transformer、LLM-sequential 和 trained random baseline 中都已统一。

### 3.4 训练/评估模式

训练：

```text
soft Gumbel-Sinkhorn
loss = -out["rate"].mean()
```

验证：

```text
soft Gumbel-Sinkhorn
monitor validation sum-rate
```

测试：

```text
hard port selection
average test sum-rate
```

这与论文“training 保留 continuous differentiable X_sink，inference 使用 hard output”的描述一致。

## 4. 仍然不完全等价的技术细节

### 4.1 数据划分

论文文字：

```text
generated 10,000 channel data samples for the training set
20% of the training data was reserved as validation
1,000 samples for the test set
```

当前代码：

```python
train = generate_channels(train_samples, seed=seed)
val = generate_channels(val_samples, seed=seed + 1)
test = generate_channels(test_samples, seed=seed + 2)
```

影响：

```text
统计分布一致，但严格操作不同。
如果 validation/test 排名和论文不同，可以尝试改成先生成 10000 再切分 8000/2000。
```

排查位置：

```text
llm_fas/data.py::build_datasets
configs/paper_default.yaml
```

### 4.2 LoRA 实现粒度

论文公式写的是：

```text
LoRA(Q_t) = L^Q_t + B^Q_t A^Q_t
LoRA(V_t) = L^V_t + B^V_t A^V_t
```

当前代码：

```text
wrap GPT-2 c_attn
split c_attn output into Q/K/V slices
add low-rank delta only to Q and V slices
K slice unchanged
```

差异：

```text
论文按 attention head 描述；
代码按 HuggingFace GPT-2 merged QKV projection 的 Q/V slice 实现。
```

排查位置：

```text
llm_fas/models.py::QVLoraConv1D
llm_fas/models.py::ProposedLLMFASModel
```

异常信号：

```text
proposed 训练不收敛
proposed 与 transformer 差距异常小
LoRA 参数没有梯度
```

建议检查：

```bash
python -m pytest tests/test_models.py -q
```

### 4.3 GPT-2 输出聚合

论文没有明确说明 `H_llm` 如何被聚合到最终 `xport` 和 `xpower`。当前代码使用：

```python
z = backbone_out.reshape(B, -1)
port_head(z)  -> n_active * N
power_head(z) -> 2 * K
```

潜在差异：

```text
作者可能使用 flatten、mean pooling、last token、attention pooling 或其它 aggregation。
```

影响：

```text
这是最可能影响 proposed 与 LLM-sequential 差距的细节之一。
如果 proposed 无法超过 LLM-sequential，应优先检查这里。
```

排查位置：

```text
llm_fas/models.py::JointFASModelBase.forward
llm_fas/models.py::port_head
llm_fas/models.py::power_head
```

可尝试方向：

```text
1. 保存 backbone_out 的 per-token 激活统计。
2. 比较 flatten head、mean pooling head、CLS/last-token head。
3. 分别观察 port logits 与 power logits 的梯度范数。
```

### 4.4 矩形 Gumbel-Sinkhorn

论文写 `X_port in R^{n x N}`，并说经过 Sinkhorn 得到 doubly stochastic matrix。但当 `n < N` 时，严格双随机矩阵不成立。

当前代码实现 partial Sinkhorn：

```text
row normalization
column sum capped around 1
row normalization again
```

hard inference：

```text
row-wise argmax
if duplicate port appears, pick next-best unused port in that row
```

潜在差异：

```text
作者可能使用 padding 成方阵、matching algorithm、straight-through estimator、或其它 repair 方式。
```

影响：

```text
端口选择质量高度依赖这里。
如果 selected ports 重复、sum-rate 波动大、hard/soft gap 大，应优先检查该模块。
```

排查位置：

```text
llm_fas/sinkhorn.py::gumbel_sinkhorn
llm_fas/sinkhorn.py::hard_topk_ports
```

建议检查：

```text
1. selection_soft row sums 是否接近 1。
2. column sums 是否出现过大值。
3. hard ports 是否重复。
4. hard rate 与 soft validation rate 是否差距过大。
```

### 4.5 LLM-sequential 与 proposed 的差异可能不足

论文核心区别：

```text
LLM-sequential: LLM only for port selection, CNN/FC for power factors
proposed: LLM jointly outputs port selection and power factors
```

当前代码确实这样区分：

```text
LLMSequentialBaselineModel freezes inherited power_head
adds EffectiveChannelPowerCNN(H_eff) -> p/q
```

但二者共享：

```text
same preprocessing
same GPT-2 + LoRA
same port_head
same Gumbel-Sinkhorn
same Eq.18 beamforming
```

影响：

```text
如果 sum-rate 主要由 selected ports 决定，而 p/q 差异较弱，二者会几乎重合。
当前 5-seed 结果正是这个现象。
```

排查位置：

```text
llm_fas/models.py::LLMSequentialBaselineModel
llm_fas/models.py::ProposedLLMFASModel
```

建议增加诊断输出：

```text
per-sample selected ports
per-sample p/q
per-sample sum-rate
port overlap between proposed and llm_sequential
p/q L2 distance between proposed and llm_sequential
```

### 4.6 Transformer baseline 细节未知

论文只说：

```text
replace backbone network with transformer
8-head MHA
same parameter settings as preprocessing MHA
```

当前代码：

```python
nn.TransformerEncoderLayer(
    d_model=d_mha,
    nhead=mha_heads,
    dim_feedforward=4*d_mha,
    dropout=0.0,
    activation="gelu",
)
```

潜在差异：

```text
encoder vs decoder
是否使用 causal mask
FFN hidden dimension
normalization placement
dropout
层数
position encoding
```

影响：

```text
如果 transformer 低于 random/CNN，不一定是论文趋势错误，也可能是 baseline 实现不等价。
```

排查位置：

```text
llm_fas/models.py::TransformerBaselineModel
```

### 4.7 CNN baseline 细节未知

论文只说：

```text
2D convolutional layer + FC layer
sequentially optimize port selection and beamforming
```

当前代码：

```text
Conv2d(2, 16, kernel=3)
Conv2d(16, 32, kernel=3)
FC
```

潜在差异：

```text
通道数、卷积层数、kernel、activation、是否 batchnorm/dropout 都没有论文细节。
```

排查位置：

```text
llm_fas/models.py::CNNBaselineModel
llm_fas/models.py::EffectiveChannelPowerCNN
```

### 4.8 Random baseline 细节未知

论文只说 random 端口 + MLP power allocation。当前 MLP：

```text
input = real/imag flattened H_eff
hidden = 256 -> 128
output = 2K
sigmoid + softmax
```

潜在差异：

```text
作者 MLP 的 hidden size、层数、输入特征可能不同。
```

影响：

```text
Random baseline 太强或太弱都会改变相对排序。
```

排查位置：

```text
llm_fas/baselines.py::RandomBaselineModel
```

## 5. 当前实验异常与定位

### 5.1 proposed 与 LLM-sequential 几乎完全相同

当前 5-seed 结果：

```text
llm_sequential mean = 19.338048
proposed mean       = 19.338045
```

优先怀疑：

```text
1. 二者选了几乎相同的 ports。
2. p/q 对最终 sum-rate 的影响很小。
3. proposed 的 power_head 没有学出不同于 CNN power head 的策略。
4. GPT-2 flatten aggregation 让 power_head 与 port_head 解耦不充分。
```

建议新增诊断：

```text
保存每个 test sample:
- method
- seed
- sample_id
- selected ports
- p
- q
- sum_rate
```

比较：

```text
port overlap(proposed, llm_sequential)
mean |p_proposed - p_llmseq|
mean |q_proposed - q_llmseq|
per-sample rate difference
```

### 5.2 proposed 不能稳定超过 random

检查顺序：

```text
1. 是否使用 paper_aligned 新代码和新输出目录。
2. random 是否是 trained random baseline，而不是 uniform fallback。
3. train/val/test 是否为 8000/2000/1000。
4. hard inference 是否从 Sinkhorn 输出 harden。
5. power head 是否 sigmoid 后再 softmax。
6. 是否至少 3 seeds。
```

相关文件：

```text
llm_fas/train.py::evaluate_checkpoints
llm_fas/baselines.py::evaluate_random_baseline_mlp
llm_fas/models.py::_power_from_logits
```

### 5.3 transformer 低于 random

可能原因：

```text
1. Transformer baseline 实现与论文不同。
2. Transformer training 早停过早。
3. Transformer encoder 不适合当前 token ordering。
4. `dim_feedforward/dropout/norm_first` 等超参数影响较大。
```

优先检查：

```text
transformer_train_history.csv
best_val_rate
hard test rate
```

相关文件：

```text
llm_fas/models.py::TransformerBaselineModel
```

### 5.4 validation 好但 test 差

可能原因：

```text
1. train/val/test 独立 seed 导致 split variance。
2. validation 用 soft selection，test 用 hard selection，soft-hard gap 大。
3. early stopping 监控 soft validation，不一定对应 hard test 最优。
```

可尝试诊断：

```text
1. 额外记录 hard validation rate。
2. 比较每个 epoch 的 soft val 与 hard val。
3. 保存 best-soft 和 best-hard 两套 checkpoint。
```

相关文件：

```text
llm_fas/train.py::_evaluate_model_rate
llm_fas/train.py::train_method
```

### 5.5 Fig.7-10 曲线不符合论文趋势

检查：

```text
1. 每个横轴点是否重新训练。
2. 每个点是否使用相同 seeds。
3. 每个点是否包含五种方法。
4. Pmax / n / d / N 是否正确写入 config_snapshot.yaml。
5. 是否混入旧 outputs。
```

相关脚本：

```text
scripts/run_fig7_pmax.py
scripts/run_fig8_active_ports.py
scripts/run_fig9_distance.py
scripts/run_fig10_ports.py
scripts/sweep_common.py
```

## 6. 建议后续增强

### 6.1 增加详细评估导出

建议新增一个可选参数，例如：

```text
--save-diagnostics
```

导出：

```text
diagnostics_<method>.csv
```

字段：

```text
seed
sample_id
method
ports
p_1,...,p_K
q_1,...,q_K
sum_rate
```

用途：

```text
定位 proposed 与 llm_sequential 是否真的做出了不同决策。
```

### 6.2 增加 hard validation

当前 early stopping 用 soft validation。建议同时记录：

```text
val_rate_soft
val_rate_hard
```

用途：

```text
判断 soft-hard mismatch 是否导致 checkpoint 选择偏差。
```

### 6.3 增加 port overlap 分析

对每个 seed 和每个 sample 计算：

```text
overlap(proposed, random)
overlap(proposed, transformer)
overlap(proposed, llm_sequential)
```

用途：

```text
如果 proposed 与 llm_sequential overlap 接近 100%，说明差异主要只剩 p/q。
```

### 6.4 对比不同 output aggregation

可实验：

```text
flatten all tokens
mean pool tokens
last token
attention pooling
separate port/power pooling
```

用途：

```text
验证 proposed 的 parallel power head 是否被当前 aggregation 限制。
```

## 7. 排查时优先查看的文件

单 seed：

```text
outputs/<experiment>/seed_<seed>/config_snapshot.yaml
outputs/<experiment>/seed_<seed>/device_info.json
outputs/<experiment>/seed_<seed>/results.csv
outputs/<experiment>/seed_<seed>/<method>_train_history.csv
```

汇总：

```text
outputs/<experiment>/summary.csv
outputs/<experiment>/all_results.csv
```

代码：

```text
llm_fas/models.py
llm_fas/sinkhorn.py
llm_fas/baselines.py
llm_fas/train.py
llm_fas/data.py
llm_fas/beamforming.py
```

## 8. 最小复查命令

```bash
python -m compileall llm_fas tests scripts
python -m pytest tests/test_sinkhorn.py tests/test_data_baselines.py tests/test_models.py -q
```

查看当前主实验：

```bash
cat outputs/paper_aligned_5seed_5methods/summary.csv
cat outputs/paper_aligned_5seed_5methods/all_results.csv
```

生成主实验图：

```bash
# 当前仓库中已生成 SVG 图：
ls outputs/paper_aligned_5seed_5methods/figures
```

## 9. 当前复现状态记录

截至 `result4` 提交，`paper_aligned_5seed_5methods` 的关键现象：

```text
LLM-sequential ≈ proposed > CNN ≈ random > transformer
```

当前已支持的结论：

```text
LLM-based methods outperform random / CNN / transformer in this default setting.
```

当前尚未支持的论文核心结论：

```text
proposed parallel framework clearly outperforms LLM-sequential.
```

后续如果要定位这个差异，优先做第 6.1 和第 6.3 的诊断导出。
