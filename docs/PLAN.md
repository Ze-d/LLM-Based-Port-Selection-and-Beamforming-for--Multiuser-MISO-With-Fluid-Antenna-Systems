下面只给你**复现这篇论文的详细思路和步骤**，不展开论文创新点评价。论文核心是：在多用户下行 Tx-MISO-FAS 系统中，输入完整 CSI，网络并行输出“激活端口选择”和“波束成形功率因子”，再用模型化波束成形公式构造最终 beamforming vector，以最大化 sum rate。论文明确使用 GPT-2 + LoRA、Gumbel-Sinkhorn、无监督 sum-rate loss 训练。

---

# 1. 复现目标先定清楚

你的复现目标不是先追求完全超过论文曲线，而是按下面三层推进：

**第一层：系统模型复现。**
先不做 LLM，只实现 MISO-FAS 信道生成、端口选择、beamforming 公式、sum rate 计算。确认随机端口选择和固定 beamforming 能跑通。

**第二层：主模型复现。**
实现论文 Fig.2 的整体框架：
CSI → FC + MHA 预处理 → GPT-2/LoRA backbone → 并行输出 port logits 和 power logits → Gumbel-Sinkhorn → beamforming derivation → sum rate loss。

**第三层：实验图复现。**
按论文 Fig.5 到 Fig.11 复现实验，包括 batch size 收敛、阵列尺寸、发射功率、激活端口数、距离、总端口数、推理时间。

---

# 2. 项目结构建议

建议项目目录这样组织：

```text
llm_fas_reproduce/
├── configs/
│   ├── default.yaml
│   ├── sweep_wx.yaml
│   ├── sweep_power.yaml
│   ├── sweep_ports.yaml
├── data/
│   ├── train_default.pt
│   ├── val_default.pt
│   ├── test_default.pt
├── models/
│   ├── preprocessing.py
│   ├── gpt2_lora_backbone.py
│   ├── postprocessing.py
│   ├── llm_fas_model.py
│   ├── baselines.py
├── utils/
│   ├── channel.py
│   ├── fas_geometry.py
│   ├── beamforming.py
│   ├── metrics.py
│   ├── sinkhorn.py
│   ├── power.py
├── train.py
├── evaluate.py
├── run_sweeps.py
├── plot_results.py
└── requirements.txt / pyproject.toml
```

优先把**信道生成、sum rate、beamforming、Gumbel-Sinkhorn**写成独立工具函数。后面调试会容易很多。

---

# 3. 环境配置

论文实验环境为 Python 3.12、PyTorch，硬件是 i9-14900K + RTX 4090。默认参数里 GPT-2 隐层维度是 768，只取前 6 层，LoRA rank=4，batch size=100，最大 200 epoch。

建议环境：

```bash
uv init llm_fas_reproduce
cd llm_fas_reproduce
uv add torch torchvision torchaudio
uv add transformers peft scipy numpy matplotlib tqdm pyyaml einops
```

如果你的 GPU 显存不足，先把 batch size 降到 16 或 32，或者先用 GPT-2 前 2 层做 sanity check。

---

# 4. 先复现系统模型

论文系统是多用户下行 Tx-MISO-FAS：基站有二维流体天线端口阵列，总端口数为
`N = Nx × Ny`，每次只激活 `n` 个端口服务 `K` 个单天线用户。默认参数为：

| 参数        | 默认值         |
| --------- | ----------- |
| 用户数 K     | 3           |
| 端口阵列      | 4 × 4       |
| 总端口数 N    | 16          |
| 激活端口数 n   | 4           |
| 天线面积 W    | 2λ × 2λ     |
| 发射功率 Pmax | 20 dBm      |
| 噪声谱密度     | -174 dBm/Hz |
| 带宽        | 10 MHz      |
| 载波频率      | 2 GHz       |
| 用户距离 d    | 0.2 km      |
| 训练样本      | 10000       |
| 测试样本      | 1000        |
| 验证集       | 训练集 20%     |

这些参数来自论文数值实验部分。

---

## 4.1 构造 FAS 端口坐标

二维端口坐标为：

```text
nx = 1, ..., Nx
ny = 1, ..., Ny
```

论文采用从左上到右下的 2D → 1D 映射：

```text
l(nx, ny) = (ny - 1) * Nx + nx
```

实现时建议用 0-based index：

```python
idx = ny * Nx + nx
```

---

## 4.2 构造空间相关矩阵 J

论文使用 Jake’s model 描述任意两个端口之间的空间相关性。对于端口 `(nx, ny)` 和 `(\tilde nx, \tilde ny)`：

```text
J = j0(2π * sqrt( ((|nx-nx'|)/(Nx-1)*Wx)^2 + ((|ny-ny'|)/(Ny-1)*Wy)^2 ))
```

其中 `j0(x)` 是 0 阶球贝塞尔函数，可以用：

```python
j0(x) = sin(x) / x
```

注意 `x=0` 时令 `j0(0)=1`。

实现步骤：

```text
for i in all ports:
    for j in all ports:
        dx = abs(nx_i - nx_j) / (Nx - 1) * Wx
        dy = abs(ny_i - ny_j) / (Ny - 1) * Wy
        r = sqrt(dx^2 + dy^2)
        J[i, j] = spherical_j0(2πr)
```

---

## 4.3 对 J 做特征分解

论文中：

```text
J = F Λ F^H
```

然后用 `F` 和 `Λ` 生成相关信道。实现：

```python
eigvals, F = torch.linalg.eigh(J)
eigvals = eigvals descending
F = F[:, sorted_idx]
Lambda_sqrt = diag(sqrt(clamp(eigvals, min=0)))
```

注意数值误差可能导致很小的负特征值，需要 `clamp(min=0)`。

---

## 4.4 生成多用户 CSI

论文信道模型为：

```text
h_k^H = sqrt(beta_k) * g_k^H * sqrt(Λ)^H * F^H
```

其中：

```text
g_k ~ CN(0, I)
```

路径损耗：

```text
PL(dB) = 128.1 + 37.6 log10(d)
```

`d` 的单位是 km。将路径损耗转为线性尺度：

```text
beta = 10^(-PL/10)
```

最终得到：

```text
H ∈ C^{K × N}
```

训练数据就是很多个独立采样的 `H`。

---

# 5. 实现 sum rate 计算

对于激活端口矩阵 `A_n` 和 beamforming 矩阵 `C`：

```text
H_eff = H A_n
```

其中：

```text
H: K × N
A_n: N × n
H_eff: K × n
C: n × K
```

第 k 个用户接收信号的 SINR：

```text
SINR_k = |h_k^H c_k|^2 / (sum_{j≠k} |h_k^H c_j|^2 + σ²)
```

sum rate：

```text
R = Σ_k log2(1 + SINR_k)
```

这个函数一定要先单独测试，后面所有训练和评估都依赖它。

噪声功率计算：

```text
noise_dBm = -174 + 10 log10(B)
```

带宽 `B = 10 MHz`，所以：

```text
noise_dBm = -174 + 70 = -104 dBm
```

再转瓦特：

```text
σ² = 10^((noise_dBm - 30) / 10)
```

发射功率同理：

```text
Pmax_W = 10^((Pmax_dBm - 30) / 10)
```

20 dBm 对应 0.1 W。

---

# 6. 实现论文的 beamforming derivation

论文后处理模块使用模型化最优 beamforming 结构，由网络输出两个功率因子 `p` 和 `q`，再推导 beamforming vector。

公式可实现为：

```text
B = I_n + (1 / σ²) Σ_i q_i h_i h_i^H
```

然后：

```text
v_k = B^{-1} h_k
c_k = sqrt(p_k) * v_k / ||v_k||
```

这里需要注意一个复现细节：论文公式中求和下标写得容易引起歧义，但从 `p,q ∈ R^K` 和多用户下行 beamforming 结构看，实际应按用户求和：

```text
i = 1, ..., K
```

而不是按激活端口数 `n` 求和。否则 `q_i` 的维度会对不上。

实现维度：

```text
H_eff: B × K × n
h_k: n × 1
B_matrix: n × n
C: B × n × K
```

训练时这个过程必须保持可微，建议用：

```python
torch.linalg.solve(B_matrix, h_k)
```

不要手动求逆。

---

# 7. 复现主模型结构

论文 Fig.2 的整体结构是：

```text
CSI H
→ real/imag 拆分
→ FC1
→ Multi-head Attention
→ FC2
→ Positional Embedding
→ GPT-2 + LoRA
→ 并行输出：
   1. port logits
   2. power logits
→ Gumbel-Sinkhorn 得到软端口选择
→ softmax 得到 p, q
→ beamforming derivation
→ sum rate loss
```

论文明确采用 preprocessing module、backbone network、post-processing module 三部分。

---

## 7.1 CSI 输入预处理

输入：

```text
H ∈ C^{K × N}
```

拆为实部和虚部：

```text
Re(H), Im(H)
```

分别 flatten：

```text
h_real ∈ R^{K N}
h_imag ∈ R^{K N}
```

然后分别经过 `FC1`：

```text
h_real → K × d_mha
h_imag → K × d_mha
```

默认：

```text
d_mha = 768
```

---

## 7.2 Multi-head Attention

对实部和虚部分别做 MHA：

```text
Hr_fc1 → MHA → Hr_mha
Hi_fc1 → MHA → Hi_mha
```

论文设置：

```text
M = 8 heads
d_mha = 768
head_dim = 96
```

然后拼接：

```text
concat(Hr_mha, Hi_mha)
```

再 flatten 成：

```text
2K d_mha
```

---

## 7.3 FC2 映射到 GPT-2 输入序列

论文将 FC2 输出 reshape 为：

```text
H_fc2 ∈ R^{Nn × d_llm}
```

默认：

```text
d_llm = 768
sequence_length = N * n
```

例如默认 `N=16, n=4`，则序列长度为：

```text
64
```

所以 FC2 输出维度为：

```text
64 × 768
```

---

## 7.4 加入 positional embedding

论文使用标准正余弦位置编码：

```text
H_em = H_fc2 + H_pe
```

维度仍然是：

```text
B × Nn × 768
```

这里注意：这是给 GPT-2 的 `inputs_embeds`，不是 token id。你要调用 GPT-2 时应使用：

```python
gpt2(inputs_embeds=H_em)
```

而不是输入文本 token。

---

# 8. GPT-2 + LoRA 复现

论文 backbone 是轻量版 GPT-2：

```text
d_llm = 768
只使用 GPT-2 前 6 层
LoRA rank = 4
```

LoRA 只加在 GPT-2 masked attention 的 Q 和 V 矩阵上，原 GPT-2 参数冻结，只训练 LoRA 矩阵、LayerNorm、预处理模块和输出头。

复现有两种方式：

## 方式 A：快速复现

用 HuggingFace + PEFT：

```python
GPT2Model.from_pretrained("gpt2")
```

然后只保留前 6 层。

PEFT 对 GPT-2 的注意力一般是 `c_attn` 合并 QKV，所以如果直接设置：

```python
target_modules=["c_attn"]
```

会和论文“只对 Q/V 加 LoRA”略有差异，但工程上最容易跑通。

## 方式 B：严格复现

自己改 GPT-2 attention，把 Q、K、V 显式拆出来，然后只对 Q 和 V 加 LoRA：

```text
Q = X W_Q + X B_Q A_Q
K = X W_K
V = X W_V + X B_V A_V
```

如果你是为了论文级复现，建议最终采用方式 B。
如果只是先验证思路，方式 A 就够了。

---

# 9. 并行输出头设计

GPT-2 输出：

```text
H_llm ∈ R^{B × Nn × 768}
```

然后做 aggregation。论文图中有 aggregation，文字中说用两个线性层分别映射到 port selection vector 和 power allocation vector。工程上建议这样：

```python
z = H_llm.reshape(B, N*n*768)
```

两个 head：

```text
port_head(z)  → x_port ∈ R^{Nn}
power_head(z) → x_power ∈ R^{2K}
```

然后：

```text
x_port reshape → X_port ∈ R^{n × N}
x_power reshape → X_power ∈ R^{2 × K}
```

---

# 10. Gumbel-Sinkhorn 端口选择

论文用 Gumbel-Sinkhorn 将离散端口选择变成可微过程，训练阶段用 soft 选择，测试阶段 argmax harden。

训练阶段：

```text
X_gum = X_port + Gumbel noise
X_s = softmax(X_gum / τ)
X_sink = Sinkhorn(X_s)
```

温度退火：

```text
τ = max(0.1, 0.95^epoch)
```

Sinkhorn iteration：

```text
10 次
```

注意这里有一个工程坑：`X_port ∈ R^{n × N}` 是矩形矩阵，严格意义上不能像方阵一样每行每列都归一到 1，因为行列总和不一致。建议实现成“近似无重复选择”的 relaxed assignment：

```text
行归一化：每个激活槽选择一个端口
列归一化：抑制多个槽选择同一端口
重复执行 10 次
最后再行归一化
```

测试阶段：

```text
每一行 argmax 得到一个端口
如果出现重复端口，用 greedy 或 Hungarian 修正为互异端口
```

最稳的测试做法：

```text
从 X_sink 中按得分选 n 个不重复端口
```

即直接取全局 top-n 不重复端口，而不是每行独立 argmax。

---

# 11. power allocation 输出

`x_power ∈ R^{2K}` reshape 为：

```text
X_power ∈ R^{2 × K}
```

第一行生成 `p`，第二行生成 `q`：

```python
p = softmax(X_power[0]) * Pmax
q = softmax(X_power[1]) * Pmax
```

这样自然满足：

```text
Σ p_k = Pmax
Σ q_k = Pmax
```

随后用 `p,q,H_eff` 代入 beamforming derivation 得到 `C`。

---

# 12. 训练损失

论文采用无监督训练，不需要最优端口标签，也不需要最优 beamforming 标签，直接用负 sum rate 做 loss。

```text
Loss = - mean_batch Σ_k log2(1 + SINR_k)
```

训练流程：

```text
for epoch in range(max_epoch):
    tau = max(0.1, 0.95 ** epoch)

    for H in train_loader:
        H_em = preprocessing(H)
        H_llm = GPT2_LoRA(H_em)

        X_sink = gumbel_sinkhorn(port_logits, tau)
        p, q = power_head(...)

        H_eff = H @ X_sink.T
        C = beamforming_derivation(H_eff, p, q)

        rate = sum_rate(H_eff, C)
        loss = -rate.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

验证阶段：

```text
不用 Gumbel noise
用 hard port selection
计算平均 sum rate
```

早停：

```text
patience = 10
max_epoch = 200
batch_size = 100
learning_rate = 1e-6
optimizer = Adam
```

---

# 13. 数据集生成步骤

每一个实验点建议重新生成数据并训练模型。论文没有明确说每个 sweep 点是否复用模型，但为了复现实验曲线，最稳妥做法是：

```text
每个横坐标点：
    生成 train/val/test
    训练 proposed model
    训练各 baseline
    测试平均 sum rate
```

默认数据量：

```text
train = 10000
val = 2000，从 train 中划出 20%
test = 1000
```

如果你先做快速验证，可以用：

```text
train = 1000
val = 200
test = 200
epoch = 20
```

确认代码无误后再放大。

---

# 14. baseline 复现顺序

论文比较了 Random、CNN、Transformer、LLM-sequential 和 Proposed Method。

建议按难度顺序复现：

## 14.1 Random baseline

```text
随机选择 n 个不重复端口
MLP 输出 p,q
用公式构造 C
用 sum rate loss 训练 MLP
```

这个 baseline 用来检查系统模型和训练逻辑。

## 14.2 Transformer baseline

用普通 Transformer encoder 替代 GPT-2 backbone：

```text
preprocessing 保持一致
backbone 换成 8-head Transformer
post-processing 保持一致
```

这条最适合和 proposed 对比，因为除了 backbone，其他模块尽量一致。

## 14.3 CNN baseline

```text
CSI real/imag → 2D CNN → FC
先输出 port selection
再输出 power allocation
最后 beamforming derivation
```

这是 sequential baseline。

## 14.4 LLM-sequential baseline

```text
GPT-2/LoRA 先输出 port selection
然后 CNN/FC 输出 power allocation
再构造 beamforming
```

它和 proposed 的区别是：
proposed 是端口和功率因子并行输出；LLM-sequential 是先端口后功率。

---

# 15. 需要复现的实验图

## Fig.5：batch size 收敛曲线

固定默认系统参数，只改变：

```text
batch size = 50, 100, 200
```

记录：

```text
train loss
val loss
```

目标现象：

```text
bs=100 收敛最好
前 5~10 epoch 快速下降
后面逐渐稳定
```

---

## Fig.6：sum rate vs 阵列尺寸 Wx

横坐标：

```text
Wx = Wy = [1, 1.5, 2, 2.5, 3] λ
```

其他参数默认：

```text
K=3
N=4×4
n=4
Pmax=20 dBm
d=0.2 km
```

目标现象：

```text
Wx 增大 → 空间相关性降低 → sum rate 上升
Proposed > LLM-sequential > Transformer > CNN > Random
```

---

## Fig.7：sum rate vs 发射功率 Pmax

横坐标：

```text
Pmax = [10, 15, 20, 25, 30] dBm
```

目标现象：

```text
功率越大，sum rate 越高
Proposed 全程最高
```

---

## Fig.8：sum rate vs 激活端口数 n

横坐标：

```text
n = [3, 4, 5, 6, 7]
```

固定：

```text
N = 4×4
K = 3
```

注意必须满足：

```text
n ≥ K
```

目标现象：

```text
n 越大，空间 DoF 越多，sum rate 越高
```

---

## Table II：不同 n 下的端口集合

固定一个测试信道样本，对不同方法输出端口集合：

```text
n = 3, 4, 5, 6, 7
```

输出形式：

```text
{4, 9, 16}
{1, 3, 13, 16}
...
```

这个表不用完全和论文一致，因为随机信道样本、模型初始化不同，但要展示 proposed 的端口选择更稳定、性能更好。

---

## Fig.9：sum rate vs 用户距离 d

横坐标：

```text
d = [100, 150, 200, 250, 300] m
```

路径损耗公式中要换成 km：

```text
d_km = d_m / 1000
```

目标现象：

```text
距离越远，路径损耗越大，sum rate 越低
```

---

## Fig.10：sum rate vs 总端口数 N

横坐标：

```text
(Nx, Ny) = (3,3), (4,4), (5,5), (6,6), (7,7)
```

即：

```text
N = 9, 16, 25, 36, 49
```

注意模型 sequence length 是 `N*n`，所以每个 N 都要重新初始化对应模型，不能直接复用同一个输出头。

---

## Fig.11：inference time vs 总端口数 N

横坐标同 Fig.10：

```text
(3,3), (4,4), (5,5), (6,6), (7,7)
```

曲线：

```text
n = 3
n = 4
n = 5
```

测试方法：

```text
model.eval()
torch.cuda.synchronize()
计时 100 或 1000 次 forward
取平均毫秒数
```

目标现象：

```text
N 和 n 增大，推理时间增加
但整体是毫秒级
```

---

# 16. 复现优先级建议

你可以按这个顺序做：

```text
Step 1：实现 power/noise/pathloss 单位转换
Step 2：实现 FAS 端口坐标和 J 矩阵
Step 3：实现信道 H 生成
Step 4：实现随机端口选择
Step 5：实现 beamforming derivation
Step 6：实现 sum rate 计算
Step 7：跑 Random baseline，确认 rate 合理
Step 8：实现 Gumbel-Sinkhorn soft port selection
Step 9：实现 power head 输出 p,q
Step 10：实现 preprocessing module
Step 11：接入 GPT-2 inputs_embeds
Step 12：冻结 GPT-2，接 LoRA
Step 13：训练 proposed model
Step 14：实现 hard inference
Step 15：复现 Fig.5 收敛曲线
Step 16：实现 Transformer baseline
Step 17：实现 CNN 和 LLM-sequential baseline
Step 18：跑 Fig.6-Fig.11 的 sweep
Step 19：整理所有图表和日志
```

---

# 17. 最容易出错的地方

第一，**路径损耗和功率单位**。
`Pmax` 是 dBm，噪声谱密度也是 dBm/Hz，必须统一转成线性瓦特。如果单位错了，sum rate 会完全不对。

第二，**Gumbel-Sinkhorn 的矩形矩阵归一化**。
`n × N` 不是方阵，不能简单照搬方阵 permutation matrix 的 Sinkhorn。复现时目标是得到“每个激活槽选一个端口、不同槽尽量不重复”的 relaxed assignment。

第三，**训练阶段必须用 soft port selection**。
如果训练时直接 argmax，梯度无法从 sum rate 回传到 port selection head。

第四，**beamforming 公式中的求和维度**。
`p,q` 是长度 K 的用户功率因子，所以 beamforming derivation 里应按用户求和构造矩阵。

第五，**GPT-2 输入不是文本 token**。
这里是把 CSI 映射成 embedding 后用 `inputs_embeds` 输入 GPT-2。

第六，**不同 N 或 n 会改变模型输出维度**。
因为 sequence length 是 `N*n`，port head 输出也是 `N*n`，所以 sweep `N` 或 `n` 时通常需要重新建模和训练。

---

# 18. 最终复现结果应该保存什么

每次实验保存：

```text
config.yaml
train_loss.npy
val_loss.npy
test_sum_rate.npy
selected_ports.json
model.pt
results.csv
figure.png
```

`results.csv` 建议字段：

```text
experiment, method, K, Nx, Ny, N, n, Wx, Wy, Pmax_dBm, d_m, seed, test_sum_rate, inference_ms
```

这样后面写论文或组会汇报时可以直接画图。

---

# 19. 你的最小可行复现版本

如果你想最快跑通，我建议先做这个版本：

```text
K = 3
N = 4×4
n = 4
W = 2λ×2λ
Pmax = 20 dBm
train = 1000
test = 200
GPT-2 layers = 2
batch size = 32
epoch = 20
只比较 Random / Transformer / Proposed
```

跑通后再切换到论文完整设置：

```text
train = 10000
test = 1000
GPT-2 layers = 6
batch size = 100
epoch = 200
全部 baseline
全部 sweep
```

这样复现风险最低。
