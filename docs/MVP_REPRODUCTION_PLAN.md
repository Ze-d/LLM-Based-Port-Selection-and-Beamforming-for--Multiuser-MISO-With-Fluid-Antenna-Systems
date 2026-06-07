# LLM-FAS MVP Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimum viable reproduction of "LLM-Based Port Selection and Beamforming for Multiuser MISO With Fluid Antenna Systems" that runs end-to-end on a small synthetic dataset and reports sum rate for Random and Proposed methods.

**Architecture:** The MVP is a compact Python package with separate modules for configuration, FAS channel generation, differentiable port selection, beamforming, metrics, model definition, training, and evaluation. It follows the paper's main pipeline: complex CSI -> preprocessing -> GPT-2 with LoRA -> parallel port and power heads -> Gumbel-Sinkhorn -> model-based beamforming -> unsupervised negative sum-rate loss.

**Tech Stack:** Python 3.12, PyTorch, Transformers, PEFT, NumPy, SciPy, PyYAML, Matplotlib, tqdm, pytest.

---

## MVP Scope

The first reproducible checkpoint must do the following:

- Generate Tx-MISO-FAS channels for the paper default system: `K=3`, `Nx=Ny=4`, `N=16`, `n=4`, `W=2lambda x 2lambda`, `Pmax=20 dBm`, `distance=0.2 km`, `bandwidth=10 MHz`.
- Implement unit-tested power conversion, path loss, FAS spatial correlation, channel generation, beamforming derivation, sum-rate calculation, Gumbel-Sinkhorn, and hard port selection.
- Train a small Proposed model for a short smoke run: `train=256`, `val=64`, `test=64`, `epochs=2`, `batch_size=16`, GPT-2 first `2` layers, LoRA rank `4`.
- Evaluate Random baseline and Proposed model on the same test dataset.
- Save metrics and artifacts under `outputs/mvp/`.

The MVP intentionally does not reproduce Fig.5-Fig.11 yet. Those sweeps start after this plan's acceptance criteria pass.

## File Structure

- Create: `requirements.txt` - minimal package dependencies.
- Create: `configs/mvp.yaml` - small reproducibility config for fast local runs.
- Create: `llm_fas/__init__.py` - package marker.
- Create: `llm_fas/config.py` - dataclasses and YAML loader.
- Create: `llm_fas/physics.py` - units, path loss, geometry, correlation matrix, channel generation.
- Create: `llm_fas/beamforming.py` - beamforming derivation and sum-rate metric.
- Create: `llm_fas/sinkhorn.py` - Gumbel-Sinkhorn relaxation and hard unique port selection.
- Create: `llm_fas/models.py` - preprocessing module, GPT-2 LoRA backbone wrapper, Proposed model.
- Create: `llm_fas/data.py` - deterministic dataset generation and DataLoader helpers.
- Create: `llm_fas/train.py` - train/evaluate loops and checkpoint writing.
- Create: `llm_fas/baselines.py` - Random baseline with equal `p,q` and model-based beamforming.
- Create: `scripts/train_mvp.py` - command line entry point for training Proposed.
- Create: `scripts/evaluate_mvp.py` - command line entry point for Random and Proposed evaluation.
- Create: `tests/test_physics.py` - unit tests for units, geometry, correlation, channels.
- Create: `tests/test_beamforming.py` - unit tests for beamforming and sum rate.
- Create: `tests/test_sinkhorn.py` - unit tests for relaxed and hard port selection.

## Acceptance Criteria

- `python -m pytest -q` passes.
- `python scripts/train_mvp.py --config configs/mvp.yaml` finishes without NaN or shape errors.
- `python scripts/evaluate_mvp.py --config configs/mvp.yaml --checkpoint outputs/mvp/proposed.pt` writes `outputs/mvp/results.csv`.
- `outputs/mvp/results.csv` contains one row for `random` and one row for `proposed`, with finite positive `test_sum_rate`.
- `outputs/mvp/train_history.csv` contains `epoch,train_loss,val_loss`.

---

### Task 1: Environment And Configuration

**Files:**
- Create: `requirements.txt`
- Create: `configs/mvp.yaml`
- Create: `llm_fas/__init__.py`
- Create: `llm_fas/config.py`

- [ ] **Step 1: Write dependency file**

Create `requirements.txt` with:

```text
torch
transformers
peft
numpy
scipy
pyyaml
matplotlib
tqdm
pytest
```

- [ ] **Step 2: Write MVP config**

Create `configs/mvp.yaml` with:

```yaml
seed: 20260606
device: auto

system:
  K: 3
  Nx: 4
  Ny: 4
  n_active: 4
  W_lambda_x: 2.0
  W_lambda_y: 2.0
  Pmax_dBm: 20.0
  noise_psd_dBm_per_Hz: -174.0
  bandwidth_Hz: 10000000.0
  carrier_Hz: 2000000000.0
  distance_km: 0.2

data:
  train_samples: 256
  val_samples: 64
  test_samples: 64

model:
  d_mha: 128
  mha_heads: 4
  backbone_name: gpt2
  gpt2_layers: 2
  lora_rank: 4
  sinkhorn_iters: 10

train:
  epochs: 2
  batch_size: 16
  lr: 0.000001
  tau_min: 0.1
  tau_decay: 0.95
  output_dir: outputs/mvp
```

- [ ] **Step 3: Write config dataclasses**

Create `llm_fas/config.py` with dataclasses named `SystemConfig`, `DataConfig`, `ModelConfig`, `TrainConfig`, and `ExperimentConfig`. The loader must expose:

```python
from pathlib import Path
import yaml


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig(
        seed=int(raw["seed"]),
        device=str(raw.get("device", "auto")),
        system=SystemConfig(**raw["system"]),
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
    )
```

The dataclasses must keep field names identical to `configs/mvp.yaml`.

- [ ] **Step 4: Run config smoke import**

Run:

```bash
python -c "from llm_fas.config import load_config; c=load_config('configs/mvp.yaml'); print(c.system.K, c.system.Nx, c.train.epochs)"
```

Expected:

```text
3 4 2
```

- [ ] **Step 5: Commit checkpoint**

Run:

```bash
git add requirements.txt configs/mvp.yaml llm_fas/__init__.py llm_fas/config.py
git commit -m "chore: add mvp reproduction configuration"
```

---

### Task 2: FAS Physics And Channel Generation

**Files:**
- Create: `llm_fas/physics.py`
- Create: `tests/test_physics.py`

- [ ] **Step 1: Write failing tests for unit conversion**

Create tests that assert:

```python
from llm_fas.physics import dbm_to_watt, noise_power_watt, path_loss_beta


def test_dbm_to_watt_for_20_dbm():
    assert abs(dbm_to_watt(20.0) - 0.1) < 1e-12


def test_noise_power_for_10_mhz():
    assert abs(noise_power_watt(-174.0, 10_000_000.0) - 10 ** ((-104.0 - 30.0) / 10.0)) < 1e-20


def test_path_loss_beta_default_distance():
    beta = path_loss_beta(0.2)
    expected_db = 128.1 + 37.6 * __import__("math").log10(0.2)
    assert abs(beta - 10 ** (-expected_db / 10.0)) < 1e-18
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_physics.py -q
```

Expected: failure because `llm_fas.physics` does not exist yet.

- [ ] **Step 3: Implement unit conversion and path loss**

Create functions in `llm_fas/physics.py`:

```python
def dbm_to_watt(dbm: float) -> float:
    return 10.0 ** ((dbm - 30.0) / 10.0)


def noise_power_watt(noise_psd_dBm_per_Hz: float, bandwidth_Hz: float) -> float:
    noise_dbm = noise_psd_dBm_per_Hz + 10.0 * math.log10(bandwidth_Hz)
    return dbm_to_watt(noise_dbm)


def path_loss_beta(distance_km: float) -> float:
    path_loss_db = 128.1 + 37.6 * math.log10(distance_km)
    return 10.0 ** (-path_loss_db / 10.0)
```

- [ ] **Step 4: Write failing tests for geometry and correlation**

Add tests that assert:

```python
from llm_fas.physics import port_coordinates, spatial_correlation_matrix


def test_port_coordinates_use_paper_mapping():
    coords = port_coordinates(4, 4)
    assert coords[0] == (0, 0)
    assert coords[1] == (1, 0)
    assert coords[4] == (0, 1)
    assert coords[15] == (3, 3)


def test_spatial_correlation_matrix_shape_and_diagonal():
    J = spatial_correlation_matrix(Nx=4, Ny=4, W_lambda_x=2.0, W_lambda_y=2.0)
    assert J.shape == (16, 16)
    assert torch.allclose(torch.diag(J), torch.ones(16), atol=1e-6)
    assert torch.allclose(J, J.T, atol=1e-6)
```

- [ ] **Step 5: Implement geometry and Jake correlation**

Implement:

```python
def port_coordinates(Nx: int, Ny: int) -> list[tuple[int, int]]:
    return [(nx, ny) for ny in range(Ny) for nx in range(Nx)]


def spherical_j0(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x.abs() < 1e-8, torch.ones_like(x), torch.sin(x) / x)


def spatial_correlation_matrix(Nx: int, Ny: int, W_lambda_x: float, W_lambda_y: float) -> torch.Tensor:
    coords = port_coordinates(Nx, Ny)
    N = Nx * Ny
    J = torch.empty(N, N, dtype=torch.float32)
    for i, (nx_i, ny_i) in enumerate(coords):
        for j, (nx_j, ny_j) in enumerate(coords):
            dx = abs(nx_i - nx_j) / max(Nx - 1, 1) * W_lambda_x
            dy = abs(ny_i - ny_j) / max(Ny - 1, 1) * W_lambda_y
            distance = math.sqrt(dx * dx + dy * dy)
            J[i, j] = spherical_j0(torch.tensor(2.0 * math.pi * distance))
    return J
```

- [ ] **Step 6: Write failing channel generation test**

Add a test:

```python
from llm_fas.physics import generate_channels


def test_generate_channels_shape_dtype_and_finite_values():
    H = generate_channels(
        num_samples=5,
        K=3,
        Nx=4,
        Ny=4,
        W_lambda_x=2.0,
        W_lambda_y=2.0,
        distance_km=0.2,
        seed=123,
    )
    assert H.shape == (5, 3, 16)
    assert H.is_complex()
    assert torch.isfinite(H.real).all()
    assert torch.isfinite(H.imag).all()
```

- [ ] **Step 7: Implement differentiable-compatible channel generation**

Implement `generate_channels` so it:

- Builds `J`.
- Computes `eigvals, eigvecs = torch.linalg.eigh(J)`.
- Clamps eigenvalues at zero before square root.
- Draws `g ~ CN(0, I)` with `(randn + 1j * randn) / sqrt(2)`.
- Returns `H` with shape `[num_samples, K, Nx * Ny]`.

- [ ] **Step 8: Run physics tests**

Run:

```bash
python -m pytest tests/test_physics.py -q
```

Expected: all physics tests pass.

- [ ] **Step 9: Commit checkpoint**

Run:

```bash
git add llm_fas/physics.py tests/test_physics.py
git commit -m "feat: implement fas channel generation"
```

---

### Task 3: Beamforming And Sum Rate

**Files:**
- Create: `llm_fas/beamforming.py`
- Create: `tests/test_beamforming.py`

- [ ] **Step 1: Write failing shape and finiteness tests**

Create `tests/test_beamforming.py` with:

```python
import torch

from llm_fas.beamforming import beamforming_from_pq, sum_rate


def test_beamforming_shape_and_power_constraint():
    H_eff = torch.randn(2, 3, 4, dtype=torch.complex64)
    p = torch.full((2, 3), 0.1 / 3.0)
    q = torch.full((2, 3), 0.1 / 3.0)
    C = beamforming_from_pq(H_eff, p, q, noise_power=1e-13)
    assert C.shape == (2, 4, 3)
    power = (C.abs() ** 2).sum(dim=(1, 2))
    assert torch.allclose(power, torch.full((2,), 0.1), rtol=1e-4, atol=1e-6)


def test_sum_rate_is_finite_and_positive():
    H_eff = torch.randn(2, 3, 4, dtype=torch.complex64)
    p = torch.full((2, 3), 0.1 / 3.0)
    q = torch.full((2, 3), 0.1 / 3.0)
    C = beamforming_from_pq(H_eff, p, q, noise_power=1e-13)
    rates = sum_rate(H_eff, C, noise_power=1e-13)
    assert rates.shape == (2,)
    assert torch.isfinite(rates).all()
    assert (rates > 0).all()
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_beamforming.py -q
```

Expected: failure because `llm_fas.beamforming` does not exist yet.

- [ ] **Step 3: Implement beamforming derivation**

Implement `beamforming_from_pq(H_eff, p, q, noise_power)` with:

- `H_eff`: complex tensor `[B, K, n]` representing rows `h_k^H`.
- `p,q`: real tensors `[B, K]`, each row summing to `Pmax`.
- Matrix:

```text
B = I_n + (1 / noise_power) * sum_i q_i h_i h_i^H
```

- Use `torch.linalg.solve`, not matrix inverse.
- Return `C` with shape `[B, n, K]`.

The implementation must add a small diagonal jitter `1e-8` in the same dtype as `B` to avoid singular smoke-test matrices.

- [ ] **Step 4: Implement sum rate**

Implement `sum_rate(H_eff, C, noise_power)` with:

```python
gains = torch.einsum("bkn,bnj->bkj", H_eff, C).abs() ** 2
signal = gains.diagonal(dim1=1, dim2=2)
interference = gains.sum(dim=2) - signal
sinr = signal / (interference + noise_power)
return torch.log2(1.0 + sinr).sum(dim=1)
```

- [ ] **Step 5: Run beamforming tests**

Run:

```bash
python -m pytest tests/test_beamforming.py -q
```

Expected: all beamforming tests pass.

- [ ] **Step 6: Commit checkpoint**

Run:

```bash
git add llm_fas/beamforming.py tests/test_beamforming.py
git commit -m "feat: implement beamforming and sum rate"
```

---

### Task 4: Differentiable Port Selection

**Files:**
- Create: `llm_fas/sinkhorn.py`
- Create: `tests/test_sinkhorn.py`

- [ ] **Step 1: Write failing tests for relaxed assignment**

Create `tests/test_sinkhorn.py` with:

```python
import torch

from llm_fas.sinkhorn import gumbel_sinkhorn, hard_topk_ports


def test_gumbel_sinkhorn_shape_and_row_sums():
    logits = torch.zeros(2, 4, 16)
    relaxed = gumbel_sinkhorn(logits, tau=1.0, iters=10, add_noise=False)
    assert relaxed.shape == (2, 4, 16)
    assert torch.allclose(relaxed.sum(dim=-1), torch.ones(2, 4), atol=1e-5)
    assert torch.isfinite(relaxed).all()


def test_hard_topk_ports_are_unique():
    scores = torch.zeros(2, 4, 16)
    scores[:, :, 3] = 10.0
    scores[:, :, 7] = 9.0
    scores[:, :, 11] = 8.0
    scores[:, :, 15] = 7.0
    ports = hard_topk_ports(scores, n_active=4)
    assert ports.shape == (2, 4)
    for row in ports.tolist():
        assert len(set(row)) == 4
```

- [ ] **Step 2: Implement rectangular Gumbel-Sinkhorn**

Implement `gumbel_sinkhorn(logits, tau, iters, add_noise)`:

- Input `logits` has shape `[B, n, N]`.
- If `add_noise` is true, add `-log(-log(U))` with clamped `U`.
- Divide by `tau`.
- Apply softmax over ports.
- Repeatedly normalize rows to sum one and scale columns so no column dominates.
- Finish with row normalization so every activation slot selects one relaxed port.

- [ ] **Step 3: Implement hard unique inference**

Implement `hard_topk_ports(scores, n_active)`:

- Collapse slot scores to per-port scores with `scores.max(dim=1).values`.
- Return top `n_active` unique port indices for each batch row.
- Keep output dtype `torch.long`.

Also implement `ports_to_selection_matrix(ports, N)` returning `[B, n, N]` one-hot rows.

- [ ] **Step 4: Run sinkhorn tests**

Run:

```bash
python -m pytest tests/test_sinkhorn.py -q
```

Expected: all sinkhorn tests pass.

- [ ] **Step 5: Commit checkpoint**

Run:

```bash
git add llm_fas/sinkhorn.py tests/test_sinkhorn.py
git commit -m "feat: implement differentiable port selection"
```

---

### Task 5: Dataset And Random Baseline

**Files:**
- Create: `llm_fas/data.py`
- Create: `llm_fas/baselines.py`

- [ ] **Step 1: Implement deterministic dataset builder**

Create `llm_fas/data.py` with:

- `build_datasets(cfg)` returning train, val, and test tensors.
- Use `generate_channels` three times with seeds `seed`, `seed + 1`, and `seed + 2`.
- `make_loader(H, batch_size, shuffle, seed)` returning a PyTorch `DataLoader`.

- [ ] **Step 2: Implement Random baseline**

Create `llm_fas/baselines.py` with `evaluate_random_baseline(H, cfg, seed)`:

- Randomly sample `n_active` unique ports per channel sample.
- Build hard selection matrix `[B, n, N]`.
- Compute `H_eff = H @ selection.transpose(-1, -2)`.
- Set `p=q=Pmax/K`.
- Compute `C` with `beamforming_from_pq`.
- Return mean sum rate as Python `float`.

- [ ] **Step 3: Run Random baseline smoke command**

Run:

```bash
python -c "from llm_fas.config import load_config; from llm_fas.data import build_datasets; from llm_fas.baselines import evaluate_random_baseline; c=load_config('configs/mvp.yaml'); _,_,test=build_datasets(c); print(round(evaluate_random_baseline(test,c,123), 6))"
```

Expected: prints a finite positive number.

- [ ] **Step 4: Commit checkpoint**

Run:

```bash
git add llm_fas/data.py llm_fas/baselines.py
git commit -m "feat: add deterministic data and random baseline"
```

---

### Task 6: Proposed Model

**Files:**
- Create: `llm_fas/models.py`

- [ ] **Step 1: Implement trainable model module**

Create `ProposedLLMFASModel` with:

```python
class ProposedLLMFASModel(nn.Module):
    def __init__(self, cfg: ExperimentConfig):
        ...

    def forward(self, H: torch.Tensor, tau: float, training: bool) -> dict[str, torch.Tensor]:
        ...
```

The returned dictionary must include:

- `port_scores`: `[B, n, N]`
- `selection_soft`: `[B, n, N]`
- `p`: `[B, K]`
- `q`: `[B, K]`
- `H_eff`: `[B, K, n]`
- `C`: `[B, n, K]`
- `rate`: `[B]`

- [ ] **Step 2: Implement preprocessing**

Preprocessing must:

- Split `H` into real and imaginary parts.
- Project each user row from `N` to `d_mha`.
- Apply `nn.MultiheadAttention(batch_first=True)` separately to real and imaginary tokens.
- Flatten `[B, 2K, d_mha]`.
- Project to GPT-2 input embeddings with shape `[B, N * n, d_llm]`.

Use `d_llm = backbone.config.n_embd` after loading GPT-2.

- [ ] **Step 3: Implement GPT-2 LoRA backbone**

Implementation requirements:

- Load `GPT2Model.from_pretrained(cfg.model.backbone_name)`.
- Keep only the first `cfg.model.gpt2_layers` transformer blocks.
- Freeze GPT-2 parameters.
- Apply PEFT LoRA with `target_modules=["c_attn"]`, `r=cfg.model.lora_rank`.
- Re-enable gradients for LayerNorm parameters whose names contain `ln_`.
- Call the backbone with `inputs_embeds=embeddings`.

This is the MVP approximation of the paper's Q/V-only LoRA. The strict Q/V split is reserved for the full reproduction stage.

- [ ] **Step 4: Implement parallel heads and physical post-processing**

The model head must:

- Flatten GPT-2 output `[B, N*n,d_llm]`.
- Produce `port_scores` using a linear layer to `n*N`.
- Produce `power_logits` using a linear layer to `2*K`.
- Convert `power_logits[:, 0, :]` and `power_logits[:, 1, :]` to `p` and `q` with row-wise softmax multiplied by `Pmax_W`.
- Use `gumbel_sinkhorn` during training and deterministic relaxed selection without Gumbel noise during evaluation.
- Compute `H_eff`, `C`, and `rate`.

- [ ] **Step 5: Run model forward smoke command**

Run:

```bash
python -c "from llm_fas.config import load_config; from llm_fas.data import build_datasets; from llm_fas.models import ProposedLLMFASModel; c=load_config('configs/mvp.yaml'); H,_,_=build_datasets(c); m=ProposedLLMFASModel(c); out=m(H[:2], tau=1.0, training=True); print(out['rate'].shape, out['selection_soft'].shape)"
```

Expected:

```text
torch.Size([2]) torch.Size([2, 4, 16])
```

- [ ] **Step 6: Commit checkpoint**

Run:

```bash
git add llm_fas/models.py
git commit -m "feat: add proposed llm fas model"
```

---

### Task 7: Training And Evaluation Scripts

**Files:**
- Create: `llm_fas/train.py`
- Create: `scripts/train_mvp.py`
- Create: `scripts/evaluate_mvp.py`

- [ ] **Step 1: Implement training loop**

Create `train_proposed(cfg)` in `llm_fas/train.py`:

- Set Python, NumPy, and PyTorch seeds.
- Select CUDA when `cfg.device == "auto"` and CUDA is available; otherwise CPU.
- Build datasets and loaders.
- Train with loss `-out["rate"].mean()`.
- Use Adam over parameters with `requires_grad=True`.
- Compute validation loss after every epoch with `training=False` and no Gumbel noise.
- Save `outputs/mvp/proposed.pt`.
- Save `outputs/mvp/train_history.csv` with columns `epoch,train_loss,val_loss`.

- [ ] **Step 2: Implement evaluation helper**

Create `evaluate_checkpoint(cfg, checkpoint_path)`:

- Build test dataset.
- Load `ProposedLLMFASModel`.
- Load checkpoint.
- Compute Proposed mean test sum rate.
- Compute Random mean test sum rate.
- Save `outputs/mvp/results.csv` with columns:

```text
method,K,Nx,Ny,N,n_active,Pmax_dBm,distance_km,seed,test_sum_rate
```

- [ ] **Step 3: Implement CLI wrappers**

`scripts/train_mvp.py` must parse:

```text
--config configs/mvp.yaml
```

and call `train_proposed`.

`scripts/evaluate_mvp.py` must parse:

```text
--config configs/mvp.yaml
--checkpoint outputs/mvp/proposed.pt
```

and call `evaluate_checkpoint`.

- [ ] **Step 4: Run full test suite**

Run:

```bash
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Run MVP training**

Run:

```bash
python scripts/train_mvp.py --config configs/mvp.yaml
```

Expected:

- No NaN losses.
- `outputs/mvp/proposed.pt` exists.
- `outputs/mvp/train_history.csv` exists.

- [ ] **Step 6: Run MVP evaluation**

Run:

```bash
python scripts/evaluate_mvp.py --config configs/mvp.yaml --checkpoint outputs/mvp/proposed.pt
```

Expected:

- `outputs/mvp/results.csv` exists.
- It contains rows for `random` and `proposed`.
- Both `test_sum_rate` values are finite and positive.

- [ ] **Step 7: Commit checkpoint**

Run:

```bash
git add llm_fas/train.py scripts/train_mvp.py scripts/evaluate_mvp.py
git commit -m "feat: add mvp training and evaluation scripts"
```

---

### Task 8: Result Review And Next Reproduction Stage

**Files:**
- Modify: `docs/MVP_REPRODUCTION_PLAN.md`
- Create after execution: `outputs/mvp/results.csv`
- Create after execution: `outputs/mvp/train_history.csv`

- [ ] **Step 1: Inspect generated artifacts**

Run:

```bash
python -c "import pandas as pd; print(pd.read_csv('outputs/mvp/results.csv')); print(pd.read_csv('outputs/mvp/train_history.csv'))"
```

Expected: CSV contents print successfully and contain finite numeric values.

If `pandas` is not installed, use:

```bash
python -c "import csv; print(list(csv.DictReader(open('outputs/mvp/results.csv', newline='')))); print(list(csv.DictReader(open('outputs/mvp/train_history.csv', newline=''))))"
```

- [ ] **Step 2: Record MVP outcome**

Append a section named `## Execution Notes` to this file with:

- Test command result.
- Train command result.
- Evaluation command result.
- Random test sum rate.
- Proposed test sum rate.
- Any runtime limitation, such as CPU-only training time or HuggingFace model download issues.

- [ ] **Step 3: Decide the next reproduction expansion**

After the MVP passes, expand in this order:

1. Increase `train_samples` to `1000`, `test_samples` to `200`, and `epochs` to `20`.
2. Switch GPT-2 layers from `2` to `6`.
3. Add Transformer baseline using the same preprocessing and post-processing.
4. Reproduce Fig.5 batch-size convergence.
5. Add CNN and LLM-sequential baselines.
6. Run Fig.6-Fig.11 sweeps.

- [ ] **Step 4: Commit final MVP result notes**

Run:

```bash
git add docs/MVP_REPRODUCTION_PLAN.md outputs/mvp/results.csv outputs/mvp/train_history.csv
git commit -m "docs: record mvp reproduction results"
```

---

## Self-Review

- Spec coverage: The plan covers the MVP subset of the existing overall plan: system model, channel generation, port selection relaxation, beamforming derivation, sum-rate loss, Proposed model, Random sanity baseline, training, and evaluation.
- Completeness scan: Every MVP task names concrete files, commands, expected outputs, and tensor contracts. Full paper sweeps are explicitly outside MVP scope and listed as the next stage after acceptance criteria pass.
- Type consistency: The core tensors use consistent shapes: `H [B,K,N]`, soft selection `[B,n,N]`, `H_eff [B,K,n]`, `C [B,n,K]`, `p/q [B,K]`, `rate [B]`.
- Reproduction risk: The MVP uses PEFT LoRA on GPT-2 `c_attn`, which is an engineering approximation of paper Q/V-only LoRA. This is acceptable for the first end-to-end reproduction and must be replaced by explicit Q/V LoRA for strict paper-level reproduction.
