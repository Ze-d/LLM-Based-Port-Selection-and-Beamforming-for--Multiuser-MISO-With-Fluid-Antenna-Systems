# Stage 1 Hard Inference And Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Proposed evaluation use hard unique activated ports and write complete experiment metadata to `results.csv`.

**Architecture:** Keep the existing MVP training pipeline and add an explicit selection-mode path inside `ProposedLLMFASModel.forward`. Training can still use differentiable soft Gumbel-Sinkhorn, while validation/evaluation use hard unique port selection built from the model's port scores.

**Tech Stack:** Python 3.12, PyTorch, Transformers, PEFT, PyYAML, pytest.

---

## File Structure

- Modify: `llm_fas/models.py` - add `selection_mode`, hard selection tensors, and actual `selection` tensor used for rate.
- Modify: `llm_fas/train.py` - route validation/evaluation through hard selection and write richer metadata rows.
- Modify: `tests/test_models.py` - test hard inference tensor contract and unique ports.
- Modify: `tests/test_train.py` - test metadata columns and `selection_mode=hard`.
- Optional modify: `tests/test_sinkhorn.py` - add explicit one-hot selection matrix test if needed.

## Task 1: Model Hard Inference Contract

**Files:**
- Modify: `tests/test_models.py`
- Modify: `llm_fas/models.py`

- [ ] **Step 1: Write failing hard inference test**

Add this test to `tests/test_models.py` after `test_proposed_model_forward_returns_expected_tensors`:

```python
def test_proposed_model_hard_inference_returns_unique_ports(monkeypatch):
    monkeypatch.setattr("llm_fas.models.GPT2Model.from_pretrained", _tiny_gpt2)
    cfg = _tiny_cfg()
    H = generate_channels(
        num_samples=3,
        K=cfg.system.K,
        Nx=cfg.system.Nx,
        Ny=cfg.system.Ny,
        W_lambda_x=cfg.system.W_lambda_x,
        W_lambda_y=cfg.system.W_lambda_y,
        distance_km=cfg.system.distance_km,
        seed=100,
    )

    model = ProposedLLMFASModel(cfg)
    out = model(H, tau=0.1, training=False, selection_mode="hard")

    N = cfg.system.Nx * cfg.system.Ny
    assert out["selection_mode"] == "hard"
    assert out["ports"].shape == (3, cfg.system.n_active)
    assert out["selection_hard"].shape == (3, cfg.system.n_active, N)
    assert out["selection"].shape == (3, cfg.system.n_active, N)
    assert torch.equal(out["selection"], out["selection_hard"])
    assert torch.allclose(out["selection_hard"].sum(dim=-1), torch.ones(3, cfg.system.n_active))
    assert torch.all((out["selection_hard"] == 0.0) | (out["selection_hard"] == 1.0))
    for ports in out["ports"].tolist():
        assert len(set(ports)) == cfg.system.n_active
    assert torch.isfinite(out["rate"]).all()
```

- [ ] **Step 2: Run model test and verify failure**

Run:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest tests/test_models.py -q
```

Expected: fail because `selection_mode` is not accepted by `ProposedLLMFASModel.forward`.

- [ ] **Step 3: Import hard selection helpers**

In `llm_fas/models.py`, replace:

```python
from llm_fas.sinkhorn import gumbel_sinkhorn
```

with:

```python
from llm_fas.sinkhorn import gumbel_sinkhorn, hard_topk_ports, ports_to_selection_matrix
```

- [ ] **Step 4: Update forward signature and return type**

In `llm_fas/models.py`, add:

```python
from typing import Literal
```

Then change:

```python
def forward(self, H: torch.Tensor, tau: float, training: bool) -> dict[str, torch.Tensor]:
```

to:

```python
def forward(
    self,
    H: torch.Tensor,
    tau: float,
    training: bool,
    selection_mode: Literal["soft", "hard"] | None = None,
) -> dict[str, torch.Tensor | str]:
```

- [ ] **Step 5: Implement selection mode logic**

Inside `forward`, immediately after `port_scores` and `p,q` are computed, replace the existing `selection_soft`, `H_eff`, `C`, and `rate` block with:

```python
        active_selection_mode = selection_mode or ("soft" if training else "hard")
        if active_selection_mode not in {"soft", "hard"}:
            raise ValueError(f"Unsupported selection_mode: {active_selection_mode}")

        selection_soft = gumbel_sinkhorn(
            port_scores,
            tau=tau,
            iters=self.cfg.model.sinkhorn_iters,
            add_noise=training and active_selection_mode == "soft",
        )

        ports = None
        selection_hard = None
        if active_selection_mode == "hard":
            ports = hard_topk_ports(port_scores, self.n_active)
            selection_hard = ports_to_selection_matrix(ports, self.N).to(
                device=H.device,
                dtype=H.real.dtype,
            )
            selection = selection_hard
        else:
            selection = selection_soft

        H_eff = H @ selection.transpose(-1, -2).to(H.dtype)
        C = beamforming_from_pq(H_eff, p, q, self.noise_power)
        rate = sum_rate(H_eff, C, self.noise_power)
```

- [ ] **Step 6: Return both soft and active selection tensors**

Replace the current return dictionary with:

```python
        result: dict[str, torch.Tensor | str] = {
            "port_scores": port_scores,
            "selection_soft": selection_soft,
            "selection": selection,
            "selection_mode": active_selection_mode,
            "p": p,
            "q": q,
            "H_eff": H_eff,
            "C": C,
            "rate": rate,
        }
        if ports is not None and selection_hard is not None:
            result["ports"] = ports
            result["selection_hard"] = selection_hard
        return result
```

- [ ] **Step 7: Run model tests**

Run:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest tests/test_models.py -q
```

Expected: all model tests pass.

## Task 2: Evaluation Routing And Metadata

**Files:**
- Modify: `tests/test_train.py`
- Modify: `llm_fas/train.py`

- [ ] **Step 1: Write failing metadata assertions**

In `tests/test_train.py`, update `test_evaluate_checkpoint_writes_results` so that after reading `rows`, it asserts the full metadata contract:

```python
    expected_columns = {
        "method",
        "selection_mode",
        "K",
        "Nx",
        "Ny",
        "N",
        "n_active",
        "W_lambda_x",
        "W_lambda_y",
        "Pmax_dBm",
        "distance_km",
        "seed",
        "train_samples",
        "val_samples",
        "test_samples",
        "epochs",
        "batch_size",
        "gpt2_layers",
        "d_mha",
        "mha_heads",
        "test_sum_rate",
    }
    assert set(rows[0].keys()) == expected_columns
    assert [row["method"] for row in rows] == ["random", "proposed"]
    assert [row["selection_mode"] for row in rows] == ["hard", "hard"]
```

Keep the existing finite positive `test_sum_rate` assertions.

- [ ] **Step 2: Run train tests and verify failure**

Run:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest tests/test_train.py -q
```

Expected: fail because `results.csv` does not yet include the new metadata columns.

- [ ] **Step 3: Add selection mode parameter to evaluator**

In `llm_fas/train.py`, change `_evaluate_model_rate` signature from:

```python
def _evaluate_model_rate(
    model: ProposedLLMFASModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    tau: float,
) -> float:
```

to:

```python
def _evaluate_model_rate(
    model: ProposedLLMFASModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    tau: float,
    selection_mode: str,
) -> float:
```

Then change:

```python
out = model(batch, tau=tau, training=False)
```

to:

```python
out = model(batch, tau=tau, training=False, selection_mode=selection_mode)
```

- [ ] **Step 4: Use hard selection for validation and test evaluation**

In `train_proposed`, change:

```python
val_rate = _evaluate_model_rate(model, val_loader, device, tau)
```

to:

```python
val_rate = _evaluate_model_rate(model, val_loader, device, tau, selection_mode="hard")
```

In `evaluate_checkpoint`, change:

```python
proposed_rate = _evaluate_model_rate(model, test_loader, device, tau)
```

to:

```python
proposed_rate = _evaluate_model_rate(model, test_loader, device, tau, selection_mode="hard")
```

- [ ] **Step 5: Add result row helper**

In `llm_fas/train.py`, add this helper above `evaluate_checkpoint`:

```python
def _result_row(cfg: ExperimentConfig, method: str, selection_mode: str, test_sum_rate: float) -> dict[str, float | int | str]:
    N = cfg.system.Nx * cfg.system.Ny
    return {
        "method": method,
        "selection_mode": selection_mode,
        "K": cfg.system.K,
        "Nx": cfg.system.Nx,
        "Ny": cfg.system.Ny,
        "N": N,
        "n_active": cfg.system.n_active,
        "W_lambda_x": cfg.system.W_lambda_x,
        "W_lambda_y": cfg.system.W_lambda_y,
        "Pmax_dBm": cfg.system.Pmax_dBm,
        "distance_km": cfg.system.distance_km,
        "seed": cfg.seed,
        "train_samples": cfg.data.train_samples,
        "val_samples": cfg.data.val_samples,
        "test_samples": cfg.data.test_samples,
        "epochs": cfg.train.epochs,
        "batch_size": cfg.train.batch_size,
        "gpt2_layers": cfg.model.gpt2_layers,
        "d_mha": cfg.model.d_mha,
        "mha_heads": cfg.model.mha_heads,
        "test_sum_rate": test_sum_rate,
    }
```

- [ ] **Step 6: Use helper in evaluate_checkpoint**

Replace the manual `N = ...` and `rows = [...]` block with:

```python
    rows = [
        _result_row(cfg, method="random", selection_mode="hard", test_sum_rate=random_rate),
        _result_row(cfg, method="proposed", selection_mode="hard", test_sum_rate=proposed_rate),
    ]
```

Then replace the `fieldnames=[...]` list with:

```python
            fieldnames=[
                "method",
                "selection_mode",
                "K",
                "Nx",
                "Ny",
                "N",
                "n_active",
                "W_lambda_x",
                "W_lambda_y",
                "Pmax_dBm",
                "distance_km",
                "seed",
                "train_samples",
                "val_samples",
                "test_samples",
                "epochs",
                "batch_size",
                "gpt2_layers",
                "d_mha",
                "mha_heads",
                "test_sum_rate",
            ],
```

- [ ] **Step 7: Run train tests**

Run:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest tests/test_train.py -q
```

Expected: all train tests pass.

## Task 3: Full Verification

**Files:**
- Verification uses the files modified in Tasks 1 and 2.

- [ ] **Step 1: Run full test suite**

Run:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest -q
```

Expected:

```text
20 passed
```

The exact count may be higher if an extra sinkhorn test is added.

- [ ] **Step 2: Run a tiny end-to-end train/evaluate smoke**

Use the existing config unless runtime is a concern. If a faster smoke is needed, temporarily run through the existing test monkeypatch only; do not commit config changes for a smaller run.

Run:

```bash
uv run --with torch --with numpy --with pyyaml --with transformers --with peft python scripts/train_mvp.py --config configs/mvp.yaml
uv run --with torch --with numpy --with pyyaml --with transformers --with peft python scripts/evaluate_mvp.py --config configs/mvp.yaml --checkpoint outputs/mvp_train1000_test200_e20/proposed.pt
```

Expected:

- Training finishes without NaN.
- Evaluation writes `outputs/mvp_train1000_test200_e20/results.csv`.
- `results.csv` contains `selection_mode` and the full metadata columns.
- Proposed `test_sum_rate` is finite and positive.

- [ ] **Step 3: Inspect result CSV**

Run:

```bash
python -c "import csv; rows=list(csv.DictReader(open('outputs/mvp_train1000_test200_e20/results.csv', newline=''))); print(rows)"
```

Expected: two rows, `method=random` and `method=proposed`, both with `selection_mode=hard`.

- [ ] **Step 4: Commit implementation**

Run:

```bash
git add llm_fas/models.py llm_fas/train.py tests/test_models.py tests/test_train.py outputs/mvp_train1000_test200_e20/results.csv outputs/mvp_train1000_test200_e20/train_history.csv
git commit -m "feat: evaluate proposed with hard port selection"
```

If the smoke run is skipped due to runtime, do not add output CSVs. Commit only code and tests:

```bash
git add llm_fas/models.py llm_fas/train.py tests/test_models.py tests/test_train.py
git commit -m "feat: evaluate proposed with hard port selection"
```

## Self-Review

- Spec coverage: This plan implements Stage 1 from the full reproduction design: hard Proposed inference and richer result metadata.
- Placeholder scan: The plan contains concrete file paths, code blocks, commands, and expected outcomes. There are no deferred implementation placeholders.
- Type consistency: The model returns tensors plus one string field, so the forward return type changes to `dict[str, torch.Tensor | str]`. Core tensors keep the existing contracts: `H [B,K,N]`, `selection [B,n,N]`, `H_eff [B,K,n]`, `C [B,n,K]`, `rate [B]`.
- Risk: Hard validation may change the recorded validation curve relative to earlier relaxed validation. This is intentional for Stage 1 because the reproduction needs deployment-faithful evaluation before scaling.
