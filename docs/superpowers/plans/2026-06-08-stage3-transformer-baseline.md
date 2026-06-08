# Stage 3 Transformer Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a trainable `transformer` baseline that shares the Proposed method's preprocessing, port-selection heads, power heads, hard-inference evaluation, and result schema.

**Architecture:** Refactor the current Proposed model into a shared joint FAS base class plus two backbones: GPT-2/LoRA for `proposed`, and a PyTorch `TransformerEncoder` for `transformer`. Generalize training, evaluation, and seed orchestration from a Proposed-only flow to a method-dispatch flow while keeping existing Stage 1 and Stage 2 entry points backward compatible.

**Tech Stack:** Python 3.12, PyTorch, Transformers, PEFT, PyYAML, pytest, stdlib `csv`, `argparse`, `dataclasses`, and `pathlib`.

---

## Scope

Stage 3 implements the Transformer baseline only. CNN, LLM-sequential, paper-scale configs, and figure sweeps remain later stages.

The Transformer baseline must reuse the following contracts:

- Input channel tensor `H`: `[B,K,N]` complex.
- Port scores: `[B,n,N]`.
- Soft selection during training via Gumbel-Sinkhorn.
- Hard unique ports during validation and test evaluation.
- Power outputs `p,q`: `[B,K]`, each softmax-normalized to sum to `Pmax_W`.
- Beamforming and rate from existing `beamforming_from_pq()` and `sum_rate()`.
- CSV schema identical to Stage 2.

The only model difference between `proposed` and `transformer` is the sequence backbone after shared CSI preprocessing.

Stage 2 showed Proposed has a small mean advantage over Random but not all-seed dominance. Stage 3 must therefore use multi-seed comparisons when interpreting results. A one-seed Stage 3 smoke run is valid for software verification only, not for a performance claim.

## File Structure

- Modify: `llm_fas/models.py` - add shared `JointFASModelBase`, keep `ProposedLLMFASModel`, and add `TransformerBaselineModel`.
- Modify: `llm_fas/train.py` - add method registry, generic `train_method()`, generic `evaluate_checkpoints()`, and compatibility wrappers.
- Modify: `llm_fas/experiments.py` - add method-list parsing and multi-method seed orchestration while preserving Stage 2 default behavior.
- Modify: `scripts/run_stage2_seeds.py` - add optional `--methods` with default `proposed`.
- Create: `scripts/run_stage3_transformer.py` - Stage 3 CLI with default methods `transformer,proposed`.
- Modify: `tests/test_models.py` - verify Transformer baseline tensor contract and hard inference.
- Modify: `tests/test_train.py` - verify generic method training and combined evaluation rows.
- Modify: `tests/test_experiments.py` - verify method parsing, multi-method orchestration, and Stage 3 CLI help.
- Create after execution: `outputs/stage3_transformer_smoke/all_results.csv`.
- Create after execution: `outputs/stage3_transformer_smoke/summary.csv`.
- Create after execution: `docs/STAGE3_TRANSFORMER_BASELINE_SUMMARY.md`.

## Task 1: Shared Model Contract And Transformer Baseline

**Files:**
- Modify: `tests/test_models.py`
- Modify: `llm_fas/models.py`

- [ ] **Step 1: Write failing Transformer model tests**

Modify the import in `tests/test_models.py`:

```python
from llm_fas.models import ProposedLLMFASModel, TransformerBaselineModel
```

Add this helper near the existing tests:

```python
def _assert_joint_model_contract(out, batch_size: int, cfg) -> None:
    N = cfg.system.Nx * cfg.system.Ny
    assert out["port_scores"].shape == (batch_size, cfg.system.n_active, N)
    assert out["selection_soft"].shape == (batch_size, cfg.system.n_active, N)
    assert out["p"].shape == (batch_size, cfg.system.K)
    assert out["q"].shape == (batch_size, cfg.system.K)
    assert out["H_eff"].shape == (batch_size, cfg.system.K, cfg.system.n_active)
    assert out["C"].shape == (batch_size, cfg.system.n_active, cfg.system.K)
    assert out["rate"].shape == (batch_size,)
    assert torch.isfinite(out["rate"]).all()
    assert torch.allclose(
        out["selection_soft"].sum(dim=-1),
        torch.ones(batch_size, cfg.system.n_active),
        atol=1e-5,
    )

    Pmax = dbm_to_watt(cfg.system.Pmax_dBm)
    assert torch.allclose(out["p"].sum(dim=1), torch.full((batch_size,), Pmax), rtol=1e-5, atol=1e-7)
    assert torch.allclose(out["q"].sum(dim=1), torch.full((batch_size,), Pmax), rtol=1e-5, atol=1e-7)
```

Replace the duplicated shape assertions inside `test_proposed_model_forward_returns_expected_tensors()` with:

```python
    _assert_joint_model_contract(out, batch_size=2, cfg=cfg)
```

Append these new tests:

```python
def test_transformer_baseline_forward_returns_expected_tensors():
    cfg = _tiny_cfg()
    H = generate_channels(
        num_samples=2,
        K=cfg.system.K,
        Nx=cfg.system.Nx,
        Ny=cfg.system.Ny,
        W_lambda_x=cfg.system.W_lambda_x,
        W_lambda_y=cfg.system.W_lambda_y,
        distance_km=cfg.system.distance_km,
        seed=102,
    )

    model = TransformerBaselineModel(cfg)
    out = model(H, tau=1.0, training=True)

    _assert_joint_model_contract(out, batch_size=2, cfg=cfg)
    assert out["selection_mode"] == "soft"


def test_transformer_baseline_hard_inference_returns_unique_ports():
    cfg = _tiny_cfg()
    H = generate_channels(
        num_samples=3,
        K=cfg.system.K,
        Nx=cfg.system.Nx,
        Ny=cfg.system.Ny,
        W_lambda_x=cfg.system.W_lambda_x,
        W_lambda_y=cfg.system.W_lambda_y,
        distance_km=cfg.system.distance_km,
        seed=103,
    )

    model = TransformerBaselineModel(cfg)
    out = model(H, tau=0.1, training=False, selection_mode="hard")

    N = cfg.system.Nx * cfg.system.Ny
    assert out["selection_mode"] == "hard"
    assert out["ports"].shape == (3, cfg.system.n_active)
    assert out["selection_hard"].shape == (3, cfg.system.n_active, N)
    assert torch.equal(out["selection"], out["selection_hard"])
    assert torch.allclose(out["selection_hard"].sum(dim=-1), torch.ones(3, cfg.system.n_active))
    assert torch.all((out["selection_hard"] == 0.0) | (out["selection_hard"] == 1.0))
    for ports in out["ports"].tolist():
        assert len(set(ports)) == cfg.system.n_active
    assert torch.isfinite(out["rate"]).all()
```

- [ ] **Step 2: Run model tests and verify the new failure**

Run:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest tests/test_models.py -q
```

Expected: fail with `ImportError` or `AttributeError` because `TransformerBaselineModel` is not defined.

- [ ] **Step 3: Refactor `llm_fas/models.py` into shared base plus two backbones**

Replace the current `ProposedLLMFASModel` class with these classes while keeping the existing imports:

```python
class JointFASModelBase(nn.Module):
    def __init__(self, cfg: ExperimentConfig, d_llm: int):
        super().__init__()
        self.cfg = cfg
        self.K = cfg.system.K
        self.N = cfg.system.Nx * cfg.system.Ny
        self.n_active = cfg.system.n_active
        self.sequence_length = self.N * self.n_active
        self.Pmax_W = dbm_to_watt(cfg.system.Pmax_dBm)
        self.noise_power = noise_power_watt(cfg.system.noise_psd_dBm_per_Hz, cfg.system.bandwidth_Hz)
        self.d_llm = int(d_llm)

        self.real_proj = nn.Linear(self.N, cfg.model.d_mha)
        self.imag_proj = nn.Linear(self.N, cfg.model.d_mha)
        self.real_mha = nn.MultiheadAttention(cfg.model.d_mha, cfg.model.mha_heads, batch_first=True)
        self.imag_mha = nn.MultiheadAttention(cfg.model.d_mha, cfg.model.mha_heads, batch_first=True)
        self.embed_proj = nn.Linear(2 * self.K * cfg.model.d_mha, self.sequence_length * self.d_llm)
        self.port_head = nn.Linear(self.sequence_length * self.d_llm, self.n_active * self.N)
        self.power_head = nn.Linear(self.sequence_length * self.d_llm, 2 * self.K)

    def _preprocess(self, H: torch.Tensor) -> torch.Tensor:
        real_tokens = self.real_proj(H.real.float())
        imag_tokens = self.imag_proj(H.imag.float())
        real_attn, _ = self.real_mha(real_tokens, real_tokens, real_tokens, need_weights=False)
        imag_attn, _ = self.imag_mha(imag_tokens, imag_tokens, imag_tokens, need_weights=False)
        features = torch.cat([real_attn, imag_attn], dim=1).reshape(H.shape[0], -1)
        return self.embed_proj(features).reshape(H.shape[0], self.sequence_length, self.d_llm)

    def _run_backbone(self, embeddings: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        H: torch.Tensor,
        tau: float,
        training: bool,
        selection_mode: Literal["soft", "hard"] | None = None,
    ) -> dict[str, torch.Tensor | str]:
        active_selection_mode = ("soft" if training else "hard") if selection_mode is None else selection_mode
        if active_selection_mode not in ("soft", "hard"):
            raise ValueError(f"selection_mode must be 'soft' or 'hard', got {active_selection_mode!r}")

        embeddings = self._preprocess(H)
        backbone_out = self._run_backbone(embeddings)
        z = backbone_out.reshape(H.shape[0], -1)

        port_scores = self.port_head(z).reshape(H.shape[0], self.n_active, self.N)
        power_logits = self.power_head(z).reshape(H.shape[0], 2, self.K)
        p = F.softmax(power_logits[:, 0, :], dim=-1) * self.Pmax_W
        q = F.softmax(power_logits[:, 1, :], dim=-1) * self.Pmax_W

        selection_soft = gumbel_sinkhorn(
            port_scores,
            tau=tau,
            iters=self.cfg.model.sinkhorn_iters,
            add_noise=training,
        )
        if active_selection_mode == "hard":
            ports = hard_topk_ports(port_scores, self.n_active)
            selection_hard = ports_to_selection_matrix(ports, self.N).to(device=H.device, dtype=H.real.dtype)
            selection = selection_hard
        else:
            selection = selection_soft

        H_eff = H @ selection.transpose(-1, -2).to(H.dtype)
        C = beamforming_from_pq(H_eff, p, q, self.noise_power)
        rate = sum_rate(H_eff, C, self.noise_power)
        out: dict[str, torch.Tensor | str] = {
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
        if active_selection_mode == "hard":
            out["ports"] = ports
            out["selection_hard"] = selection_hard
        return out


class ProposedLLMFASModel(JointFASModelBase):
    def __init__(self, cfg: ExperimentConfig):
        backbone = GPT2Model.from_pretrained(cfg.model.backbone_name)
        backbone.h = nn.ModuleList(list(backbone.h[: cfg.model.gpt2_layers]))
        backbone.config.n_layer = len(backbone.h)
        for param in backbone.parameters():
            param.requires_grad = False

        super().__init__(cfg, d_llm=int(backbone.config.n_embd))

        lora_config = LoraConfig(
            r=cfg.model.lora_rank,
            lora_alpha=cfg.model.lora_rank,
            target_modules=["c_attn"],
            bias="none",
            fan_in_fan_out=True,
        )
        self.backbone = get_peft_model(backbone, lora_config)
        for name, param in self.backbone.named_parameters():
            if "ln_" in name:
                param.requires_grad = True

    def _run_backbone(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs_embeds=embeddings).last_hidden_state


class TransformerBaselineModel(JointFASModelBase):
    def __init__(self, cfg: ExperimentConfig):
        super().__init__(cfg, d_llm=cfg.model.d_mha)
        self.register_buffer(
            "position_encoding",
            self._make_sinusoidal_position_encoding(self.sequence_length, self.d_llm),
            persistent=False,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_llm,
            nhead=cfg.model.mha_heads,
            dim_feedforward=4 * self.d_llm,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=cfg.model.gpt2_layers)

    @staticmethod
    def _make_sinusoidal_position_encoding(sequence_length: int, d_model: int) -> torch.Tensor:
        positions = torch.arange(sequence_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        encoding = torch.zeros(1, sequence_length, d_model)
        encoding[0, :, 0::2] = torch.sin(positions * div_term)
        if d_model > 1:
            encoding[0, :, 1::2] = torch.cos(positions * div_term[: encoding[0, :, 1::2].shape[-1]])
        return encoding

    def _run_backbone(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.backbone(embeddings + self.position_encoding.to(device=embeddings.device, dtype=embeddings.dtype))
```

- [ ] **Step 4: Run model tests**

Run:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest tests/test_models.py -q
```

Expected: all model tests pass.

- [ ] **Step 5: Commit model baseline**

Run:

```bash
git add llm_fas/models.py tests/test_models.py
git commit -m "feat: add transformer baseline model"
```

## Task 2: Generic Train And Evaluation Methods

**Files:**
- Modify: `tests/test_train.py`
- Modify: `llm_fas/train.py`

- [ ] **Step 1: Write failing generic training and evaluation tests**

Modify the imports in `tests/test_train.py`:

```python
from llm_fas.train import evaluate_checkpoint, evaluate_checkpoints, train_method, train_proposed
```

Append these tests:

```python
def test_train_method_writes_transformer_checkpoint_and_history(tmp_path):
    cfg = _tiny_cfg(tmp_path)

    checkpoint_path = train_method(cfg, method="transformer")

    assert checkpoint_path == Path(cfg.train.output_dir) / "transformer.pt"
    assert checkpoint_path.exists()
    history_path = Path(cfg.train.output_dir) / "transformer_train_history.csv"
    assert history_path.exists()
    rows = list(csv.DictReader(history_path.open(newline="")))
    assert len(rows) == 1
    assert rows[0].keys() == {"epoch", "train_loss", "val_loss"}
    assert math.isfinite(float(rows[0]["train_loss"]))
    assert math.isfinite(float(rows[0]["val_loss"]))


def test_evaluate_checkpoints_writes_random_transformer_proposed(monkeypatch, tmp_path):
    monkeypatch.setattr("llm_fas.models.GPT2Model.from_pretrained", _tiny_gpt2)
    cfg = _tiny_cfg(tmp_path)
    transformer_checkpoint = train_method(cfg, method="transformer")
    proposed_checkpoint = train_proposed(cfg)

    results_path = evaluate_checkpoints(
        cfg,
        {
            "transformer": transformer_checkpoint,
            "proposed": proposed_checkpoint,
        },
    )

    rows = list(csv.DictReader(results_path.open(newline="")))
    assert [row["method"] for row in rows] == ["random", "transformer", "proposed"]
    assert [row["selection_mode"] for row in rows] == ["hard", "hard", "hard"]
    assert {tuple(row.keys()) for row in rows} == {tuple(rows[0].keys())}
    for row in rows:
        assert row["K"] == str(cfg.system.K)
        assert row["N"] == str(cfg.system.Nx * cfg.system.Ny)
        assert math.isfinite(float(row["test_sum_rate"]))
        assert float(row["test_sum_rate"]) > 0.0
```

- [ ] **Step 2: Run train tests and verify failure**

Run:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest tests/test_train.py -q
```

Expected: fail because `train_method` and `evaluate_checkpoints` are not defined.

- [ ] **Step 3: Add method registry and generic functions to `llm_fas/train.py`**

Update imports in `llm_fas/train.py`:

```python
from collections.abc import Mapping

from llm_fas.models import JointFASModelBase, ProposedLLMFASModel, TransformerBaselineModel
```

Add constants after imports:

```python
MODEL_REGISTRY = {
    "proposed": ProposedLLMFASModel,
    "transformer": TransformerBaselineModel,
}

RESULT_FIELDNAMES = [
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
]

EVALUATION_METHOD_ORDER = ["transformer", "proposed"]
```

Add this helper after `_tau_for_epoch()`:

```python
def build_model(method: str, cfg: ExperimentConfig) -> JointFASModelBase:
    try:
        model_cls = MODEL_REGISTRY[method]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported trainable method {method!r}. Supported methods: {supported}") from exc
    return model_cls(cfg)
```

Change `_evaluate_model_rate()` signature to accept the shared base:

```python
def _evaluate_model_rate(
    model: JointFASModelBase,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    tau: float,
    selection_mode: str,
) -> float:
```

Rename `train_proposed()` to `train_method()` and use the method registry:

```python
def train_method(cfg: ExperimentConfig, method: str = "proposed") -> Path:
    _set_seeds(cfg.seed)
    device = _select_device(cfg.device)
    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_H, val_H, _ = build_datasets(cfg)
    train_loader = make_loader(train_H, cfg.train.batch_size, shuffle=True, seed=cfg.seed)
    val_loader = make_loader(val_H, cfg.train.batch_size, shuffle=False, seed=cfg.seed + 1)

    model = build_model(method, cfg).to(device)
    optimizer = torch.optim.Adam((param for param in model.parameters() if param.requires_grad), lr=cfg.train.lr)
    history: list[dict[str, float | int]] = []

    for epoch in range(cfg.train.epochs):
        model.train()
        tau = _tau_for_epoch(cfg, epoch)
        total_loss = 0.0
        total_samples = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            out = model(batch, tau=tau, training=True)
            loss = -out["rate"].mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch + 1}: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.shape[0]
            total_samples += batch.shape[0]

        train_loss = total_loss / total_samples
        val_rate = _evaluate_model_rate(model, val_loader, device, tau, selection_mode="hard")
        val_loss = -val_rate
        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}: train={train_loss}, val={val_loss}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

    checkpoint_path = output_dir / f"{method}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    method_history_path = output_dir / f"{method}_train_history.csv"
    _write_history(method_history_path, history)
    if method == "proposed":
        _write_history(output_dir / "train_history.csv", history)
    return checkpoint_path


def train_proposed(cfg: ExperimentConfig) -> Path:
    return train_method(cfg, method="proposed")


def train_transformer(cfg: ExperimentConfig) -> Path:
    return train_method(cfg, method="transformer")
```

Replace `evaluate_checkpoint()` with generic evaluation plus the compatibility wrapper:

```python
def _ordered_methods(methods: list[str]) -> list[str]:
    order = {method: index for index, method in enumerate(EVALUATION_METHOD_ORDER)}
    return sorted(methods, key=lambda method: (order.get(method, len(order)), method))


def _write_results(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_checkpoints(cfg: ExperimentConfig, checkpoints: Mapping[str, str | Path]) -> Path:
    if not checkpoints:
        raise ValueError("At least one checkpoint must be provided")

    _set_seeds(cfg.seed)
    device = _select_device(cfg.device)
    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_H = build_datasets(cfg)
    test_loader = make_loader(test_H, cfg.train.batch_size, shuffle=False, seed=cfg.seed + 2)
    random_rate = evaluate_random_baseline(test_H.to(device), cfg, seed=cfg.seed + 3)

    rows: list[dict[str, float | int | str]] = [
        _result_row(cfg, method="random", selection_mode="hard", test_sum_rate=random_rate)
    ]
    tau = cfg.train.tau_min
    for method in _ordered_methods(list(checkpoints.keys())):
        checkpoint_path = Path(checkpoints[method])
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        model = build_model(method, cfg).to(device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        method_rate = _evaluate_model_rate(model, test_loader, device, tau, selection_mode="hard")
        rows.append(_result_row(cfg, method=method, selection_mode="hard", test_sum_rate=method_rate))

    results_path = output_dir / "results.csv"
    _write_results(results_path, rows)
    return results_path


def evaluate_checkpoint(cfg: ExperimentConfig, checkpoint_path: str | Path) -> Path:
    return evaluate_checkpoints(cfg, {"proposed": checkpoint_path})
```

Remove the old duplicated `evaluate_checkpoint()` implementation after the wrapper is in place.

- [ ] **Step 4: Run train tests**

Run:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest tests/test_train.py -q
```

Expected: all train tests pass.

- [ ] **Step 5: Run model and train tests together**

Run:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest tests/test_models.py tests/test_train.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit generic train/evaluation flow**

Run:

```bash
git add llm_fas/train.py tests/test_train.py
git commit -m "feat: generalize training for transformer baseline"
```

## Task 3: Multi-Method Experiment Runner And CLIs

**Files:**
- Modify: `tests/test_experiments.py`
- Modify: `llm_fas/experiments.py`
- Modify: `scripts/run_stage2_seeds.py`
- Create: `scripts/run_stage3_transformer.py`

- [ ] **Step 1: Write failing method parsing and orchestration tests**

Modify the import from `llm_fas.experiments` in `tests/test_experiments.py`:

```python
from llm_fas.experiments import (
    clone_config_for_seed,
    parse_method_list,
    parse_seed_list,
    run_seed_experiments,
    summarize_results,
    write_aggregate_outputs,
)
```

Append these tests:

```python
def test_parse_method_list_accepts_transformer_and_proposed():
    assert parse_method_list("transformer, proposed") == ["transformer", "proposed"]


def test_parse_method_list_rejects_random_as_trainable_method():
    with pytest.raises(ValueError, match="Unsupported method"):
        parse_method_list("random,proposed")


def test_run_seed_experiments_trains_multiple_methods_together(monkeypatch, tmp_path):
    calls: list[tuple[str, int, str]] = []

    def fake_train_method(cfg, method):
        calls.append(("train", cfg.seed, method))
        output_dir = Path(cfg.train.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = output_dir / f"{method}.pt"
        checkpoint.write_text(f"fake {method} checkpoint", encoding="utf-8")
        return checkpoint

    def fake_evaluate_checkpoints(cfg, checkpoints):
        calls.append(("evaluate", cfg.seed, ",".join(checkpoints.keys())))
        output_dir = Path(cfg.train.output_dir)
        result_path = output_dir / "results.csv"
        with result_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(_result("random", cfg.seed, 1.0).keys()))
            writer.writeheader()
            writer.writerow(_result("random", cfg.seed, 10.0 + cfg.seed))
            writer.writerow(_result("transformer", cfg.seed, 10.5 + cfg.seed))
            writer.writerow(_result("proposed", cfg.seed, 11.0 + cfg.seed))
        return result_path

    monkeypatch.setattr("llm_fas.experiments.train_method", fake_train_method)
    monkeypatch.setattr("llm_fas.experiments.evaluate_checkpoints", fake_evaluate_checkpoints)

    all_results_path, summary_path = run_seed_experiments(
        config_path="configs/mvp.yaml",
        seeds=[1, 2],
        output_root=tmp_path / "stage3",
        methods=["transformer", "proposed"],
    )

    assert calls == [
        ("train", 1, "transformer"),
        ("train", 1, "proposed"),
        ("evaluate", 1, "transformer,proposed"),
        ("train", 2, "transformer"),
        ("train", 2, "proposed"),
        ("evaluate", 2, "transformer,proposed"),
    ]
    assert all_results_path.exists()
    assert summary_path.exists()
    all_rows = list(csv.DictReader(all_results_path.open(newline="")))
    assert [row["method"] for row in all_rows] == [
        "random",
        "transformer",
        "proposed",
        "random",
        "transformer",
        "proposed",
    ]


def test_stage3_transformer_runner_cli_exposes_expected_arguments():
    help_result = subprocess.run(
        [sys.executable, "scripts/run_stage3_transformer.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert help_result.returncode == 0
    assert "--config" in help_result.stdout
    assert "--seeds" in help_result.stdout
    assert "--output-root" in help_result.stdout
    assert "--methods" in help_result.stdout
```

Update `test_stage2_seed_runner_cli_exposes_expected_arguments()` to also assert:

```python
    assert "--methods" in help_result.stdout
```

- [ ] **Step 2: Run experiment tests and verify failure**

Run:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest tests/test_experiments.py -q
```

Expected: fail because `parse_method_list`, `train_method`, `evaluate_checkpoints`, and `scripts/run_stage3_transformer.py` are not wired.

- [ ] **Step 3: Add method parsing and multi-method orchestration to `llm_fas/experiments.py`**

Add this function after `parse_seed_list()`:

```python
def parse_method_list(raw: str) -> list[str]:
    methods = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not methods:
        raise ValueError("At least one trainable method must be provided")
    supported = {"proposed", "transformer"}
    invalid = [method for method in methods if method not in supported]
    if invalid:
        raise ValueError(f"Unsupported method(s): {', '.join(invalid)}. Supported methods: proposed, transformer")
    return methods
```

Add lazy wrappers after the existing `evaluate_checkpoint()` wrapper:

```python
def train_method(cfg: ExperimentConfig, method: str) -> Path:
    from llm_fas.train import train_method as _train_method

    return _train_method(cfg, method=method)


def evaluate_checkpoints(cfg: ExperimentConfig, checkpoints: dict[str, str | Path]) -> Path:
    from llm_fas.train import evaluate_checkpoints as _evaluate_checkpoints

    return _evaluate_checkpoints(cfg, checkpoints)
```

Replace `run_seed_experiments()` with:

```python
def run_seed_experiments(
    config_path: str | Path,
    seeds: Iterable[int],
    output_root: str | Path,
    methods: Iterable[str] | None = None,
) -> tuple[Path, Path]:
    from llm_fas.config import load_config

    base_cfg = load_config(config_path)
    seed_values = [int(seed) for seed in seeds]
    if not seed_values:
        raise ValueError("At least one seed must be provided")
    method_values = list(methods) if methods is not None else ["proposed"]
    if not method_values:
        raise ValueError("At least one trainable method must be provided")

    all_rows: list[dict[str, str]] = []
    for seed in seed_values:
        cfg = clone_config_for_seed(base_cfg, seed, output_root)
        write_config_snapshot(cfg, cfg.train.output_dir)
        if method_values == ["proposed"]:
            checkpoint_path = train_proposed(cfg)
            result_path = evaluate_checkpoint(cfg, checkpoint_path)
        else:
            checkpoints = {method: train_method(cfg, method) for method in method_values}
            result_path = evaluate_checkpoints(cfg, checkpoints)
        all_rows.extend(read_result_rows(result_path))
    return write_aggregate_outputs(all_rows, output_root)
```

- [ ] **Step 4: Update Stage 2 CLI with optional methods**

Modify `scripts/run_stage2_seeds.py` imports:

```python
from llm_fas.experiments import parse_method_list, parse_seed_list, run_seed_experiments
```

Add this parser argument after `--output-root`:

```python
    parser.add_argument(
        "--methods",
        default="proposed",
        help="Comma-separated trainable methods. Supported: proposed, transformer.",
    )
```

Parse methods next to seeds:

```python
    try:
        seeds = parse_seed_list(args.seeds)
        methods = parse_method_list(args.methods)
    except ValueError as exc:
        parser.error(str(exc))
    all_results_path, summary_path = run_seed_experiments(args.config, seeds, args.output_root, methods=methods)
```

- [ ] **Step 5: Create Stage 3 CLI**

Create `scripts/run_stage3_transformer.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_fas.experiments import parse_method_list, parse_seed_list, run_seed_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 3 Transformer baseline experiments over multiple seeds.")
    parser.add_argument("--config", default="configs/mvp.yaml", help="Path to the base experiment config.")
    parser.add_argument(
        "--seeds",
        default="20260606",
        help="Comma-separated integer seeds.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/stage3_transformer_smoke",
        help="Directory for per-seed outputs and aggregate CSVs.",
    )
    parser.add_argument(
        "--methods",
        default="transformer,proposed",
        help="Comma-separated trainable methods. Supported: proposed, transformer.",
    )
    args = parser.parse_args()

    try:
        seeds = parse_seed_list(args.seeds)
        methods = parse_method_list(args.methods)
    except ValueError as exc:
        parser.error(str(exc))
    all_results_path, summary_path = run_seed_experiments(args.config, seeds, args.output_root, methods=methods)
    print(all_results_path)
    print(summary_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run experiment tests**

Run:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest tests/test_experiments.py -q
```

Expected: all experiment tests pass.

- [ ] **Step 7: Run CLI help commands**

Run:

```bash
uv run --with torch --with numpy --with pyyaml --with transformers --with peft python scripts/run_stage2_seeds.py --help
uv run --with torch --with numpy --with pyyaml --with transformers --with peft python scripts/run_stage3_transformer.py --help
```

Expected: both commands exit `0`; both print `--methods`.

- [ ] **Step 8: Commit multi-method runner**

Run:

```bash
git add llm_fas/experiments.py scripts/run_stage2_seeds.py scripts/run_stage3_transformer.py tests/test_experiments.py
git commit -m "feat: run multi-method seed experiments"
```

## Task 4: Full Test Verification

**Files:**
- No source edits expected.

- [ ] **Step 1: Run the full test suite**

Run:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Inspect git status**

Run:

```bash
git status --short --branch
```

Expected: only intentional source/test/script changes are present. Existing local edits under `outputs/stage2_mvp_stability/` must remain untouched unless the user explicitly asks to refresh Stage 2 artifacts.

## Task 5: Stage 3 Smoke Run And Chinese Summary

**Files:**
- Create after execution: `outputs/stage3_transformer_smoke/all_results.csv`
- Create after execution: `outputs/stage3_transformer_smoke/summary.csv`
- Create after execution: `docs/STAGE3_TRANSFORMER_BASELINE_SUMMARY.md`

- [ ] **Step 1: Run one-seed Stage 3 smoke experiment**

Run:

```bash
uv run --with torch --with numpy --with pyyaml --with transformers --with peft python scripts/run_stage3_transformer.py --config configs/mvp.yaml --seeds 20260606 --output-root outputs/stage3_transformer_smoke --methods transformer,proposed
```

Expected:

- `outputs/stage3_transformer_smoke/seed_20260606/transformer.pt` exists but remains ignored by git.
- `outputs/stage3_transformer_smoke/seed_20260606/proposed.pt` exists but remains ignored by git.
- `outputs/stage3_transformer_smoke/seed_20260606/results.csv` exists.
- `outputs/stage3_transformer_smoke/all_results.csv` exists.
- `outputs/stage3_transformer_smoke/summary.csv` exists.
- `all_results.csv` contains methods `random`, `transformer`, and `proposed`.

- [ ] **Step 2: Inspect smoke outputs**

Run:

```bash
python -c "import csv; rows=list(csv.DictReader(open('outputs/stage3_transformer_smoke/all_results.csv', newline=''))); print([(r['method'], r['selection_mode'], r['test_sum_rate']) for r in rows]); print(list(csv.DictReader(open('outputs/stage3_transformer_smoke/summary.csv', newline=''))))"
```

Expected:

- The first printed list has exactly three rows.
- Method order is `random`, `transformer`, `proposed`.
- Each `test_sum_rate` parses as a positive finite float.
- Summary contains rows for `random`, `transformer`, and `proposed`.

- [ ] **Step 3: Write Chinese Stage 3 summary document**

Create `docs/STAGE3_TRANSFORMER_BASELINE_SUMMARY.md` with this structure and numeric values copied from the smoke output:

```markdown
# Stage 3：Transformer Baseline 接入总结

## 阶段目标

本阶段目标是将论文中的 Transformer baseline 接入现有 MVP 复现实验链路，使其与 Proposed 共用同一套 CSI 预处理、端口选择、功率因子预测、hard inference、beamforming 和 CSV schema。

## 实现范围

- 新增 `TransformerBaselineModel`。
- 保留 `ProposedLLMFASModel` 的 GPT-2 + LoRA backbone。
- 抽出共享 joint FAS 模型前后处理逻辑。
- 新增 `train_method()` 和 `evaluate_checkpoints()`，支持在同一个 `results.csv` 中写入 `random`、`transformer`、`proposed`。
- 新增 Stage 3 CLI：`scripts/run_stage3_transformer.py`。

## Smoke 实验设置

| 项目 | 值 |
|---|---:|
| Config | `configs/mvp.yaml` |
| Seeds | `20260606` |
| Methods | `transformer, proposed` |
| Selection mode | `hard` |
| Output root | `outputs/stage3_transformer_smoke` |

## Smoke 结果

| Method | Selection mode | Test sum rate |
|---|---|---:|
| random | hard | 填入 `all_results.csv` 中的数值 |
| transformer | hard | 填入 `all_results.csv` 中的数值 |
| proposed | hard | 填入 `all_results.csv` 中的数值 |

## 结论

本阶段完成的是软件链路验证：Transformer baseline 已能训练、保存 checkpoint，并与 Random/Proposed 共同写入同 schema 结果。由于 Smoke 只运行 1 个 seed，本结果不能作为方法优劣结论。下一步应运行至少 3 个 seed 的 Stage 3 比较，再判断 Transformer 与 Proposed 的稳定差异。

## 后续动作

- 运行 `scripts/run_stage3_transformer.py --seeds 20260606,20260607,20260608` 获得三 seed 结果。
- 若 Proposed 仍不稳定优于 Random 或 Transformer，优先诊断端口选择分布、训练动态和模型容量，再进入论文完整配置。
```

- [ ] **Step 4: Run full tests after generated artifacts**

Run:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Stage 3 smoke artifacts**

Run:

```bash
git add outputs/stage3_transformer_smoke/all_results.csv outputs/stage3_transformer_smoke/summary.csv docs/STAGE3_TRANSFORMER_BASELINE_SUMMARY.md
git commit -m "exp: record stage 3 transformer smoke results"
```

Do not commit `outputs/stage3_transformer_smoke/**/*.pt`.

## Task 6: Optional Three-Seed Stage 3 Comparison

**Files:**
- Create after execution: `outputs/stage3_transformer_3seed/all_results.csv`
- Create after execution: `outputs/stage3_transformer_3seed/summary.csv`

- [ ] **Step 1: Run three-seed Stage 3 comparison if runtime is acceptable**

Run:

```bash
uv run --with torch --with numpy --with pyyaml --with transformers --with peft python scripts/run_stage3_transformer.py --config configs/mvp.yaml --seeds 20260606,20260607,20260608 --output-root outputs/stage3_transformer_3seed --methods transformer,proposed
```

Expected:

- `outputs/stage3_transformer_3seed/all_results.csv` contains 9 rows: 3 seeds times `random`, `transformer`, `proposed`.
- `outputs/stage3_transformer_3seed/summary.csv` contains rows for all three methods.

- [ ] **Step 2: Commit three-seed CSVs when complete**

Run:

```bash
git add outputs/stage3_transformer_3seed/all_results.csv outputs/stage3_transformer_3seed/summary.csv
git commit -m "exp: record stage 3 transformer multi-seed results"
```

## Self-Review

- Spec coverage: This plan implements Stage 3 from the approved reproduction design: Transformer baseline, shared preprocessing and heads, same hard inference, one CSV with `random`, `transformer`, and `proposed`, and identical result schema.
- Placeholder scan: The implementation steps define exact tests, functions, CLIs, commands, expected outputs, and commit commands. The Stage 3 summary document includes explicit table locations for values that can only exist after the smoke run creates CSV output.
- Type consistency: `train_method(cfg, method) -> Path`; `evaluate_checkpoints(cfg, checkpoints) -> Path`; `run_seed_experiments(..., methods=None) -> tuple[Path, Path]`; `parse_method_list(raw) -> list[str]`; model forward outputs match the existing `dict[str, torch.Tensor | str]` contract.
- Backward compatibility: `train_proposed()` and `evaluate_checkpoint()` stay available for `scripts/train_mvp.py`, `scripts/evaluate_mvp.py`, Stage 1 tests, and Stage 2 default runner behavior.
- Risk: The one-seed smoke run may take longer than a unit test because it trains both Transformer and Proposed. If runtime is too high, complete Tasks 1-4 first, then run Task 5 when compute is available; do not claim Stage 3 experimental completion until the smoke CSV and summary document exist.
