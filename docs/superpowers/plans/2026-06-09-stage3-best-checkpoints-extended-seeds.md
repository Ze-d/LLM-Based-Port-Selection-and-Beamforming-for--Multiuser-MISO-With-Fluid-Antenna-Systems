# Stage 3 Best Checkpoints And Extended Seeds Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve Stage 3 reliability by evaluating best-validation checkpoints and adding a convenient 5/10 seed runner.

**Architecture:** Keep the existing model and evaluation contracts unchanged. Update `train_method()` so `{method}.pt` always contains the best hard-validation checkpoint, and add a thin Stage 3 extended CLI that expands deterministic seed ranges before calling the existing experiment runner.

**Tech Stack:** Python 3.12, PyTorch, Transformers, PEFT, PyYAML, pytest, argparse, pathlib.

---

## Scope

This enhancement is intentionally narrow:

- Save best-validation checkpoints for trainable methods.
- Preserve existing checkpoint names: `proposed.pt` and `transformer.pt`.
- Preserve existing history CSV schema: `epoch,train_loss,val_loss`.
- Add a CLI for 5/10 seed Stage 3 comparison without automatically starting long training.
- Do not modify existing result CSVs under `outputs/stage2_mvp_stability`, `outputs/stage3_transformer_smoke`, or `outputs/stage3_transformer_3seed`.

## Files

- Modify: `llm_fas/train.py`
- Modify: `tests/test_train.py`
- Modify: `tests/test_experiments.py`
- Create: `scripts/run_stage3_extended.py`

## Task 1: Save Best-Validation Checkpoints

**Files:**
- Modify: `tests/test_train.py`
- Modify: `llm_fas/train.py`

- [ ] **Step 1: Write failing test for best checkpoint saves**

Append to `tests/test_train.py`:

```python
def test_train_method_saves_checkpoint_each_time_validation_improves(monkeypatch, tmp_path):
    cfg = replace(_tiny_cfg(tmp_path), train=replace(_tiny_cfg(tmp_path).train, epochs=2))
    val_rates = iter([10.0, 11.0])
    saved_paths: list[Path] = []

    def fake_evaluate_model_rate(*args, **kwargs):
        return next(val_rates)

    def fake_save(state_dict, path):
        saved_paths.append(Path(path))
        Path(path).write_text("checkpoint", encoding="utf-8")

    monkeypatch.setattr("llm_fas.train._evaluate_model_rate", fake_evaluate_model_rate)
    monkeypatch.setattr("llm_fas.train.torch.save", fake_save)

    checkpoint_path = train_method(cfg, method="transformer")

    assert checkpoint_path == Path(cfg.train.output_dir) / "transformer.pt"
    assert saved_paths == [checkpoint_path, checkpoint_path]
    rows = list(csv.DictReader((Path(cfg.train.output_dir) / "transformer_train_history.csv").open(newline="")))
    assert [float(row["val_loss"]) for row in rows] == [-10.0, -11.0]
```

- [ ] **Step 2: Run the new test and confirm failure**

Run:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest tests/test_train.py::test_train_method_saves_checkpoint_each_time_validation_improves -q
```

Expected: fail because current code saves only once after the loop.

- [ ] **Step 3: Implement best checkpoint saving**

In `llm_fas/train.py`, define `checkpoint_path = output_dir / f"{method}.pt"` before the epoch loop. Track `best_val_rate = -float("inf")`. After computing finite `val_rate`, save the model when `val_rate > best_val_rate`.

The training function should still write history at the end and return `checkpoint_path`.

- [ ] **Step 4: Run train tests**

Run:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest tests/test_train.py -q
```

Expected: all train tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add llm_fas/train.py tests/test_train.py
git commit -m "feat: save best validation checkpoints"
```

## Task 2: Stage 3 Extended Seed Runner

**Files:**
- Modify: `tests/test_experiments.py`
- Create: `scripts/run_stage3_extended.py`

- [ ] **Step 1: Write failing CLI tests**

Append to `tests/test_experiments.py`:

```python
def test_stage3_extended_runner_cli_exposes_seed_count():
    help_result = subprocess.run(
        [sys.executable, "scripts/run_stage3_extended.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert help_result.returncode == 0
    assert "--seed-count" in help_result.stdout
    assert "--seeds" in help_result.stdout
    assert "--output-root" in help_result.stdout
    assert "--methods" in help_result.stdout


def test_stage3_extended_runner_seed_count_dry_run():
    dry_result = subprocess.run(
        [sys.executable, "scripts/run_stage3_extended.py", "--seed-count", "5", "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert dry_result.returncode == 0
    assert "20260606,20260607,20260608,20260609,20260610" in dry_result.stdout
    assert "outputs/stage3_transformer_5seed" in dry_result.stdout
```

- [ ] **Step 2: Run the new CLI tests and confirm failure**

Run:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest tests/test_experiments.py::test_stage3_extended_runner_cli_exposes_seed_count tests/test_experiments.py::test_stage3_extended_runner_seed_count_dry_run -q
```

Expected: fail because `scripts/run_stage3_extended.py` does not exist.

- [ ] **Step 3: Create `scripts/run_stage3_extended.py`**

Create a CLI with these behaviors:

- `--config` default `configs/mvp.yaml`
- `--seed-start` default `20260606`
- `--seed-count` default `5`, choices `5` or `10`
- `--seeds` optional comma-separated override
- `--output-root` optional; if omitted, use `outputs/stage3_transformer_{seed_count}seed`
- `--methods` default `transformer,proposed`
- `--dry-run` prints resolved seeds, methods, output root, and config, then exits without training
- normal execution calls `run_seed_experiments()`

- [ ] **Step 4: Run experiment tests**

Run:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest tests/test_experiments.py -q
```

Expected: all experiment tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add scripts/run_stage3_extended.py tests/test_experiments.py
git commit -m "feat: add stage 3 extended seed runner"
```

## Task 3: Final Verification

**Files:**
- No source edits expected.

- [ ] **Step 1: Run full test suite**

Run:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run CLI dry-runs**

Run:

```bash
uv run --with torch --with numpy --with pyyaml --with transformers --with peft python scripts/run_stage3_extended.py --seed-count 5 --dry-run
uv run --with torch --with numpy --with pyyaml --with transformers --with peft python scripts/run_stage3_extended.py --seed-count 10 --dry-run
```

Expected: commands exit `0`; 5-seed output resolves to `20260606..20260610`; 10-seed output resolves to `20260606..20260615`.

## Self-Review

- Spec coverage: Implements best-validation checkpoint saving and 5/10 seed runner convenience.
- Backward compatibility: Existing `train_proposed()`, `evaluate_checkpoint()`, result schema, history schema, and Stage 3 runner remain compatible.
- Long-running experiments: This plan intentionally adds commands but does not run 5/10 seed training automatically.
