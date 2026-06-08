# Stage 2 MVP Stability Runs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible multi-seed runner for the current MVP+ configuration and record whether hard-inference Proposed is stable against Random.

**Architecture:** Introduce a focused `llm_fas.experiments` module for seed expansion, per-seed output directory management, config snapshots, result aggregation, and markdown summary generation. Add `scripts/run_stage2_seeds.py` as a thin CLI wrapper. Keep existing `train_proposed` and `evaluate_checkpoint` unchanged except for using them from the experiment layer.

**Tech Stack:** Python 3.12, PyTorch, Transformers, PEFT, PyYAML, pytest, stdlib `csv`, `statistics`, `dataclasses`, and `argparse`.

---

## Scope

Stage 2 keeps the current MVP+ computational scale:

- `train_samples=1000`
- `val_samples=200`
- `test_samples=200`
- `epochs=20`
- GPT-2 layers `2`
- `d_mha=128`
- `batch_size=16`
- hard inference for validation and test evaluation

The runner must execute the same config over a seed list without editing `configs/mvp.yaml`. Each seed writes to its own output directory. Aggregated CSVs live under the Stage 2 output root.

## File Structure

- Create: `llm_fas/experiments.py` - seed parsing, config cloning, config snapshots, CSV aggregation, markdown summary generation, and seed-run orchestration.
- Create: `scripts/run_stage2_seeds.py` - CLI for running Stage 2 seeds.
- Create: `tests/test_experiments.py` - fast tests for pure helpers and runner orchestration with monkeypatched training/evaluation.
- Create after execution: `outputs/stage2_mvp_stability/all_results.csv`
- Create after execution: `outputs/stage2_mvp_stability/summary.csv`
- Create after execution: `docs/STAGE2_MVP_STABILITY_SUMMARY.md`

## Task 1: Experiment Helper Module

**Files:**
- Create: `tests/test_experiments.py`
- Create: `llm_fas/experiments.py`

- [ ] **Step 1: Write failing tests for seed parsing and config cloning**

Create `tests/test_experiments.py` with:

```python
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from llm_fas.config import load_config
from llm_fas.experiments import clone_config_for_seed, parse_seed_list


def test_parse_seed_list_accepts_comma_separated_values():
    assert parse_seed_list("20260606, 20260607,20260608") == [20260606, 20260607, 20260608]


def test_parse_seed_list_rejects_empty_input():
    with pytest.raises(ValueError, match="At least one seed"):
        parse_seed_list(" , ")


def test_clone_config_for_seed_changes_seed_and_output_dir(tmp_path):
    cfg = load_config("configs/mvp.yaml")
    cloned = clone_config_for_seed(cfg, seed=123, output_root=tmp_path)

    assert cloned.seed == 123
    assert cloned.train.output_dir == str(tmp_path / "seed_123")
    assert cfg.seed != cloned.seed
    assert cfg.train.output_dir != cloned.train.output_dir
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest tests/test_experiments.py -q
```

Expected: fail because `llm_fas.experiments` does not exist.

- [ ] **Step 3: Implement parsing and config cloning**

Create `llm_fas/experiments.py` with:

```python
from __future__ import annotations

import csv
import statistics
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

import yaml

from llm_fas.config import ExperimentConfig
from llm_fas.train import evaluate_checkpoint, train_proposed


def parse_seed_list(raw: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise ValueError("At least one seed must be provided")
    return seeds


def clone_config_for_seed(cfg: ExperimentConfig, seed: int, output_root: str | Path) -> ExperimentConfig:
    seed_dir = Path(output_root) / f"seed_{seed}"
    return replace(cfg, seed=seed, train=replace(cfg.train, output_dir=str(seed_dir)))
```

- [ ] **Step 4: Run tests**

Run:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest tests/test_experiments.py -q
```

Expected: the first three tests pass.

## Task 2: Aggregation And Summary Helpers

**Files:**
- Modify: `tests/test_experiments.py`
- Modify: `llm_fas/experiments.py`

- [ ] **Step 1: Write failing tests for result aggregation**

Append to `tests/test_experiments.py`:

```python
from llm_fas.experiments import summarize_results, write_aggregate_outputs


def _result(method: str, seed: int, rate: float) -> dict[str, str]:
    return {
        "method": method,
        "selection_mode": "hard",
        "K": "3",
        "Nx": "4",
        "Ny": "4",
        "N": "16",
        "n_active": "4",
        "W_lambda_x": "2.0",
        "W_lambda_y": "2.0",
        "Pmax_dBm": "20.0",
        "distance_km": "0.2",
        "seed": str(seed),
        "train_samples": "1000",
        "val_samples": "200",
        "test_samples": "200",
        "epochs": "20",
        "batch_size": "16",
        "gpt2_layers": "2",
        "d_mha": "128",
        "mha_heads": "4",
        "test_sum_rate": str(rate),
    }


def test_summarize_results_reports_mean_std_and_win_count():
    rows = [
        _result("random", 1, 10.0),
        _result("proposed", 1, 11.0),
        _result("random", 2, 12.0),
        _result("proposed", 2, 11.5),
        _result("random", 3, 9.0),
        _result("proposed", 3, 9.5),
    ]

    summary = summarize_results(rows)

    proposed = next(row for row in summary if row["method"] == "proposed")
    random = next(row for row in summary if row["method"] == "random")
    assert proposed["num_seeds"] == 3
    assert proposed["mean_test_sum_rate"] == pytest.approx((11.0 + 11.5 + 9.5) / 3.0)
    assert proposed["std_test_sum_rate"] == pytest.approx(statistics.stdev([11.0, 11.5, 9.5]))
    assert proposed["wins_vs_random"] == 2
    assert random["wins_vs_random"] == 0


def test_write_aggregate_outputs_writes_all_results_and_summary(tmp_path):
    rows = [
        _result("random", 1, 10.0),
        _result("proposed", 1, 11.0),
        _result("random", 2, 12.0),
        _result("proposed", 2, 11.5),
    ]

    all_results_path, summary_path = write_aggregate_outputs(rows, tmp_path)

    assert all_results_path == tmp_path / "all_results.csv"
    assert summary_path == tmp_path / "summary.csv"
    assert all_results_path.exists()
    assert summary_path.exists()
    all_rows = list(csv.DictReader(all_results_path.open(newline="")))
    summary_rows = list(csv.DictReader(summary_path.open(newline="")))
    assert len(all_rows) == 4
    assert {row["method"] for row in summary_rows} == {"random", "proposed"}
```

Also add `import statistics` near the top of `tests/test_experiments.py`.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest tests/test_experiments.py -q
```

Expected: fail because `summarize_results` and `write_aggregate_outputs` do not exist.

- [ ] **Step 3: Implement aggregation helpers**

Append to `llm_fas/experiments.py`:

```python
SUMMARY_FIELDNAMES = [
    "method",
    "selection_mode",
    "num_seeds",
    "mean_test_sum_rate",
    "std_test_sum_rate",
    "min_test_sum_rate",
    "max_test_sum_rate",
    "wins_vs_random",
]


def _sorted_fieldnames(rows: list[dict[str, str]]) -> list[str]:
    preferred = [
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
    keys = set().union(*(row.keys() for row in rows))
    return [key for key in preferred if key in keys] + sorted(keys - set(preferred))


def summarize_results(rows: list[dict[str, str]]) -> list[dict[str, float | int | str]]:
    by_method: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)

    random_by_seed = {
        int(row["seed"]): float(row["test_sum_rate"])
        for row in by_method.get("random", [])
    }
    summary: list[dict[str, float | int | str]] = []
    for method in sorted(by_method):
        method_rows = by_method[method]
        rates = [float(row["test_sum_rate"]) for row in method_rows]
        wins = 0
        if method != "random":
            for row in method_rows:
                seed = int(row["seed"])
                if seed in random_by_seed and float(row["test_sum_rate"]) > random_by_seed[seed]:
                    wins += 1
        summary.append(
            {
                "method": method,
                "selection_mode": method_rows[0].get("selection_mode", ""),
                "num_seeds": len(rates),
                "mean_test_sum_rate": statistics.mean(rates),
                "std_test_sum_rate": statistics.stdev(rates) if len(rates) > 1 else 0.0,
                "min_test_sum_rate": min(rates),
                "max_test_sum_rate": max(rates),
                "wins_vs_random": wins,
            }
        )
    return summary


def write_aggregate_outputs(rows: list[dict[str, str]], output_root: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results_path = output_dir / "all_results.csv"
    summary_path = output_dir / "summary.csv"

    with all_results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_sorted_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(summarize_results(rows))

    return all_results_path, summary_path
```

- [ ] **Step 4: Run tests**

Run:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest tests/test_experiments.py -q
```

Expected: aggregation tests pass.

## Task 3: Seed Runner Orchestration

**Files:**
- Modify: `tests/test_experiments.py`
- Modify: `llm_fas/experiments.py`

- [ ] **Step 1: Write failing orchestration test**

Append to `tests/test_experiments.py`:

```python
from llm_fas.experiments import run_seed_experiments


def test_run_seed_experiments_trains_evaluates_and_aggregates(monkeypatch, tmp_path):
    calls: list[tuple[str, int, str]] = []

    def fake_train(cfg):
        calls.append(("train", cfg.seed, cfg.train.output_dir))
        output_dir = Path(cfg.train.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = output_dir / "proposed.pt"
        checkpoint.write_text("fake checkpoint", encoding="utf-8")
        return checkpoint

    def fake_evaluate(cfg, checkpoint_path):
        calls.append(("evaluate", cfg.seed, str(checkpoint_path)))
        output_dir = Path(cfg.train.output_dir)
        result_path = output_dir / "results.csv"
        with result_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(_result("random", cfg.seed, 1.0).keys()))
            writer.writeheader()
            writer.writerow(_result("random", cfg.seed, 10.0 + cfg.seed))
            writer.writerow(_result("proposed", cfg.seed, 11.0 + cfg.seed))
        return result_path

    monkeypatch.setattr("llm_fas.experiments.train_proposed", fake_train)
    monkeypatch.setattr("llm_fas.experiments.evaluate_checkpoint", fake_evaluate)

    all_results_path, summary_path = run_seed_experiments(
        config_path="configs/mvp.yaml",
        seeds=[1, 2],
        output_root=tmp_path / "stage2",
    )

    assert calls == [
        ("train", 1, str(tmp_path / "stage2" / "seed_1")),
        ("evaluate", 1, str(tmp_path / "stage2" / "seed_1" / "proposed.pt")),
        ("train", 2, str(tmp_path / "stage2" / "seed_2")),
        ("evaluate", 2, str(tmp_path / "stage2" / "seed_2" / "proposed.pt")),
    ]
    assert all_results_path.exists()
    assert summary_path.exists()
    assert (tmp_path / "stage2" / "seed_1" / "config_snapshot.yaml").exists()
    assert (tmp_path / "stage2" / "seed_2" / "config_snapshot.yaml").exists()
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest tests/test_experiments.py -q
```

Expected: fail because `run_seed_experiments` does not exist.

- [ ] **Step 3: Implement config snapshots and runner**

Append to `llm_fas/experiments.py`:

```python
def write_config_snapshot(cfg: ExperimentConfig, output_dir: str | Path) -> Path:
    path = Path(output_dir) / "config_snapshot.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)
    return path


def read_result_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_seed_experiments(
    config_path: str | Path,
    seeds: Iterable[int],
    output_root: str | Path,
) -> tuple[Path, Path]:
    from llm_fas.config import load_config

    base_cfg = load_config(config_path)
    all_rows: list[dict[str, str]] = []
    for seed in seeds:
        cfg = clone_config_for_seed(base_cfg, int(seed), output_root)
        write_config_snapshot(cfg, cfg.train.output_dir)
        checkpoint_path = train_proposed(cfg)
        result_path = evaluate_checkpoint(cfg, checkpoint_path)
        all_rows.extend(read_result_rows(result_path))
    return write_aggregate_outputs(all_rows, output_root)
```

- [ ] **Step 4: Run experiment helper tests**

Run:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest tests/test_experiments.py -q
```

Expected: all experiment helper tests pass.

- [ ] **Step 5: Run relevant existing tests**

Run:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest tests/test_train.py tests/test_experiments.py -q
```

Expected: train and experiment tests pass.

- [ ] **Step 6: Commit helper module**

Run:

```bash
git add llm_fas/experiments.py tests/test_experiments.py
git commit -m "feat: add stage 2 seed experiment helpers"
```

## Task 4: Stage 2 CLI

**Files:**
- Modify: `tests/test_experiments.py`
- Create: `scripts/run_stage2_seeds.py`

- [ ] **Step 1: Write failing CLI help test**

Append to `tests/test_experiments.py`:

```python
import subprocess
import sys


def test_stage2_seed_runner_cli_exposes_expected_arguments():
    help_result = subprocess.run(
        [sys.executable, "scripts/run_stage2_seeds.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert help_result.returncode == 0
    assert "--config" in help_result.stdout
    assert "--seeds" in help_result.stdout
    assert "--output-root" in help_result.stdout
```

- [ ] **Step 2: Run CLI test and verify failure**

Run:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest tests/test_experiments.py::test_stage2_seed_runner_cli_exposes_expected_arguments -q
```

Expected: fail because `scripts/run_stage2_seeds.py` does not exist.

- [ ] **Step 3: Implement CLI script**

Create `scripts/run_stage2_seeds.py`:

```python
from __future__ import annotations

import argparse

from llm_fas.experiments import parse_seed_list, run_seed_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 2 MVP stability experiments over multiple seeds.")
    parser.add_argument("--config", default="configs/mvp.yaml", help="Path to the base experiment config.")
    parser.add_argument(
        "--seeds",
        default="20260606,20260607,20260608",
        help="Comma-separated integer seeds.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/stage2_mvp_stability",
        help="Directory for per-seed outputs and aggregate CSVs.",
    )
    args = parser.parse_args()

    seeds = parse_seed_list(args.seeds)
    all_results_path, summary_path = run_seed_experiments(args.config, seeds, args.output_root)
    print(all_results_path)
    print(summary_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run CLI help test**

Run:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest tests/test_experiments.py::test_stage2_seed_runner_cli_exposes_expected_arguments -q
```

Expected: CLI help test passes.

- [ ] **Step 5: Run all experiment tests**

Run:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest tests/test_experiments.py -q
```

Expected: all experiment tests pass.

- [ ] **Step 6: Commit CLI**

Run:

```bash
git add scripts/run_stage2_seeds.py tests/test_experiments.py
git commit -m "feat: add stage 2 seed runner cli"
```

## Task 5: Full Verification

**Files:**
- No source file edits expected.

- [ ] **Step 1: Run full test suite**

Run:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run CLI help command**

Run:

```bash
uv run --with torch --with numpy --with pyyaml --with transformers --with peft python scripts/run_stage2_seeds.py --help
```

Expected: command exits `0` and prints `--config`, `--seeds`, and `--output-root`.

## Task 6: Stage 2 Real Seed Runs And Summary

**Files:**
- Create after execution: `outputs/stage2_mvp_stability/all_results.csv`
- Create after execution: `outputs/stage2_mvp_stability/summary.csv`
- Create after execution: `docs/STAGE2_MVP_STABILITY_SUMMARY.md`

- [ ] **Step 1: Run the three-seed Stage 2 experiment**

Run:

```bash
uv run --with torch --with numpy --with pyyaml --with transformers --with peft python scripts/run_stage2_seeds.py --config configs/mvp.yaml --seeds 20260606,20260607,20260608 --output-root outputs/stage2_mvp_stability
```

Expected:

- `outputs/stage2_mvp_stability/seed_20260606/results.csv` exists.
- `outputs/stage2_mvp_stability/seed_20260607/results.csv` exists.
- `outputs/stage2_mvp_stability/seed_20260608/results.csv` exists.
- `outputs/stage2_mvp_stability/all_results.csv` exists.
- `outputs/stage2_mvp_stability/summary.csv` exists.

- [ ] **Step 2: Inspect aggregate outputs**

Run:

```bash
python -c "import csv; print(list(csv.DictReader(open('outputs/stage2_mvp_stability/summary.csv', newline='')))); print(len(list(csv.DictReader(open('outputs/stage2_mvp_stability/all_results.csv', newline='')))))"
```

Expected:

- Summary contains rows for `random` and `proposed`.
- `all_results.csv` has `6` rows.

- [ ] **Step 3: Write Chinese Stage 2 summary document**

Create `docs/STAGE2_MVP_STABILITY_SUMMARY.md` with this structure, filling the numeric values from `summary.csv` and `all_results.csv`:

```markdown
# Stage 2：MVP+ 多 Seed 稳定性实验总结

## 阶段目标

本阶段目标是在不扩大模型和数据规模的前提下，验证 hard-inference Proposed 相对 Random 的结果是否具有 seed 稳定性。

## 实验设置

| 项目 | 值 |
|---|---:|
| Seeds | `20260606, 20260607, 20260608` |
| Train samples | `1000` |
| Val samples | `200` |
| Test samples | `200` |
| Epochs | `20` |
| Batch size | `16` |
| GPT-2 layers | `2` |
| d_mha | `128` |
| Selection mode | `hard` |

## 结果文件

- `outputs/stage2_mvp_stability/all_results.csv`
- `outputs/stage2_mvp_stability/summary.csv`

## 聚合结果

在这里填写 Random 和 Proposed 的 mean/std/min/max/wins_vs_random。

## 结论

如果 Proposed 的 `wins_vs_random` 等于 `3`，写明当前 MVP+ 设置下 hard Proposed 在 3 个 seed 上均优于 Random，可以进入 Transformer baseline。

如果 Proposed 的 `wins_vs_random` 小于 `3`，写明当前结果不稳定，下一步应优先诊断端口选择、训练动态或模型容量，而不是直接扩大到论文完整配置。

## 后续动作

根据本阶段结果决定进入 Stage 3 或先做诊断。
```

- [ ] **Step 4: Run final test suite after generated artifacts**

Run:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Stage 2 artifacts**

If the three-seed run completes, run:

```bash
git add outputs/stage2_mvp_stability/all_results.csv outputs/stage2_mvp_stability/summary.csv docs/STAGE2_MVP_STABILITY_SUMMARY.md
git commit -m "exp: record stage 2 mvp stability results"
```

Do not commit `outputs/stage2_mvp_stability/**/proposed.pt`.

## Self-Review

- Spec coverage: The plan covers Stage 2 from the approved design: multi-seed execution without manually editing `configs/mvp.yaml`, per-seed outputs, aggregate CSVs with mean/std, and a document recording whether hard Proposed is consistently better than Random.
- Placeholder scan: The only template-like section is the explicit markdown content to create after real numeric outputs exist; it does not contain undefined implementation steps and gives exact branches for the conclusion based on `wins_vs_random`.
- Type consistency: `parse_seed_list` returns `list[int]`; `clone_config_for_seed` returns `ExperimentConfig`; `run_seed_experiments` returns `(all_results_path, summary_path)` as `tuple[Path, Path]`; result rows are read as `dict[str, str]`; summary rows use `dict[str, float | int | str]`.
- Risk: The real three-seed run may take roughly three times the Stage 1 smoke training time. If runtime is too high, keep code/tests committed and run the seeds later, but do not mark Stage 2 experimentally complete until `summary.csv` and `docs/STAGE2_MVP_STABILITY_SUMMARY.md` exist.
