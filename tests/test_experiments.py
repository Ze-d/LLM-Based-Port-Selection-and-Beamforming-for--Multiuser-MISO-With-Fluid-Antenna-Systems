from __future__ import annotations

import csv
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

from llm_fas.config import load_config
from llm_fas.experiments import (
    clone_config_for_seed,
    parse_method_list,
    parse_seed_list,
    run_seed_experiments,
    summarize_results,
    write_aggregate_outputs,
)


def test_parse_seed_list_accepts_comma_separated_values():
    assert parse_seed_list("20260606, 20260607,20260608") == [20260606, 20260607, 20260608]


def test_parse_seed_list_rejects_empty_input():
    with pytest.raises(ValueError, match="At least one seed"):
        parse_seed_list(" , ")


def test_parse_method_list_accepts_transformer_and_proposed():
    assert parse_method_list("transformer, proposed") == ["transformer", "proposed"]


def test_parse_method_list_rejects_random_as_trainable_method():
    with pytest.raises(ValueError, match="Unsupported method"):
        parse_method_list("random,proposed")


def test_clone_config_for_seed_changes_seed_and_output_dir(tmp_path):
    cfg = load_config("configs/mvp.yaml")
    cloned = clone_config_for_seed(cfg, seed=123, output_root=tmp_path)

    assert cloned.seed == 123
    assert cloned.train.output_dir == str(tmp_path / "seed_123")
    assert cfg.seed != cloned.seed
    assert cfg.train.output_dir != cloned.train.output_dir


def _result(method: str, seed: int, rate: float, selection_mode: str = "hard") -> dict[str, str]:
    return {
        "method": method,
        "selection_mode": selection_mode,
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


def test_summarize_results_keeps_selection_modes_separate_for_wins():
    rows = [
        _result("random", 1, 10.0, selection_mode="hard"),
        _result("proposed", 1, 11.0, selection_mode="hard"),
        _result("random", 1, 12.0, selection_mode="soft"),
        _result("proposed", 1, 11.5, selection_mode="soft"),
    ]

    summary = summarize_results(rows)

    assert len(summary) == 4
    proposed_hard = next(
        row for row in summary if row["method"] == "proposed" and row["selection_mode"] == "hard"
    )
    proposed_soft = next(
        row for row in summary if row["method"] == "proposed" and row["selection_mode"] == "soft"
    )
    assert proposed_hard["num_seeds"] == 1
    assert proposed_hard["mean_test_sum_rate"] == 11.0
    assert proposed_hard["wins_vs_random"] == 1
    assert proposed_soft["num_seeds"] == 1
    assert proposed_soft["mean_test_sum_rate"] == 11.5
    assert proposed_soft["wins_vs_random"] == 0


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


def test_write_aggregate_outputs_writes_headers_for_empty_rows(tmp_path):
    all_results_path, summary_path = write_aggregate_outputs([], tmp_path)

    assert all_results_path.read_text(encoding="utf-8").splitlines() == [
        "method,selection_mode,K,Nx,Ny,N,n_active,W_lambda_x,W_lambda_y,Pmax_dBm,distance_km,"
        "seed,train_samples,val_samples,test_samples,epochs,batch_size,gpt2_layers,d_mha,mha_heads,test_sum_rate"
    ]
    assert summary_path.read_text(encoding="utf-8").splitlines() == [
        "method,selection_mode,num_seeds,mean_test_sum_rate,std_test_sum_rate,min_test_sum_rate,"
        "max_test_sum_rate,wins_vs_random"
    ]
    assert list(csv.DictReader(all_results_path.open(newline=""))) == []
    assert list(csv.DictReader(summary_path.open(newline=""))) == []


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


def test_run_seed_experiments_rejects_empty_seeds(monkeypatch, tmp_path):
    calls: list[str] = []

    def fake_train(cfg):
        calls.append("train")
        return Path(cfg.train.output_dir) / "proposed.pt"

    def fake_evaluate(cfg, checkpoint_path):
        calls.append("evaluate")
        return Path(cfg.train.output_dir) / "results.csv"

    monkeypatch.setattr("llm_fas.experiments.train_proposed", fake_train)
    monkeypatch.setattr("llm_fas.experiments.evaluate_checkpoint", fake_evaluate)

    with pytest.raises(ValueError, match="At least one seed"):
        run_seed_experiments(
            config_path="configs/mvp.yaml",
            seeds=[],
            output_root=tmp_path,
        )

    assert calls == []


def test_run_seed_experiments_accepts_one_shot_seed_generator(monkeypatch, tmp_path):
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

    run_seed_experiments(
        config_path="configs/mvp.yaml",
        seeds=(seed for seed in [1, 2]),
        output_root=tmp_path / "stage2",
    )

    assert calls == [
        ("train", 1, str(tmp_path / "stage2" / "seed_1")),
        ("evaluate", 1, str(tmp_path / "stage2" / "seed_1" / "proposed.pt")),
        ("train", 2, str(tmp_path / "stage2" / "seed_2")),
        ("evaluate", 2, str(tmp_path / "stage2" / "seed_2" / "proposed.pt")),
    ]


def test_run_seed_experiments_applies_device_override(monkeypatch, tmp_path):
    devices: list[str] = []

    def fake_train(cfg):
        devices.append(cfg.device)
        output_dir = Path(cfg.train.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = output_dir / "proposed.pt"
        checkpoint.write_text("fake checkpoint", encoding="utf-8")
        return checkpoint

    def fake_evaluate(cfg, checkpoint_path):
        devices.append(cfg.device)
        output_dir = Path(cfg.train.output_dir)
        result_path = output_dir / "results.csv"
        with result_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(_result("random", cfg.seed, 1.0).keys()))
            writer.writeheader()
            writer.writerow(_result("random", cfg.seed, 10.0))
            writer.writerow(_result("proposed", cfg.seed, 11.0))
        return result_path

    monkeypatch.setattr("llm_fas.experiments.train_proposed", fake_train)
    monkeypatch.setattr("llm_fas.experiments.evaluate_checkpoint", fake_evaluate)

    run_seed_experiments(
        config_path="configs/mvp.yaml",
        seeds=[1],
        output_root=tmp_path,
        device="cuda",
    )

    assert devices == ["cuda", "cuda"]


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
    assert "--methods" in help_result.stdout
    assert "--device" in help_result.stdout


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
    assert "--device" in help_result.stdout


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
    assert "--device" in help_result.stdout


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


def test_stage2_seed_runner_cli_reports_bad_seed_without_traceback():
    bad_result = subprocess.run(
        [sys.executable, "scripts/run_stage2_seeds.py", "--seeds", "abc"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert bad_result.returncode == 2
    assert "invalid literal" in bad_result.stderr or "At least one seed" in bad_result.stderr
    assert "Traceback" not in bad_result.stderr
