from __future__ import annotations

import csv
import statistics
from pathlib import Path

import pytest

from llm_fas.config import load_config
from llm_fas.experiments import clone_config_for_seed, parse_seed_list, summarize_results, write_aggregate_outputs


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
