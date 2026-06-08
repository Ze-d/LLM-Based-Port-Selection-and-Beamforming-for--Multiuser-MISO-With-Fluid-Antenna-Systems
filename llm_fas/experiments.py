from __future__ import annotations

import csv
import statistics
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

import yaml

from llm_fas.config import ExperimentConfig


def parse_seed_list(raw: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise ValueError("At least one seed must be provided")
    return seeds


def clone_config_for_seed(cfg: ExperimentConfig, seed: int, output_root: str | Path) -> ExperimentConfig:
    seed_dir = Path(output_root) / f"seed_{seed}"
    return replace(cfg, seed=seed, train=replace(cfg.train, output_dir=str(seed_dir)))


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

ALL_RESULTS_FIELDNAMES = [
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


def _sorted_fieldnames(rows: list[dict[str, str]]) -> list[str]:
    keys = set().union(*(row.keys() for row in rows))
    return ALL_RESULTS_FIELDNAMES + sorted(keys - set(ALL_RESULTS_FIELDNAMES))


def summarize_results(rows: list[dict[str, str]]) -> list[dict[str, float | int | str]]:
    by_method_mode: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["method"], row.get("selection_mode", ""))
        by_method_mode.setdefault(key, []).append(row)

    random_by_mode_seed = {
        (row.get("selection_mode", ""), int(row["seed"])): float(row["test_sum_rate"])
        for (method, _selection_mode), method_rows in by_method_mode.items()
        if method == "random"
        for row in method_rows
    }
    summary: list[dict[str, float | int | str]] = []
    for method, selection_mode in sorted(by_method_mode):
        method_rows = by_method_mode[(method, selection_mode)]
        rates = [float(row["test_sum_rate"]) for row in method_rows]
        wins = 0
        if method != "random":
            for row in method_rows:
                seed = int(row["seed"])
                random_rate = random_by_mode_seed.get((selection_mode, seed))
                if random_rate is not None and float(row["test_sum_rate"]) > random_rate:
                    wins += 1
        summary.append(
            {
                "method": method,
                "selection_mode": selection_mode,
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
