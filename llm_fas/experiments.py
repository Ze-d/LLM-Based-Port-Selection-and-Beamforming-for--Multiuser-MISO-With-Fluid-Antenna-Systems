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
