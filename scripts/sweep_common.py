"""Shared helpers for paper figure parameter sweeps."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict
from pathlib import Path
from typing import TypeVar

import yaml

from llm_fas.config import ExperimentConfig, load_config
from llm_fas.experiments import parse_method_list, parse_seed_list, run_seed_experiments


T = TypeVar("T")


def value_token(value: float | int | str) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def parse_int_values(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one integer sweep value is required")
    return values


def parse_float_values(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one float sweep value is required")
    return values


def parse_grid_values(raw: str) -> list[tuple[int, int]]:
    values: list[tuple[int, int]] = []
    for part in [part.strip().lower() for part in raw.split(",") if part.strip()]:
        if "x" not in part:
            raise ValueError(f"Grid value must use NxNy form such as 4x4, got {part!r}")
        nx_raw, ny_raw = part.split("x", 1)
        values.append((int(nx_raw), int(ny_raw)))
    if not values:
        raise ValueError("At least one grid sweep value is required")
    return values


def resolve_seeds_and_methods(seeds_raw: str, methods_raw: str) -> tuple[list[int], list[str]]:
    return parse_seed_list(seeds_raw), parse_method_list(methods_raw)


def write_temp_config(cfg: ExperimentConfig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)
    return path


def run_parameter_sweep(
    *,
    figure_name: str,
    base_config_path: str,
    output_root: str | Path,
    seeds: Iterable[int],
    methods: Iterable[str],
    values: Iterable[T],
    value_name: Callable[[T], str],
    apply_value: Callable[[ExperimentConfig, T], ExperimentConfig],
    device: str | None,
    dry_run: bool,
) -> None:
    base_cfg = load_config(base_config_path)
    if device is not None:
        from dataclasses import replace

        base_cfg = replace(base_cfg, device=device)

    output_root = Path(output_root)
    seed_values = list(seeds)
    method_values = list(methods)
    sweep_values = list(values)

    if dry_run:
        print(f"figure={figure_name}")
        print(f"base_config={base_config_path}")
        print(f"seeds={','.join(str(seed) for seed in seed_values)}")
        print(f"methods={','.join(method_values)}")
        print(f"device={base_cfg.device}")
        print(f"output_root={output_root}")
        print("sweep:")
        for value in sweep_values:
            print(f"  {value_name(value)}")
        return

    for value in sweep_values:
        label = value_name(value)
        cfg = apply_value(base_cfg, value)
        config_path = write_temp_config(cfg, output_root / "_configs" / f"{label}.yaml")
        point_output = output_root / label
        print(f"\n=== {figure_name}: {label} ===")
        run_seed_experiments(
            config_path=config_path,
            seeds=seed_values,
            output_root=point_output,
            methods=method_values,
            device=None,
        )

    print(f"\n{figure_name} sweep outputs written under {output_root}")
