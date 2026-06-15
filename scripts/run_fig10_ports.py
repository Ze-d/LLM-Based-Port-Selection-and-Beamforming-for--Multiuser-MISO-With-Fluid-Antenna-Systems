"""Run Fig.10 sweep: sum rate versus total number of ports N."""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_fas.config import ExperimentConfig
from sweep_common import parse_grid_values, resolve_seeds_and_methods, run_parameter_sweep


def _apply_grid(cfg: ExperimentConfig, grid: tuple[int, int]) -> ExperimentConfig:
    nx, ny = grid
    if cfg.system.n_active > nx * ny:
        raise ValueError(f"n_active={cfg.system.n_active} cannot exceed N={nx * ny} for grid {nx}x{ny}")
    return replace(cfg, system=replace(cfg.system, Nx=nx, Ny=ny))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Fig.10 total-port-count parameter sweep.")
    parser.add_argument("--config", default="configs/paper_default.yaml", help="Base experiment config.")
    parser.add_argument("--grids", default="3x3,4x4,5x5,6x6", help="Comma-separated Nx x Ny grids.")
    parser.add_argument("--seeds", default="20260606", help="Comma-separated seeds.")
    parser.add_argument(
        "--methods",
        default="random,cnn,transformer,llm_sequential,proposed",
        help="Comma-separated methods.",
    )
    parser.add_argument("--output-root", default="outputs/fig10_ports", help="Output root directory.")
    parser.add_argument("--device", default=None, help="Override config device, e.g. cuda or cuda:0.")
    parser.add_argument("--dry-run", action="store_true", help="Print sweep matrix without training.")
    args = parser.parse_args()

    try:
        grids = parse_grid_values(args.grids)
        seeds, methods = resolve_seeds_and_methods(args.seeds, args.methods)
    except ValueError as exc:
        parser.error(str(exc))

    run_parameter_sweep(
        figure_name="Fig.10 ports",
        base_config_path=args.config,
        output_root=args.output_root,
        seeds=seeds,
        methods=methods,
        values=grids,
        value_name=lambda grid: f"N_{grid[0]}x{grid[1]}",
        apply_value=_apply_grid,
        device=args.device,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
