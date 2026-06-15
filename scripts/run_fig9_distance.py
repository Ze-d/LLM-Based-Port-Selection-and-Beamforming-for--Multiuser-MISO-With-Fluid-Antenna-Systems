"""Run Fig.9 sweep: sum rate versus user-BS distance."""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_fas.config import ExperimentConfig
from sweep_common import parse_float_values, resolve_seeds_and_methods, run_parameter_sweep, value_token


def _apply_distance(cfg: ExperimentConfig, distance_km: float) -> ExperimentConfig:
    return replace(cfg, system=replace(cfg.system, distance_km=float(distance_km)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Fig.9 distance parameter sweep.")
    parser.add_argument("--config", default="configs/paper_default.yaml", help="Base experiment config.")
    parser.add_argument("--values", default="0.1,0.15,0.2,0.25,0.3", help="Comma-separated distances in km.")
    parser.add_argument("--seeds", default="20260606", help="Comma-separated seeds.")
    parser.add_argument(
        "--methods",
        default="random,cnn,transformer,llm_sequential,proposed",
        help="Comma-separated methods.",
    )
    parser.add_argument("--output-root", default="outputs/fig9_distance", help="Output root directory.")
    parser.add_argument("--device", default=None, help="Override config device, e.g. cuda or cuda:0.")
    parser.add_argument("--dry-run", action="store_true", help="Print sweep matrix without training.")
    args = parser.parse_args()

    try:
        values = parse_float_values(args.values)
        seeds, methods = resolve_seeds_and_methods(args.seeds, args.methods)
    except ValueError as exc:
        parser.error(str(exc))

    run_parameter_sweep(
        figure_name="Fig.9 distance",
        base_config_path=args.config,
        output_root=args.output_root,
        seeds=seeds,
        methods=methods,
        values=values,
        value_name=lambda value: f"d_{value_token(value)}",
        apply_value=_apply_distance,
        device=args.device,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
