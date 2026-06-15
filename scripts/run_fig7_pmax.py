"""Run Fig.7 sweep: sum rate versus BS power Pmax."""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_fas.config import ExperimentConfig
from sweep_common import parse_float_values, resolve_seeds_and_methods, run_parameter_sweep, value_token


def _apply_pmax(cfg: ExperimentConfig, pmax: float) -> ExperimentConfig:
    return replace(cfg, system=replace(cfg.system, Pmax_dBm=float(pmax)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Fig.7 Pmax parameter sweep.")
    parser.add_argument("--config", default="configs/paper_default.yaml", help="Base experiment config.")
    parser.add_argument("--values", default="10,15,20,25,30", help="Comma-separated Pmax values in dBm.")
    parser.add_argument("--seeds", default="20260606", help="Comma-separated seeds.")
    parser.add_argument(
        "--methods",
        default="random,cnn,transformer,llm_sequential,proposed",
        help="Comma-separated methods.",
    )
    parser.add_argument("--output-root", default="outputs/fig7_Pmax", help="Output root directory.")
    parser.add_argument("--device", default=None, help="Override config device, e.g. cuda or cuda:0.")
    parser.add_argument("--dry-run", action="store_true", help="Print sweep matrix without training.")
    args = parser.parse_args()

    try:
        values = parse_float_values(args.values)
        seeds, methods = resolve_seeds_and_methods(args.seeds, args.methods)
    except ValueError as exc:
        parser.error(str(exc))

    run_parameter_sweep(
        figure_name="Fig.7 Pmax",
        base_config_path=args.config,
        output_root=args.output_root,
        seeds=seeds,
        methods=methods,
        values=values,
        value_name=lambda value: f"Pmax_{value_token(int(value) if value.is_integer() else value)}",
        apply_value=_apply_pmax,
        device=args.device,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
