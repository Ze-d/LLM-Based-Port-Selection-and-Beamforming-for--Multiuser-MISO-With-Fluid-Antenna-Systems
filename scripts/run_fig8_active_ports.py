"""Run Fig.8 sweep: sum rate versus number of activated ports n."""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_fas.config import ExperimentConfig
from sweep_common import parse_int_values, resolve_seeds_and_methods, run_parameter_sweep


def _apply_active_ports(cfg: ExperimentConfig, n_active: int) -> ExperimentConfig:
    return replace(cfg, system=replace(cfg.system, n_active=int(n_active)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Fig.8 active-port parameter sweep.")
    parser.add_argument("--config", default="configs/paper_default.yaml", help="Base experiment config.")
    parser.add_argument("--values", default="3,4,5,6", help="Comma-separated active port counts.")
    parser.add_argument("--seeds", default="20260606", help="Comma-separated seeds.")
    parser.add_argument(
        "--methods",
        default="random,cnn,transformer,llm_sequential,proposed",
        help="Comma-separated methods.",
    )
    parser.add_argument("--output-root", default="outputs/fig8_active_ports", help="Output root directory.")
    parser.add_argument("--device", default=None, help="Override config device, e.g. cuda or cuda:0.")
    parser.add_argument("--dry-run", action="store_true", help="Print sweep matrix without training.")
    args = parser.parse_args()

    try:
        values = parse_int_values(args.values)
        seeds, methods = resolve_seeds_and_methods(args.seeds, args.methods)
    except ValueError as exc:
        parser.error(str(exc))

    run_parameter_sweep(
        figure_name="Fig.8 active ports",
        base_config_path=args.config,
        output_root=args.output_root,
        seeds=seeds,
        methods=methods,
        values=values,
        value_name=lambda value: f"n_{value}",
        apply_value=_apply_active_ports,
        device=args.device,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
