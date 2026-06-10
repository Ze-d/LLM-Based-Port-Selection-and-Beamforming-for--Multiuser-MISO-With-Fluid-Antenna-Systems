"""Fig.5 batch-size convergence experiment runner.

Trains the Proposed model at batch_sizes in {50, 100, 200} and records
per-epoch training and validation losses. Output CSVs can be used to
generate paper-style convergence curves.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_fas.config import ExperimentConfig, load_config
from llm_fas.experiments import parse_seed_list
from llm_fas.train import evaluate_checkpoints, train_method

FIG5_BATCH_SIZES = [50, 100, 200]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Fig.5 batch-size convergence experiment for the Proposed model."
    )
    parser.add_argument(
        "--config",
        default="configs/paper_default.yaml",
        help="Base config (paper-scale). Batch size is overridden per run.",
    )
    parser.add_argument(
        "--seeds",
        default="20260606",
        help="Comma-separated seeds.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/fig5_convergence",
        help="Directory for per-batch-size outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved sweep matrix without starting training.",
    )
    args = parser.parse_args()

    try:
        seeds = parse_seed_list(args.seeds)
    except ValueError as exc:
        parser.error(str(exc))

    base_cfg = load_config(args.config)
    output_root = Path(args.output_root)

    if args.dry_run:
        print(f"base_config={args.config}")
        print(f"seeds={seeds}")
        print(f"output_root={output_root}")
        print("batch_sizes:")
        for bs in FIG5_BATCH_SIZES:
            print(
                f"  {bs}: train_samples={base_cfg.data.train_samples}, "
                f"epochs={base_cfg.train.epochs}, "
                f"d_mha={base_cfg.model.d_mha}, "
                f"layers={base_cfg.model.gpt2_layers}"
            )
        return

    for seed in seeds:
        for bs in FIG5_BATCH_SIZES:
            cfg: ExperimentConfig = replace(
                base_cfg,
                seed=seed,
                train=replace(
                    base_cfg.train,
                    batch_size=bs,
                    output_dir=str(output_root / f"bs{bs}_seed{seed}"),
                ),
            )
            print(f"\n=== Fig.5 batch_size={bs}, seed={seed} ===")
            checkpoint_path = train_method(cfg, method="proposed")
            evaluate_checkpoints(cfg, {"proposed": checkpoint_path})

    print(f"\nConvergence histories written to subdirs of {output_root}")


if __name__ == "__main__":
    main()
