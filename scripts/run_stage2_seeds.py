from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_fas.experiments import parse_seed_list, run_seed_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 2 MVP stability experiments over multiple seeds.")
    parser.add_argument("--config", default="configs/mvp.yaml", help="Path to the base experiment config.")
    parser.add_argument(
        "--seeds",
        default="20260606,20260607,20260608",
        help="Comma-separated integer seeds.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/stage2_mvp_stability",
        help="Directory for per-seed outputs and aggregate CSVs.",
    )
    args = parser.parse_args()

    try:
        seeds = parse_seed_list(args.seeds)
    except ValueError as exc:
        parser.error(str(exc))
    all_results_path, summary_path = run_seed_experiments(args.config, seeds, args.output_root)
    print(all_results_path)
    print(summary_path)


if __name__ == "__main__":
    main()
