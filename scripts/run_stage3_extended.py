from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_fas.experiments import parse_method_list, parse_seed_list, run_seed_experiments


def _seed_range(seed_start: int, seed_count: int) -> list[int]:
    return list(range(seed_start, seed_start + seed_count))


def _default_output_root(seed_count: int) -> str:
    return f"outputs/stage3_transformer_{seed_count}seed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run extended Stage 3 Transformer baseline seed experiments.")
    parser.add_argument("--config", default="configs/mvp.yaml", help="Path to the base experiment config.")
    parser.add_argument("--seed-start", type=int, default=20260606, help="First seed for generated seed ranges.")
    parser.add_argument(
        "--seed-count",
        type=int,
        choices=[5, 10],
        default=5,
        help="Number of generated seeds when --seeds is not provided.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated integer seeds. Overrides --seed-start and --seed-count when provided.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Directory for per-seed outputs and aggregate CSVs. Defaults to outputs/stage3_transformer_{seed_count}seed.",
    )
    parser.add_argument(
        "--methods",
        default="transformer,proposed",
        help="Comma-separated trainable methods. Supported: random, cnn, transformer, llm_sequential, proposed.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override config device. Examples: auto, cpu, cuda, cuda:0.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved settings without starting training.")
    args = parser.parse_args()

    try:
        seeds = parse_seed_list(args.seeds) if args.seeds else _seed_range(args.seed_start, args.seed_count)
        methods = parse_method_list(args.methods)
    except ValueError as exc:
        parser.error(str(exc))

    output_root = args.output_root or _default_output_root(args.seed_count)
    seed_text = ",".join(str(seed) for seed in seeds)
    method_text = ",".join(methods)
    if args.dry_run:
        print(f"config={args.config}")
        print(f"seeds={seed_text}")
        print(f"methods={method_text}")
        print(f"device={args.device or 'config'}")
        print(f"output_root={output_root}")
        return

    all_results_path, summary_path = run_seed_experiments(
        args.config,
        seeds,
        output_root,
        methods=methods,
        device=args.device,
    )
    print(all_results_path)
    print(summary_path)


if __name__ == "__main__":
    main()
