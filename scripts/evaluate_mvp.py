from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_fas.config import load_config
from llm_fas.train import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MVP Random and Proposed methods.")
    parser.add_argument("--config", default="configs/mvp.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", default="outputs/mvp/proposed.pt", help="Path to Proposed checkpoint.")
    parser.add_argument(
        "--device",
        default=None,
        help="Override config device. Examples: auto, cpu, cuda, cuda:0.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device is not None:
        cfg = replace(cfg, device=args.device)
    results_path = evaluate_checkpoint(cfg, args.checkpoint)
    print(results_path)


if __name__ == "__main__":
    main()
