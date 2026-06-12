from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_fas.config import load_config
from llm_fas.train import train_proposed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MVP Proposed LLM-FAS model.")
    parser.add_argument("--config", default="configs/mvp.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--device",
        default=None,
        help="Override config device. Examples: auto, cpu, cuda, cuda:0.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device is not None:
        cfg = replace(cfg, device=args.device)
    checkpoint_path = train_proposed(cfg)
    print(checkpoint_path)


if __name__ == "__main__":
    main()
