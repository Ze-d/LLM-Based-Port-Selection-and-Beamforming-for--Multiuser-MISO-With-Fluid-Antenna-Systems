from __future__ import annotations

import argparse
import sys
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
    args = parser.parse_args()

    results_path = evaluate_checkpoint(load_config(args.config), args.checkpoint)
    print(results_path)


if __name__ == "__main__":
    main()
