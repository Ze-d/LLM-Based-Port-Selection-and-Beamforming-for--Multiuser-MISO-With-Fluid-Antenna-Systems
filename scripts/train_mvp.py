from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_fas.config import load_config
from llm_fas.train import train_proposed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MVP Proposed LLM-FAS model.")
    parser.add_argument("--config", default="configs/mvp.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    checkpoint_path = train_proposed(load_config(args.config))
    print(checkpoint_path)


if __name__ == "__main__":
    main()
