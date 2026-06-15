"""Plot Fig.7: sum rate versus BS power Pmax."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_sweep_common import plot_sum_rate_sweep
from sweep_common import parse_float_values, value_token


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Fig.7 from Pmax sweep outputs.")
    parser.add_argument("--output-root", default="outputs/fig7_Pmax", help="Sweep output root.")
    parser.add_argument("--values", default="10,15,20,25,30", help="Comma-separated Pmax values in dBm.")
    parser.add_argument("--output-prefix", default="fig7_Pmax", help="Output figure filename prefix.")
    args = parser.parse_args()

    values = parse_float_values(args.values)
    point_dirs = [f"Pmax_{value_token(int(value) if value.is_integer() else value)}" for value in values]
    plot_sum_rate_sweep(
        output_root=ROOT / args.output_root,
        point_dirs=point_dirs,
        x_values=[int(value) if value.is_integer() else value for value in values],
        xlabel="BS Power Pmax (dBm)",
        title="Fig. 7: Sum Rate versus BS Power Pmax",
        output_prefix=args.output_prefix,
        outputs_dir=ROOT / "outputs",
    )


if __name__ == "__main__":
    main()
