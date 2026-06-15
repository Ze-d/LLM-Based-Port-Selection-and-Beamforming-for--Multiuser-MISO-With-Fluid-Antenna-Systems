"""Plot Fig.9: sum rate versus user-BS distance."""
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
    parser = argparse.ArgumentParser(description="Plot Fig.9 from distance sweep outputs.")
    parser.add_argument("--output-root", default="outputs/fig9_distance", help="Sweep output root.")
    parser.add_argument("--values", default="0.1,0.15,0.2,0.25,0.3", help="Comma-separated distances in km.")
    parser.add_argument("--output-prefix", default="fig9_distance", help="Output figure filename prefix.")
    args = parser.parse_args()

    values = parse_float_values(args.values)
    plot_sum_rate_sweep(
        output_root=ROOT / args.output_root,
        point_dirs=[f"d_{value_token(value)}" for value in values],
        x_values=values,
        xlabel="Distance between User and BS (km)",
        title="Fig. 9: Sum Rate versus User-BS Distance",
        output_prefix=args.output_prefix,
        outputs_dir=ROOT / "outputs",
    )


if __name__ == "__main__":
    main()
