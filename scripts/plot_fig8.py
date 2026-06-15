"""Plot Fig.8: sum rate versus number of activated ports n."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_sweep_common import plot_sum_rate_sweep
from sweep_common import parse_int_values


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Fig.8 from active-port sweep outputs.")
    parser.add_argument("--output-root", default="outputs/fig8_active_ports", help="Sweep output root.")
    parser.add_argument("--values", default="3,4,5,6", help="Comma-separated active port counts.")
    parser.add_argument("--output-prefix", default="fig8_active_ports", help="Output figure filename prefix.")
    args = parser.parse_args()

    values = parse_int_values(args.values)
    plot_sum_rate_sweep(
        output_root=ROOT / args.output_root,
        point_dirs=[f"n_{value}" for value in values],
        x_values=values,
        xlabel="Number of Activated Ports n",
        title="Fig. 8: Sum Rate versus Number of Activated Ports",
        output_prefix=args.output_prefix,
        outputs_dir=ROOT / "outputs",
    )


if __name__ == "__main__":
    main()
