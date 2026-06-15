"""Plot Fig.10: sum rate versus total number of ports N."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_sweep_common import plot_sum_rate_sweep
from sweep_common import parse_grid_values


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Fig.10 from total-port-count sweep outputs.")
    parser.add_argument("--output-root", default="outputs/fig10_ports", help="Sweep output root.")
    parser.add_argument("--grids", default="3x3,4x4,5x5,6x6", help="Comma-separated Nx x Ny grids.")
    parser.add_argument("--output-prefix", default="fig10_ports", help="Output figure filename prefix.")
    args = parser.parse_args()

    grids = parse_grid_values(args.grids)
    x_values = [nx * ny for nx, ny in grids]
    plot_sum_rate_sweep(
        output_root=ROOT / args.output_root,
        point_dirs=[f"N_{nx}x{ny}" for nx, ny in grids],
        x_values=x_values,
        x_tick_labels=[str(value) for value in x_values],
        xlabel="Number of Ports N",
        title="Fig. 10: Sum Rate versus Number of Ports",
        output_prefix=args.output_prefix,
        outputs_dir=ROOT / "outputs",
    )


if __name__ == "__main__":
    main()
