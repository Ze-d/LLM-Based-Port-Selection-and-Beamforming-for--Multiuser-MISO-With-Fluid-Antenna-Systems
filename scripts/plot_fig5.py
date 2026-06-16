"""Plot Fig.5: Convergence performance for different batch sizes.

Reproduces the paper-style convergence curves comparing train/val loss
across batch sizes 50, 100, 200.  Produces two layouts:
  - fig5_combined: all curves on a single axes
  - fig5_subplots: one subplot per batch size
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Configuration ──────────────────────────────────────────────────────────
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "fig5_fixed"
DEFAULT_OUTPUT_PREFIX = "fig5_convergence"
BATCH_SIZES = [50, 100, 200]
DEFAULT_SEED = 20260606

# Color palette: one colour per batch size
BS_COLORS = {50: "#2c7bb6", 100: "#d7191c", 200: "#1a9641"}   # blue, red, green


def load_history(bs: int, seed: int, output_root: Path) -> tuple[list[int], list[float], list[float]]:
    """Load train/val loss from CSV. Returns (epochs, train_loss, val_loss)."""
    path = output_root / f"bs{bs}_seed{seed}" / "train_history.csv"
    epochs: list[int] = []
    train: list[float] = []
    val: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train.append(float(row["train_loss"]))
            val.append(float(row["val_loss"]))
    return epochs, train, val


def _setup_style():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    return plt


def plot_combined(output_root: Path, output_prefix: str, seed: int) -> None:
    """All six curves (3 batch-sizes × {train,val}) on one panel."""
    import matplotlib.ticker as ticker

    plt = _setup_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for bs in BATCH_SIZES:
        epochs, train_loss, val_loss = load_history(bs, seed, output_root)
        color = BS_COLORS[bs]

        ax.plot(epochs, train_loss, color=color, linestyle="-", linewidth=1.5,
                label=f"bs={bs} Train")
        ax.plot(epochs, val_loss, color=color, linestyle="--", linewidth=1.8,
                label=f"bs={bs} Val")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss Value")
    ax.set_title("Fig. 5: Convergence Performance for Different Batch Sizes")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Tight y-range
    all_losses: list[float] = []
    for bs in BATCH_SIZES:
        _, t, v = load_history(bs, seed, output_root)
        all_losses.extend(t + v)
    y_min, y_max = min(all_losses), max(all_losses)
    margin = (y_max - y_min) * 0.10 if y_max > y_min else 1.0
    ax.set_ylim(y_min - margin, y_max + margin)

    fig.tight_layout()
    out_pdf = ROOT / "outputs" / f"{output_prefix}_combined.pdf"
    out_png = ROOT / "outputs" / f"{output_prefix}_combined.png"
    fig.savefig(out_pdf, format="pdf")
    fig.savefig(out_png, format="png")
    print(f"Combined figure: {out_pdf}")
    print(f"Combined PNG  : {out_png}")
    plt.close(fig)


def plot_subplots(output_root: Path, output_prefix: str, seed: int) -> None:
    """Three side-by-side subplots, one per batch size."""
    import matplotlib.ticker as ticker

    plt = _setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for idx, bs in enumerate(BATCH_SIZES):
        ax = axes[idx]
        epochs, train_loss, val_loss = load_history(bs, seed, output_root)

        ax.plot(epochs, train_loss, color=BS_COLORS[bs], linestyle="-",
                linewidth=1.5, label="Training")
        ax.plot(epochs, val_loss, color=BS_COLORS[bs], linestyle="--",
                linewidth=1.5, label="Validation")

        ax.set_xlabel("Epoch")
        if idx == 0:
            ax.set_ylabel("Loss Value")
        ax.set_title(f"Batch Size = {bs}")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="upper right")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        all_losses = train_loss + val_loss
        y_min, y_max = min(all_losses), max(all_losses)
        margin = (y_max - y_min) * 0.15 if y_max > y_min else 1.0
        ax.set_ylim(y_min - margin, y_max + margin)

    fig.suptitle("Fig. 5: Convergence Performance for Different Batch Sizes", fontsize=15, y=1.02)
    fig.tight_layout()
    out_pdf = ROOT / "outputs" / f"{output_prefix}_subplots.pdf"
    out_png = ROOT / "outputs" / f"{output_prefix}_subplots.png"
    fig.savefig(out_pdf, format="pdf")
    fig.savefig(out_png, format="png")
    print(f"Subplot figure: {out_pdf}")
    print(f"Subplot PNG  : {out_png}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Fig.5 convergence curves.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Directory produced by run_fig5_convergence.py.")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX, help="Prefix for generated figure files.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed to plot.")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    plot_combined(output_root, args.output_prefix, args.seed)
    plot_subplots(output_root, args.output_prefix, args.seed)


if __name__ == "__main__":
    main()
