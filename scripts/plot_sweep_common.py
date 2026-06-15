"""Shared plotting helpers for paper-style sum-rate sweeps."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Sequence


METHODS = ["random", "cnn", "transformer", "llm_sequential", "proposed"]
COLORS = {
    "random": "#999999",
    "cnn": "#fdae61",
    "transformer": "#d7191c",
    "llm_sequential": "#7b3294",
    "proposed": "#2c7bb6",
}
MARKERS = {"random": "s", "cnn": "^", "transformer": "D", "llm_sequential": "v", "proposed": "o"}
LABELS = {
    "random": "Random",
    "cnn": "CNN",
    "transformer": "Transformer",
    "llm_sequential": "LLM-sequential",
    "proposed": "Proposed",
}


def setup_style():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    return plt


def load_rates(point_dir: Path) -> dict[str, tuple[float, float]]:
    summary_path = point_dir / "summary.csv"
    if summary_path.exists():
        rates: dict[str, tuple[float, float]] = {}
        with summary_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                method = row["method"]
                rates[method] = (
                    float(row["mean_test_sum_rate"]),
                    float(row.get("std_test_sum_rate", 0.0) or 0.0),
                )
        return rates

    all_results_path = point_dir / "all_results.csv"
    grouped: dict[str, list[float]] = defaultdict(list)
    with all_results_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            grouped[row["method"]].append(float(row["test_sum_rate"]))

    rates = {}
    for method, values in grouped.items():
        mean = sum(values) / len(values)
        if len(values) > 1:
            variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            std = variance**0.5
        else:
            std = 0.0
        rates[method] = (mean, std)
    return rates


def plot_sum_rate_sweep(
    *,
    output_root: Path,
    point_dirs: Sequence[str],
    x_values: Sequence[float | int],
    xlabel: str,
    title: str,
    output_prefix: str,
    outputs_dir: Path,
    x_tick_labels: Sequence[str] | None = None,
) -> None:
    import matplotlib.ticker as ticker

    plt = setup_style()

    data: dict[str, list[float | None]] = {method: [] for method in METHODS}
    errors: dict[str, list[float | None]] = {method: [] for method in METHODS}
    for point_dir in point_dirs:
        rates = load_rates(output_root / point_dir)
        for method in METHODS:
            value = rates.get(method)
            data[method].append(value[0] if value is not None else None)
            errors[method].append(value[1] if value is not None else None)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    plotted_methods = [method for method in METHODS if all(value is not None for value in data[method])]
    for method in plotted_methods:
        y_values = [float(value) for value in data[method] if value is not None]
        y_errors = [float(value) for value in errors[method] if value is not None]
        has_error = any(error > 0.0 for error in y_errors)
        plot_kwargs = {
            "color": COLORS[method],
            "marker": MARKERS[method],
            "markersize": 7,
            "linewidth": 2,
            "label": LABELS[method],
            "markerfacecolor": "white" if method != "proposed" else COLORS[method],
            "markeredgewidth": 1.4,
        }
        if has_error:
            ax.errorbar(x_values, y_values, yerr=y_errors, capsize=3, **plot_kwargs)
        else:
            ax.plot(x_values, y_values, **plot_kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Sum Rate (bps/Hz)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best")
    if x_tick_labels is not None:
        ax.set_xticks(list(x_values))
        ax.set_xticklabels(list(x_tick_labels))
    else:
        ax.xaxis.set_major_locator(ticker.FixedLocator(list(x_values)))

    fig.tight_layout()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        out = outputs_dir / f"{output_prefix}.{ext}"
        fig.savefig(out, format=ext)
        print(f"Saved: {out}")
    plt.close(fig)
