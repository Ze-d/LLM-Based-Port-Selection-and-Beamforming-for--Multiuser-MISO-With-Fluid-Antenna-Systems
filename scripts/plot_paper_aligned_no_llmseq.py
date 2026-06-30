"""Plot paper-aligned 5-seed results without the LLM-sequential baseline."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("outputs/paper_aligned_5seed_5methods")
ALL_RESULTS = ROOT / "all_results.csv"
FIGURES = ROOT / "figures"

METHODS = ["random", "cnn", "transformer", "proposed"]
LABELS = {
    "random": "Random",
    "cnn": "CNN",
    "transformer": "Transformer",
    "proposed": "Proposed",
}
COLORS = {
    "random": "#8c8c8c",
    "cnn": "#e69f00",
    "transformer": "#cc3311",
    "proposed": "#0072b2",
}
MARKERS = {
    "random": "o",
    "cnn": "s",
    "transformer": "^",
    "proposed": "D",
}
SEED_LABEL = {
    20260606: "S1",
    20260607: "S2",
    20260608: "S3",
    20260609: "S4",
    20260610: "S5",
}


def load_results() -> dict[str, dict[int, float]]:
    grouped: dict[str, dict[int, float]] = defaultdict(dict)
    with ALL_RESULTS.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            method = row["method"]
            if method in METHODS:
                grouped[method][int(row["seed"])] = float(row["test_sum_rate"])
    return grouped


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return (sum((value - avg) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def save_all(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGURES / f"{stem}.{ext}", bbox_inches="tight", dpi=300)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_mean_sum_rate(results: dict[str, dict[int, float]]) -> None:
    values = {method: list(results[method].values()) for method in METHODS}
    means = [mean(values[method]) for method in METHODS]
    stds = [sample_std(values[method]) for method in METHODS]
    baseline_best = max(means[:-1])
    proposed_gain = means[-1] - baseline_best

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    x = list(range(len(METHODS)))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        color=[COLORS[method] for method in METHODS],
        edgecolor="#222222",
        linewidth=0.8,
        alpha=0.72,
    )
    bars[-1].set_alpha(0.95)
    bars[-1].set_linewidth(1.6)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[method] for method in METHODS])
    ax.set_ylabel("Test sum rate (bps/Hz)")
    ax.set_title("Mean performance across five seeds")
    ax.grid(axis="y", alpha=0.28, linestyle="--")
    ax.set_ylim(min(means) - 0.10, max(means) + max(stds) + 0.08)

    for bar, value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold" if bar is bars[-1] else "normal",
        )

    ax.annotate(
        f"+{proposed_gain:.4f} bps/Hz\nvs best baseline",
        xy=(x[-1], means[-1]),
        xytext=(x[-1] - 0.62, means[-1] + max(stds) + 0.035),
        arrowprops={"arrowstyle": "->", "color": COLORS["proposed"], "lw": 1.5},
        color=COLORS["proposed"],
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

    fig.tight_layout()
    save_all(fig, "mean_sum_rate_no_llmseq")
    plt.close(fig)


def plot_gain_vs_random(results: dict[str, dict[int, float]]) -> None:
    seeds = sorted(results["proposed"])
    compared_methods = ["cnn", "transformer", "proposed"]
    gains_pct: dict[str, list[float]] = {}
    wins: dict[str, int] = {}
    for method in compared_methods:
        gains = [
            results[method][seed] - results["random"][seed]
            for seed in seeds
        ]
        gains_pct[method] = [
            100.0 * gain / results["random"][seed]
            for gain, seed in zip(gains, seeds)
        ]
        wins[method] = sum(gain > 0.0 for gain in gains)

    mean_pct = [mean(gains_pct[method]) for method in compared_methods]
    std_pct = [sample_std(gains_pct[method]) for method in compared_methods]
    mean_abs = [
        mean([
            results[method][seed] - results["random"][seed]
            for seed in seeds
        ])
        for method in compared_methods
    ]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = list(range(len(compared_methods)))
    bars = ax.bar(
        x,
        mean_pct,
        yerr=std_pct,
        capsize=4,
        color=[COLORS[method] for method in compared_methods],
        edgecolor="#222222",
        linewidth=0.8,
        alpha=0.78,
    )
    bars[-1].set_alpha(0.95)
    bars[-1].set_linewidth(1.6)
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[method] for method in compared_methods])
    ax.set_ylabel("Gain over Random (%)")
    ax.set_title("Gain over Random across five seeds")
    ax.grid(axis="y", alpha=0.28, linestyle="--")

    for bar, method, pct_value, abs_value in zip(bars, compared_methods, mean_pct, mean_abs):
        label_y = pct_value + 0.045 if pct_value >= 0.0 else pct_value - 0.045
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{pct_value:+.2f}%\n{abs_value:+.4f} bps/Hz\n{wins[method]}/5 wins",
            ha="center",
            va="bottom" if pct_value >= 0.0 else "top",
            fontsize=9.5,
            fontweight="bold" if method == "proposed" else "normal",
        )

    lower = min(0.0, min(value - error for value, error in zip(mean_pct, std_pct))) - 0.22
    upper = max(0.0, max(value + error for value, error in zip(mean_pct, std_pct))) + 0.36
    ax.set_ylim(lower, upper)
    fig.tight_layout()
    save_all(fig, "gain_vs_random_no_llmseq")
    plt.close(fig)


def plot_per_seed_sum_rate(results: dict[str, dict[int, float]]) -> None:
    seeds = sorted(results["proposed"])
    x = list(range(len(seeds)))

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for method in METHODS:
        y_values = [results[method][seed] for seed in seeds]
        is_proposed = method == "proposed"
        ax.plot(
            x,
            y_values,
            marker=MARKERS[method],
            markersize=7.5 if is_proposed else 6.5,
            linewidth=2.8 if is_proposed else 1.9,
            color=COLORS[method],
            label=LABELS[method],
            markerfacecolor=COLORS[method] if is_proposed else "white",
            markeredgewidth=1.4,
            zorder=3 if is_proposed else 2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([SEED_LABEL.get(seed, str(seed)) for seed in seeds])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Test sum rate (bps/Hz)")
    ax.set_title("Per-seed test sum-rate")
    ax.grid(axis="y", alpha=0.28, linestyle="--")
    ax.legend(loc="upper left", ncols=4, frameon=False)
    ax.set_xlim(-0.2, len(seeds) - 0.8)

    all_values = [
        results[method][seed]
        for method in METHODS
        for seed in seeds
    ]
    ax.set_ylim(min(all_values) - 0.06, max(all_values) + 0.06)

    fig.tight_layout()
    save_all(fig, "per_seed_sum_rate_no_llmseq")
    plt.close(fig)


def plot_per_seed_gain(results: dict[str, dict[int, float]]) -> None:
    seeds = sorted(results["proposed"])
    baselines = ["random", "cnn", "transformer"]
    markers = {"random": "o", "cnn": "s", "transformer": "^"}

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = list(range(len(seeds)))
    for baseline in baselines:
        gains = [results["proposed"][seed] - results[baseline][seed] for seed in seeds]
        ax.plot(
            x,
            gains,
            marker=markers[baseline],
            markersize=7,
            linewidth=2.1,
            color=COLORS[baseline],
            label=f"vs {LABELS[baseline]}",
        )

    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.fill_between([-0.25, len(seeds) - 0.75], 0.0, 0.30, color=COLORS["proposed"], alpha=0.08)
    ax.set_xticks(x)
    ax.set_xticklabels([SEED_LABEL.get(seed, str(seed)) for seed in seeds])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Sum-rate gain (bps/Hz)")
    ax.set_title("Per-seed paired gain of Proposed")
    ax.grid(axis="y", alpha=0.28, linestyle="--")
    ax.legend(loc="upper left", ncols=3, frameon=False)
    ax.set_xlim(-0.25, len(seeds) - 0.75)
    ax.set_ylim(-0.08, 0.30)

    fig.tight_layout()
    save_all(fig, "per_seed_gain_no_llmseq")
    plt.close(fig)


def main() -> None:
    setup_style()
    results = load_results()
    missing = [method for method in METHODS if method not in results]
    if missing:
        raise SystemExit(f"Missing methods in {ALL_RESULTS}: {', '.join(missing)}")

    plot_mean_sum_rate(results)
    plot_gain_vs_random(results)
    plot_per_seed_sum_rate(results)
    plot_per_seed_gain(results)
    print(f"Saved no-LLM-seq figures to {FIGURES}")


if __name__ == "__main__":
    main()
