import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# chart colors (categorical slots 1+2) and recessive chart chrome
SERIES_COLORS = {"Precision": "#2a78d6", "Recall": "#eb6834"}
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def adjust_positions(items, min_gap=0.08):
    """ Avoid overlap in donut plot """
    items = sorted(items, key=lambda d: d["end_y"])

    for j in range(1, len(items)):
        prev = items[j - 1]
        curr = items[j]

        if curr["end_y"] - prev["end_y"] < min_gap:
            curr["end_y"] = prev["end_y"] + min_gap

    return items


def calc_pooled_scores(counts: pd.DataFrame) -> pd.DataFrame:
    """Pool TPs, FPs, FNs across all documents and compute precision, recall and F1 per impact column.

    counts: one row per impact column (or per impact column and document) with columns
        "impact", "tps", "fps", "fns"
    """
    pooled = counts.groupby("impact", sort=False)[["tps", "fps", "fns"]].sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        pooled["precision"] = pooled["tps"] / (pooled["tps"] + pooled["fps"])
        pooled["recall"] = pooled["tps"] / (pooled["tps"] + pooled["fns"])
        pooled["f1"] = 2 * pooled["precision"] * pooled["recall"] / (pooled["precision"] + pooled["recall"])
    pooled = pooled.fillna(0.0)
    pooled["n"] = pooled["tps"] + pooled["fns"]  # number of validation records

    return pooled.reset_index()


def plot_pooled_precision_recall(
    counts: pd.DataFrame,
    title: str = "Precision and recall scores",
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot precision and recall per impact column, pooled across all documents.

    counts: DataFrame with columns "impact", "tps", "fps", "fns", e.g. collected in the
        evaluation loop as {"impact": p_impact, "tps": sum(tps_list), "fps": sum(fps_list), "fns": sum(fns_list)}
    save_path: if given, the figure is stored there (e.g. "../figures/precision_recall.png")
    """
    scores = calc_pooled_scores(counts)
    metrics = ["Precision", "Recall"]

    n_impacts = len(scores)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.7 * n_impacts + 1.4))
    fig = ax.figure
    fig.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # horizontal grouped bars: one row per impact column, first impact on top
    bar_height = 0.34
    gap = 0.03  # surface gap between the two bars of a group
    y = np.arange(n_impacts)[::-1]
    offsets = {"Precision": bar_height / 2 + gap / 2, "Recall": -(bar_height / 2 + gap / 2)}

    for metric in metrics:
        values = scores[metric.lower()].to_numpy()
        ax.barh(
            y + offsets[metric], values, height=bar_height,
            color=SERIES_COLORS[metric], label=metric, zorder=2,
        )
        # direct value labels at the bar ends, in text ink
        for yi, v in zip(y + offsets[metric], values):
            ax.text(v + 0.01, yi, f"{v:.2f}", va="center", ha="left", fontsize=8.5, color=INK_SECONDARY)

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{imp}" for imp in scores["impact"]],
        color=INK_PRIMARY, fontsize=9.5,
    )
    ax.set_xlim(0, 1.08)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Score", color=INK_MUTED, fontsize=9)
    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=8.5, length=0)
    ax.tick_params(axis="y", length=0)

    # recessive grid and axes
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    for side in ["top", "right", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(BASELINE)

    ax.set_title(title, loc="left", color=INK_PRIMARY, fontsize=11.5, pad=24)
    ax.legend(
        loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(metrics), frameon=False,
        fontsize=9, labelcolor=INK_SECONDARY, handlelength=1.0, borderaxespad=0.2,
    )

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)

    return ax
