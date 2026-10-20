import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.settings import init_logger


logger = init_logger(__name__)


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
    pooled["n_pred"] = pooled["tps"] + pooled["fps"]  # number of predicted records

    return pooled.reset_index()


def plot_pooled_precision_recall(
    counts: pd.DataFrame,
    title: str = "Precision and recall scores",
    save_path: str | None = None,
    ax: plt.Axes | None = None,
    show_counts: bool = True,
) -> plt.Axes:
    """Plot precision and recall per impact column, pooled across all documents.

    counts: DataFrame with columns "impact", "tps", "fps", "fns", e.g. collected in the
        evaluation loop as {"impact": p_impact, "tps": sum(tps_list), "fps": sum(fps_list), "fns": sum(fns_list)}
    save_path: if given, the figure is stored there (e.g. "../figures/precision_recall.png")
    show_counts: label each bar also with its number of cases: predicted cases (TPs + FPs) for the
        precision bar, validation cases (TPs + FNs) for the recall bar
    """
    scores = calc_pooled_scores(counts)
    metrics = ["Precision", "Recall"]
    # number of cases each score is based on: precision -> prediction set, recall -> validation set
    counts_per_metric = {"Precision": (scores["n_pred"].to_numpy(), "pred."), "Recall": (scores["n"].to_numpy(), "valid.")}

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
        n_cases, set_name = counts_per_metric[metric]
        for yi, v, n in zip(y + offsets[metric], values, n_cases):
            label = f"{v:.2f}" + (f" · {int(n)} {set_name}" if show_counts else "")
            ax.text(v + 0.01, yi, label, va="center", ha="left", fontsize=8.5, color=INK_SECONDARY)

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{imp}" for imp in scores["impact"]],
        color=INK_PRIMARY, fontsize=9.5,
    )
    ax.set_xlim(0, 1.25 if show_counts else 1.08)  # room for the bar labels
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
    if show_counts:
        for handle in ax.get_legend_handles_labels()[0]:
            handle.set_label(f"{handle.get_label()} (n = {'predicted' if handle.get_label() == 'Precision' else 'validation'} cases)")
    ax.legend(
        loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(metrics), frameon=False,
        fontsize=9, labelcolor=INK_SECONDARY, handlelength=1.0, borderaxespad=0.2,
    )

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)

    return ax


def plot_pooled_precision_recall_step1_vs_judge(
    counts_step1: pd.DataFrame,
    counts_judge: pd.DataFrame,
    title: str = "Precision and recall scores: Step 1 vs. LLM-judge",
    save_path: str | None = None,
    ax: plt.Axes | None = None,
    show_counts: bool = True,
) -> plt.Axes:
    """Plot precision and recall per impact column for Step 1 and LLM-judge predictions, pooled across all documents.

    Step 1 scores are drawn as empty, dashed bars; the LLM-judge scores as filled bars at the same position,
    so the change introduced by the judge is directly visible per impact column.

    counts_step1, counts_judge: DataFrames with columns "impact", "tps", "fps", "fns" (see plot_pooled_precision_recall),
        both evaluated against the same validation set
    save_path: if given, the figure is stored there (e.g. "../figures/precision_recall_step1_vs_judge.png")
    show_counts: label each bar also with its number of cases: predicted cases (TPs + FPs) for the
        precision bar, validation cases (TPs + FNs) for the recall bar
    """
    scores_judge = calc_pooled_scores(counts_judge).set_index("impact")
    # align Step 1 scores to the impact order of the judge scores, impacts missing in Step 1 get score 0
    scores_step1 = (
        calc_pooled_scores(counts_step1).set_index("impact")
        .reindex(scores_judge.index).fillna(0.0)
    )
    metrics = ["Precision", "Recall"]
    # number of cases each score is based on: precision -> prediction set, recall -> validation set
    n_col = {"Precision": ("n_pred", "pred."), "Recall": ("n", "valid.")}

    n_impacts = len(scores_judge)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.8 * n_impacts + 1.6))
    fig = ax.figure
    fig.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # horizontal grouped bars: one row per impact column, first impact on top
    bar_height = 0.34
    gap = 0.03  # surface gap between the two bars of a group
    y = np.arange(n_impacts)[::-1]
    offsets = {"Precision": bar_height / 2 + gap / 2, "Recall": -(bar_height / 2 + gap / 2)}

    for metric in metrics:
        values_judge = scores_judge[metric.lower()].to_numpy()
        values_step1 = scores_step1[metric.lower()].to_numpy()
        y_metric = y + offsets[metric]

        # LLM-judge: filled bars, narrower than the Step 1 bars so that the dashed Step 1 outline stays visible
        ax.barh(
            y_metric, values_judge, height=bar_height * 0.6,
            color=SERIES_COLORS[metric], label=f"{metric} (LLM-judge)", zorder=2,
        )
        # Step 1: empty, dashed bars framing the LLM-judge bars
        ax.barh(
            y_metric, values_step1, height=bar_height,
            fill=False, edgecolor=SERIES_COLORS[metric], linestyle=(0, (3, 2)), linewidth=1.3,
            label=f"{metric} (Step 1)", zorder=3,
        )

        # direct value labels behind the longer of both bars, in text ink
        col, set_name = n_col[metric]
        n_judge = scores_judge[col].to_numpy()
        n_step1 = scores_step1[col].to_numpy()
        for yi, vj, vs, nj, ns in zip(y_metric, values_judge, values_step1, n_judge, n_step1):
            label = f"{vj:.2f}"
            if show_counts:
                label += f" · {int(nj)} {set_name}"
            label += f"  (Step 1: {vs:.2f}" + (f" · {int(ns)} {set_name})" if show_counts else ")")
            ax.text(max(vj, vs) + 0.01, yi, label, va="center", ha="left", fontsize=8, color=INK_SECONDARY)

    ax.set_yticks(y)
    ax.set_yticklabels(list(scores_judge.index), color=INK_PRIMARY, fontsize=9.5)
    ax.set_xlim(0, 1.6 if show_counts else 1.3)  # room for the bar labels
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Score", color=INK_MUTED, fontsize=9)
    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=8.5, length=0)
    ax.tick_params(axis="y", length=0)

    # recessive grid and axes
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    for side in ["top", "right", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(BASELINE)

    ax.set_title(title, loc="left", color=INK_PRIMARY, fontsize=11.5, pad=40)
    handles, labels = ax.get_legend_handles_labels()
    if show_counts:
        labels = [
            f"{lab} (n = {'predicted' if lab.startswith('Precision') else 'validation'} cases)" for lab in labels
        ]
    ax.legend(
        handles, labels, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False,
        fontsize=8.5, labelcolor=INK_SECONDARY, handlelength=1.6, borderaxespad=0.2,
    )

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)

    return ax
