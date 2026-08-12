"""Figures for the bakeoff.

Light-mode PNGs, deliberately a single committed look rather than a theme pair, since these
are checked into the repo and embedded in `REPORT.md`.

Colour is assigned by *job*, never cycled: the three model families get the three
categorical slots, and the two diagnosis classes get the first two. Curve plots show at
most four series and direct-label every one of them, which is also what discharges the
aqua slot's sub-3:1 contrast against the light surface.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

import diagnostics
from data import Dataset
from models import BASELINE, EULAR_MAX_SCORE, MONOTONE, UNCONSTRAINED, ModelSpec

mpl.use("Agg")
import matplotlib.pyplot as plt  # the Agg backend above must be selected before this loads

FIGURE_DIR = Path(__file__).parent / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e6e5e0"

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
FAMILY_COLOR = {MONOTONE: SERIES[0], UNCONSTRAINED: SERIES[1], BASELINE: SERIES[2]}
CLASS_COLOR = {1: SERIES[0], 0: SERIES[1]}

STATUS_BAD = "#e34948"


def _style() -> None:
    """Apply the shared recessive-chrome style to matplotlib's global state."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "figure.dpi": 130,
            "font.size": 9,
        }
    )


def _finish(fig: plt.Figure, name: str) -> Path:
    """Trim spines, save under `FIGURE_DIR`, and close the figure."""
    for ax in fig.axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_data_profile(dataset: Dataset, separation: pd.DataFrame) -> Path:
    """Class-conditional criterion rates and the pattern-frequency tail.

    Args:
        dataset: The `d9` feature set.
        separation: Output of `diagnostics.separation_report`.

    Returns:
        Path to the written PNG.
    """
    _style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6), width_ratios=[1.15, 1])

    rows = separation.iloc[: len(dataset.criterion_index)]
    y = np.arange(len(rows))
    left.barh(y + 0.19, rows["p_given_sle"], height=0.34, color=CLASS_COLOR[1])
    left.barh(y - 0.19, rows["p_given_not_sle"], height=0.34, color=CLASS_COLOR[0])
    left.set_yticks(y, rows["feature"])
    left.invert_yaxis()
    left.set_xlabel("proportion of the group reporting the criterion")
    left.set_title("Criterion rates by diagnosis")
    left.text(
        rows["p_given_sle"].iloc[0] + 0.012,
        0.19,
        "SLE",
        color=CLASS_COLOR[1],
        fontsize=9,
        va="center",
        fontweight="bold",
    )
    left.text(
        rows["p_given_not_sle"].iloc[0] + 0.012,
        -0.19,
        "not SLE",
        color=CLASS_COLOR[0],
        fontsize=9,
        va="center",
        fontweight="bold",
    )
    inverted = np.flatnonzero(rows["p_given_not_sle"] > rows["p_given_sle"])
    worst = int(max(inverted, key=lambda i: rows["p_given_not_sle"].iloc[i]))
    names = " and ".join(rows["feature"].iloc[i] for i in inverted)
    left.annotate(
        f"{names} run the wrong way:\ncommoner in the controls than in the cases",
        xy=(rows["p_given_not_sle"].iloc[worst], worst - 0.19),
        xytext=(0.30, worst - 2.35),
        color=STATUS_BAD,
        fontsize=8.5,
        arrowprops={"arrowstyle": "-|>", "color": STATUS_BAD, "lw": 1.2},
    )

    frame = pd.DataFrame(dataset.x[:, : len(dataset.criterion_index)])
    frame["y"] = dataset.y
    grouped = frame.groupby(list(range(len(dataset.criterion_index))))["y"]
    counts = grouped.agg(["size", "sum"]).sort_values("size", ascending=False).head(8)
    labels = ["".join("1" if v else "0" for v in idx) for idx in counts.index]
    pos = np.arange(len(counts))
    right.bar(pos, counts["sum"], width=0.62, color=CLASS_COLOR[1])
    right.bar(
        pos,
        counts["size"] - counts["sum"],
        width=0.62,
        bottom=counts["sum"],
        color=CLASS_COLOR[0],
        linewidth=2,
        edgecolor=SURFACE,
    )
    right.set_xticks(pos, labels, rotation=45, ha="right", family="monospace", fontsize=7.5)
    right.set_ylabel("patients")
    right.set_title("The 8 most common tick-box patterns")
    right.set_xlabel("Fever · ACL · SCL/DL · Ulcer · Alopecia · Joint · Proteinuria")
    right.annotate(
        f"{int(counts['size'].iloc[0])} patients tick joint involvement alone\n"
        f"— {int(counts['size'].iloc[0] - counts['sum'].iloc[0])} of them are controls",
        xy=(0, counts["size"].iloc[0]),
        xytext=(1.4, counts["size"].iloc[0] * 0.86),
        color=INK_SECONDARY,
        fontsize=8.5,
        arrowprops={"arrowstyle": "-|>", "color": INK_MUTED, "lw": 1.2},
    )

    fig.suptitle(
        "Why this cohort cannot be taken at face value for booth screening",
        fontsize=13,
        fontweight="bold",
        color=INK,
        x=0.5,
        y=1.03,
    )
    return _finish(fig, "01_data_profile.png")


def curve_selection(summary: pd.DataFrame, predictions: pd.DataFrame) -> list[str]:
    """Pick at most four models for the curve plots: best per family, plus the shipped rule."""
    available = set(predictions["model"])
    chosen: list[str] = []
    for family in (MONOTONE, UNCONSTRAINED):
        rows = summary[(summary["family"] == family) & (summary["model"].isin(available))]
        if len(rows):
            chosen.append(rows.sort_values("f1_mean", ascending=False)["model"].iloc[0])
    for name in ("EULAR/ACR rule (shipped)", "Any non-joint criterion"):
        if name in available:
            chosen.append(name)
    return chosen[:4]


def series_color(position: int) -> str:
    """Colour for the nth curve in a plot.

    The categorical slots are never cycled, so a fourth series falls back to muted ink
    rather than repeating slot 1 and implying it is the same entity.

    Args:
        position: Zero-based index of the series.

    Returns:
        A hex colour.
    """
    return SERIES[position] if position < len(SERIES) else INK_MUTED


def _label_block(ax: plt.Axes, entries: list[tuple[str, str]], x: float, y: float) -> None:
    """Draw a stacked colour-coded label list at axes coordinates `(x, y)`.

    Used instead of leader lines on the curve plots: with four overlapping curves the
    leaders cross each other and the data, which reads worse than a plain keyed list.

    Args:
        ax: Axes to draw on.
        entries: `(text, colour)` pairs, drawn top to bottom.
        x: Left edge, in axes fraction.
        y: Top edge, in axes fraction.
    """
    for i, (text, color) in enumerate(entries):
        ax.text(
            x,
            y - 0.062 * i,
            text,
            transform=ax.transAxes,
            color=color,
            fontsize=8.5,
            fontweight="bold",
            va="top",
        )


def plot_roc(predictions: pd.DataFrame, summary: pd.DataFrame) -> Path:
    """Pooled out-of-fold ROC curves for the selected models."""
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.5, linestyle=(0, (4, 4)), zorder=1)
    ax.text(0.58, 0.53, "chance", color=INK_MUTED, fontsize=8.5, rotation=38, ha="center")

    entries = []
    for i, name in enumerate(curve_selection(summary, predictions)):
        rows = predictions[predictions["model"] == name].dropna(subset=["proba"])
        fpr, tpr, _ = roc_curve(rows["y"], rows["proba"])
        auc = summary.loc[summary["model"] == name, "roc_auc_mean"].iloc[0]
        ax.plot(fpr, tpr, color=series_color(i), zorder=3)
        entries.append((f"{name}  ·  AUC {auc:.3f}", series_color(i)))
    _label_block(ax, entries, x=0.30, y=0.30)

    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate (sensitivity)")
    ax.set_title("Out-of-fold ROC — best of each family vs the shipped rule")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    return _finish(fig, "02_roc_curves.png")


def plot_precision_recall(predictions: pd.DataFrame, summary: pd.DataFrame) -> Path:
    """Pooled out-of-fold precision-recall curves, with the prevalence baseline drawn in."""
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    first_model = predictions["model"].iloc[0]
    prevalence = float(predictions.loc[predictions["model"] == first_model, "y"].mean())
    ax.axhline(prevalence, color=GRID, linewidth=1.5, linestyle=(0, (4, 4)))
    ax.text(
        0.02,
        prevalence + 0.015,
        f"cohort base rate {prevalence:.3f}",
        color=INK_MUTED,
        fontsize=8.5,
    )

    entries = []
    for i, name in enumerate(curve_selection(summary, predictions)):
        rows = predictions[predictions["model"] == name].dropna(subset=["proba"])
        precision, recall, _ = precision_recall_curve(rows["y"], rows["proba"])
        ap = summary.loc[summary["model"] == name, "pr_auc_mean"].iloc[0]
        ax.plot(recall, precision, color=series_color(i), zorder=3)
        entries.append((f"{name}  ·  AP {ap:.3f}", series_color(i)))
    _label_block(ax, entries, x=0.03, y=0.30)

    ax.set_xlabel("recall (sensitivity)")
    ax.set_ylabel("precision")
    ax.set_title("Out-of-fold precision-recall")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    return _finish(fig, "03_pr_curves.png")


def plot_calibration(predictions: pd.DataFrame, summary: pd.DataFrame) -> Path:
    """Small-multiple reliability diagrams, one panel per model."""
    _style()
    names = list(summary.sort_values("f1_mean", ascending=False)["model"])
    names = [n for n in names if n in set(predictions["model"])]
    cols = 4
    rows_n = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.15 * cols, 3.0 * rows_n), squeeze=False)

    for ax, name in zip(axes.ravel(), names, strict=False):
        rows = predictions[predictions["model"] == name].dropna(subset=["proba"])
        stats = summary[summary["model"] == name].iloc[0]
        color = FAMILY_COLOR[stats["family"]]
        ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.4, linestyle=(0, (4, 4)))
        n_bins = min(10, rows["proba"].nunique())
        if n_bins >= 2:
            observed, predicted = calibration_curve(rows["y"], rows["proba"], n_bins=n_bins)
            ax.plot(
                predicted,
                observed,
                color=color,
                marker="o",
                markersize=5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
            )
        ax.set_title(name, fontsize=9.5)
        # Bottom-right: a reliability curve runs from bottom-left to top-right, so this
        # corner is the one reliably free of the data.
        ax.text(
            0.96,
            0.06,
            f"Brier {stats['brier_mean']:.3f}\nECE {stats['ece_mean']:.3f}",
            transform=ax.transAxes,
            fontsize=8,
            color=INK_SECONDARY,
            ha="right",
            va="bottom",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])

    for ax in axes.ravel()[len(names) :]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("predicted probability")
    for ax in axes[:, 0]:
        ax.set_ylabel("observed frequency")

    fig.suptitle(
        "Reliability at the cohort's 50% base rate — not at booth prevalence",
        fontsize=13,
        fontweight="bold",
        color=INK,
        y=1.0,
    )
    fig.tight_layout()
    return _finish(fig, "04_calibration.png")


def plot_metric_comparison(fold_metrics: pd.DataFrame, summary: pd.DataFrame) -> Path:
    """Dot-and-whisker leaderboard across the outer folds, one panel per metric."""
    _style()
    order = list(summary.sort_values("f1_mean")["model"])
    panels = [
        ("f1", "F1 at the tuned cut-off"),
        ("sensitivity", "Sensitivity (1 − missed cases)"),
        ("roc_auc", "ROC AUC"),
        ("brier", "Brier score (lower is better)"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 6.0), sharey=True)

    for ax, (metric, title) in zip(axes, panels, strict=True):
        for i, name in enumerate(order):
            rows = fold_metrics[fold_metrics["model"] == name][metric]
            family = summary.loc[summary["model"] == name, "family"].iloc[0]
            color = FAMILY_COLOR[family]
            # Every metric here is a rate, so a mean ± sd whisker is clipped at 1: drawing
            # a sensitivity of 1.08 would advertise an impossible value.
            ax.plot(
                [max(0.0, rows.mean() - rows.std()), min(1.0, rows.mean() + rows.std())],
                [i, i],
                color=color,
                linewidth=2,
                alpha=0.42,
                solid_capstyle="round",
            )
            ax.plot(
                rows.mean(),
                i,
                "o",
                color=color,
                markersize=7,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                zorder=3,
            )
        ax.set_title(title, fontsize=10)
        ax.set_yticks(range(len(order)), order)
        ax.grid(axis="y", visible=False)

    handles = [
        plt.Line2D([], [], color=FAMILY_COLOR[f], marker="o", linestyle="", markersize=7, label=f)
        for f in (MONOTONE, UNCONSTRAINED, BASELINE)
    ]
    fig.legend(handles=handles, loc="lower center", ncols=3, bbox_to_anchor=(0.5, -0.045))
    fig.suptitle(
        "Model leaderboard — mean ± sd over 50 outer folds (5-fold × 10 repeats)",
        fontsize=13,
        fontweight="bold",
        color=INK,
        y=1.0,
    )
    fig.tight_layout()
    return _finish(fig, "05_metric_comparison.png")


def plot_monotonicity(audit: pd.DataFrame) -> Path:
    """Lattice violations and the empty-form probability, side by side."""
    _style()
    audit = audit.sort_values("violation_rate")
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    y = np.arange(len(audit))
    colors = [FAMILY_COLOR[f] for f in audit["family"]]

    left.barh(y, audit["violation_rate"], height=0.62, color=colors)
    left.set_yticks(y, audit["model"])
    left.set_xlabel("share of the 448 lattice edges where adding a symptom lowers risk")
    left.set_title("Monotonicity violations")
    left.grid(axis="y", visible=False)

    right.barh(y, audit["p_all_zero"], height=0.62, color=colors)
    right.axvline(0.921, color=STATUS_BAD, linewidth=1.6, linestyle=(0, (4, 3)))
    right.text(
        0.905,
        len(audit) - 0.4,
        "v1's withdrawn model: 0.921",
        color=STATUS_BAD,
        fontsize=8.5,
        ha="right",
    )
    right.set_xlabel("predicted P(SLE) when nothing at all is ticked")
    right.set_title("The empty-form probe")
    right.grid(axis="y", visible=False)

    handles = [
        plt.Line2D([], [], color=FAMILY_COLOR[f], marker="s", linestyle="", markersize=8, label=f)
        for f in (MONOTONE, UNCONSTRAINED, BASELINE)
    ]
    fig.legend(handles=handles, loc="lower center", ncols=3, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(
        "Structural safety: every unconstrained model can be talked out of a referral",
        fontsize=13,
        fontweight="bold",
        color=INK,
        y=1.0,
    )
    fig.tight_layout()
    return _finish(fig, "06_monotonicity_audit.png")


def plot_threshold_sweep(predictions: pd.DataFrame, summary: pd.DataFrame) -> Path:
    """Sensitivity, specificity and precision against the cut-off, faceted by model."""
    _style()
    names = curve_selection(summary, predictions)
    fig, axes = plt.subplots(1, len(names), figsize=(3.7 * len(names), 4.2), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, name in zip(axes, names, strict=True):
        rows = predictions[predictions["model"] == name].dropna(subset=["proba"])
        grid = np.linspace(0, 1, 101)
        y = rows["y"].to_numpy()
        proba = rows["proba"].to_numpy()
        curves = {"sensitivity": [], "specificity": [], "precision": []}
        for t in grid:
            flagged = proba >= t
            tp = float(np.sum(flagged & (y == 1)))
            fp = float(np.sum(flagged & (y == 0)))
            tn = float(np.sum(~flagged & (y == 0)))
            fn = float(np.sum(~flagged & (y == 1)))
            curves["sensitivity"].append(tp / (tp + fn) if tp + fn else 0.0)
            curves["specificity"].append(tn / (tn + fp) if tn + fp else 0.0)
            curves["precision"].append(tp / (tp + fp) if tp + fp else np.nan)
        for i, (label, values) in enumerate(curves.items()):
            ax.plot(grid, values, color=SERIES[i], label=label)
        ax.axhline(0.90, color=INK_MUTED, linewidth=1.2, linestyle=(0, (3, 3)))
        ax.set_title(name, fontsize=9.5)
        ax.set_xlabel("cut-off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)

    axes[0].set_ylabel("rate")
    axes[0].text(0.30, 0.80, "90% sensitivity floor", color=INK_MUTED, fontsize=8)
    for i, label in enumerate(("sensitivity", "specificity", "precision")):
        axes[-1].text(
            1.03,
            0.92 - 0.09 * i,
            label,
            transform=axes[-1].transAxes,
            color=SERIES[i],
            fontsize=9,
            fontweight="bold",
        )
    fig.suptitle(
        "Where the cut-off lands — on this cohort's 50/50 mix",
        fontsize=13,
        fontweight="bold",
        color=INK,
        y=1.02,
    )
    fig.tight_layout()
    return _finish(fig, "07_threshold_sweep.png")


def plot_prevalence_shift(operating: pd.DataFrame) -> Path:
    """Positive predictive value against base rate, for the selected operating points."""
    _style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.6, 4.8))
    grid = np.logspace(-3.3, np.log10(0.6), 220)

    entries = []
    for i, (name, row) in enumerate(operating.iterrows()):
        color = series_color(i)
        tp = row["sensitivity"] * grid
        fp = (1 - row["specificity"]) * (1 - grid)
        left.plot(grid, tp / (tp + fp), color=color)
        right.plot(grid, (tp + fp) * 1000, color=color)
        entries.append(
            (
                f"{name}  ·  sens {row['sensitivity']:.2f} · spec {row['specificity']:.2f}",
                color,
            )
        )
    _label_block(left, entries, x=0.04, y=0.97)

    for ax, label in (
        (left, "positive predictive value"),
        (right, "people referred per 1,000 screened"),
    ):
        ax.set_xscale("log")
        ax.set_xlabel("true prevalence in the screened population")
        ax.set_ylabel(label)
        ax.axvspan(0.0005, 0.002, color=STATUS_BAD, alpha=0.07, zorder=0)
        ax.axvline(0.4975, color=GRID, linewidth=1.4, linestyle=(0, (4, 4)))
    left.text(0.0006, 0.44, "public\nbooth\n(~0.1%)", color=STATUS_BAD, fontsize=8.5)
    left.text(0.40, 0.10, "this cohort\n(49.8%)", color=INK_MUTED, fontsize=8.5, ha="right")
    left.set_ylim(0, 1)
    right.set_ylim(0, 1000)

    fig.suptitle(
        "The same model, re-read at the base rate it will actually meet",
        fontsize=13,
        fontweight="bold",
        color=INK,
        y=1.02,
    )
    fig.tight_layout()
    return _finish(fig, "08_prevalence_shift.png")


def plot_points_model(spec: ModelSpec, dataset: Dataset, criteria: list[dict]) -> Path:
    """The fitted non-negative logistic weights beside the published EULAR/ACR weights."""
    _style()
    model = clone(spec.estimator).fit(dataset.x, dataset.y)
    coefficients = model.coef_.ravel()[: len(dataset.criterion_index)]
    published = np.array([c["score"] for c in criteria], dtype=float)
    # Rescale the learned log-odds onto the published 0-18 range so the two point systems
    # are read on one axis; only the relative spacing is meaningful.
    scaled = coefficients / coefficients.max() * published.max()
    names = [c["key"] for c in criteria]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    y = np.arange(len(names))
    ax.barh(y + 0.19, published, height=0.34, color=SERIES[2])
    ax.barh(y - 0.19, scaled, height=0.34, color=SERIES[0])
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("points (learned weights rescaled to the published maximum)")
    ax.set_title("What the cohort would weight, versus what EULAR/ACR 2019 publishes")
    _label_block(
        ax,
        [
            ("learned from this cohort", SERIES[0]),
            ("EULAR/ACR 2019", SERIES[2]),
        ],
        x=0.52,
        y=0.30,
    )
    zeroed = [names[i] for i in np.flatnonzero(scaled <= 1e-6)]
    ax.annotate(
        f"the non-negativity constraint pins {' and '.join(zeroed)} at zero;\n"
        "unconstrained, this cohort would give them negative weight",
        xy=(0.12, names.index(zeroed[-1]) - 0.19),
        xytext=(0.52, 0.62),
        textcoords="axes fraction",
        color=STATUS_BAD,
        fontsize=8.5,
        arrowprops={"arrowstyle": "-|>", "color": STATUS_BAD, "lw": 1.2},
    )
    ax.set_xlim(0, EULAR_MAX_SCORE)
    return _finish(fig, "09_points_model.png")


def plot_all(
    dataset: Dataset,
    fold_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    predictions: pd.DataFrame,
    audit: pd.DataFrame,
    operating: pd.DataFrame,
    points_spec: ModelSpec,
    criteria: list[dict],
) -> list[Path]:
    """Render every figure.

    Args:
        dataset: The `d9` feature set.
        fold_metrics: Per-fold metrics for the `d9` models.
        summary: Aggregated `d9` leaderboard.
        predictions: Pooled out-of-fold predictions for the `d9` models.
        audit: Monotonicity audit for the `d9` models.
        operating: Sensitivity/specificity per model at the sensitivity-floor cut-off,
            indexed by model name.
        points_spec: The non-negative logistic spec, for the weights figure.
        criteria: `core.load_criteria()` output.

    Returns:
        Paths of the written PNGs, in figure order.
    """
    separation = diagnostics.separation_report(dataset)
    return [
        plot_data_profile(dataset, separation),
        plot_roc(predictions, summary),
        plot_precision_recall(predictions, summary),
        plot_calibration(predictions, summary),
        plot_metric_comparison(fold_metrics, summary),
        plot_monotonicity(audit),
        plot_threshold_sweep(predictions, summary),
        plot_prevalence_shift(operating),
        plot_points_model(points_spec, dataset, criteria),
    ]
