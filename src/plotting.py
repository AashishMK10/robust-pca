"""Generate IEEE-sized figures from the saved experiment data."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, ScalarFormatter

BLUE = "#155e89"
ORANGE = "#ab4929"
INK = "#203040"
COLUMN = 3.47
PAGE = 7.06
FIGURE_NAMES = ("reconstruction_error", "matrix_recovery", "corruption_detection", "convergence")

STYLE = {
    "font.family": "DejaVu Serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "axes.edgecolor": "#607181",
    "axes.labelcolor": INK,
    "axes.linewidth": 0.6,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "legend.fontsize": 7.5,
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
}


def _save(figure: Figure, path: Path) -> None:
    figure.savefig(path.with_suffix(".png"), dpi=320)
    figure.savefig(path.with_suffix(".pdf"), dpi=320)
    plt.close(figure)


def _curve_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d8dfe5", linewidth=0.55)
    ax.set_axisbelow(True)


def reconstruction_error(summary: list[dict], path: Path) -> None:
    """Compare the means and sample standard deviations at each density."""
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(COLUMN, 2.80), layout="constrained")
    percent = 100 * np.array([row["density"] for row in summary])
    methods = (("pca", "SVD", ORANGE, "s"), ("pcp", "PCP", BLUE, "o"))
    for ax, (name, label, color, marker) in zip(axes, methods, strict=True):
        means = np.array([row[f"{name}_error_mean"] for row in summary])
        sd = np.nan_to_num([row[f"{name}_error_sd"] for row in summary], nan=0.0)
        ax.errorbar(
            percent, means, yerr=[np.minimum(means, sd), sd],
            color=color, marker=marker, markersize=3.5, linewidth=1.15,
            capsize=2.2, clip_on=False,
        )
        _curve_axes(ax)
        ax.set(ylabel=rf"{label} $E_L$", ylim=(0, None))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(formatter)
    axes[-1].set(xlabel="Corrupted entries (%)", xticks=percent, xlim=(-0.8, 20.8))
    _save(fig, path)


def matrix_recovery(components: dict, path: Path) -> None:
    """Show component families and errors with shared symmetric color limits."""
    fig, axes = plt.subplots(2, 4, figsize=(PAGE, 3.95), layout="constrained")
    truth = components["l0"]
    pca_error = components["l_pca"] - truth
    pcp_error = components["l_pcp"] - truth
    groups = (
        (axes[0, :3], (truth, components["l_pca"], components["l_pcp"]),
         (r"(a) $L_0$", r"(b) $\hat L_{\mathrm{PCA}}$", r"(c) $\hat L_{\mathrm{PCP}}$"),
         "Low-rank value"),
        (axes[0, 3:], (components["observed"],), (r"(d) $M$",), "Observation"),
        (axes[1, :2], (components["s0"], components["s_pcp"]),
         (r"(e) $S_0$", r"(f) $\hat S_{\mathrm{PCP}}$"), "Sparse value"),
        (axes[1, 2:], (pca_error, pcp_error),
         (r"(g) $\hat L_{\mathrm{PCA}}-L_0$", r"(h) $\hat L_{\mathrm{PCP}}-L_0$"),
         "Signed error"),
    )
    for group_axes, matrices, titles, label in groups:
        limit = max(float(np.max(np.abs(values))) for values in matrices) or 1.0
        for ax, values, title in zip(group_axes, matrices, titles, strict=True):
            image = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
            ax.set_title(title)
            ax.set(xticks=[], yticks=[])
        colorbar = fig.colorbar(image, ax=list(group_axes), orientation="horizontal", pad=0.08, aspect=28)
        colorbar.locator = MaxNLocator(nbins=4, symmetric=True)
        colorbar.update_ticks()
        colorbar.set_label(label, labelpad=2)
    _save(fig, path)


def corruption_detection(components: dict, path: Path) -> None:
    """Display the true and predicted support and both kinds of error."""
    fig, axes = plt.subplots(2, 2, figsize=(COLUMN, 3.28), layout="constrained")
    actual = components["actual"]
    predicted = components["predicted"]
    for ax, mask, title, color in zip(
        axes.flat,
        (actual, predicted, ~actual & predicted, actual & ~predicted),
        ("(a) True support", "(b) Predicted support", "(c) False positives", "(d) False negatives"),
        (BLUE, BLUE, ORANGE, "#743885"),
        strict=True,
    ):
        ax.imshow(mask, cmap=ListedColormap(["#f4f7f9", color]), vmin=0, vmax=1, interpolation="nearest")
        count = int(mask.sum())
        ax.set_title(f"{title}\n{count:,} entries")
        ax.set(xticks=[], yticks=[])
    _save(fig, path)


def convergence(history: list[dict], tolerance: float, path: Path) -> None:
    """Plot both stopping residuals for the preselected representative trial."""
    fig, ax = plt.subplots(figsize=(COLUMN, 2.38), layout="constrained")
    iterations = [int(row["iteration"]) for row in history]
    _curve_axes(ax)
    ax.plot(
        iterations, [float(row["relative_primal"]) for row in history],
        color=BLUE, linewidth=1.35, label="Relative primal",
    )
    ax.plot(
        iterations, [float(row["relative_dual"]) for row in history],
        color=ORANGE, linestyle="--", linewidth=1.35, label="Relative dual",
    )
    ax.axhline(tolerance, color="#586575", linestyle=":", linewidth=1.15, label="Stopping tolerance")
    ax.set(xlabel="ADMM iteration", ylabel="Normalized residual")
    ax.set_yscale("symlog", linthresh=tolerance / 100)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper right", frameon=True, edgecolor="#d8dfe5", framealpha=0.95)
    _save(fig, path)


def make_figures(
    summary: list[dict], components: dict, history: list[dict], tolerance: float, destination: Path,
) -> None:
    """Export matching 320-DPI PNG and vector PDF figures."""
    destination.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        reconstruction_error(summary, destination / FIGURE_NAMES[0])
        matrix_recovery(components, destination / FIGURE_NAMES[1])
        corruption_detection(components, destination / FIGURE_NAMES[2])
        convergence(history, tolerance, destination / FIGURE_NAMES[3])
