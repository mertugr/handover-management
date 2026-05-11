"""
Visualisation module. Saves all plots as PNG files to results/.

Plots:
  01_cell_grid            - base station layout and ML-chosen serving cell along trajectory
  02_rssi_over_time       - RSSI signals over time, with serving-cell shading for both controllers
  03_handover_timeline    - optimal vs threshold vs ML serving cell over time
  04_metric_comparison    - grouped bar chart of Threshold vs ML metrics
  05_rssi_heatmap         - spatial RSSI coverage for one base station
  06_confusion_matrix     - Random Forest test-set confusion matrix
  07_feature_importance   - Random Forest feature importances
"""

import os
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from simulation.cell_grid import BASE_STATIONS, NUM_CELLS, GRID_WIDTH, GRID_HEIGHT
from simulation.rssi      import rssi_from_cell

RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "results")
CELL_COLOURS = plt.cm.tab10(np.linspace(0, 0.9, NUM_CELLS))

os.makedirs(RESULTS_DIR, exist_ok=True)


def _savefig(name: str, fig: plt.Figure):
    path = os.path.join(RESULTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved -> {path}")


def plot_cell_grid(trajectory_xy: Optional[np.ndarray] = None,
                   serving_cells: Optional[np.ndarray] = None):
    """Plot the 3x3 base station layout with an optional user trajectory."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, GRID_WIDTH)
    ax.set_ylim(0, GRID_HEIGHT)
    ax.set_aspect("equal")
    ax.set_title("3x3 Cellular Grid", fontsize=13, fontweight="bold")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    for x in [1000, 2000]:
        ax.axvline(x, color="lightgray", linewidth=0.8, linestyle="--")
    for y in [1000, 2000]:
        ax.axhline(y, color="lightgray", linewidth=0.8, linestyle="--")

    for i, (bx, by) in enumerate(BASE_STATIONS):
        ax.plot(bx, by, marker="^", markersize=14, color=CELL_COLOURS[i],
                zorder=5, markeredgecolor="black", markeredgewidth=0.8)
        ax.text(bx + 35, by + 35, f"Cell {i}", fontsize=8, zorder=6)

    if trajectory_xy is not None and len(trajectory_xy) > 0:
        xs, ys = trajectory_xy[:, 0], trajectory_xy[:, 1]
        if serving_cells is not None:
            for t in range(len(xs) - 1):
                ax.plot(xs[t:t+2], ys[t:t+2],
                        color=CELL_COLOURS[serving_cells[t]], linewidth=1.0, alpha=0.7)
        else:
            ax.plot(xs, ys, color="navy", linewidth=1.0, alpha=0.6)
        ax.plot(xs[0],  ys[0],  "go", markersize=8, zorder=7)
        ax.plot(xs[-1], ys[-1], "rs", markersize=8, zorder=7)

    patches = [mpatches.Patch(color=CELL_COLOURS[i], label=f"Cell {i}")
               for i in range(NUM_CELLS)]
    patches += [
        Line2D([0], [0], marker="o", color="g", linestyle="", markersize=6, label="Start"),
        Line2D([0], [0], marker="s", color="r", linestyle="", markersize=6, label="End"),
    ]
    ax.legend(handles=patches, fontsize=7, loc="lower right",
              title="Serving Cell", title_fontsize=8)
    _savefig("01_cell_grid", fig)


def _shade_serving(ax, served: np.ndarray, T: int):
    """Shade the background of an axis with the colour of the serving cell."""
    prev, start = served[0], 0
    for t in range(1, T):
        if served[t] != prev:
            ax.axvspan(start, t, alpha=0.12, color=CELL_COLOURS[prev])
            start, prev = t, served[t]
    ax.axvspan(start, T, alpha=0.12, color=CELL_COLOURS[prev])


def plot_rssi_over_time(rssi_matrix: np.ndarray,
                        served_thr: np.ndarray,
                        served_ml:  np.ndarray):
    """Plot RSSI signals over time; two stacked panels for threshold and ML."""
    T      = len(rssi_matrix)
    t_axis = np.arange(T)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.suptitle("RSSI Over Time – Threshold vs ML Handover",
                 fontsize=13, fontweight="bold")

    for ax, served, title in zip(axes, [served_thr, served_ml],
                                 ["Threshold Baseline", "ML-Based"]):
        for i in range(NUM_CELLS):
            ax.plot(t_axis, rssi_matrix[:, i],
                    color=CELL_COLOURS[i], linewidth=0.7, alpha=0.6, label=f"Cell {i}")
        _shade_serving(ax, served, T)
        ax.set_ylabel("RSSI (dBm)")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-120, -30)
        ax.grid(True, linewidth=0.4, alpha=0.5)

    axes[-1].set_xlabel("Time Step (s)")
    handles = [Line2D([0], [0], color=CELL_COLOURS[i], linewidth=2, label=f"Cell {i}")
               for i in range(NUM_CELLS)]
    fig.legend(handles=handles, loc="right", fontsize=8,
               title="Cell", title_fontsize=9)
    plt.tight_layout(rect=[0, 0, 0.88, 0.96])
    _savefig("02_rssi_over_time", fig)


def plot_handover_timeline(served_thr: np.ndarray,
                           served_ml:  np.ndarray,
                           true_cells: np.ndarray):
    """Plot serving cell assignment over time for optimal/threshold/ML."""
    T      = len(true_cells)
    t_axis = np.arange(T)

    fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
    fig.suptitle("Serving Cell Assignment Over Time",
                 fontsize=13, fontweight="bold")

    for ax, data, label, colour in zip(
            axes,
            [true_cells, served_thr, served_ml],
            ["Optimal (Ground Truth)", "Threshold Baseline", "ML-Based"],
            ["gray", "tomato", "steelblue"]):
        ax.step(t_axis, data, where="post", color=colour, linewidth=1.5)
        ax.set_ylabel("Cell ID")
        ax.set_yticks(range(NUM_CELLS))
        ax.set_title(label, fontsize=10)
        ax.grid(True, linewidth=0.3, alpha=0.5)

    axes[-1].set_xlabel("Time Step (s)")
    plt.tight_layout()
    _savefig("03_handover_timeline", fig)


def plot_metric_comparison(thr_metrics, ml_metrics):
    """One small bar panel per metric so different scales don't crush each other."""
    metric_pairs = [
        ("Success Rate (%)",          "success_rate_pct",        "higher better"),
        ("Ping-Pong Count",           "ping_pong_count",         "lower better"),
        ("Unnecessary HO Count",      "unnecessary_count",       "lower better"),
        ("Total Handovers",           "total_handovers",         "lower better"),
        ("HO Rate / 100 steps",       "ho_rate_per_100_steps",   "lower better"),
        ("Total Interruption (ms)",   "total_interruption_ms",   "lower better"),
    ]

    n     = len(metric_pairs)
    cols  = 3
    rows  = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4 * rows))
    axes = axes.flatten()

    for ax, (label, attr, hint) in zip(axes, metric_pairs):
        thr_v = getattr(thr_metrics, attr)
        ml_v  = getattr(ml_metrics,  attr)
        bars  = ax.bar(["Threshold", "ML-Based"], [thr_v, ml_v],
                       color=["tomato", "steelblue"], alpha=0.85, width=0.55)
        ax.set_title(f"{label}\n({hint})", fontsize=10, fontweight="bold")
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)
        ax.margins(y=0.18)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f"{h:.2f}" if isinstance(h, float) and h % 1 else f"{int(h)}",
                    ha="center", va="bottom", fontsize=9)

    # Hide any unused panel(s).
    for ax in axes[len(metric_pairs):]:
        ax.axis("off")

    fig.suptitle("Threshold vs ML Handover – Performance Metrics",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _savefig("04_metric_comparison", fig)


def plot_rssi_heatmap(cell_id: int = 4):
    """Spatial RSSI heatmap for one base station using the path-loss model."""
    res = 80
    xs  = np.linspace(0, GRID_WIDTH,  res)
    ys  = np.linspace(0, GRID_HEIGHT, res)
    Z   = np.zeros((res, res))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            Z[j, i] = rssi_from_cell(np.array([x, y]), cell_id, add_noise=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(Z, origin="lower", extent=[0, GRID_WIDTH, 0, GRID_HEIGHT],
                   cmap="RdYlGn", vmin=-115, vmax=-40, aspect="auto")
    plt.colorbar(im, ax=ax, label="RSSI (dBm)")

    for i, (bx, by) in enumerate(BASE_STATIONS):
        marker = "^" if i == cell_id else "."
        size   = 12  if i == cell_id else 7
        ax.plot(bx, by, marker=marker, color="white", markersize=size,
                markeredgecolor="black", markeredgewidth=0.8, zorder=5)
        ax.text(bx + 30, by + 30, str(i), color="white", fontsize=7, zorder=6)

    ax.set_title(f"RSSI Heatmap – Cell {cell_id} (COST-231 Path-Loss)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.tight_layout()
    _savefig("05_rssi_heatmap", fig)


def plot_confusion_matrix(cm: np.ndarray, class_labels: list[int]):
    """Heatmap of the Random Forest test-set confusion matrix."""
    row_sum = cm.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = np.where(row_sum > 0, cm / row_sum, 0.0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Row-normalised fraction")

    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, fontsize=8)
    ax.set_yticklabels(class_labels, fontsize=8)
    ax.set_xlabel("Predicted next cell")
    ax.set_ylabel("True next cell")
    ax.set_title("Random Forest – Confusion Matrix (test set)",
                 fontsize=11, fontweight="bold")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if val > 0:
                colour = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{int(val)}", ha="center", va="center",
                        fontsize=7, color=colour)

    plt.tight_layout()
    _savefig("06_confusion_matrix", fig)


def plot_feature_importance(feature_names: list[str],
                            importances:  np.ndarray):
    """Horizontal bar chart of Random Forest feature importances."""
    order = np.argsort(importances)
    names_sorted = [feature_names[i] for i in order]
    vals_sorted  = importances[order]

    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.25)))
    ax.barh(np.arange(len(names_sorted)), vals_sorted,
            color="seagreen", alpha=0.85)
    ax.set_yticks(np.arange(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest – Feature Importance",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    _savefig("07_feature_importance", fig)
