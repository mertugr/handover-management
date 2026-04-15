"""
Visualisation module. Saves all plots as PNG files to results/.

Plots:
  01_cell_grid        - base station layout and sample user trajectory
  02_rssi_over_time   - RSSI signals over time with serving cell shading
  03_handover_timeline - serving cell vs optimal cell over time
  04_metric_comparison - bar chart of controller metrics
  05_rssi_heatmap     - spatial RSSI coverage for one base station
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


def plot_rssi_over_time(rssi_matrix: np.ndarray, served_thr: np.ndarray):
    """Plot RSSI signals over time with serving cell shading."""
    T      = len(rssi_matrix)
    t_axis = np.arange(T)

    fig, ax = plt.subplots(figsize=(13, 4))
    fig.suptitle("RSSI Over Time – Threshold Controller", fontsize=13, fontweight="bold")

    for i in range(NUM_CELLS):
        ax.plot(t_axis, rssi_matrix[:, i],
                color=CELL_COLOURS[i], linewidth=0.7, alpha=0.6, label=f"Cell {i}")

    prev, start = served_thr[0], 0
    for t in range(1, T):
        if served_thr[t] != prev or t == T - 1:
            ax.axvspan(start, t, alpha=0.12, color=CELL_COLOURS[prev])
            start, prev = t, served_thr[t]

    ax.set_ylabel("RSSI (dBm)")
    ax.set_xlabel("Time Step (s)")
    ax.set_ylim(-120, -30)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    handles = [Line2D([0], [0], color=CELL_COLOURS[i], linewidth=2, label=f"Cell {i}")
               for i in range(NUM_CELLS)]
    fig.legend(handles=handles, loc="right", fontsize=8, title="Cell", title_fontsize=9)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    _savefig("02_rssi_over_time", fig)


def plot_handover_timeline(served_thr: np.ndarray, true_cells: np.ndarray):
    """Plot serving cell assignment over time vs ground truth."""
    T      = len(true_cells)
    t_axis = np.arange(T)

    fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
    fig.suptitle("Serving Cell Assignment Over Time", fontsize=13, fontweight="bold")

    for ax, data, label, colour in zip(
            axes,
            [true_cells, served_thr],
            ["Optimal (Ground Truth)", "Threshold Baseline"],
            ["gray", "tomato"]):
        ax.step(t_axis, data, where="post", color=colour, linewidth=1.5)
        ax.set_ylabel("Cell ID")
        ax.set_yticks(range(NUM_CELLS))
        ax.set_title(label, fontsize=10)
        ax.grid(True, linewidth=0.3, alpha=0.5)

    axes[-1].set_xlabel("Time Step (s)")
    plt.tight_layout()
    _savefig("03_handover_timeline", fig)


def plot_metric_comparison(thr_metrics):
    """Bar chart of controller performance metrics."""
    metric_pairs = [
        ("Ping-Pong\nCount",        "ping_pong_count",       ""),
        ("Unnecessary\nHO Count",   "unnecessary_count",     ""),
        ("Total\nInterruption (ms)","total_interruption_ms", "ms"),
        ("HO Rate\n(per 100 steps)","ho_rate_per_100_steps", ""),
        ("Success\nRate (%)",       "success_rate_pct",      "%"),
    ]
    labels   = [m[0] for m in metric_pairs]
    thr_vals = [getattr(thr_metrics, m[1]) for m in metric_pairs]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(np.arange(len(labels)), thr_vals, 0.5,
                  label="Threshold", color="tomato", alpha=0.85)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Value")
    ax.set_title("Threshold Handover – Performance Metrics",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
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
