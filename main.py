"""
Main pipeline entry point.

Steps:
  1. Generate or load the synthetic mobility dataset
  2. Run threshold handover simulation for each user
  3. Compute performance metrics
  4. Generate plots

Command-line flags:
  --regenerate   force re-generation of traces (ignore cache)
  --users N      number of users to evaluate (default 20)
  --plot-user ID user whose trace is shown in the timeline plots (default 0)
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config import RANDOM_SEED
from data.mock_data_generator    import load_or_generate, NUM_USERS
from handover.threshold_handover import ThresholdHandoverController
from evaluation.metrics          import compute_metrics
from simulation.cell_grid        import NUM_CELLS
from visualization.plots         import (plot_cell_grid, plot_rssi_over_time,
                                          plot_handover_timeline,
                                          plot_metric_comparison,
                                          plot_rssi_heatmap)


def step1_generate_data(regenerate: bool):
    print("\n" + "=" * 60)
    print("STEP 1 - Mobility Simulation & RSSI Generation")
    print("=" * 60)
    df = load_or_generate(force_regenerate=regenerate)
    print(f"  Dataset rows: {len(df):,}   columns: {len(df.columns)}")
    return df


def simulate_single_user(df, user_id):
    """Run threshold simulation for one user. Returns per-step arrays."""
    user_df = df[df["user_id"] == user_id].reset_index(drop=True)
    T       = len(user_df)

    rssi_cols   = [f"rssi_{i}" for i in range(NUM_CELLS)]
    rssi_matrix = user_df[rssi_cols].values
    true_cells  = user_df["current_cell"].values.astype(int)
    trajectory  = user_df[["x", "y"]].values

    thr_ctrl = ThresholdHandoverController()
    thr_ctrl.reset(initial_cell=int(true_cells[0]))
    served_thr = np.zeros(T, dtype=int)
    for t in range(T):
        served_thr[t], _ = thr_ctrl.process_step(t, rssi_matrix[t])

    return {
        "thr_log":    thr_ctrl.log,
        "served_thr": served_thr,
        "true_cells": true_cells,
        "rssi_matrix": rssi_matrix,
        "trajectory": trajectory,
        "T":          T,
    }


def step2_simulate_handovers(df, n_users: int, plot_user: int):
    print("\n" + "=" * 60)
    print(f"STEP 2 - Handover Simulation ({n_users} users)")
    print("=" * 60)

    all_thr_logs   = []
    all_served_thr = []
    all_true       = []
    all_rssi       = []
    total_steps    = 0
    plot_result    = None

    for uid in range(min(n_users, NUM_USERS)):
        res = simulate_single_user(df, uid)
        all_thr_logs.extend(res["thr_log"])
        all_served_thr.append(res["served_thr"])
        all_true.append(res["true_cells"])
        all_rssi.append(res["rssi_matrix"])
        total_steps += res["T"]
        if uid == plot_user:
            plot_result = res

    print(f"  Total steps : {total_steps:,}")
    print(f"  Total HOs   : {len(all_thr_logs)}")

    return (all_thr_logs,
            np.concatenate(all_served_thr),
            np.concatenate(all_true),
            np.concatenate(all_rssi, axis=0),
            total_steps, plot_result)


def step3_evaluate(thr_log, served_thr, true_cells, rssi_matrix, total_steps):
    print("\n" + "=" * 60)
    print("STEP 3 - Performance Evaluation")
    print("=" * 60)

    thr_metrics = compute_metrics(thr_log, total_steps,
                                  served_thr, true_cells, rssi_matrix)

    print(f"  Success rate     : {thr_metrics.success_rate_pct:.2f}%")
    print(f"  Ping-pong count  : {thr_metrics.ping_pong_count}")
    print(f"  Unnecessary HOs  : {thr_metrics.unnecessary_count}")
    print(f"  Total HOs        : {thr_metrics.total_handovers}")
    print(f"  Avg serving RSSI : {thr_metrics.avg_serving_rssi:.2f} dBm")

    return thr_metrics


def step4_visualise(plot_result, thr_metrics):
    print("\n" + "=" * 60)
    print("STEP 4 - Generating Plots")
    print("=" * 60)

    r = plot_result
    plot_rssi_heatmap(cell_id=4)
    plot_cell_grid(trajectory_xy=r["trajectory"], serving_cells=r["served_thr"])
    plot_rssi_over_time(r["rssi_matrix"], r["served_thr"])
    plot_handover_timeline(r["served_thr"], r["true_cells"])
    plot_metric_comparison(thr_metrics)

    print("\n  5 plots saved to results/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--users",      type=int, default=20)
    parser.add_argument("--plot-user",  type=int, default=0)
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)

    df = step1_generate_data(args.regenerate)

    thr_log, served_thr, true_cells, rssi_matrix, total_steps, plot_result = \
        step2_simulate_handovers(df, args.users, args.plot_user)

    thr_metrics = step3_evaluate(
        thr_log, served_thr, true_cells, rssi_matrix, total_steps)

    step4_visualise(plot_result, thr_metrics)

    print("\n" + "=" * 60)
    print("RESULTS – Threshold Baseline")
    print("=" * 60)
    print(f"  Users     : {args.users}")
    print(f"  Steps     : {total_steps:,}")
    print(f"  Success   : {thr_metrics.success_rate_pct:.2f}%")
    print(f"  Ping-pong : {thr_metrics.ping_pong_count}")
    print(f"  Unn. HOs  : {thr_metrics.unnecessary_count}")
    print(f"  Total HOs : {thr_metrics.total_handovers}")
    print("=" * 60)


if __name__ == "__main__":
    main()
