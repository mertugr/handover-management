"""
Main pipeline entry point.

Steps:
  1. Generate or load the synthetic mobility dataset
  2. Train (or load) the Random Forest predictor
  3. Simulate BOTH handover controllers (Threshold + ML) for each user
  4. Compute performance metrics for both controllers
  5. Generate plots comparing the two strategies

Command-line flags:
  --regenerate   force re-generation of traces (ignore cache)
  --retrain      force re-training of the Random Forest model
  --users N      number of users to evaluate (default 20). Use --users 60 for
                 the full dataset (slower; includes held-out test users).
  --plot-user ID user whose trace is shown in the timeline plots (default 0)
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config import RANDOM_SEED, NUM_USERS
from data.mock_data_generator    import load_or_generate
from handover.threshold_handover import ThresholdHandoverController
from handover.ml_handover        import MLHandoverController
from evaluation.metrics          import compute_metrics
from ml.trainer                  import train_or_load
from ml.predictor                import HandoverPredictor
from simulation.cell_grid        import NUM_CELLS
from visualization.plots         import (plot_cell_grid, plot_rssi_over_time,
                                          plot_handover_timeline,
                                          plot_metric_comparison,
                                          plot_rssi_heatmap,
                                          plot_confusion_matrix,
                                          plot_feature_importance)


def step1_generate_data(regenerate: bool):
    print("\n" + "=" * 60)
    print("STEP 1 - Mobility Simulation & RSSI Generation")
    print("=" * 60)
    df = load_or_generate(force_regenerate=regenerate)
    print(f"  Dataset rows: {len(df):,}   columns: {len(df.columns)}")
    return df


def step2_train_model(df, retrain: bool):
    print("\n" + "=" * 60)
    print("STEP 2 - Random Forest Training")
    print("=" * 60)
    model, report = train_or_load(df, force_retrain=retrain)
    print(f"  Train samples : {report.n_train:,}")
    print(f"  Val   samples : {report.n_val:,}")
    print(f"  Test  samples : {report.n_test:,}")
    print(f"  Train accuracy: {report.train_accuracy*100:.2f}%")
    print(f"  Val   accuracy: {report.val_accuracy*100:.2f}%")
    print(f"  Test  accuracy: {report.test_accuracy*100:.2f}%")
    print(f"  Test  F1 macro: {report.test_f1_macro:.3f}")
    return model, report


def simulate_single_user(df, user_id, predictor: HandoverPredictor):
    """Run BOTH controllers for one user. Returns per-step arrays."""
    user_df = df[df["user_id"] == user_id].reset_index(drop=True)
    T       = len(user_df)

    rssi_cols   = [f"rssi_{i}" for i in range(NUM_CELLS)]
    rssi_matrix = user_df[rssi_cols].values
    # Ground truth for evaluation: noiseless best-by-RSSI cell at each step.
    # (`current_cell` in the dataset is the controller-served cell — used as an
    # RF input feature, not as a label.)
    true_cells  = user_df["optimal_cell"].values.astype(int)
    trajectory  = user_df[["x", "y"]].values
    speeds      = user_df["speed"].values
    dir_sin     = user_df["direction_sin"].values
    dir_cos     = user_df["direction_cos"].values
    directions  = np.arctan2(dir_sin, dir_cos)

    initial_cell = int(true_cells[0])

    thr_ctrl = ThresholdHandoverController()
    thr_ctrl.reset(initial_cell=initial_cell, user_id=int(user_id))
    served_thr = np.zeros(T, dtype=int)
    for t in range(T):
        served_thr[t], _ = thr_ctrl.process_step(t, rssi_matrix[t])

    ml_ctrl = MLHandoverController(predictor)
    ml_ctrl.reset(initial_cell=initial_cell, user_id=int(user_id))
    # Batch RF inference for the whole trace (~100x faster than per-step).
    ml_ctrl.precompute(speeds, directions, rssi_matrix)
    served_ml = np.zeros(T, dtype=int)
    for t in range(T):
        served_ml[t], _ = ml_ctrl.process_step(
            t, float(speeds[t]), float(directions[t]), rssi_matrix[t]
        )

    return {
        "thr_log":     thr_ctrl.log,
        "ml_log":      ml_ctrl.log,
        "served_thr":  served_thr,
        "served_ml":   served_ml,
        "true_cells":  true_cells,
        "rssi_matrix": rssi_matrix,
        "trajectory":  trajectory,
        "T":           T,
    }


def step3_simulate_handovers(df, n_users: int, plot_user: int,
                              predictor: HandoverPredictor):
    print("\n" + "=" * 60)
    print(f"STEP 3 - Handover Simulation ({n_users} users)")
    print("=" * 60)

    all_thr_logs, all_ml_logs = [], []
    all_served_thr, all_served_ml = [], []
    all_true, all_rssi = [], []
    total_steps = 0
    plot_result = None

    for uid in range(min(n_users, NUM_USERS)):
        res = simulate_single_user(df, uid, predictor)
        all_thr_logs.extend(res["thr_log"])
        all_ml_logs.extend(res["ml_log"])
        all_served_thr.append(res["served_thr"])
        all_served_ml.append(res["served_ml"])
        all_true.append(res["true_cells"])
        all_rssi.append(res["rssi_matrix"])
        total_steps += res["T"]
        if uid == plot_user:
            plot_result = res

    print(f"  Total steps    : {total_steps:,}")
    print(f"  Threshold HOs  : {len(all_thr_logs)}")
    print(f"  ML HOs         : {len(all_ml_logs)}")

    return {
        "thr_log":     all_thr_logs,
        "ml_log":      all_ml_logs,
        "served_thr":  np.concatenate(all_served_thr),
        "served_ml":   np.concatenate(all_served_ml),
        "true_cells":  np.concatenate(all_true),
        "rssi_matrix": np.concatenate(all_rssi, axis=0),
        "total_steps": total_steps,
        "plot_result": plot_result,
    }


def step4_evaluate(results):
    print("\n" + "=" * 60)
    print("STEP 4 - Performance Evaluation")
    print("=" * 60)

    thr_metrics = compute_metrics(
        results["thr_log"], results["total_steps"],
        results["served_thr"], results["true_cells"], results["rssi_matrix"])

    ml_metrics = compute_metrics(
        results["ml_log"], results["total_steps"],
        results["served_ml"], results["true_cells"], results["rssi_matrix"])

    hdr = f"{'Metric':<28}{'Threshold':>14}{'ML':>14}"
    print(hdr)
    print("-" * len(hdr))
    rows = [
        ("Success rate (%)",      f"{thr_metrics.success_rate_pct:.2f}",
                                  f"{ml_metrics.success_rate_pct:.2f}"),
        ("Ping-pong count",       thr_metrics.ping_pong_count,
                                  ml_metrics.ping_pong_count),
        ("Unnecessary HOs",       thr_metrics.unnecessary_count,
                                  ml_metrics.unnecessary_count),
        ("Total HOs",             thr_metrics.total_handovers,
                                  ml_metrics.total_handovers),
        ("HO rate / 100 steps",   f"{thr_metrics.ho_rate_per_100_steps:.2f}",
                                  f"{ml_metrics.ho_rate_per_100_steps:.2f}"),
        ("Interruption (ms)",     f"{thr_metrics.total_interruption_ms:.0f}",
                                  f"{ml_metrics.total_interruption_ms:.0f}"),
        ("Avg serving RSSI (dBm)",f"{thr_metrics.avg_serving_rssi:.2f}",
                                  f"{ml_metrics.avg_serving_rssi:.2f}"),
    ]
    for name, t, m in rows:
        print(f"{name:<28}{str(t):>14}{str(m):>14}")

    return thr_metrics, ml_metrics


def step5_visualise(plot_result, thr_metrics, ml_metrics, train_report):
    print("\n" + "=" * 60)
    print("STEP 5 - Generating Plots")
    print("=" * 60)

    r = plot_result
    plot_rssi_heatmap(cell_id=4)
    plot_cell_grid(trajectory_xy=r["trajectory"],
                   serving_cells=r["served_ml"])
    plot_rssi_over_time(r["rssi_matrix"], r["served_thr"], r["served_ml"])
    plot_handover_timeline(r["served_thr"], r["served_ml"], r["true_cells"])
    plot_metric_comparison(thr_metrics, ml_metrics)

    plot_confusion_matrix(train_report.confusion, train_report.class_labels)
    plot_feature_importance(train_report.feature_names,
                            train_report.feature_importances)

    print("\n  Plots saved to results/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate", action="store_true",
                        help="force re-generation of synthetic traces")
    parser.add_argument("--retrain",    action="store_true",
                        help="force re-training of the Random Forest model")
    parser.add_argument("--users",      type=int, default=20,
                        help="number of users to evaluate (default: 20)")
    parser.add_argument("--plot-user",  type=int, default=0)
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)

    args.users = max(1, min(args.users, NUM_USERS))
    if not (0 <= args.plot_user < args.users):
        print(f"[WARN] --plot-user={args.plot_user} out of range "
              f"[0, {args.users}); clamping to 0")
        args.plot_user = 0

    df = step1_generate_data(args.regenerate)
    # Retrain whenever traces change to keep the model aligned with the data.
    model, train_report = step2_train_model(df, retrain=args.retrain or args.regenerate)
    predictor = HandoverPredictor(model)

    results = step3_simulate_handovers(df, args.users, args.plot_user, predictor)
    thr_metrics, ml_metrics = step4_evaluate(results)
    step5_visualise(results["plot_result"], thr_metrics, ml_metrics, train_report)

    print("\n" + "=" * 60)
    print("SUMMARY – Threshold vs ML Handover")
    print("=" * 60)
    print(f"  Users : {args.users}    Steps: {results['total_steps']:,}")
    print(f"  Success  : Threshold {thr_metrics.success_rate_pct:6.2f}% "
          f"| ML {ml_metrics.success_rate_pct:6.2f}%")
    print(f"  Ping-pong: Threshold {thr_metrics.ping_pong_count:6d}   "
          f"| ML {ml_metrics.ping_pong_count:6d}")
    print(f"  Unn. HOs : Threshold {thr_metrics.unnecessary_count:6d}   "
          f"| ML {ml_metrics.unnecessary_count:6d}")
    print(f"  Total HOs: Threshold {thr_metrics.total_handovers:6d}   "
          f"| ML {ml_metrics.total_handovers:6d}")
    print("=" * 60)


if __name__ == "__main__":
    main()
