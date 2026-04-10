"""
data/mock_data_generator.py
============================
WHY: The proposal mentions using synthetic traces for training and evaluation
     (GeoLife is listed as optional).  Generating our own synthetic data means
     the project is 100% offline, reproducible, and avoids any data-download
     dependency.  The generator is the only place in the codebase that calls the
     mobility and RSSI simulation – all downstream modules simply consume the
     resulting DataFrame.

WHAT:
  - Spawns NUM_USERS mobile users, each simulated for NUM_STEPS seconds.
  - At every time step records:
      * Position (x, y) – for visualisation
      * Speed, direction (sin/cos) – mobility features
      * RSSI from all 9 cells – primary physical-layer features
      * RSSI trend (delta from previous step) – temporal context feature
      * current_cell – cell serving the user at this step (label at t)
      * next_cell    – cell that will serve the user LOOKAHEAD steps later
                       (the supervised classification target)
  - Saves the dataset to data/traces.csv so it is reusable across runs.
  - Also returns the raw DataFrame so main.py can pass it directly without
    reading from disk.

Design choices:
  - LOOKAHEAD = 5 s: far enough ahead that proactive HO is meaningful,
    short enough that the future cell is still predictable.
  - Seeded RNG per user → fully reproducible dataset.
  - We deliberately introduce RSSI measurement noise so the ML model must
    learn robustness, not just memorise distance thresholds.
"""

import os
import numpy as np
import pandas as pd

from simulation.mobility import MobileUser
from simulation.rssi     import rssi_all_cells, best_cell_by_rssi
from simulation.cell_grid import NUM_CELLS

# ── Dataset parameters ─────────────────────────────────────────────────────────
NUM_USERS  = 60    # total mobile users to simulate
NUM_STEPS  = 700   # seconds per user (burn-in + recording)
BURN_IN    = 100   # initial steps discarded to avoid cold-start bias
LOOKAHEAD  = 5     # steps ahead for the 'next_cell' label
RANDOM_SEED = 42

# ── Column names ───────────────────────────────────────────────────────────────
RSSI_COLS  = [f"rssi_{i}"       for i in range(NUM_CELLS)]
TREND_COLS = [f"rssi_trend_{i}" for i in range(NUM_CELLS)]

SAVE_PATH = os.path.join(os.path.dirname(__file__), "traces.csv")


def _simulate_user(uid: int) -> list[dict]:
    """
    Simulate one user for BURN_IN + NUM_STEPS + LOOKAHEAD ticks and return a
    list of feature dictionaries (one per recorded time step).

    WHY separate function: makes it easy to parallelise later and keeps the
    main generator function readable.
    """
    rng  = np.random.RandomState(RANDOM_SEED + uid * 1000)
    user = MobileUser(uid, rng=rng)

    # ── Warm-up: run BURN_IN steps but discard them ────────────────────────────
    for _ in range(BURN_IN):
        user.step()

    # ── Recording: NUM_STEPS + LOOKAHEAD steps ─────────────────────────────────
    history = []
    for _ in range(NUM_STEPS + LOOKAHEAD):
        rssi  = rssi_all_cells(user.position, add_noise=True, rng=rng)
        cell  = best_cell_by_rssi(user.position)   # noiseless ground truth
        history.append({
            "x":          float(user.position[0]),
            "y":          float(user.position[1]),
            "speed":      float(user.speed),
            "direction":  float(user.direction),
            "rssi":       rssi,
            "cell":       cell,
        })
        user.step()

    # ── Build feature records ──────────────────────────────────────────────────
    records = []
    for t in range(NUM_STEPS):
        h          = history[t]
        h_prev     = history[t - 1] if t > 0 else history[t]
        h_future   = history[t + LOOKAHEAD]

        rssi_trend = h["rssi"] - h_prev["rssi"]   # positive = signal growing

        record = {
            "user_id":       uid,
            "time_step":     t,
            "x":             h["x"],
            "y":             h["y"],
            "speed":         h["speed"],
            # Encode direction as sin/cos to avoid 0°/360° discontinuity
            "direction_sin": float(np.sin(h["direction"])),
            "direction_cos": float(np.cos(h["direction"])),
            "current_cell":  h["cell"],
            "next_cell":     h_future["cell"],   # supervised label
        }
        for i in range(NUM_CELLS):
            record[RSSI_COLS[i]]  = float(h["rssi"][i])
            record[TREND_COLS[i]] = float(rssi_trend[i])

        records.append(record)

    return records


def generate_dataset(save: bool = True) -> pd.DataFrame:
    """
    Generate the full synthetic dataset.

    Parameters
    ----------
    save : bool – if True, writes traces.csv to the data/ directory.

    Returns
    -------
    df : pd.DataFrame with shape (NUM_USERS * NUM_STEPS, n_features)
    """
    print(f"[DataGen] Generating {NUM_USERS} users × {NUM_STEPS} steps "
          f"(lookahead={LOOKAHEAD}s) …")

    all_records = []
    for uid in range(NUM_USERS):
        all_records.extend(_simulate_user(uid))
        if (uid + 1) % 10 == 0:
            print(f"  … {uid + 1}/{NUM_USERS} users done")

    df = pd.DataFrame(all_records)

    # ── Sanity checks ──────────────────────────────────────────────────────────
    assert len(df) == NUM_USERS * NUM_STEPS, "Unexpected row count"
    n_labels = df["next_cell"].nunique()
    assert n_labels >= 7, (
        f"Only {n_labels}/9 cells appear as labels – grid coverage too sparse"
    )
    if n_labels < NUM_CELLS:
        missing = set(range(NUM_CELLS)) - set(df["next_cell"].unique())
        print(f"[DataGen] Warning: cells {missing} never appear as next_cell "
              f"(corner cells with low coverage – acceptable)")
    assert df.isna().sum().sum() == 0, "NaN values found in dataset"

    print(f"[DataGen] Dataset shape: {df.shape}")
    print(f"[DataGen] Label distribution (next_cell):\n"
          f"{df['next_cell'].value_counts().sort_index().to_string()}")

    if save:
        df.to_csv(SAVE_PATH, index=False)
        print(f"[DataGen] Saved to {SAVE_PATH}")

    return df


def load_or_generate(force_regenerate: bool = False) -> pd.DataFrame:
    """
    Load traces.csv if it exists; otherwise generate and save it.
    WHY: Avoids re-running the 60-user simulation on every main.py call
         while still allowing forced regeneration for experiments.
    """
    if not force_regenerate and os.path.exists(SAVE_PATH):
        print(f"[DataGen] Loading cached dataset from {SAVE_PATH} …")
        return pd.read_csv(SAVE_PATH)
    return generate_dataset(save=True)
