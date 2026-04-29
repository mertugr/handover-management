"""
Synthetic dataset generator.
Simulates NUM_USERS mobile users for NUM_STEPS seconds each and records
position, speed, direction, RSSI from all 9 cells, and the optimal cell
at each time step. Results are cached to data/traces.csv.
"""

import hashlib
import json
import os
import numpy as np
import pandas as pd

import config
from simulation.mobility  import MobileUser
from simulation.rssi      import rssi_all_cells, best_cell_by_rssi
from simulation.cell_grid import NUM_CELLS
from config import NUM_USERS, NUM_STEPS, BURN_IN, LOOKAHEAD, RANDOM_SEED

RSSI_COLS  = [f"rssi_{i}"       for i in range(NUM_CELLS)]
TREND_COLS = [f"rssi_trend_{i}" for i in range(NUM_CELLS)]

SAVE_PATH = os.path.join(os.path.dirname(__file__), "traces.csv")
META_PATH = SAVE_PATH + ".meta.json"

# Constants whose value changes the simulated traces. Touching any of these
# must invalidate the cached dataset; controller / model / split knobs are
# excluded so tweaking them doesn't trigger a slow regeneration.
_FINGERPRINT_KEYS = (
    "RANDOM_SEED",
    # Grid geometry
    "GRID_ROWS", "GRID_COLS", "GRID_WIDTH", "GRID_HEIGHT",
    # Mobility model
    "MIN_SPEED_MPS", "MAX_SPEED_MPS", "TIME_STEP_S",
    "PAUSE_PROB", "MIN_PAUSE_S", "MAX_PAUSE_S",
    "DIR_NOISE_STD", "MARGIN",
    # RSSI / path-loss
    "P_TX_DBM", "CABLE_LOSS_DB", "ANTENNA_GAIN",
    "PL_CONST", "PL_SLOPE", "SHADOWING_STD",
    "MIN_RSSI", "MAX_RSSI", "MIN_DISTANCE",
    # Dataset shape
    "NUM_USERS", "NUM_STEPS", "BURN_IN", "LOOKAHEAD",
)


def _config_fingerprint() -> str:
    """Stable hash of every config knob that affects the generated dataset."""
    payload = {k: getattr(config, k) for k in _FINGERPRINT_KEYS}
    payload["NUM_CELLS"] = NUM_CELLS  # derived from GRID_ROWS * GRID_COLS
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _simulate_user(uid: int) -> list[dict]:
    """Simulate one user and return a list of feature dicts (one per step)."""
    rng  = np.random.RandomState(RANDOM_SEED + uid * 1000)
    user = MobileUser(uid, rng=rng)

    # Discard the first BURN_IN steps to avoid cold-start bias
    for _ in range(BURN_IN):
        user.step()

    # Record NUM_STEPS + LOOKAHEAD steps so we can compute the future cell label
    history = []
    for _ in range(NUM_STEPS + LOOKAHEAD):
        rssi = rssi_all_cells(user.position, add_noise=True, rng=rng)
        cell = best_cell_by_rssi(user.position)
        history.append({
            "x":         float(user.position[0]),
            "y":         float(user.position[1]),
            "speed":     float(user.speed),
            "direction": float(user.direction),
            "rssi":      rssi,
            "cell":      cell,
        })
        user.step()

    # Build the final feature records
    records = []
    for t in range(NUM_STEPS):
        h        = history[t]
        h_prev   = history[t - 1] if t > 0 else history[t]
        h_future = history[t + LOOKAHEAD]

        rssi_trend = h["rssi"] - h_prev["rssi"]

        record = {
            "user_id":       uid,
            "time_step":     t,
            "x":             h["x"],
            "y":             h["y"],
            "speed":         h["speed"],
            # Direction encoded as sin/cos to avoid 0/360 discontinuity
            "direction_sin": float(np.sin(h["direction"])),
            "direction_cos": float(np.cos(h["direction"])),
            "current_cell":  h["cell"],
            "next_cell":     h_future["cell"],  # supervised label
        }
        for i in range(NUM_CELLS):
            record[RSSI_COLS[i]]  = float(h["rssi"][i])
            record[TREND_COLS[i]] = float(rssi_trend[i])

        records.append(record)

    return records


def generate_dataset(save: bool = True) -> pd.DataFrame:
    """Generate the full dataset for all users and optionally save to CSV."""
    print(f"[DataGen] Generating {NUM_USERS} users x {NUM_STEPS} steps ...")

    all_records = []
    for uid in range(NUM_USERS):
        all_records.extend(_simulate_user(uid))
        if (uid + 1) % 10 == 0:
            print(f"  ... {uid + 1}/{NUM_USERS} users done")

    df = pd.DataFrame(all_records)

    assert len(df) == NUM_USERS * NUM_STEPS
    assert df["next_cell"].nunique() >= 7, "Too few cells covered – check grid"
    assert df.isna().sum().sum() == 0, "NaN values found in dataset"

    print(f"[DataGen] Shape: {df.shape}")

    if save:
        df.to_csv(SAVE_PATH, index=False)
        with open(META_PATH, "w") as f:
            json.dump({"fingerprint": _config_fingerprint()}, f)
        print(f"[DataGen] Saved to {SAVE_PATH}")

    return df


def load_or_generate(force_regenerate: bool = False) -> pd.DataFrame:
    """Load cached traces.csv if it exists, otherwise generate it.

    The cache is invalidated automatically when any config constant that
    affects the dataset shape (NUM_USERS, NUM_STEPS, BURN_IN, LOOKAHEAD,
    NUM_CELLS, RANDOM_SEED) changes.
    """
    if not force_regenerate and os.path.exists(SAVE_PATH):
        cached_fp = None
        if os.path.exists(META_PATH):
            try:
                with open(META_PATH) as f:
                    cached_fp = json.load(f).get("fingerprint")
            except (OSError, ValueError):
                cached_fp = None

        if cached_fp == _config_fingerprint():
            print(f"[DataGen] Loading cached dataset ...")
            return pd.read_csv(SAVE_PATH)

        print("[DataGen] Config changed since last run - regenerating dataset")

    return generate_dataset(save=True)
