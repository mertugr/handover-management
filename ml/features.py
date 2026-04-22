"""
Feature extraction for the Random Forest handover predictor.

The feature vector matches the dataset columns written by
data.mock_data_generator so the RF model can be trained offline from CSV
and queried online inside the ML handover controller.

Feature layout (length = 3 + 2*NUM_CELLS):
    speed, direction_sin, direction_cos,
    rssi_0 ... rssi_{N-1},
    rssi_trend_0 ... rssi_trend_{N-1}

Note on current_cell: the raw trace includes a `current_cell` column, but it
is NOT used as a feature. At training time the recorded current_cell is
always the noiseless optimal cell (best_cell_by_rssi with no shadowing),
while at inference time the ML controller may be serving a stale cell. Mixing
those two semantics teaches the RF to echo current_cell, so the controller
learns to "stay put" even when it shouldn't. Dropping it fixes that
distribution shift; the RSSI vector already encodes which cell is strongest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from simulation.cell_grid import NUM_CELLS

RSSI_COLS  = [f"rssi_{i}"       for i in range(NUM_CELLS)]
TREND_COLS = [f"rssi_trend_{i}" for i in range(NUM_CELLS)]

FEATURE_COLS: list[str] = (
    ["speed", "direction_sin", "direction_cos"]
    + RSSI_COLS
    + TREND_COLS
)

LABEL_COL = "next_cell"


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) arrays ready for scikit-learn."""
    missing = [c for c in FEATURE_COLS + [LABEL_COL] if c not in df.columns]
    if missing:
        raise KeyError(f"Dataset is missing required columns: {missing}")

    X = df[FEATURE_COLS].to_numpy(dtype=np.float64, copy=False)
    y = df[LABEL_COL].to_numpy(dtype=np.int64,   copy=False)
    return X, y


def build_feature_vector(speed: float, direction_rad: float,
                         rssi: np.ndarray, rssi_prev: np.ndarray) -> np.ndarray:
    """Build one feature row for online inference (ML handover controller)."""
    if rssi.shape != (NUM_CELLS,) or rssi_prev.shape != (NUM_CELLS,):
        raise ValueError(
            f"RSSI arrays must have shape ({NUM_CELLS},); "
            f"got rssi={rssi.shape}, rssi_prev={rssi_prev.shape}"
        )

    trend = rssi - rssi_prev
    vec = np.empty(len(FEATURE_COLS), dtype=np.float64)
    vec[0] = float(speed)
    vec[1] = float(np.sin(direction_rad))
    vec[2] = float(np.cos(direction_rad))
    vec[3:3 + NUM_CELLS]                = rssi
    vec[3 + NUM_CELLS:3 + 2 * NUM_CELLS] = trend
    return vec
