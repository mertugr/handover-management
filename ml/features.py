"""
Feature extraction for the Random Forest handover predictor.

Feature set follows the project proposal: current cell ID, user speed,
movement direction, and RSSI values from every base station. The feature
vector matches the dataset columns written by data.mock_data_generator so
the RF model can be trained offline from CSV and queried online inside the
ML handover controller.

Feature layout (length = 4 + NUM_CELLS = 13 for a 3x3 grid):
    current_cell,
    speed,
    direction_sin, direction_cos,   # one "direction" feature, sin/cos encoded
    rssi_0 ... rssi_{N-1}
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from simulation.cell_grid import NUM_CELLS

RSSI_COLS = [f"rssi_{i}" for i in range(NUM_CELLS)]

FEATURE_COLS: list[str] = (
    ["current_cell", "speed", "direction_sin", "direction_cos"]
    + RSSI_COLS
)

LABEL_COL = "next_cell"

# Index of `current_cell` inside the feature vector (used by the controller
# to swap in its served cell at inference time without rebuilding the row).
CURRENT_CELL_IDX = 0


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) arrays ready for scikit-learn."""
    missing = [c for c in FEATURE_COLS + [LABEL_COL] if c not in df.columns]
    if missing:
        raise KeyError(f"Dataset is missing required columns: {missing}")

    X = df[FEATURE_COLS].to_numpy(dtype=np.float64, copy=False)
    y = df[LABEL_COL].to_numpy(dtype=np.int64,   copy=False)
    return X, y


def build_feature_vector(current_cell: int, speed: float,
                         direction_rad: float, rssi: np.ndarray) -> np.ndarray:
    """Build one feature row for online inference (ML handover controller)."""
    if rssi.shape != (NUM_CELLS,):
        raise ValueError(
            f"RSSI array must have shape ({NUM_CELLS},); got {rssi.shape}"
        )

    vec = np.empty(len(FEATURE_COLS), dtype=np.float64)
    vec[0] = float(current_cell)
    vec[1] = float(speed)
    vec[2] = float(np.sin(direction_rad))
    vec[3] = float(np.cos(direction_rad))
    vec[4:4 + NUM_CELLS] = rssi
    return vec
