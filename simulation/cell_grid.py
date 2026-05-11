"""
3x3 base station grid layout and helper utilities.
Cell ID = row * GRID_COLS + col (row-major order).
"""

import numpy as np
from config import GRID_ROWS, GRID_COLS, GRID_WIDTH, GRID_HEIGHT

NUM_CELLS = GRID_ROWS * GRID_COLS  # 9 base stations
ISD       = GRID_WIDTH / GRID_COLS  # inter-site distance = 1000 m

# Positions of the 9 base stations (eNBs) in metres.
# Each cell is placed at the centre of its 1000 x 1000 m tile.
BASE_STATIONS = np.array([
    [500.0,  500.0],   # Cell 0
    [1500.0, 500.0],   # Cell 1
    [2500.0, 500.0],   # Cell 2
    [500.0,  1500.0],  # Cell 3
    [1500.0, 1500.0],  # Cell 4 (centre)
    [2500.0, 1500.0],  # Cell 5
    [500.0,  2500.0],  # Cell 6
    [1500.0, 2500.0],  # Cell 7
    [2500.0, 2500.0],  # Cell 8
], dtype=np.float64)


def cell_position(cell_id: int) -> np.ndarray:
    """Return the (x, y) position of the given base station."""
    return BASE_STATIONS[cell_id]
