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

CELL_LABELS = [f"Cell-{i}" for i in range(NUM_CELLS)]


def cell_position(cell_id: int) -> np.ndarray:
    """Return the (x, y) position of the given base station."""
    return BASE_STATIONS[cell_id]


def nearest_cell(position: np.ndarray) -> int:
    """Return the cell ID closest to the given position (Euclidean distance)."""
    dists = np.linalg.norm(BASE_STATIONS - position, axis=1)
    return int(np.argmin(dists))


def get_neighbours(cell_id: int, max_dist: float = ISD * 1.5) -> list:
    """Return cell IDs within max_dist metres of the given cell."""
    origin = BASE_STATIONS[cell_id]
    return [i for i, pos in enumerate(BASE_STATIONS)
            if i != cell_id and np.linalg.norm(pos - origin) <= max_dist]
